"""
Stalker-Predictor — Baseline-Deviation Features
=================================================
Pro Coin live: aktueller Wert vs. coin_baseline_profile (hour×weekday×btc_regime)
→ z-score-Features (dev_<metric>_z).

Quellen:
  - coins.agg_1m       — aktuelle Werte
  - coins.agg_1h       — BTC für Regime-Klassifikation
  - analyser_app.coin_baseline_profile — Baseline-Quantile

Settings: predictor.stalker.{metrics, btc_regime, baseline_min_samples_per_bucket}

Returnt None wenn Baseline für (Coin, h, wd, regime) fehlt → Caller skip Coin
(NO-FALLBACK).
"""
import logging
from datetime import datetime, timedelta

import pytz

log = logging.getLogger(__name__)

BERLIN_TZ = pytz.timezone("Europe/Berlin")


def _must(cfg, key):
    v = cfg.get(key)
    if v is None:
        log.error(f"FALLBACK_TRIGGERED feature_baseline_dev: '{key}' fehlt")
        raise RuntimeError(f"settings.predictor.stalker.{key} missing")
    return v


def get_data_days(coins_conn) -> float:
    """Effektive Tage mit voller agg_1m-Coverage (BTC-Row-Count / 1440 min/d).

    Misst Datendichte, nicht Zeitspanne — sonst zählen leere Tage mit.
    """
    with coins_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM agg_1m WHERE symbol='BTC'")
        r = cur.fetchone()
    if not r or r[0] is None:
        log.error("FALLBACK_TRIGGERED feature_baseline_dev.get_data_days: agg_1m BTC leer -> 0")
        return 0.0
    return float(r[0]) / 1440.0


def get_effective_min_samples(stalker_cfg: dict, data_days: float) -> int:
    """Liest baseline_min_samples_schedule aus settings, wählt höchste Stufe
    deren after_days_data <= data_days. No-fallback bei fehlendem Schedule.
    """
    sched = stalker_cfg.get("baseline_min_samples_schedule")
    if not sched or not isinstance(sched, list):
        log.error(
            "FALLBACK_TRIGGERED feature_baseline_dev.get_effective_min_samples: "
            "predictor.stalker.baseline_min_samples_schedule fehlt/leer -> abort"
        )
        raise RuntimeError("settings.predictor.stalker.baseline_min_samples_schedule missing")
    applicable = [it for it in sched if float(it.get("after_days_data", 0)) <= data_days]
    if not applicable:
        # data_days unter kleinster Schwelle → kleinste Stufe nehmen
        chosen = min(sched, key=lambda it: float(it.get("after_days_data", 0)))
    else:
        chosen = max(applicable, key=lambda it: float(it.get("after_days_data", 0)))
    v = chosen.get("value")
    if v is None:
        log.error(
            "FALLBACK_TRIGGERED feature_baseline_dev.get_effective_min_samples: "
            "Schedule-Stufe ohne 'value' -> abort"
        )
        raise RuntimeError("baseline_min_samples_schedule item missing 'value'")
    return int(v)


def _classify_btc_regime(coins_conn, regime_cfg):
    """Berechnet aktuelles BTC-Regime aus den letzten N+1 Stunden agg_1h."""
    win_h = _must(regime_cfg, "source_window_hours")
    bull_th = _must(regime_cfg, "bull_threshold_pct")
    bear_th = _must(regime_cfg, "bear_threshold_pct")
    with coins_conn.cursor() as cur:
        cur.execute(
            "SELECT bucket, close FROM agg_1h WHERE symbol='BTC' "
            "ORDER BY bucket DESC LIMIT %s",
            (win_h + 1,),
        )
        rows = cur.fetchall()
    if len(rows) < win_h + 1:
        log.error("FALLBACK_TRIGGERED feature_baseline_dev._classify_btc_regime: zu wenig BTC-1h-Daten")
        return None
    now_close = rows[0][1]
    old_close = rows[win_h][1]
    if not old_close or not now_close:
        return None
    chg = (now_close - old_close) / old_close * 100.0
    if chg > bull_th:
        return "bull"
    if chg < bear_th:
        return "bear"
    return "range"


def _fetch_current_metrics(coins_conn, symbol):
    """Aktuelle Metric-Werte aus den letzten 6 agg_1m-Buckets."""
    with coins_conn.cursor() as cur:
        cur.execute(
            """SELECT bucket, open, high, low, close, volume, trades,
                      spread_bps, book_imbalance_5, book_depth_5,
                      funding, open_interest
               FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 6""",
            (symbol,),
        )
        rows = cur.fetchall()
    if len(rows) < 6:
        return None
    # Reihenfolge umdrehen: ältester first
    rows = list(reversed(rows))
    cur_r, old_r = rows[-1], rows[0]
    last5 = rows[-5:]

    close = cur_r[4]
    old_close = old_r[4]
    oi = cur_r[11]
    old_oi = old_r[11]

    abs_close_pct_5m = (
        abs((close - old_close) / old_close * 100.0)
        if (close is not None and old_close)
        else None
    )
    oi_delta_5m = (
        (oi - old_oi) / old_oi * 100.0
        if (oi is not None and old_oi)
        else None
    )
    hs = [r[2] for r in last5 if r[2] is not None]
    ls = [r[3] for r in last5 if r[3] is not None]
    volatility_pct_5m = (
        (max(hs) - min(ls)) / close * 100.0
        if (hs and ls and close)
        else None
    )
    return {
        "bucket_ts": cur_r[0],
        "volume": cur_r[5],
        "volatility_pct_5m": volatility_pct_5m,
        "abs_close_pct_5m": abs_close_pct_5m,
        "spread_bps": cur_r[7],
        "book_imbalance_5": cur_r[8],
        "book_depth_5": cur_r[9],
        "funding": cur_r[10],
        "oi_delta_5m": oi_delta_5m,
        "trades": float(cur_r[6]) if cur_r[6] is not None else None,
    }


def _fetch_baseline_row(app_conn, symbol, hour, wd, regime):
    """Liefert {metric -> (n,p25,p50,p75,mad)} für (Coin,h,wd,regime)."""
    with app_conn.cursor() as cur:
        cur.execute(
            """SELECT metric, n_samples, p25, p50, p75, mad
               FROM coin_baseline_profile
               WHERE symbol=%s AND hour_of_day=%s AND weekday=%s AND btc_regime=%s""",
            (symbol, hour, wd, regime),
        )
        out = {}
        for metric, n, p25, p50, p75, mad in cur.fetchall():
            out[metric] = (n, p25, p50, p75, mad)
        return out


def compute_baseline_dev_features(coins_conn, app_conn, symbol, stalker_cfg,
                                   btc_regime=None, effective_min_samples=None):
    """
    Returnt dict mit dev_<metric>_z (zscore = (current - p50) / max(mad, 1e-9))
    sowie regime-Indicator (regime_bull/bear/range Binary).

    `effective_min_samples` kann vom Caller vorberechnet übergeben werden
    (für scan_pass: 1× pro Pass via get_effective_min_samples(cfg, data_days)).
    Wenn None: Helper wird hier aufgerufen (lookup data_days nochmal).

    Returnt None wenn keine Baseline für aktuelle (h,wd,regime)-Kombo existiert
    UND min_samples nicht erfüllt → Caller skip.
    """
    metrics_wanted = _must(stalker_cfg, "metrics")
    if effective_min_samples is None:
        data_days = get_data_days(coins_conn)
        effective_min_samples = get_effective_min_samples(stalker_cfg, data_days)
    min_samples = effective_min_samples
    regime_cfg = _must(stalker_cfg, "btc_regime")
    z_clip = _must(stalker_cfg, "dev_zscore_clip")  # cap gegen Magnitude-Regressor-Explosion

    if btc_regime is None:
        btc_regime = _classify_btc_regime(coins_conn, regime_cfg)
        if btc_regime is None:
            return None

    current = _fetch_current_metrics(coins_conn, symbol)
    if current is None:
        log.warning(f"FALLBACK_TRIGGERED feature_baseline_dev({symbol}): <6 agg_1m rows -> skip")
        return None

    bucket_ts = current["bucket_ts"]
    local_dt = bucket_ts.astimezone(BERLIN_TZ)
    h = local_dt.hour
    wd = local_dt.weekday()

    baseline = _fetch_baseline_row(app_conn, symbol, h, wd, btc_regime)
    fallback_used = False
    if not baseline and btc_regime != "range":
        # Aktuelles Regime hat keine Coverage → Fallback auf 'range' (größte
        # Bucket-Population). regime_<X>-Features bleiben unverändert auf
        # current_regime, damit das Modell weiß welches Regime gerade aktiv ist.
        baseline = _fetch_baseline_row(app_conn, symbol, h, wd, "range")
        fallback_used = bool(baseline)
        if fallback_used:
            log.debug(
                "feature_baseline_dev %s: regime=%s leer für (h=%d, wd=%d) -> fallback 'range'",
                symbol, btc_regime, h, wd,
            )
    if not baseline:
        return None

    feat = {
        "regime_bull": 1.0 if btc_regime == "bull" else 0.0,
        "regime_bear": 1.0 if btc_regime == "bear" else 0.0,
        "regime_range": 1.0 if btc_regime == "range" else 0.0,
        # Indicator-Feature: 1.0 wenn Baseline-Bucket war ein Fallback (range statt aktuelles regime),
        # 0.0 wenn der exakte Regime-Bucket genommen wurde
        "regime_fallback": 1.0 if fallback_used else 0.0,
    }

    valid_count = 0
    for metric in metrics_wanted:
        cur_val = current.get(metric)
        bl = baseline.get(metric)
        if cur_val is None or bl is None:
            feat[f"dev_{metric}_z"] = 0.0
            continue
        n, p25, p50, p75, mad = bl
        if n < min_samples or p50 is None:
            feat[f"dev_{metric}_z"] = 0.0
            continue
        denom = max(mad if mad is not None else 0.0, 1e-9)
        z = (cur_val - p50) / denom
        # Cap gegen extreme zscores (wenn mad~0): sonst explodiert der
        # Magnitude-Regressor (tp_pct = Milliarden %). settings: dev_zscore_clip.
        if z > z_clip: z = z_clip
        elif z < -z_clip: z = -z_clip
        feat[f"dev_{metric}_z"] = z
        valid_count += 1

    if valid_count == 0:
        return None

    return feat


def feature_keys(stalker_cfg: dict) -> list:
    metrics_wanted = _must(stalker_cfg, "metrics")
    keys = ["regime_bull", "regime_bear", "regime_range", "regime_fallback"]
    keys += [f"dev_{m}_z" for m in metrics_wanted]
    return keys
