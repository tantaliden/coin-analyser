#!/usr/bin/env python3
"""
Stalker-Predictor Baseline Refresher
=====================================
Berechnet pro (Coin, hour_of_day, weekday, btc_regime, metric) robuste Quantile
aus agg_1m (volume, volatility, spread, imbalance, depth, funding, oi_delta, trades).
Schreibt nach analyser_app.coin_baseline_profile.

Quellen 1:1 aus settings.json:predictor.stalker (no hardcoded defaults).

Refresh-Loop:
    bei Start → einmal alle Coins → sleep(baseline_refresh_hours) → wiederholen
"""
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import pytz

SETTINGS_PATH = "/opt/coin/settings.json"
BERLIN_TZ = pytz.timezone("Europe/Berlin")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s baseline_refresher: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_settings():
    with open(SETTINGS_PATH) as f:
        return json.load(f)


def get_cfg(s):
    cfg = s.get("predictor", {}).get("stalker")
    if cfg is None:
        log.error("FALLBACK_TRIGGERED baseline_refresher.get_cfg: predictor.stalker fehlt -> abort")
        raise RuntimeError("settings.predictor.stalker missing")
    return cfg


def must(cfg, key):
    v = cfg.get(key)
    if v is None:
        log.error(f"FALLBACK_TRIGGERED baseline_refresher.must: '{key}' fehlt -> abort")
        raise RuntimeError(f"settings.predictor.stalker.{key} missing")
    return v


def db_coins(s):
    c = s["databases"]["coins"]
    return psycopg2.connect(host=c["host"], port=c["port"], user=c["user"],
                            password=c["password"], dbname=c["name"])


def db_app(s):
    a = s["databases"]["app"]
    return psycopg2.connect(host=a["host"], port=a["port"], user=a["user"],
                            password=a["password"], dbname=a["name"])


def load_btc_regime_map(coins_conn, lookback_days, regime_cfg):
    """1h-Bucket -> ('bull'|'bear'|'range') Map aus BTC agg_1h."""
    win_h = must(regime_cfg, "source_window_hours")
    bull_th = must(regime_cfg, "bull_threshold_pct")
    bear_th = must(regime_cfg, "bear_threshold_pct")
    with coins_conn.cursor() as cur:
        cur.execute(
            "SELECT bucket, close FROM agg_1h WHERE symbol='BTC' "
            "AND bucket >= now() - (%s || ' days')::interval ORDER BY bucket",
            (str(lookback_days + 1),),
        )
        rows = cur.fetchall()
    out = {}
    for i, (b, c) in enumerate(rows):
        if i < win_h or c is None:
            out[b] = "range"
            continue
        old = rows[i - win_h][1]
        if not old:
            out[b] = "range"
            continue
        chg = (c - old) / old * 100.0
        if chg > bull_th:
            out[b] = "bull"
        elif chg < bear_th:
            out[b] = "bear"
        else:
            out[b] = "range"
    return out


def hour_bucket(ts):
    """Truncate timestamptz auf volle Stunde."""
    return ts.replace(minute=0, second=0, microsecond=0)


def compute_per_row_metrics(rows):
    """rows = ordered list of agg_1m DictRows. Yields (bucket, dict-of-metrics)."""
    for i, row in enumerate(rows):
        if i < 5:
            continue
        old = rows[i - 5]
        last5 = rows[i - 4 : i + 1]

        close = row["close"]
        old_close = old["close"]
        old_oi = old["open_interest"]
        oi = row["open_interest"]

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
        hs = [r["high"] for r in last5 if r["high"] is not None]
        ls = [r["low"] for r in last5 if r["low"] is not None]
        volatility_pct_5m = (
            (max(hs) - min(ls)) / close * 100.0 if (hs and ls and close) else None
        )
        yield row["bucket"], {
            "volume": row["volume"],
            "volatility_pct_5m": volatility_pct_5m,
            "abs_close_pct_5m": abs_close_pct_5m,
            "spread_bps": row["spread_bps"],
            "book_imbalance_5": row["book_imbalance_5"],
            "book_depth_5": row["book_depth_5"],
            "funding": row["funding"],
            "oi_delta_5m": oi_delta_5m,
            "trades": float(row["trades"]) if row["trades"] is not None else None,
        }


def refresh_coin(coins_conn, app_conn, symbol, lookback_days, min_samples,
                 metrics_wanted, regime_map):
    with coins_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT bucket, open, high, low, close, volume, trades,
                      spread_bps, book_imbalance_5, book_depth_5,
                      funding, open_interest
               FROM agg_1m
               WHERE symbol=%s AND bucket >= now() - (%s || ' days')::interval
               ORDER BY bucket""",
            (symbol, str(lookback_days)),
        )
        rows = cur.fetchall()

    if len(rows) < min_samples + 5:
        return 0

    buckets = defaultdict(lambda: defaultdict(list))
    for bucket_ts, mvs in compute_per_row_metrics(rows):
        local_dt = bucket_ts.astimezone(BERLIN_TZ)
        h = local_dt.hour
        wd = local_dt.weekday()
        regime = regime_map.get(hour_bucket(bucket_ts), "range")
        for metric in metrics_wanted:
            v = mvs.get(metric)
            if v is not None:
                buckets[(h, wd, regime)][metric].append(v)

    upserts = []
    now_iso = datetime.now(BERLIN_TZ)
    for (h, wd, regime), metric_map in buckets.items():
        for metric, vals in metric_map.items():
            n = len(vals)
            if n < min_samples:
                continue
            arr = np.asarray(vals, dtype=float)
            p25, p50, p75 = np.percentile(arr, [25, 50, 75])
            mad = float(np.median(np.abs(arr - p50)))
            upserts.append(
                (symbol, h, wd, regime, metric, n, float(p25), float(p50), float(p75), mad, now_iso)
            )

    if not upserts:
        return 0

    with app_conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """INSERT INTO coin_baseline_profile
               (symbol, hour_of_day, weekday, btc_regime, metric,
                n_samples, p25, p50, p75, mad, updated_at)
               VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol, hour_of_day, weekday, btc_regime, metric)
               DO UPDATE SET n_samples=EXCLUDED.n_samples,
                             p25=EXCLUDED.p25, p50=EXCLUDED.p50, p75=EXCLUDED.p75,
                             mad=EXCLUDED.mad, updated_at=EXCLUDED.updated_at""",
            upserts,
            page_size=200,
        )
    app_conn.commit()
    return len(upserts)


def refresh_all(s, cfg):
    lookback_days = must(cfg, "baseline_lookback_days")
    metrics_wanted = must(cfg, "metrics")
    regime_cfg = must(cfg, "btc_regime")

    coins_conn = db_coins(s)
    app_conn = db_app(s)
    try:
        # Effective min_samples per Schedule basierend auf tatsächlicher data_days
        import sys
        sys.path.insert(0, "/opt/coin/backend")
        from services.feature_baseline_dev import get_data_days, get_effective_min_samples
        data_days = get_data_days(coins_conn)
        min_samples = get_effective_min_samples(cfg, data_days)
        log.info(f"data_days={data_days:.2f} → effective min_samples={min_samples} (via schedule)")

        regime_map = load_btc_regime_map(coins_conn, lookback_days, regime_cfg)
        log.info(f"BTC-regime map: {len(regime_map)} hourly buckets")

        with coins_conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM agg_1m "
                "WHERE bucket >= now() - (%s || ' days')::interval ORDER BY symbol",
                (str(lookback_days),),
            )
            symbols = [r[0] for r in cur.fetchall()]
        log.info(f"refreshing baseline for {len(symbols)} coins, lookback={lookback_days}d, min_samples={min_samples}")

        total_upserts = 0
        for i, sym in enumerate(symbols, 1):
            try:
                n = refresh_coin(coins_conn, app_conn, sym, lookback_days,
                                 min_samples, metrics_wanted, regime_map)
                total_upserts += n
                if i % 25 == 0:
                    log.info(f"  progress: {i}/{len(symbols)} (last {sym}={n})")
            except Exception as e:
                log.error(f"refresh_coin({sym}) failed: {e}")
                app_conn.rollback()
        log.info(f"refresh complete: {total_upserts} bucket-rows upserted")
    finally:
        coins_conn.close()
        app_conn.close()


def main():
    s = load_settings()
    cfg = get_cfg(s)
    if not cfg.get("enabled"):
        log.warning("predictor.stalker.enabled=false -> exit")
        return
    interval_s = int(must(cfg, "baseline_refresh_hours")) * 3600
    log.info(f"baseline_refresher starting, refresh-interval={interval_s}s")

    while True:
        t0 = time.time()
        try:
            s = load_settings()
            cfg = get_cfg(s)
            if not cfg.get("enabled"):
                log.warning("stalker disabled at runtime -> sleep")
            else:
                refresh_all(s, cfg)
        except Exception as e:
            log.exception(f"refresh cycle failed: {e}")
        dt = time.time() - t0
        sleep_s = max(60, interval_s - int(dt))
        log.info(f"cycle done in {dt:.1f}s, sleep {sleep_s}s")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
