"""Predictor Service v5 — Multi-Head Online-Learner (River).

Drei Köpfe auf gemeinsamer Feature-Basis:
  - DirectionClassifier  (3-Klassen: long/short/skip, kalibrierte P)
  - MagnitudeRegressor_long  (TP=Peak%, SL=Trough%)
  - MagnitudeRegressor_short (TP=Trough%, SL=Peak%)

Selection: max(P(long),P(short)) >= min_direction_confidence (default 0.95)
TP/SL: aus Magnitude-Regressor × Sicherheits-Faktor (settings.json).
Lern-Signal beim Close direkt aus realisiertem peak/trough.

Settings: settings.json -> 'predictor.multi_head'.
Tabellen unverändert: open_predictions, predictor_state, prediction_feedback.

Trader-Welt (modify_bandit) bleibt unverändert (separate LinTSBandit, eigene Tabellen).
"""

import json
import logging
import math
import os
import pickle
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Backend-Root in sys.path, damit `from rl_agent.trader` und
# `from predictor.order_executor` funktionieren (auto-trade aus scan_pass_v4).
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Lazy-loaded torch im NeuralBandit (Modul global verfuegbar):
try:
    import torch  # noqa: F401  (Trader nutzt das, Predictor nicht)
except ImportError:
    torch = None

# Multi-Head Predictor (v5) — DirectionClassifier + MagnitudeRegressor
from services.predictor_model import MultiHeadPredictor

# Feature-Module v5 (Stufe 3: A=Orderflow, B=Whale, E=Funding, F=Sektor)
from services.feature_orderflow import compute_orderflow_features
from services.feature_funding import (compute_funding_features,
                                       compute_universe_funding_median)
from services.feature_sector import (build_coin_sector_map,
                                      compute_sector_close_pcts,
                                      compute_sector_features)
from services.feature_whale import compute_whale_features

# Stalker (v5.1, 18.05.2026): pro-Coin Baseline + Cross-Coin + Coin-Identity
from services.feature_baseline_dev import (
    compute_baseline_dev_features,
    feature_keys as stalker_baseline_keys,
    _classify_btc_regime as stalker_classify_btc_regime,
    get_data_days as stalker_data_days,
    get_effective_min_samples as stalker_effective_min_samples,
)
from services.feature_cross_coin import (
    build_reference_cache as stalker_build_ref_cache,
    compute_cross_coin_features,
    feature_keys as stalker_cross_keys,
)
from services.feature_coin_identity import (
    compute_identity_features,
    feature_keys as stalker_identity_keys,
)

# Paper-Trading-Engine (Variante B, 20.05.2026)
from services.paper_engine import paper_open, paper_count_open, paper_watch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("predictor")

SETTINGS_PATH = '/opt/coin/settings.json'
ACTIVE_VERSION = 'v4-bandit'


def load_settings():
    with open(SETTINGS_PATH) as fp:
        return json.load(fp)


def db_coins(s):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def db_app(s):
    db = s["databases"]["app"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def db_learner(s):
    db = s["databases"]["learner"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def f(x):
    if x is None: return None
    return float(x)


def fseries(series):
    return [f(x) for x in series]


# =============================================================================
# Operatoren (für Rule-Flags als Features)
# =============================================================================

def op_gt(val, threshold, *_):
    val = f(val); threshold = float(threshold)
    return val is not None and val > threshold

def op_lt(val, threshold, *_):
    val = f(val); threshold = float(threshold)
    return val is not None and val < threshold

def op_rising_pct(series, threshold_pct, *_):
    s = fseries(series)
    if not s or len(s) < 2: return False
    first = next((x for x in s if x is not None), None)
    last = s[-1]
    if first is None or last is None or first == 0: return False
    return (last / first - 1) * 100 >= float(threshold_pct)

def op_falling_pct(series, threshold_pct, *_):
    s = fseries(series)
    if not s or len(s) < 2: return False
    first = next((x for x in s if x is not None), None)
    last = s[-1]
    if first is None or last is None or first == 0: return False
    return (last / first - 1) * 100 <= -float(threshold_pct)

def op_abs_change_pct_max(series, threshold_pct, *_):
    s = fseries(series)
    if not s or len(s) < 2: return False
    first = next((x for x in s if x is not None), None)
    if first is None or first == 0: return False
    clean = [x for x in s if x is not None]
    if not clean: return False
    mx = max(clean); mn = min(clean)
    return max(abs(mx/first - 1), abs(mn/first - 1)) * 100 <= float(threshold_pct)

def op_z_score(series, threshold_z, *_):
    s = fseries(series)
    if not s or len(s) < 10: return False
    baseline = [x for x in s[:-1] if x is not None]
    if len(baseline) < 3: return False
    mean = sum(baseline) / len(baseline)
    var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    std = math.sqrt(var)
    if std == 0: return False
    last = s[-1]
    if last is None: return False
    return (last - mean) / std >= float(threshold_z)

def op_bollinger_pos(series, threshold, period, *_):
    s = fseries(series)
    period = int(period)
    if not s or len(s) < period: return False
    win = s[-period:]
    clean = [x for x in win if x is not None]
    if len(clean) < period: return False
    mean = sum(clean) / period
    var = sum((x - mean) ** 2 for x in clean) / period
    std = math.sqrt(var)
    if std == 0: return False
    pos = (clean[-1] - mean) / (2 * std)
    threshold = float(threshold)
    if threshold < 0:
        return pos <= threshold
    return pos >= threshold

OP_FUNCS = {
    ">": op_gt, "<": op_lt,
    "rising_pct": op_rising_pct, "falling_pct": op_falling_pct,
    "abs_change_pct_max": op_abs_change_pct_max,
    "z_score": op_z_score, "bollinger_pos": op_bollinger_pos,
}

SERIES_FIELDS = (
    "open","close","volume","trades","taker_buy_base","taker_buy_quote",
    "funding","open_interest","premium","spread_bps","book_imbalance_5",
    "book_depth_5","mark_px","mid_px","oracle_px",
    "bbo_bid_px","bbo_ask_px","bbo_bid_sz","bbo_ask_sz",
)


def load_series(cur, symbol, field, minutes):
    lookback = max(int(minutes), 30)
    cur.execute(f"""
        SELECT bucket, {", ".join(SERIES_FIELDS)}
        FROM agg_1m WHERE symbol=%s AND bucket >= now() - (%s || ' minutes')::interval
        ORDER BY bucket
    """, (symbol, lookback))
    rows = cur.fetchall()
    if not rows: return []
    if field == "taker_buy_base_ratio":
        out = []
        for r in rows:
            v = f(r['volume']); t = f(r['taker_buy_base'])
            if v and v > 0 and t is not None: out.append(t / v)
            else: out.append(None)
        return out
    return [f(r.get(field)) for r in rows]


def evaluate_rule(cur, symbol, rule, base_lookback):
    details = []
    for cond in rule["conditions"]:
        field = cond["field"]; op = cond["op"]; val = cond["value"]
        window = cond.get("window_min", base_lookback)
        series = load_series(cur, symbol, field, window)
        series_window = series[-max(1, int(window)):] if series else []
        fn = OP_FUNCS.get(op)
        if fn is None:
            return {"ok": False, "details": details, "reason": f"unknown op {op}"}
        if op in (">", "<"):
            last = next((s for s in reversed(series_window) if s is not None), None)
            ok = fn(last, val)
        elif op == "bollinger_pos":
            period = cond.get("window_min", 20)
            ok = op_bollinger_pos(series, val, period)
        else:
            ok = fn(series_window, val)
        if not ok:
            return {"ok": False, "details": details}
    return {"ok": True, "details": details}


# =============================================================================
# Indikatoren
# =============================================================================

def _ema(values, period):
    clean = [v for v in values if v is not None]
    if len(clean) < period: return None
    k = 2.0 / (period + 1)
    e = clean[0]
    for v in clean[1:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes, period=14):
    clean = [c for c in closes if c is not None]
    if len(clean) < period + 1:
        log.error("FALLBACK_TRIGGERED _rsi: zu wenig Daten (n=%d, brauche >=%d) -> None", len(clean), period+1)
        return None
    gains = []; losses = []
    for i in range(1, len(clean)):
        diff = clean[i] - clean[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if len(gains) < period:
        log.error("FALLBACK_TRIGGERED _rsi: zu wenig gains (n=%d, brauche %d) -> None", len(gains), period)
        return None
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0  # legitimer Edge-Case: keine Verluste -> RSI=100
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes, fast=12, slow=26):
    clean = [c for c in closes if c is not None]
    if len(clean) < slow:
        log.error("FALLBACK_TRIGGERED _macd: zu wenig Daten (n=%d, brauche >=%d) -> None", len(clean), slow)
        return None
    ef = _ema(clean, fast); es = _ema(clean, slow)
    if ef is None or es is None:
        log.error("FALLBACK_TRIGGERED _macd: _ema returnt None -> None")
        return None
    return ef - es


def _cvd_horizon(rows_window):
    cvd = 0.0
    for r in rows_window:
        v = f(r['volume']) or 0.0
        tb = f(r['taker_buy_base']) or 0.0
        ts = v - tb
        cvd += (tb - ts)
    return cvd


def _cvd_ratio(rows_window):
    total_v = sum((f(r['volume']) or 0.0) for r in rows_window)
    if total_v <= 0: return 0.0
    return _cvd_horizon(rows_window) / total_v


def get_timeout_hours(s):
    """Single Source of Truth: predictor.timeout_hours.
    Predictor (Multi-Head) + Trader nutzen IMMER den gleichen Wert.
    Reihenfolge: predictor.timeout_hours -> predictor.multi_head.timeout_hours
    -> predictor.bandit.timeout_hours (legacy).
    KEIN silent default — FALLBACK_TRIGGERED + raise wenn Setting fehlt.
    s = volle settings.json-Struktur."""
    p = s.get("predictor", {})
    v = p.get("timeout_hours")
    if v is None:
        v = p.get("multi_head", {}).get("timeout_hours")
    if v is None:
        v = p.get("bandit", {}).get("timeout_hours")
    if v is None:
        log.error("FALLBACK_TRIGGERED get_timeout_hours: predictor.timeout_hours (oder multi_head/bandit) fehlt in settings -> raise")
        raise RuntimeError("predictor.timeout_hours missing in settings.json")
    return float(v)


_FEATURE_RANGES = {
    # name: (min, max) — wenn Werte ausserhalb -> log.warning DATA_QUALITY
    'close_pct_1m': (-30, 30), 'close_pct_5m': (-50, 50), 'close_pct_15m': (-60, 60),
    'close_pct_30m': (-70, 70), 'close_pct_60m': (-80, 80), 'close_pct_240m': (-100, 100),
    'btc_close_pct_5m': (-30, 30), 'btc_close_pct_15m': (-30, 30), 'btc_close_pct_60m': (-50, 50),
    'rel_strength_5m': (-50, 50), 'rel_strength_15m': (-60, 60), 'rel_strength_60m': (-80, 80),
    'atr_pct_15m': (0, 10), 'atr_pct_1h': (0, 15), 'atr_pct_4h': (0, 30),
    'rsi_14_1m': (0, 100), 'rsi_14_5m': (0, 100),
    'spread_bps': (0, 5000), 'impact_spread_bps': (0, 1000),
    'funding': (-0.005, 0.005), 'premium': (-0.05, 0.05),
    'taker_buy_ratio': (0, 1), 'taker_buy_ratio_15m': (0, 1),
    'hour_sin': (-1, 1), 'hour_cos': (-1, 1), 'weekday': (0, 6),
    'book_imbalance_5': (-1, 1), 'book_imbalance_10': (-1, 1), 'book_imbalance_20': (-1, 1),
    'bbo_size_ratio': (-1, 1),
    # Position-Features (Trader)
    'pnl_now_pct': (-100, 100), 'peak_pct_now': (0, 200), 'trough_pct_now': (-100, 0),
    'leverage': (1, 50), 'margin_pnl_pct': (-500, 500),
    'time_in_trade_h': (0, 48), 'time_remaining_h': (0, 48),
    'dist_to_tp_pct': (-100, 100), 'dist_to_sl_pct': (-100, 100),
    'original_action_idx': (-1, 200),
}


def data_quality_check(s, send_telegram_report=False):
    """Periodischer Check: hat ein Feature in den letzten N Samples konstanten Wert
    oder Out-of-Range-Werte? Schreibt DATA_QUALITY-Logs.
    Wenn send_telegram_report: zusätzlich aggregierten Bericht via send_telegram().
    Welt-Trennung: liest open_predictions.features (Predictor) und trader_decisions.features (Trader) separat."""
    findings_predictor = []
    findings_trader = []
    N = 100  # Sample-Size

    def analyse(rows, label, target_list):
        if not rows: return
        per_feat = {}
        for r in rows:
            feat = r[0] if isinstance(r, tuple) else r
            if not isinstance(feat, dict): continue
            for k, v in feat.items():
                if k.startswith('_'): continue  # Diagnose-Felder ueberspringen
                try: fv = float(v) if v is not None else None
                except (TypeError, ValueError): continue
                per_feat.setdefault(k, []).append(fv)
        for k, vals in per_feat.items():
            n = len(vals)
            n_null = sum(1 for x in vals if x is None)
            real = [x for x in vals if x is not None]
            if n_null == n:
                target_list.append(f"DATA_QUALITY {label} {k}: ALWAYS NULL ({n}/{n})")
                continue
            if not real: continue
            mn = min(real); mx = max(real)
            if mn == mx:
                target_list.append(f"DATA_QUALITY {label} {k}: CONST {mn} ({n}/{n})")
            elif n_null / n > 0.8:
                target_list.append(f"DATA_QUALITY {label} {k}: NULL in {n_null}/{n} ({n_null/n*100:.0f}%)")
            else:
                # check for >80% gleicher Wert
                from collections import Counter
                c = Counter(real).most_common(1)
                if c and c[0][1] / len(real) > 0.8:
                    target_list.append(f"DATA_QUALITY {label} {k}: dominanter Wert {c[0][0]} in {c[0][1]}/{len(real)} ({c[0][1]/len(real)*100:.0f}%)")
                # Range-Check
                rng = _FEATURE_RANGES.get(k)
                if rng:
                    lo, hi = rng
                    out = [x for x in real if x < lo or x > hi]
                    if out:
                        target_list.append(f"DATA_QUALITY {label} {k}: {len(out)} Werte ausserhalb [{lo},{hi}] (min={mn} max={mx})")

    try:
        with db_app(s) as app:
            with app.cursor() as cur:
                cur.execute("""SELECT features FROM open_predictions WHERE status IN ('win','loss','timeout','open')
                              ORDER BY created_at DESC LIMIT %s""", (N,))
                analyse(cur.fetchall(), "PREDICTOR", findings_predictor)
                cur.execute("""SELECT features FROM trader_decisions
                              ORDER BY decided_at DESC LIMIT %s""", (N,))
                analyse(cur.fetchall(), "TRADER", findings_trader)
    except Exception as e:
        log.exception("data_quality_check DB-Fehler: %s", e)
        return

    for msg in findings_predictor + findings_trader:
        log.warning(msg)

    if send_telegram_report:
        n_p = len(findings_predictor); n_t = len(findings_trader)
        if n_p == 0 and n_t == 0:
            txt = f"Daten-Quality Report 12h: alles sauber. Predictor + Trader Features OK."
        else:
            txt = f"Daten-Quality Report 12h: {n_p} Predictor-, {n_t} Trader-Befunde.\n"
            for m in (findings_predictor + findings_trader)[:20]:
                txt += "- " + m + "\n"
            if n_p + n_t > 20:
                txt += f"... und {n_p + n_t - 20} weitere im Log."
        try:
            send_telegram(s, txt)
        except Exception as e:
            log.warning("data_quality_check send_telegram failed: %s", e)
    return findings_predictor, findings_trader


def load_btc_moves(coins_conn):
    """Returns {'5m','15m','60m':pct} oder None bei fehlenden Daten.
    Caller MUSS None handhaben (Scan-Pass kann ohne BTC-Referenz nicht arbeiten)."""
    try:
        with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT bucket, close FROM agg_1m WHERE symbol='BTC'
                ORDER BY bucket DESC LIMIT 65
            """)
            rows = list(cur.fetchall())
        if len(rows) < 61:
            log.error("FALLBACK_TRIGGERED load_btc_moves: nur %d BTC-Candles (brauche >=61) -> None", len(rows))
            return None
        cl_now = f(rows[0]['close'])
        if not cl_now:
            log.error("FALLBACK_TRIGGERED load_btc_moves: BTC.close=NULL/0 in juengstem Bucket -> None")
            return None
        out = {}
        for h in (5, 15, 60):
            c0 = f(rows[h]['close'])
            if not c0 or c0 <= 0:
                log.error("FALLBACK_TRIGGERED load_btc_moves: BTC.close vor %dm NULL/0 -> None", h)
                return None
            out[f'{h}m'] = (cl_now - c0) / c0 * 100.0
        return out
    except Exception as e:
        log.error("FALLBACK_TRIGGERED load_btc_moves: Exception '%s' -> None", e)
        return None


# =============================================================================
# Feature-Snapshot v2 (unverändert von v3)
# =============================================================================

def feature_snapshot_v2(coins_conn, symbol, rule_flags=None, btc_moves=None):
    feat = {}
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT bucket, open, high, low, close, volume, trades,
                   taker_buy_base, taker_buy_quote,
                   funding, open_interest, premium,
                   spread_bps, book_imbalance_5, book_depth_5,
                   mark_px, mid_px, oracle_px, bbo_bid_sz, bbo_ask_sz
            FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 241
        """, (symbol,))
        rows = list(reversed(cur.fetchall()))
        # Strenge Untergrenze: 241 = exakt das was close_pct_240m braucht.
        # Bei weniger Daten wuerden close_pct_* in 0.0-Fallback laufen und der
        # Bandit lernt aus Datenluecken statt aus echten Marktbewegungen.
        # Coins ohne vollstaendige 4h-Historie (frisch listed / Datenluecke)
        # werden geskippt bis genuegend agg_1m vorhanden ist.
        if not rows or len(rows) < 241: return None
        last = rows[-1]
        if last['close'] is None: return None
        cl = f(last['close'])

        for k in ("funding","open_interest","premium","spread_bps","book_imbalance_5","book_depth_5"):
            v = f(last[k])
            if v is None:
                log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: %s NULL in last agg_1m -> coin skip", symbol, k)
                return None
            feat[k] = v

        bidsz = f(last['bbo_bid_sz'])
        asksz = f(last['bbo_ask_sz'])
        if bidsz is None or asksz is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: bbo_bid_sz/bbo_ask_sz NULL -> coin skip", symbol)
            return None
        feat['bbo_size_ratio'] = (bidsz - asksz) / (bidsz + asksz + 1e-9)

        mark = f(last['mark_px']); mid = f(last['mid_px']); oracle = f(last['oracle_px'])
        if mark is None or mid is None or oracle is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: mark/mid/oracle_px NULL -> coin skip", symbol)
            return None
        feat['mark_vs_mid_bps'] = (mark - mid) / mid * 10000
        feat['mark_vs_oracle_bps'] = (mark - oracle) / oracle * 10000

        def close_at(ago_min):
            idx = -1 - ago_min
            if abs(idx) > len(rows): return None
            v = f(rows[idx]['close'])
            return v if v else None

        for h in (1, 5, 15, 30, 60, 240):
            c0 = close_at(h)
            if c0 is None:
                log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: close vor %dm NULL/0 -> coin skip", symbol, h)
                return None
            feat[f'close_pct_{h}m'] = (cl - c0) / c0 * 100

        vols_15m = [f(r['volume']) or 0 for r in rows[-15:]]
        vols_1h = [f(r['volume']) or 0 for r in rows[-60:]] if len(rows) >= 60 else vols_15m
        feat['vol_15m_sum'] = sum(vols_15m)
        feat['vol_spike_ratio'] = (sum(vols_15m) / max(sum(vols_1h) / 4, 1e-9)) if sum(vols_1h) else 1.0
        if len(vols_1h) >= 10:
            m = sum(vols_1h[:-1]) / len(vols_1h[:-1])
            v = sum((x - m) ** 2 for x in vols_1h[:-1]) / len(vols_1h[:-1])
            sd = math.sqrt(v)
            feat['vol_zscore_1h'] = ((vols_1h[-1] - m) / sd) if sd > 0 else 0.0
        else:
            feat['vol_zscore_1h'] = 0.0

        vol_now = f(last['volume']) or 0
        tb_now = f(last['taker_buy_base']) or 0
        feat['taker_buy_ratio'] = (tb_now / vol_now) if vol_now > 0 else 0.5
        tb_15 = sum(f(r['taker_buy_base']) or 0 for r in rows[-15:])
        v_15 = sum(f(r['volume']) or 0 for r in rows[-15:])
        feat['taker_buy_ratio_15m'] = (tb_15 / v_15) if v_15 > 0 else 0.5

        def at(ago_min, key):
            idx = -1 - ago_min
            if abs(idx) > len(rows): return None
            return f(rows[idx][key])

        for h in (5, 15, 60):
            oi_then = at(h, 'open_interest') or 0
            oi_now = feat['open_interest']
            feat[f'oi_pct_{h}m'] = ((oi_now - oi_then) / oi_then * 100) if oi_then else 0.0

        f30 = at(30, 'funding')
        feat['funding_delta_30m'] = (feat['funding'] - f30) if f30 is not None else 0.0
        sp5 = at(5, 'spread_bps')
        feat['spread_delta_5m'] = (feat['spread_bps'] - sp5) if sp5 is not None else 0.0
        pr5 = at(5, 'premium')
        feat['premium_delta_5m'] = (feat['premium'] - pr5) if pr5 is not None else 0.0

        oi_n = feat.get('open_interest') or 0.0
        def _oi_drop_pct(ago_min):
            oi_then = at(ago_min, 'open_interest')
            if not oi_then or oi_then <= 0 or oi_n is None: return 0.0
            return (oi_then - oi_n) / oi_then * 100.0

        feat['oi_drop_1m'] = max(0.0, _oi_drop_pct(1))
        feat['oi_drop_5m'] = max(0.0, _oi_drop_pct(5))
        feat['oi_drop_15m'] = max(0.0, _oi_drop_pct(15))

        cl_1m = at(1, 'close'); cl_5m = at(5, 'close')
        m1 = ((cl - cl_1m) / cl_1m * 100.0) if (cl_1m and cl_1m > 0) else 0.0
        m5 = ((cl - cl_5m) / cl_5m * 100.0) if (cl_5m and cl_5m > 0) else 0.0
        feat['liq_long_proxy_1m'] = feat['oi_drop_1m'] * max(0.0, -m1)
        feat['liq_long_proxy_5m'] = feat['oi_drop_5m'] * max(0.0, -m5)
        feat['liq_short_proxy_1m'] = feat['oi_drop_1m'] * max(0.0, m1)
        feat['liq_short_proxy_5m'] = feat['oi_drop_5m'] * max(0.0, m5)

        def atr_local(window_rows):
            trs = []; prev_c = None
            for r in window_rows:
                h_, l_, c_ = f(r['high']), f(r['low']), f(r['close'])
                if h_ is None or l_ is None or c_ is None: continue
                if prev_c is None:
                    prev_c = c_; continue
                tr = max(h_ - l_, abs(h_ - prev_c), abs(l_ - prev_c))
                trs.append(tr); prev_c = c_
            if not trs or not cl: return 0.0
            return (sum(trs) / len(trs)) / cl * 100

        feat['atr_pct_15m'] = atr_local(rows[-15:])
        feat['atr_pct_1h'] = atr_local(rows[-60:]) if len(rows) >= 60 else feat['atr_pct_15m']
        feat['atr_pct_4h'] = atr_local(rows[-240:]) if len(rows) >= 240 else feat['atr_pct_1h']

        cur.execute("""
            SELECT impact_bid, impact_ask, day_ntl_vlm, day_base_vlm, prev_day_px
            FROM hl_asset_ctx WHERE symbol=%s ORDER BY ts DESC LIMIT 1
        """, (symbol,))
        ctx = cur.fetchone()
        if ctx:
            ib = f(ctx['impact_bid']); ia = f(ctx['impact_ask'])
            feat['impact_spread_bps'] = ((ia - ib) / ((ia + ib) / 2) * 10000) if (ia and ib and (ia + ib) > 0) else 0.0
            feat['day_ntl_vlm'] = f(ctx['day_ntl_vlm']) or 0.0
            pdy = f(ctx['prev_day_px'])
            feat['prev_day_change_pct'] = ((cl - pdy) / pdy * 100) if pdy else 0.0
        else:
            feat['impact_spread_bps'] = 0.0
            feat['day_ntl_vlm'] = 0.0
            feat['prev_day_change_pct'] = 0.0

        cur.execute("""
            SELECT book_imbalance_10, book_depth_10, book_imbalance_20, book_depth_20
            FROM klines WHERE symbol=%s AND interval='10s' ORDER BY open_time DESC LIMIT 1
        """, (symbol,))
        l2ext = cur.fetchone()
        if l2ext:
            for k in ('book_imbalance_10', 'book_depth_10', 'book_imbalance_20', 'book_depth_20'):
                v = f(l2ext[k])
                feat[k] = v if v is not None else 0.0
        else:
            feat['book_imbalance_10'] = 0.0; feat['book_depth_10'] = 0.0
            feat['book_imbalance_20'] = 0.0; feat['book_depth_20'] = 0.0

        feat['cvd_5m'] = _cvd_horizon(rows[-5:])
        feat['cvd_15m'] = _cvd_horizon(rows[-15:])
        feat['cvd_60m'] = _cvd_horizon(rows[-60:]) if len(rows) >= 60 else feat['cvd_15m']
        feat['cvd_ratio_15m'] = _cvd_ratio(rows[-15:])
        feat['cvd_ratio_60m'] = _cvd_ratio(rows[-60:]) if len(rows) >= 60 else feat['cvd_ratio_15m']

        closes_1m = [f(r['close']) for r in rows]
        rsi_1m = _rsi(closes_1m, 14)
        if rsi_1m is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: rsi_14_1m None -> coin skip", symbol)
            return None
        feat['rsi_14_1m'] = rsi_1m
        closes_5m = closes_1m[::-5][::-1] if len(closes_1m) >= 70 else closes_1m
        rsi_5m = _rsi(closes_5m, 14)
        if rsi_5m is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: rsi_14_5m None -> coin skip", symbol)
            return None
        feat['rsi_14_5m'] = rsi_5m
        macd = _macd(closes_1m, 12, 26)
        if macd is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: macd None -> coin skip", symbol)
            return None
        feat['macd_line'] = macd
        if not cl:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: cl=0/None bei macd_pct -> coin skip", symbol)
            return None
        feat['macd_pct'] = feat['macd_line'] / cl * 100.0

        cur.execute("""
            SELECT bucket, funding FROM agg_1h WHERE symbol=%s ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        f24 = [f(r['funding']) for r in cur.fetchall()]
        clean_f = [x for x in f24 if x is not None]
        if len(clean_f) < 6:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: funding_zscore_24h zu wenig Daten (n=%d) -> coin skip", symbol, len(clean_f))
            return None
        f_mean = sum(clean_f) / len(clean_f)
        f_var = sum((x - f_mean) ** 2 for x in clean_f) / len(clean_f)
        f_std = math.sqrt(f_var)
        cur_f = clean_f[0]
        if f_std == 0:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: funding_std=0, ergebnis undefiniert -> coin skip", symbol)
            return None
        feat['funding_zscore_24h'] = (cur_f - f_mean) / f_std
        streak = 0; sign = None
        for v in f24:
            if v is None: continue
            sg = 1 if v > 0 else (-1 if v < 0 else 0)
            if sign is None: sign = sg; streak = 1
            elif sg == sign: streak += 1
            else: break
        feat['funding_streak_signed'] = streak * (sign or 0)

        cur.execute("""
            SELECT bucket, high, low, close FROM agg_1h WHERE symbol=%s ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        hl_rows = cur.fetchall()
        ranges = []
        for r in hl_rows:
            h_, l_, c_ = f(r['high']), f(r['low']), f(r['close'])
            if h_ and l_ and c_ and c_ > 0: ranges.append((h_ - l_) / c_ * 100.0)
        if len(ranges) >= 4:
            last_1h = ranges[0]
            avg_24h = sum(ranges) / len(ranges)
            feat['vol_1h_vs_24h_ratio'] = (last_1h / avg_24h) if avg_24h > 0 else 1.0
            r_var = sum((x - avg_24h) ** 2 for x in ranges) / len(ranges)
            r_std = math.sqrt(r_var)
            feat['vol_zscore_24h'] = ((last_1h - avg_24h) / r_std) if r_std > 0 else 0.0
        else:
            feat['vol_1h_vs_24h_ratio'] = 1.0
            feat['vol_zscore_24h'] = 0.0

    if symbol == 'BTC':
        # BTC scannt sich nicht selbst gegen BTC-Bewegung -> rel_strength immer 0,
        # btc_close_pct = eigene close_pct
        feat['btc_close_pct_5m'] = feat['close_pct_5m']
        feat['btc_close_pct_15m'] = feat['close_pct_15m']
        feat['btc_close_pct_60m'] = feat['close_pct_60m']
        feat['rel_strength_5m'] = 0.0
        feat['rel_strength_15m'] = 0.0
        feat['rel_strength_60m'] = 0.0
    else:
        if btc_moves is None:
            log.error("FALLBACK_TRIGGERED feature_snapshot_v2 %s: btc_moves None -> coin skip", symbol)
            return None
        feat['btc_close_pct_5m'] = btc_moves['5m']
        feat['btc_close_pct_15m'] = btc_moves['15m']
        feat['btc_close_pct_60m'] = btc_moves['60m']
        feat['rel_strength_5m'] = feat['close_pct_5m'] - btc_moves['5m']
        feat['rel_strength_15m'] = feat['close_pct_15m'] - btc_moves['15m']
        feat['rel_strength_60m'] = feat['close_pct_60m'] - btc_moves['60m']

    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0
    feat['hour_sin'] = math.sin(2 * math.pi * hour / 24)
    feat['hour_cos'] = math.cos(2 * math.pi * hour / 24)
    feat['weekday'] = float(now_utc.weekday())

    # Wallet-Drawdown: wenn die virtuelle Wallet unter ihrem Peak liegt, weiss
    # der Bandit "ich habe gerade eine Verlustphase". Wert ist <= 0 (0 = kein
    # Drawdown, -10 = 10% unter Peak). Bandit kann lernen bei Drawdown z.B.
    # konservativere TP/SL-Buckets zu waehlen.
    try:
        # feature_snapshot_v2 nutzt nur coins_conn — wir oeffnen kurz app
        from psycopg2 import connect as _pgconnect
        s_app = load_settings()['databases']['app']
        with _pgconnect(host=s_app['host'], port=s_app['port'], dbname=s_app['name'],
                          user=s_app['user'], password=s_app['password']) as app_conn:
            feat['wallet_drawdown_pct'] = get_wallet_drawdown_pct(app_conn)
    except Exception:
        feat['wallet_drawdown_pct'] = 0.0

    if rule_flags:
        for name, val in rule_flags.items():
            feat[f'rule_{name}'] = float(1 if val else 0)
    return feat


# =============================================================================
# Bandit (Linear Thompson Sampling)
# =============================================================================

# =============================================================================
# Virtuelle Wallet — Bandit lernt Wallet-State (drawdown) als Feature.
# =============================================================================

def get_wallet_drawdown_pct(app_conn) -> float:
    """Aktueller Drawdown der virtuellen Wallet in % (relativ zum Peak).
    Returns 0.0 wenn keine Tabelle oder balance >= peak."""
    try:
        with app_conn.cursor() as cur:
            cur.execute("SELECT balance, peak_balance FROM virtual_wallet_state WHERE id=1")
            r = cur.fetchone()
            if r and r[1] and float(r[1]) > 0:
                return (float(r[0]) / float(r[1]) - 1.0) * 100.0
    except Exception:
        pass
    return 0.0


def update_virtual_wallet(app_conn, pnl_dollar: float) -> float:
    """Aktualisiert virtuelle Wallet nach realisiertem $-PnL. Setzt peak hoch
    wenn neuer Hoechststand. Returns neuer Drawdown-Wert."""
    try:
        with app_conn.cursor() as cur:
            cur.execute("""
                UPDATE virtual_wallet_state
                SET balance = balance + %s,
                    peak_balance = GREATEST(peak_balance, balance + %s),
                    last_update = now(),
                    last_pnl_dollar = %s,
                    n_trades = COALESCE(n_trades, 0) + 1
                WHERE id=1
                RETURNING balance, peak_balance
            """, (pnl_dollar, pnl_dollar, pnl_dollar))
            r = cur.fetchone()
        app_conn.commit()
        if r and r[1] and float(r[1]) > 0:
            return (float(r[0]) / float(r[1]) - 1.0) * 100.0
    except Exception as e:
        log.warning("update_virtual_wallet failed: %s", e)
    return 0.0


def reset_virtual_wallet(app_conn, start_balance: float):
    """Reset auf Start-Balance — fuer Backfill-Replay."""
    with app_conn.cursor() as cur:
        cur.execute("""
            UPDATE virtual_wallet_state
            SET balance = %s, peak_balance = %s, start_balance = %s,
                last_update = now(), n_trades = 0, last_pnl_dollar = 0
            WHERE id=1
        """, (start_balance, start_balance, start_balance))
    app_conn.commit()


# Non-Rule-Features (stabil). Rule-Flags werden in build_feature_keys() dynamisch
# aus settings.json ergaenzt, damit neue/umbenannte Regeln nicht stillschweigend
# aus dem Feature-Vektor herausfallen.
_BASE_FEATURE_KEYS = sorted([
    'atr_pct_15m','atr_pct_1h','atr_pct_4h','bbo_size_ratio',
    'book_depth_10','book_depth_20','book_depth_5',
    'book_imbalance_10','book_imbalance_20','book_imbalance_5',
    'btc_close_pct_15m','btc_close_pct_5m','btc_close_pct_60m',
    'close_pct_15m','close_pct_1m','close_pct_240m','close_pct_30m','close_pct_5m','close_pct_60m',
    'cvd_15m','cvd_5m','cvd_60m','cvd_ratio_15m','cvd_ratio_60m',
    'day_ntl_vlm','funding','funding_delta_30m','funding_streak_signed','funding_zscore_24h',
    'hour_cos','hour_sin','impact_spread_bps',
    'liq_long_proxy_1m','liq_long_proxy_5m','liq_short_proxy_1m','liq_short_proxy_5m',
    'macd_line','macd_pct','mark_vs_mid_bps','mark_vs_oracle_bps',
    'oi_drop_15m','oi_drop_1m','oi_drop_5m','oi_pct_15m','oi_pct_5m','oi_pct_60m',
    'open_interest','premium','premium_delta_5m','prev_day_change_pct',
    'rel_strength_15m','rel_strength_5m','rel_strength_60m',
    'rsi_14_1m','rsi_14_5m',
    'spread_bps','spread_delta_5m','taker_buy_ratio','taker_buy_ratio_15m',
    'vol_15m_sum','vol_1h_vs_24h_ratio','vol_spike_ratio','vol_zscore_1h','vol_zscore_24h',
    'wallet_drawdown_pct',  # virtueller Wallet-Drawdown (negativer Wert = unter Peak)
    'weekday',
])


def build_feature_keys(s):
    """Liste der Feature-Keys = base + sortierte rule_<name>-Keys aus settings.

    Mit predictor.stalker.enabled=true werden die Stalker-Features angehängt:
    Baseline-Devs + Cross-Coin + Coin-Identity (Reihenfolge stabil, damit
    Vector-Position deterministisch ist).
    """
    rule_keys = sorted([f"rule_{r['name']}" for r in s.get("predictor", {}).get("rules", [])])
    keys = _BASE_FEATURE_KEYS + rule_keys
    stalker_cfg = s.get("predictor", {}).get("stalker", {}) or {}
    if stalker_cfg.get("enabled"):
        keys = keys + stalker_baseline_keys(stalker_cfg)
        keys = keys + stalker_cross_keys(stalker_cfg["cross_coin"])
        keys = keys + stalker_identity_keys(stalker_cfg["coin_identity"])
    return keys


# Werden in main() konkret gesetzt. Bias-Term sitzt in Index 0, Features ab Index 1.
FEATURE_KEYS = []
N_FEAT = 1


class OnlineScaler:
    """Welford online mean/var fuer feature normalization. Index 0 = Bias-Term,
    bleibt von der Skalierung ausgenommen und nach transform() konstant 1.0."""
    def __init__(self, n):
        self.n = 0
        self.mean = np.zeros(n)
        self.M2 = np.zeros(n)
    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.n += 1
        # Index 0 (Bias) wird nicht in die Statistik aufgenommen — sonst landet
        # mean[0]=1, std[0]=0 -> nach Skalierung waere der Bias 0 statt 1.
        delta = x[1:] - self.mean[1:]
        self.mean[1:] += delta / self.n
        self.M2[1:] += delta * (x[1:] - self.mean[1:])
    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if self.n < 30:
            # Cold-Start: roh, aber clip extreme
            out = np.clip(x, -10.0, 10.0)
        else:
            var = self.M2 / max(1, self.n - 1)
            std = np.sqrt(var)
            std[std < 1e-9] = 1.0
            out = (x - self.mean) / std
            out = np.clip(out, -10.0, 10.0)
        out = out.copy()
        out[0] = 1.0  # Bias-Term immer 1.0
        return out


def vectorize(feat_dict):
    """Feat-Dict → np.array shape (N_FEAT,) mit bias=1.0 an Index 0."""
    x = np.zeros(N_FEAT)
    x[0] = 1.0
    for i, k in enumerate(FEATURE_KEYS, 1):
        v = feat_dict.get(k)
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                x[i] = 0.0
            else:
                x[i] = float(v)
        except (TypeError, ValueError):
            x[i] = 0.0
    return x


class LinTSBandit:
    """Linear Thompson Sampling Contextual Bandit.

    Pro Action ein Bayesian Linear Regression-Modell:
      r = beta_a · x + noise,  beta_a ~ N(mu_a, Sigma_a)
    Update via Sherman-Morrison (rekursiv).
    """
    def __init__(self, n_features, action_specs, alpha=1.0, sigma2=1.0):
        self.n = n_features
        self.actions = action_specs
        self.alpha = alpha
        self.sigma2 = sigma2
        self.A = [alpha * np.eye(n_features) for _ in action_specs]
        self.b = [np.zeros(n_features) for _ in action_specs]
        self._A_inv_cache = [None] * len(action_specs)
        self.n_obs = [0] * len(action_specs)
        self.cum_reward = [0.0] * len(action_specs)

    def _A_inv(self, i):
        if self._A_inv_cache[i] is None:
            try:
                self._A_inv_cache[i] = np.linalg.inv(self.A[i])
            except np.linalg.LinAlgError:
                self._A_inv_cache[i] = np.linalg.pinv(self.A[i])
        return self._A_inv_cache[i]

    def predict_mean(self, i, x):
        return float((self._A_inv(i) @ self.b[i]) @ x)

    def thompson_sample(self, i, x, rng):
        A_inv = self._A_inv(i)
        mu = A_inv @ self.b[i]
        cov = self.sigma2 * A_inv
        try:
            L = np.linalg.cholesky(cov + 1e-9 * np.eye(self.n))
            beta = mu + L @ rng.standard_normal(self.n)
        except np.linalg.LinAlgError:
            beta = mu
        return float(beta @ x)

    def select(self, x, exploration=1.0, rng=None):
        """Wähle Action. exploration ∈ [0,1]: 1 = volle TS-Sampling, 0 = greedy.
        Mischung: mit Wahrscheinlichkeit `exploration` wird TS-sampled, sonst greedy."""
        if rng is None: rng = np.random.default_rng()
        x = np.asarray(x, dtype=float)
        use_sample = (exploration > 0) and (rng.random() < exploration)
        if use_sample:
            scores = np.array([self.thompson_sample(i, x, rng) for i in range(len(self.actions))])
        else:
            scores = np.array([self.predict_mean(i, x) for i in range(len(self.actions))])
        idx = int(np.argmax(scores))
        return idx, float(scores[idx]), scores

    def update(self, i, x, r):
        x = np.asarray(x, dtype=float)
        self.A[i] += np.outer(x, x)
        self.b[i] += float(r) * x
        self._A_inv_cache[i] = None
        self.n_obs[i] += 1
        self.cum_reward[i] += float(r)


class NeuralBandit:
    """Neural Network Bandit fuer den Trader.

    MLP mit Dropout-basiertem Thompson Sampling. Pro Forward-Pass mit aktivem
    Dropout entsteht eine andere Prediction → mean+std ueber K Samples gibt
    Posterior-Approximation. exploration controls noise-Faktor.

    Replay Buffer: alle (features, action_idx, reward) Tupel. Training:
    SGD-Batches via torch.optim.Adam.

    NICHT ueber pickle.dumps direkt persistieren — torch.save/load wird im
    save_state/load_state-Wrapper verwendet."""

    def __init__(self, n_features, n_actions, action_specs,
                 hidden=(128, 64), dropout=0.2, lr=1e-3,
                 buffer_max=50000, ts_samples=5):
        import torch.nn as _nn
        import torch.optim as _optim
        self.n_features = n_features
        self.n_actions = n_actions
        self.actions = action_specs
        self.dropout_p = dropout
        self.ts_samples = ts_samples
        self.buffer_max = buffer_max
        self.lr = lr
        self.hidden = list(hidden)
        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(_nn.Linear(in_dim, h))
            layers.append(_nn.ReLU())
            layers.append(_nn.Dropout(dropout))
            in_dim = h
        layers.append(_nn.Linear(in_dim, n_actions))
        self.net = _nn.Sequential(*layers)
        self.optimizer = _optim.Adam(self.net.parameters(), lr=lr)
        self.replay_buffer = []
        self.n_obs = [0] * n_actions
        self.cum_reward = [0.0] * n_actions
        self.n_train_steps = 0

    def select(self, x, exploration=1.0, rng=None):
        import torch as _torch
        x_arr = np.asarray(x, dtype=np.float32)
        x_t = _torch.from_numpy(x_arr).unsqueeze(0)
        # Dropout aktiv lassen (model.train()) für TS-Samples
        self.net.train()
        samples = []
        with _torch.no_grad():
            for _ in range(self.ts_samples):
                out = self.net(x_t).cpu().numpy()[0]
                samples.append(out)
        self.net.eval()
        samples = np.array(samples)
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
        if rng is None: rng = np.random.default_rng()
        if exploration > 0:
            noise = rng.standard_normal(self.n_actions) * float(exploration)
            scores = means + stds * noise
        else:
            scores = means
        idx = int(np.argmax(scores))
        return idx, float(scores[idx]), scores

    def add_observation(self, x, action_idx, reward):
        x_arr = np.asarray(x, dtype=np.float32).copy()
        self.replay_buffer.append((x_arr, int(action_idx), float(reward)))
        if len(self.replay_buffer) > self.buffer_max:
            del self.replay_buffer[: len(self.replay_buffer) - self.buffer_max]
        self.n_obs[action_idx] += 1
        self.cum_reward[action_idx] += float(reward)

    def train_steps(self, batch_size=64, n_epochs=5, min_buffer=100):
        import torch as _torch
        import torch.nn.functional as _F
        if len(self.replay_buffer) < max(batch_size, min_buffer):
            return 0, 0.0
        self.net.train()
        total = 0.0
        n = 0
        for _ in range(n_epochs):
            idxs = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
            xs = _torch.from_numpy(np.stack([self.replay_buffer[i][0] for i in idxs]))
            actions = _torch.tensor([self.replay_buffer[i][1] for i in idxs], dtype=_torch.long)
            rewards = _torch.tensor([self.replay_buffer[i][2] for i in idxs], dtype=_torch.float32)
            preds = self.net(xs).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = _F.mse_loss(preds, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n += 1
            self.n_train_steps += 1
        self.net.eval()
        return n, total / max(n, 1)

    def state_dict(self):
        return {
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'n_obs': self.n_obs,
            'cum_reward': self.cum_reward,
            'n_train_steps': self.n_train_steps,
            'config': {
                'n_features': self.n_features,
                'n_actions': self.n_actions,
                'hidden': self.hidden,
                'dropout': self.dropout_p,
                'lr': self.lr,
                'buffer_max': self.buffer_max,
                'ts_samples': self.ts_samples,
            },
        }

    @classmethod
    def from_state_dict(cls, sd, action_specs):
        cfg = sd['config']
        b = cls(cfg['n_features'], cfg['n_actions'], action_specs,
                hidden=cfg['hidden'], dropout=cfg['dropout'], lr=cfg['lr'],
                buffer_max=cfg['buffer_max'], ts_samples=cfg['ts_samples'])
        b.net.load_state_dict(sd['net_state'])
        b.optimizer.load_state_dict(sd['optimizer_state'])
        b.replay_buffer = sd['replay_buffer']
        b.n_obs = sd['n_obs']
        b.cum_reward = sd['cum_reward']
        b.n_train_steps = sd.get('n_train_steps', 0)
        return b


def build_actions(tp_buckets, sl_buckets, sides=('long', 'short')):
    actions = [{'name': 'skip', 'side': None, 'tp_pct': None, 'sl_pct': None}]
    seen = set()
    for side in sides:
        for tp in sorted(set(float(x) for x in tp_buckets)):
            for sl in sorted(set(float(x) for x in sl_buckets)):
                name = f'{side}_tp{tp:g}_sl{sl:g}'
                if name in seen: continue
                seen.add(name)
                actions.append({'name': name, 'side': side, 'tp_pct': tp, 'sl_pct': sl})
    return actions


def save_state(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fp:
        pickle.dump(state, fp)
    os.replace(tmp, path)


def load_state(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'rb') as fp: return pickle.load(fp)
    except Exception as e:
        log.warning("load_state %s failed: %s", path, e)
        return None


# =============================================================================
# Helpers (DB-Schreiben / Universe / Telegram)
# =============================================================================

def has_open(app_conn, symbol):
    with app_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM open_predictions WHERE symbol=%s AND status='open' LIMIT 1", (symbol,))
        return cur.fetchone() is not None


def open_prediction(app_conn, symbol, side, entry, sl, tp, score, rule_or_action, source, features,
                     pred_up=None, pred_down=None):
    with app_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO open_predictions
              (symbol, side, entry_px, sl_px, tp_px, score, rule_name, source, features,
               last_px, last_check_at, peak_px, trough_px, predicted_up_pct, predicted_down_pct)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,now(),%s,%s,%s,%s)
            ON CONFLICT DO NOTHING RETURNING id
        """, (symbol, side, entry, sl, tp, score, rule_or_action, source, json.dumps(features),
               entry, entry, entry, pred_up, pred_down))
        r = cur.fetchone()
    app_conn.commit()
    return r[0] if r else None


def refresh_universe(coins_conn, top_n):
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT ON (symbol) symbol, day_ntl_vlm
            FROM hl_asset_ctx ORDER BY symbol, ts DESC
        """)
        rows = cur.fetchall()
    rows = [r for r in rows if r['day_ntl_vlm'] is not None]
    rows.sort(key=lambda r: f(r['day_ntl_vlm']) or 0, reverse=True)
    return [r['symbol'] for r in rows[:top_n]]


def send_telegram(s, text):
    tg = s.get("telegram") or {}
    if not tg.get("bot_token") or not tg.get("chat_id"): return
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": tg["chat_id"], "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=10).read()
    except Exception as e:
        log.warning("telegram failed: %s", e)


# =============================================================================
# Scan-Pass v5 — Multi-Head Predictor (DirectionClassifier + MagnitudeRegressor)
# =============================================================================

def scan_pass_mh(s, mh_model, rng, mh_model_lock):
    """Multi-Head Scan-Pass.

    Pro Coin:
      1. Features berechnen
      2. DirectionClassifier predict_proba → P(long), P(short), P(skip)
      3. side = argmax(long,short); confidence = P(side)
      4. confidence < min_direction_confidence  →  skip
      5. tp_pct = MagnitudeRegressor_<side>.tp(features) × tp_safety_factor
      6. sl_pct = MagnitudeRegressor_<side>.sl(features) × sl_safety_factor
      7. tp_pct < min_tp_pct  oder  sl_pct > max_sl_pct  →  skip
      8. Kandidaten nach confidence DESC, Top-slots_free öffnen
    """
    cfg = s["predictor"]
    mh_cfg = cfg.get("multi_head", {})
    rules = cfg.get("rules", [])
    lookback = cfg["lookback_minutes"]
    cooldown = cfg.get("cooldown_seconds_per_symbol", 300)

    # Threshold-Staffel je nach Modell-Reife
    n_obs = mh_model.direction.n_obs
    cold_start_min_n = int(mh_cfg.get("cold_start_min_n", 200))
    cold_start_threshold = float(mh_cfg.get("cold_start_threshold", 0.50))
    target_threshold = float(mh_cfg.get("min_direction_confidence", 0.95))
    if n_obs < cold_start_min_n:
        direction_threshold = cold_start_threshold
    else:
        direction_threshold = target_threshold

    min_n_for_predict = int(mh_cfg.get("min_n_for_predict", 30))
    tp_safety = float(mh_cfg.get("tp_safety_factor", 0.6))
    sl_safety = float(mh_cfg.get("sl_safety_factor", 1.3))
    min_tp_pct = float(mh_cfg.get("min_tp_pct", 0.5))
    max_sl_pct = float(mh_cfg.get("max_sl_pct", 2.5))
    _mtp = mh_cfg.get("max_tp_pct")  # Sanity-Cap gegen Magnitude-Regressor-Explosion (Billionen %)
    if _mtp is None:
        log.error("FALLBACK_TRIGGERED scan_pass_mh: predictor.multi_head.max_tp_pct fehlt -> abort")
        return 0
    max_tp_pct = float(_mtp)
    max_open = int(mh_cfg.get("max_open_predictions", 100))

    matches = 0

    with db_coins(s) as coins, db_app(s) as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur_a:
            cur_a.execute("SELECT universe, universe_refreshed_at FROM predictor_state WHERE id=1")
            st = cur_a.fetchone()
        need_refresh = (
            not st or not st['universe'] or not st['universe_refreshed_at']
            or (datetime.now(timezone.utc) - st['universe_refreshed_at']).total_seconds()
                > cfg["universe_refresh_minutes"] * 60
        )
        if need_refresh:
            uni = refresh_universe(coins, cfg["universe_top_n"])
            with app.cursor() as cur_a:
                cur_a.execute(
                    "UPDATE predictor_state SET universe=%s::jsonb, universe_refreshed_at=now(), updated_at=now() WHERE id=1",
                    (json.dumps(uni),))
            app.commit()
            log.info("universe refreshed: %d symbols", len(uni))
        else:
            uni = st['universe']

        # Slot-Logik (Volker 19.05.): Predictions immer in DB, HL-Trade nur
        # solange offene trader_positions < max_open. Predictions die wegen
        # Slot-Limit nicht getradet werden bekommen direkt auto_trade_skipped=TRUE
        # (UI hidet, Predictor lernt trotzdem aus Klines).
        with app.cursor() as cur_a:
            cur_a.execute("SELECT COUNT(*) FROM open_predictions WHERE status='open'")
            n_open = cur_a.fetchone()[0]
            cur_a.execute("SELECT COUNT(*) FROM trader_positions WHERE status='open'")
            hl_open_at_start = cur_a.fetchone()[0]
        auto_trade_active = bool(cfg.get("trading", {}).get("auto_trade"))
        paper_mode = bool(cfg.get("trading", {}).get("paper_mode"))
        if n_obs < cold_start_min_n:
            cold_cap = int(mh_cfg.get("cold_start_max_per_scan", 10))
            hl_quota_this_scan = min(max(0, max_open - hl_open_at_start), cold_cap)
        else:
            hl_quota_this_scan = max(0, max_open - hl_open_at_start)
        hl_traded_this_scan = 0

        # Cooldown: letzten Close pro Symbol bulk-fetchen.
        last_closes = {}
        if cooldown > 0 and uni:
            with app.cursor(cursor_factory=RealDictCursor) as cur_a:
                cur_a.execute("""
                    SELECT symbol, MAX(closed_at) AS last_close
                    FROM open_predictions
                    WHERE symbol = ANY(%s) AND closed_at IS NOT NULL
                    GROUP BY symbol
                """, (uni,))
                last_closes = {r['symbol']: r['last_close'] for r in cur_a.fetchall()}

        btc_moves = load_btc_moves(coins)
        if btc_moves is None:
            log.error("FALLBACK_TRIGGERED scan_pass_mh: load_btc_moves None -> scan-pass abgebrochen")
            return 0

        # Universe-level pre-computes für Sektor + Funding-Universe-Median
        sector_priority = cfg.get("sector_priority", [])
        coin_sector_map = build_coin_sector_map(app, uni, sector_priority)
        sector_stats, coin_pcts = compute_sector_close_pcts(coins, coin_sector_map)
        universe_funding_median = compute_universe_funding_median(coins, uni)
        whale_enabled = bool(cfg.get("whale_tracker", {}).get("enabled", False))

        # Stalker (v5.1): Reference-Cache + BTC-Regime + effective min_samples 1× pro scan-pass
        stalker_cfg = cfg.get("stalker", {}) or {}
        stalker_on = bool(stalker_cfg.get("enabled"))
        stalker_ref_cache = None
        stalker_btc_regime = None
        stalker_eff_min_samples = None
        if stalker_on:
            stalker_ref_cache = stalker_build_ref_cache(coins, stalker_cfg["cross_coin"])
            stalker_btc_regime = stalker_classify_btc_regime(coins, stalker_cfg["btc_regime"])
            if stalker_btc_regime is None:
                log.error("FALLBACK_TRIGGERED scan_pass_mh: Stalker-BTC-Regime nicht ermittelbar -> Stalker für diesen scan-pass aus")
                stalker_on = False
            else:
                _dd = stalker_data_days(coins)
                stalker_eff_min_samples = stalker_effective_min_samples(stalker_cfg, _dd)

        # === Worker-Pool: Features parallel, KI/HL seriell ===
        # Volker-Direktive 19.05.2026: „nach dem Prinzip dass Daten aufbereitet werden,
        # damit der prediktor schneller entscheiden kann, und die aktuellsten Daten
        # verwendet werden". Worker-Anzahl aus settings.predictor.multi_head.scan_workers.
        import threading as _th
        from concurrent.futures import ThreadPoolExecutor as _Pool
        _n_w = mh_cfg.get("scan_workers")
        if _n_w is None:
            log.error("FALLBACK_TRIGGERED scan_pass_mh: predictor.multi_head.scan_workers fehlt -> abort")
            return 0
        n_workers = int(_n_w)
        stats_lock = _th.Lock()
        slot_lock = _th.Lock()
        stats = {"cold_skip":0,"below":0,"no_mag":0,"filt":0,
                 "no_of":0,"no_fb":0,"no_sec":0,"no_bl":0,
                 "hl_traded":0,"matches":0}
        cold_start_active = (n_obs < cold_start_min_n)

        def _bump(key, n=1):
            with stats_lock:
                stats[key] += n

        def _process_one(sym):
            try:
                with db_coins(s) as coins_w, db_app(s) as app_w:
                    if has_open(app_w, sym): return
                    last_close = last_closes.get(sym)
                    if last_close is not None:
                        age = (datetime.now(timezone.utc) - last_close).total_seconds()
                        if age < cooldown:
                            return

                    rule_flags = {}
                    with coins_w.cursor(cursor_factory=RealDictCursor) as cur_w:
                        for rule in rules:
                            try:
                                ok = evaluate_rule(cur_w, sym, rule, lookback)["ok"]
                            except Exception:
                                ok = False
                            rule_flags[rule['name']] = 1 if ok else 0

                    feat = feature_snapshot_v2(coins_w, sym, rule_flags=rule_flags, btc_moves=btc_moves)
                    if feat is None: return

                    of_feat = compute_orderflow_features(coins_w, sym)
                    if of_feat is None: _bump("no_of"); return
                    feat.update(of_feat)

                    fb_feat = compute_funding_features(coins_w, sym, universe_funding_median)
                    if fb_feat is None: _bump("no_fb"); return
                    feat.update(fb_feat)

                    sec_feat = compute_sector_features(sym, coin_sector_map, sector_stats, coin_pcts)
                    if sec_feat is None: _bump("no_sec"); return
                    feat.update(sec_feat)

                    if whale_enabled:
                        wh_feat = compute_whale_features(app_w, sym)
                        if wh_feat: feat.update(wh_feat)

                    if stalker_on:
                        bl_feat = compute_baseline_dev_features(
                            coins_w, app_w, sym, stalker_cfg,
                            btc_regime=stalker_btc_regime,
                            effective_min_samples=stalker_eff_min_samples,
                        )
                        if bl_feat is None: _bump("no_bl"); return
                        feat.update(bl_feat)
                        cc_feat = compute_cross_coin_features(coins_w, sym, stalker_cfg["cross_coin"], stalker_ref_cache)
                        feat.update(cc_feat)
                        id_feat = compute_identity_features(sym, app_w, stalker_cfg["coin_identity"])
                        feat.update(id_feat)

                    # KI-Calls — River nicht thread-safe → mh_model_lock
                    with mh_model_lock:
                        probs = mh_model.predict_proba(feat, min_n_for_predict=min_n_for_predict)
                    _process_after_predict(sym, feat, probs, coins_w, app_w)
            except Exception as e:
                log.exception("worker %s failed: %s", sym, e)

        def _process_after_predict(sym, feat, probs, coins_w, app_w):
            # Direction-Klassifikator (oder Cold-Start: random side)
            if probs is None:
                if not cold_start_active:
                    _bump("cold_skip"); return
                side = 'long' if rng.random() < 0.5 else 'short'
                confidence = 0.5
                p_long = 0.5; p_short = 0.5; p_skip = 0.0
            else:
                p_long = probs['long']; p_short = probs['short']
                if p_long >= p_short:
                    side = 'long'; confidence = p_long
                else:
                    side = 'short'; confidence = p_short
                p_skip = probs['skip']
                if not cold_start_active and confidence < direction_threshold:
                    _bump("below"); return

            # Magnitude-Regressoren mit Lock
            with mh_model_lock:
                tp_raw = mh_model.predict_tp(feat, side, min_n_for_predict=min_n_for_predict)
                sl_raw = mh_model.predict_sl(feat, side, min_n_for_predict=min_n_for_predict)
            if tp_raw is None or sl_raw is None:
                if not cold_start_active:
                    _bump("no_mag"); return
                cs_tp = mh_cfg.get("cold_start_tp_raw_pct")
                cs_sl = mh_cfg.get("cold_start_sl_raw_pct")
                if cs_tp is None or cs_sl is None:
                    log.error("FALLBACK_TRIGGERED scan_pass_mh cold-start: cold_start_tp_raw_pct/cold_start_sl_raw_pct fehlen")
                    return
                tp_raw = float(cs_tp); sl_raw = float(cs_sl)

            tp_pct = tp_raw * tp_safety
            sl_pct = max(sl_raw * sl_safety, 0.1)

            if cold_start_active:
                cs_tp = float(mh_cfg.get("cold_start_tp_raw_pct"))
                cs_sl = float(mh_cfg.get("cold_start_sl_raw_pct"))
                if tp_pct < min_tp_pct: tp_pct = cs_tp * tp_safety
                if sl_pct > max_sl_pct: sl_pct = cs_sl * sl_safety
                if tp_pct > max_tp_pct: tp_pct = cs_tp * tp_safety
            elif tp_pct < min_tp_pct or tp_pct > max_tp_pct or sl_pct > max_sl_pct:
                _bump("filt"); return

            with coins_w.cursor(cursor_factory=RealDictCursor) as cur_w:
                cur_w.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
                p = cur_w.fetchone()
            if not p or p['close'] is None: return
            entry = f(p['close'])
            if side == 'long':
                tp = entry * (1 + tp_pct / 100.0); sl = entry * (1 - sl_pct / 100.0)
            else:
                tp = entry * (1 - tp_pct / 100.0); sl = entry * (1 + sl_pct / 100.0)

            feat_with_meta = dict(feat)
            feat_with_meta['_predicted_side'] = side
            feat_with_meta['_confidence'] = float(confidence)
            feat_with_meta['_tp_raw_pct'] = float(tp_raw); feat_with_meta['_sl_raw_pct'] = float(sl_raw)
            feat_with_meta['_p_long'] = float(p_long); feat_with_meta['_p_short'] = float(p_short); feat_with_meta['_p_skip'] = float(p_skip)

            action_name = f"{side}_tp{tp_pct:.2f}_sl{sl_pct:.2f}"
            pid = open_prediction(app_w, sym, side, entry, sl, tp,
                                   confidence, action_name, 'multi_head', feat_with_meta,
                                   pred_up=tp_pct, pred_down=sl_pct)
            if not pid: return
            _bump("matches")

            # Slot-Frage + Trade unter slot_lock. Reservierung NUR im Lock (ms);
            # Order-Ausführung (HL-API / Paper) AUSSERHALB — sonst blockiert ein
            # HL-Hang alle Worker (Deadlock, 20.05.2026).
            # paper_mode: Paper-Position (paper_positions) statt HL, Slot = max_open
            # paper_positions. Sonst echte HL-Order. Gleiche Slot/Hebel/Size-Regeln.
            cold_burst_max = mh_cfg.get("cold_start_max_per_scan")
            if cold_burst_max is None:
                log.error("FALLBACK_TRIGGERED scan_pass_mh: predictor.multi_head.cold_start_max_per_scan fehlt")
                return
            active_world = paper_mode or auto_trade_active  # öffnet diese Welt überhaupt Trades?
            do_trade = False
            hl_now = hl_open_at_start
            with slot_lock:
                if paper_mode:
                    hl_now = paper_count_open(app_w)
                elif auto_trade_active:
                    with app_w.cursor() as cur_sl:
                        cur_sl.execute("SELECT COUNT(*) FROM trader_positions WHERE status='open'")
                        hl_now = cur_sl.fetchone()[0]
                hl_effective = hl_now + stats.get("hl_reserved", 0)
                cold_burst_ok = (n_obs >= cold_start_min_n) or (stats["hl_traded"] < int(cold_burst_max))
                if active_world and hl_effective < max_open and cold_burst_ok:
                    stats["hl_reserved"] = stats.get("hl_reserved", 0) + 1
                    do_trade = True

            if do_trade and paper_mode:
                # Paper: Entry = frischer Preis (entry wurde gerade aus agg_1m geholt)
                try:
                    paper_open(app_w, coins_w, pid, sym, side, entry, tp, sl, cfg)
                finally:
                    with slot_lock:
                        stats["hl_reserved"] = max(0, stats.get("hl_reserved", 0) - 1)
                        stats["hl_traded"] += 1
            elif do_trade:
                log.info("OPEN %s %s entry=%.6g tp=%.6g sl=%.6g (tp%%=%.3f sl%%=%.3f) P=%.3f [%d/%d HL-open]",
                         sym, side, entry, tp, sl, tp_pct, sl_pct, confidence, hl_now, max_open)
                try:
                    try_auto_trade(s, pid, sym, side, entry, tp, sl, tp_pct, sl_pct)
                finally:
                    with slot_lock:
                        stats["hl_reserved"] = max(0, stats.get("hl_reserved", 0) - 1)
                        stats["hl_traded"] += 1
            elif active_world:
                # Slot voll / cold-burst → skipped (nur wenn diese Welt überhaupt tradet)
                with app_w.cursor() as cur_sk:
                    cur_sk.execute("UPDATE open_predictions SET auto_trade_skipped=TRUE WHERE id=%s", (pid,))
                app_w.commit()
                reason = f"slot_full({hl_now}/{max_open})" if hl_now >= max_open else "cold_burst_cap"
                log.info("OPEN+SKIP %s %s pid=%s (%s) entry=%.6g P=%.3f",
                         sym, side, pid, reason, entry, confidence)
            # else (kein paper, kein auto_trade): manuelle Welt — Prediction bleibt
            # sichtbar (kein skip), Volker traded selbst.

        # === Worker-Pool starten ===
        with _Pool(max_workers=n_workers) as pool:
            list(pool.map(_process_one, uni))

        log.info("scan: hl_open=%d max=%d quota_this_scan=%d traded=%d, %d preds geöffnet "
                 "(n_obs=%d, workers=%d, cold=%d, below=%d, no_mag=%d, no_of=%d, no_fb=%d, no_sec=%d, no_bl=%d, filt=%d, stalker=%s min_s=%s)",
                 hl_open_at_start, max_open, hl_quota_this_scan, stats["hl_traded"], stats["matches"],
                 n_obs, n_workers, stats["cold_skip"], stats["below"], stats["no_mag"],
                 stats["no_of"], stats["no_fb"], stats["no_sec"], stats["no_bl"], stats["filt"],
                 stalker_btc_regime if stalker_on else "off",
                 stalker_eff_min_samples if stalker_on else "-")
        matches = stats["matches"]
    return matches


def try_auto_trade(s, pid, symbol, side, predicted_entry, predicted_tp, predicted_sl,
                    predicted_tp_pct, predicted_sl_pct):
    """Wenn predictor.trading.auto_trade=True und Side erlaubt: Order ausfuehren.
    TP/SL kommen entweder aus der Prediction (use_predictor_targets=True) oder
    aus globalen %-Vorgaben (mit optional Long/Short-Split)."""
    cfg = s.get("predictor", {})
    t = cfg.get("trading", {})
    if not t.get("auto_trade"):
        return
    if side == "long" and not t.get("auto_long_enabled", True):
        return
    if side == "short" and not t.get("auto_short_enabled", True):
        return

    # TP/SL-Strategie:
    #   use_predictor_targets=True: nimm Bandit-TP/SL absolut, KEIN re-anchor
    #     auf Fill-Preis (Bandit-Levels bleiben erhalten)
    #   use_predictor_targets=False: nutze globale %, re-anchor auf Fill-Preis
    use_pred = t.get("use_predictor_targets", True)
    tp_pct = sl_pct = None
    if use_pred:
        tp_px = predicted_tp; sl_px = predicted_sl
    else:
        if t.get("long_short_split", False):
            if side == "long":
                tp_pct = float(t.get("global_tp_pct_long", 3.0))
                sl_pct = float(t.get("global_sl_pct_long", 1.5))
            else:
                tp_pct = float(t.get("global_tp_pct_short", 3.0))
                sl_pct = float(t.get("global_sl_pct_short", 1.5))
        else:
            tp_pct = float(t.get("global_tp_pct", 3.0))
            sl_pct = float(t.get("global_sl_pct", 1.5))
        # Initial-Werte gegen mid (werden in execute_order_with_failsafe gegen
        # actual fill-price neu gerechnet — siehe tp_pct/sl_pct-Param unten).
        if side == "long":
            tp_px = predicted_entry * (1 + tp_pct / 100.0)
            sl_px = predicted_entry * (1 - sl_pct / 100.0)
        else:
            tp_px = predicted_entry * (1 - tp_pct / 100.0)
            sl_px = predicted_entry * (1 + sl_pct / 100.0)

    leverage = int(t.get("default_leverage", 5))
    size_usd = float(t.get("default_size_usd", 20))
    slippage_pct = float(t.get("order_slippage_pct", 1.0))

    # Order-Strategie: Limit-Order zum predicted_entry direkt (oder besser).
    # skip_mid_lookup=true spart den HL-/all-mids-Roundtrip vor jedem Auto-Trade
    # (~200-500ms) — der Predictor-Entry IST der Soll-Preis, alles schlechter als
    # max_slippage_pct wird abgelehnt → Prediction bleibt 'open' aber
    # auto_trade_skipped=TRUE (UI hidet, Klines-Tracking + Hindsight-Lernen bleibt).
    skip_mid = bool(t.get("skip_mid_lookup", True))
    max_slip = float(t.get("max_slippage_pct"))  # NO-FALLBACK: settings.predictor.trading.max_slippage_pct
    try:
        from rl_agent.trader import get_current_prices_hl, get_hl_credentials
        from predictor.order_executor import execute_order_with_failsafe
        creds = get_hl_credentials(user_id=1)
        if not creds:
            log.warning("auto-trade %s: keine HL-Creds", symbol); return
        if skip_mid:
            entry_live = float(predicted_entry)
        else:
            mids = get_current_prices_hl()
            live_px = mids.get(symbol) if mids else None
            if not live_px:
                log.warning("auto-trade %s: kein Live-Preis von HL", symbol); return
            entry_live = float(live_px)
        res = execute_order_with_failsafe(
            creds, symbol, is_long=(side == "long"),
            leverage=leverage, size_usd=size_usd,
            tp_px=tp_px, sl_px=sl_px,
            entry_px=entry_live, slippage_pct=max_slip,
            tp_pct=tp_pct, sl_pct=sl_pct,
        )
        eff_lev = res.get("effective_leverage", leverage)
        if res["success"]:
            # NUR effective_leverage in DB schreiben. tp_px/sl_px bleiben die
            # Bandit-Werte (fuer Bandit-Lernen + Hindsight-Replay) — die HL-Order
            # darf andere TP/SL haben (use_predictor_targets=False), das ist OK.
            # Bandit-Welt und Trade-Welt sind getrennt.
            try:
                with db_app(s) as app:
                    with app.cursor() as cur:
                        cur.execute("UPDATE open_predictions SET effective_leverage=%s WHERE id=%s",
                                    (eff_lev, pid))
                    app.commit()
            except Exception as e:
                log.warning("DB effective_leverage pid=%s failed: %s", pid, e)

            # OPEN-HOOK Predictor -> Trader: nach erfolgreichem Auto-Trade
            # einen trader_positions Eintrag anlegen. Trader uebernimmt ab hier.
            actual_tp = res.get("tp_px") or tp_px
            actual_sl = res.get("sl_px") or sl_px
            actual_qty = float(res.get("filled_qty") or 0)
            try:
                features_snapshot = None
                with db_app(s) as app:
                    with app.cursor(cursor_factory=RealDictCursor) as cur_f:
                        cur_f.execute("SELECT features FROM open_predictions WHERE id=%s", (pid,))
                        rf = cur_f.fetchone()
                        features_snapshot = rf['features'] if rf else None
                    with app.cursor() as cur_t:
                        cur_t.execute("""
                            INSERT INTO trader_positions
                              (prediction_id, symbol, side, entry_px, qty, leverage,
                               original_tp_px, original_sl_px, current_tp_px, current_sl_px,
                               peak_px, trough_px, status, features_at_open, timeout_enabled)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', %s::jsonb, TRUE)
                            RETURNING id
                        """, (pid, symbol, side, float(predicted_entry), actual_qty, eff_lev,
                               float(actual_tp), float(actual_sl),
                               float(actual_tp), float(actual_sl),
                               float(predicted_entry), float(predicted_entry),
                               json.dumps(features_snapshot) if features_snapshot else None))
                        trader_pid = cur_t.fetchone()[0]
                    app.commit()
                log.info("AUTO-TRADE OK %s %s pid=%s trader_pid=%s margin=$%.0f lev=%dx (notional=$%.0f) tp=%.6g sl=%.6g",
                         symbol, side, pid, trader_pid, size_usd, eff_lev, size_usd * eff_lev, actual_tp, actual_sl)
            except Exception as e:
                log.exception("OPEN-HOOK trader_positions insert pid=%s failed: %s", pid, e)
                log.info("AUTO-TRADE OK %s %s pid=%s margin=$%.0f lev=%dx (notional=$%.0f) tp=%.6g sl=%.6g",
                         symbol, side, pid, size_usd, eff_lev, size_usd * eff_lev, actual_tp, actual_sl)
        else:
            # CARDINAL RULE_1b: status bleibt 'open' — Predictor lernt weiter aus Klines.
            # ZUSATZ-Flag auto_trade_skipped=TRUE → UI hidet die Prediction im Default-View,
            # aber watch_pass + Hindsight + Bandit-Lernen laufen unverändert weiter.
            try:
                with db_app(s) as app:
                    with app.cursor() as cur:
                        cur.execute(
                            "UPDATE open_predictions SET auto_trade_skipped=TRUE WHERE id=%s",
                            (pid,),
                        )
                    app.commit()
            except Exception as e:
                log.warning("set auto_trade_skipped pid=%s failed: %s", pid, e)
            log.error("AUTO-TRADE SKIPPED %s %s pid=%s lev=%dx: %s (status='open', UI hidden, Klines-Tracking weiter)",
                      symbol, side, pid, eff_lev, res.get("error"))
    except Exception as e:
        # HL-API-Fehler (Rate-Limit, Timeout, etc.) beruehren die Predictor-Welt nicht.
        # open_predictions laeuft via watch_pass_v4 mit Klines-Tracking weiter.
        # Auch hier: skipped-Flag setzen damit UI konsistent ist.
        try:
            with db_app(s) as app:
                with app.cursor() as cur:
                    cur.execute(
                        "UPDATE open_predictions SET auto_trade_skipped=TRUE WHERE id=%s",
                        (pid,),
                    )
                app.commit()
        except Exception:
            pass
        log.exception("auto-trade %s scheiterte (UI hidden, Predictor laeuft via Klines weiter): %s", symbol, e)


def _mark_predict_closed(s, pid, status, exit_px):
    """Schliesst die DB-Row mit gegebenem Status und exit_px, pnl_pct=0.
    Wird bei Failsafe-Close, Order-Failure und Exception aus try_auto_trade aufgerufen."""
    with db_app(s) as app:
        with app.cursor() as cur:
            cur.execute("""
                UPDATE open_predictions
                SET status=%s, exit_px=%s, pnl_pct=0, closed_at=now(), last_check_at=now()
                WHERE id=%s AND status='open'
            """, (status, exit_px, pid))
        app.commit()


# =============================================================================
# Hindsight-Replay — Bandit lernt aus jedem Trade fuer ALLE 31 Actions
# =============================================================================

def _win_mult_from_tiers(hit_quote_pct, tiers):
    """Bestimmt Reward-Multiplikator anhand Trefferquote (gewaehlter_TP / max_moeglich * 100).

    tiers: Liste [[schwelle_pct, mult], ...] sortiert nach Schwelle aufsteigend.
    Letzter Eintrag = "darueber". Beispiel default:
      [[50, 0.75], [70, 1.0], [80, 1.2], [90, 1.5], [9999, 2.5]]
    bedeutet: <=50% -> 0.75, 51-70% -> 1.0, 71-80% -> 1.2, 81-90% -> 1.5, >90% -> 2.5
    """
    if not tiers:
        return 1.0
    for threshold, mult in tiers:
        if hit_quote_pct <= float(threshold):
            return float(mult)
    return float(tiers[-1][1])


def replay_action(klines_rows, entry, side, tp_pct, sl_pct, timeout_h,
                   time_penalty_per_h, max_possible_pct, reward_cfg):
    """Simuliert einen hypothetischen Trade durch die echte 10s-Klines-Trajectory.

    klines_rows: list of dict mit open_time, high, low, close (sortiert ASC)
    side: 'long' / 'short' / None (skip — wird vom Caller separat berechnet)
    max_possible_pct: max moegliche Coin-Bewegung in % im timeout-Fenster fuer diese Side
                     (vom Caller berechnet, gleich fuer alle Actions derselben Side).
    reward_cfg: dict mit reward_mult_tiers, reward_timeout_mult_pos, reward_timeout_mult_neg
    Returns: dict mit status, pnl_pct, duration_h, reward.
      reward=None signalisiert "keinen Bandit-Update fuer diese Action durchfuehren"
      (skip-Platzhalter oder Daten reichen nicht aus fuer faire Bewertung).
    """
    if side is None:
        # Skip-Reward wird im Caller relativ zu den Trade-Action-Rewards berechnet.
        return {'status': 'skip', 'pnl_pct': 0.0, 'duration_h': 0.0, 'reward': None}

    if not klines_rows:
        return {'status': 'no_data', 'pnl_pct': 0.0, 'duration_h': 0.0, 'reward': None}

    if side == 'long':
        tp_price = entry * (1 + tp_pct / 100.0)
        sl_price = entry * (1 - sl_pct / 100.0)
    else:
        tp_price = entry * (1 - tp_pct / 100.0)
        sl_price = entry * (1 + sl_pct / 100.0)

    start_time = klines_rows[0]['open_time']

    mult_tiers = reward_cfg.get('reward_mult_tiers') or [[50, 0.75], [70, 1.0], [80, 1.2], [90, 1.5], [9999, 2.5]]
    timeout_mult_pos = float(reward_cfg.get('reward_timeout_mult_pos', 0.3))
    timeout_mult_neg = float(reward_cfg.get('reward_timeout_mult_neg', 2.0))

    for r in klines_rows:
        h = r.get('high'); l = r.get('low'); c = r.get('close')
        if h is None or l is None: continue
        h = float(h); l = float(l)

        elapsed_h = (r['open_time'] - start_time).total_seconds() / 3600.0
        if elapsed_h >= timeout_h:
            # Echter Timeout: Trade haengt timeout_h ohne TP/SL-Hit
            # Volker-Regel (16.05.2026): pnl >=0 -> *0.3 (weniger belohnen), pnl <0 -> *2 (haerter bestrafen)
            close_v = float(c) if c is not None else entry
            pnl = ((close_v - entry) / entry * 100) if side == 'long' else ((entry - close_v) / entry * 100)
            base = pnl - elapsed_h * time_penalty_per_h
            reward = base * (timeout_mult_pos if pnl >= 0 else timeout_mult_neg)
            return {'status': 'timeout', 'pnl_pct': pnl, 'duration_h': elapsed_h, 'reward': reward}

        # TP/SL-Check pro Bucket
        if side == 'long':
            sl_hit = l <= sl_price
            tp_hit = h >= tp_price
        else:
            sl_hit = h >= sl_price
            tp_hit = l <= tp_price

        # Beide im selben Bucket → konservativ SL annehmen (worst case)
        if sl_hit:
            pnl = -float(sl_pct)
            reward = pnl - elapsed_h * time_penalty_per_h
            return {'status': 'loss', 'pnl_pct': pnl, 'duration_h': elapsed_h, 'reward': reward}
        if tp_hit:
            pnl = float(tp_pct)
            base = pnl - elapsed_h * time_penalty_per_h
            # Trefferquoten-Multiplikator nur fuer positive Wins (negative bei Slippage-Edge: unveraendert)
            if base > 0 and max_possible_pct and max_possible_pct > 0:
                hit_quote_pct = (tp_pct / max_possible_pct) * 100.0
                mult = _win_mult_from_tiers(hit_quote_pct, mult_tiers)
                reward = base * mult
            else:
                reward = base
            return {'status': 'win', 'pnl_pct': pnl, 'duration_h': elapsed_h, 'reward': reward}

    # Trajectory zu Ende, kein TP/SL-Hit und kein echter Timeout (z.B. realer Trade
    # schloss frueh, Range zu kurz fuer diese tp/sl-Kombi). Faire Bewertung nicht
    # moeglich -> reward=None, kein Bandit-Update.
    last = klines_rows[-1]
    last_close = float(last.get('close')) if last.get('close') is not None else entry
    elapsed_h = (last['open_time'] - start_time).total_seconds() / 3600.0
    pnl = ((last_close - entry) / entry * 100) if side == 'long' else ((entry - last_close) / entry * 100)
    return {'status': 'incomplete', 'pnl_pct': pnl, 'duration_h': elapsed_h, 'reward': None}


def hindsight_replay_for_prediction(coins_conn, bandit, scaler, prediction, cfg):
    """Holt die Klines im Trade-Range, replay alle Actions, bandit-update fuer alle
    Actions mit valider Bewertung.

    Range = [created_at, min(created_at + timeout_h, now())] — damit hat jede
    Action die volle Bewertungs-Spanne, unabhaengig davon wann der reale Trade schloss.

    Skip-Reward = -max(positive_trade_rewards) wenn es eine profitable Trade-Action
    gab, sonst reward_skip. Heisst: skip wird nur dann belohnt wenn keine Trade-Action
    profitabel war (Coin hat im Range nicht hinreichend gezappelt).

    Returns: (n_actions_updated, dict pro Action mit reward, dict mit chosen_action_replay)
    """
    bandit_cfg = cfg.get("bandit", {})
    # cfg ist die predictor-Sektion -> get_timeout_hours braucht volle settings:
    timeout_h = get_timeout_hours({"predictor": cfg})
    # Hindsight-Eval-Fenster = timeout_h + extended_minutes (sieht ob Trade in
    # den extra Minuten noch getriggert haette -> Bandit lernt "zu frueh eroeffnet").
    extended_min = float(bandit_cfg.get("hindsight_extended_minutes", 30))
    eval_timeout_h = timeout_h + extended_min / 60.0
    time_penalty = float(bandit_cfg.get("reward_time_penalty_per_hour", 0.15))
    reward_skip = float(bandit_cfg.get("reward_skip", 0.1))
    # Reward-Multiplikator-Config (Volker-Spec 16.05.2026)
    reward_cfg = {
        'reward_mult_tiers': bandit_cfg.get('reward_mult_tiers',
                                            [[50, 0.75], [70, 1.0], [80, 1.2], [90, 1.5], [9999, 2.5]]),
        'reward_timeout_mult_pos': bandit_cfg.get('reward_timeout_mult_pos', 0.3),
        'reward_timeout_mult_neg': bandit_cfg.get('reward_timeout_mult_neg', 2.0),
    }

    feat = prediction.get('features') or {}
    if not isinstance(feat, dict) or not feat:
        return 0, {}, None
    entry = float(prediction['entry_px'])

    # Volle Bewertungsspanne. min(now()) damit wir nicht in die Zukunft fragen,
    # wenn der Trade noch jung ist.
    full_range_end = prediction['created_at'] + timedelta(hours=eval_timeout_h)
    now_utc = datetime.now(timezone.utc)
    range_end = min(full_range_end, now_utc)

    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, high, low, close FROM klines
            WHERE symbol=%s AND interval='10s'
              AND open_time BETWEEN %s AND %s
            ORDER BY open_time
        """, (prediction['symbol'], prediction['created_at'], range_end))
        klines_rows = cur.fetchall()

    if len(klines_rows) < 3:
        return 0, {}, None

    # Max moegliche Bewegung pro Side im timeout-Fenster (global pro Prediction,
    # gleicher Wert fuer alle Actions derselben Side -> kein Decision-Loophole).
    max_long_pct = 0.0
    max_short_pct = 0.0
    for r in klines_rows:
        h = r.get('high'); l = r.get('low')
        if h is None or l is None: continue
        long_move = (float(h) - entry) / entry * 100.0
        short_move = (entry - float(l)) / entry * 100.0
        if long_move > max_long_pct: max_long_pct = long_move
        if short_move > max_short_pct: max_short_pct = short_move

    x_raw = vectorize(feat)
    scaler.update(x_raw)
    x = scaler.transform(x_raw)

    chosen_idx = feat.get('_action_idx')
    n_updated = 0
    rewards_by_action = {}
    chosen_replay = None
    skip_idx = None

    # Phase 1: alle Trade-Actions replayen, Bandit nur fuer Actions mit reward!=None updaten.
    for idx, action in enumerate(bandit.actions):
        if action['side'] is None:
            skip_idx = idx
            continue
        max_possible = max_long_pct if action['side'] == 'long' else max_short_pct
        replay = replay_action(
            klines_rows, entry,
            action['side'], action['tp_pct'] or 0.0, action['sl_pct'] or 0.0,
            eval_timeout_h, time_penalty, max_possible, reward_cfg,
        )
        rewards_by_action[action['name']] = replay
        if replay['reward'] is not None:
            bandit.update(idx, x, replay['reward'])
            n_updated += 1
        if isinstance(chosen_idx, int) and idx == chosen_idx:
            chosen_replay = replay

    # Phase 2: Skip-Reward = -max(positive_trade_rewards). Wenn keine Trade-Action
    # positives Reward hatte: skip war richtig -> reward_skip.
    if skip_idx is not None:
        positive_rewards = [r['reward'] for r in rewards_by_action.values()
                            if r.get('reward') is not None and r['reward'] > 0]
        if positive_rewards:
            skip_reward = -max(positive_rewards)
        else:
            skip_reward = reward_skip
        bandit.update(skip_idx, x, skip_reward)
        n_updated += 1
        skip_replay = {'status': 'skip', 'pnl_pct': 0.0, 'duration_h': 0.0,
                       'reward': skip_reward}
        rewards_by_action[bandit.actions[skip_idx]['name']] = skip_replay
        if isinstance(chosen_idx, int) and skip_idx == chosen_idx:
            chosen_replay = skip_replay

    return n_updated, rewards_by_action, chosen_replay


def backfill_hindsight(s, bandit, scaler):
    """Einmalig: replay alle bisher geschlossenen Predictions mit Hindsight.
    Rekonstruiert dabei die virtuelle Wallet chronologisch und stempelt
    den Drawdown-Wert in jedes Features-Snapshot vor dem Bandit-Update."""
    log.info("Hindsight-Backfill startet...")
    cfg = s["predictor"]
    bandit_cfg = cfg.get("bandit", {})
    cutoff = bandit_cfg.get("backfill_closed_at_cutoff")  # ISO-Date z.B. '2026-05-12' fuer Soft-Restart
    vwallet_cfg = cfg.get("virtual_wallet", {})
    start_balance = float(vwallet_cfg.get("start_balance", 200.0))
    margin = float(vwallet_cfg.get("margin_per_trade", 20.0))
    slip_pct = float(vwallet_cfg.get("slippage_pct_per_trade", 0.1))

    n_preds = 0; n_updates = 0
    with db_app(s) as app, db_coins(s) as coins:
        # Wallet auf Start-Balance zuruecksetzen — Rekonstruktion startet bei Null-Drawdown
        try:
            reset_virtual_wallet(app, start_balance)
            log.info("Virtuelle Wallet reset auf $%.2f", start_balance)
        except Exception as e:
            log.warning("reset_virtual_wallet failed: %s", e)

        with app.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT id, symbol, side, entry_px, sl_px, tp_px, features,
                       created_at, closed_at, rule_name, status, pnl_pct,
                       effective_leverage
                FROM open_predictions
                WHERE status IN ('win','loss','timeout')
                  AND closed_at IS NOT NULL AND features IS NOT NULL
            """
            params = []
            if cutoff:
                query += " AND closed_at >= %s"
                params.append(cutoff)
                log.info("Backfill-Cutoff aktiv: nur Closes ab %s", cutoff)
            query += " ORDER BY closed_at"
            cur.execute(query, params)
            preds = cur.fetchall()
        if not preds:
            log.info("Backfill: keine geschlossenen Predictions vorhanden")
            return 0, 0
        for p in preds:
            try:
                # Aktuellen Drawdown-Wert holen (vor diesem Trade) und in
                # Features-Snapshot einstempeln.
                feat = dict(p['features']) if isinstance(p['features'], dict) else {}
                feat['wallet_drawdown_pct'] = get_wallet_drawdown_pct(app)
                pred_dict = {
                    'symbol': p['symbol'], 'side': p['side'],
                    'entry_px': p['entry_px'], 'features': feat,
                    'created_at': p['created_at'], 'closed_at': p['closed_at'],
                }
                n, _, _ = hindsight_replay_for_prediction(coins, bandit, scaler, pred_dict, cfg)
                n_updates += n
                if n > 0: n_preds += 1

                # Wallet aktualisieren (netto: pnl - slippage)
                eff_lev = int(p.get('effective_leverage') or 5)
                pnl_pct_real = float(p.get('pnl_pct') or 0.0)
                gross = pnl_pct_real / 100.0 * eff_lev * margin
                slip = (eff_lev * margin) * slip_pct / 100.0
                update_virtual_wallet(app, gross - slip)
            except Exception as e:
                log.warning("backfill %s id=%s failed: %s", p['symbol'], p['id'], e)
    log.info("Hindsight-Backfill done: %d Predictions replayed, %d Action-Updates total",
             n_preds, n_updates)
    return n_preds, n_updates


def process_due_predictor_hindsight(s, bandit, scaler, batch_size=20):
    """Liest faellige Eintraege aus predictor_hindsight_queue und ruft fuer
    jeden hindsight_replay_for_prediction mit voller Klines-Range (timeout_h
    + extended_minutes). Markiert processed nach Erfolg.

    Lernsignal kommt damit ~timeout_h + extended_min nach Open — der Bandit
    sieht ob seine Action in der GESAMTEN Bewertungsperiode getroffen
    haette (auch wenn der reale Trade frueh Win/Loss oder Timeout war)."""
    n_processed = 0; n_total_updates = 0
    cfg = s["predictor"]
    with db_app(s) as app, db_coins(s) as coins:
        while True:
            with app.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT q.id AS qid, q.prediction_id,
                           p.symbol, p.side, p.entry_px, p.features, p.created_at
                    FROM predictor_hindsight_queue q
                    JOIN open_predictions p ON p.id = q.prediction_id
                    WHERE q.processed_at IS NULL AND q.ready_at <= now()
                    ORDER BY q.ready_at
                    LIMIT %s
                """, (batch_size,))
                due = cur.fetchall()
            if not due: break
            for d in due:
                feat = d['features'] if isinstance(d['features'], dict) else {}
                pred_dict = {
                    'symbol': d['symbol'], 'side': d['side'],
                    'entry_px': d['entry_px'], 'features': feat,
                    'created_at': d['created_at'],
                }
                try:
                    n, rewards_map, chosen_replay = hindsight_replay_for_prediction(
                        coins, bandit, scaler, pred_dict, cfg)
                except Exception as e:
                    log.warning("predictor-hindsight pid=%s failed: %s", d['prediction_id'], e)
                    n = 0; rewards_map = {}; chosen_replay = None

                if n > 0:
                    chosen_name = feat.get('_action_name', '?')
                    chosen_r = chosen_replay.get('reward') if chosen_replay else None
                    rated = [(k, v) for k, v in rewards_map.items()
                             if v.get('reward') is not None]
                    top = sorted(rated, key=lambda kv: -kv[1]['reward'])[:3]
                    log.info("HINDSIGHT %s chosen=%s r=%s | top3: %s | n_updates=%d",
                             d['symbol'], chosen_name,
                             f"{chosen_r:+.3f}" if chosen_r is not None else "?",
                             ", ".join(f"{k}:{v['reward']:+.2f}" for k, v in top),
                             n)
                    n_total_updates += n
                    n_processed += 1

                with app.cursor() as cur2:
                    cur2.execute("""
                        UPDATE predictor_hindsight_queue
                        SET processed_at=now(), n_updates=%s
                        WHERE id=%s
                    """, (n, d['qid']))
                app.commit()
            if len(due) < batch_size: break
    return n_processed, n_total_updates


# =============================================================================
# Watch-Pass v4 — TP/SL/Timeout-Check + Hindsight-Update fuer ALLE Actions
# =============================================================================

def watch_pass_mh(s, mh_model, state_path):
    """Watch-Pass v5: trackt peak/trough, schließt bei TP/SL/Timeout,
    learnt Multi-Head direkt aus realisiertem peak_pct/trough_pct beim Close."""
    cfg = s["predictor"]
    mh_cfg = cfg.get("multi_head", {})
    timeout_h = get_timeout_hours(s)
    # Virtuelle Wallet — Drawdown-Feature für Multi-Head
    vwallet_cfg = cfg.get("virtual_wallet", {})
    vwallet_margin = float(vwallet_cfg.get("margin_per_trade", 20.0))
    vwallet_slip = float(vwallet_cfg.get("slippage_pct_per_trade", 0.1))
    closes = []
    with db_coins(s) as coins, db_app(s) as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur_a:
            cur_a.execute("""
                SELECT id, symbol, side, entry_px, sl_px, tp_px, score, rule_name, source, features,
                       created_at, last_check_at, peak_px, trough_px, effective_leverage
                FROM open_predictions WHERE status='open'
            """)
            opens = cur_a.fetchall()
        if not opens: return 0

        symbols = list({o['symbol'] for o in opens})
        with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
            cur_c.execute("""
                SELECT DISTINCT ON (symbol) symbol, mid_px, close FROM klines
                WHERE symbol = ANY(%s) AND interval='10s' ORDER BY symbol, open_time DESC
            """, (symbols,))
            prices = {r['symbol']: (f(r['mid_px']) or f(r['close'])) for r in cur_c.fetchall()}

            # Range-Aggregat seit last_check_at pro Position: deckt intra-bucket
            # Wicks ab, die der reine close-Vergleich verpasst.
            sym_lc = [(o['symbol'], o['last_check_at']) for o in opens if o.get('last_check_at')]
            ranges = {}
            if sym_lc:
                cur_c.execute("""
                    SELECT k.symbol, MAX(k.high) AS hi, MIN(k.low) AS lo
                    FROM klines k
                    JOIN (VALUES %s) AS o(sym, last_check)
                      ON k.symbol = o.sym AND k.open_time > o.last_check::timestamptz
                    WHERE k.interval = '10s'
                    GROUP BY k.symbol
                """ % ",".join(["(%s,%s)"] * len(sym_lc)),
                    [v for pair in sym_lc for v in pair])
                ranges = {r['symbol']: (f(r['hi']), f(r['lo'])) for r in cur_c.fetchall()}

        for o in opens:
            cur_px = prices.get(o['symbol'])
            if cur_px is None: continue
            entry = f(o['entry_px']); sl = f(o['sl_px']); tp = f(o['tp_px']); side = o['side']
            hi, lo = ranges.get(o['symbol'], (None, None))
            # peak/trough mit Range-Extremes UND aktuellem cur_px erweitern
            peak_px = max(f(o['peak_px']) or entry, cur_px, hi or 0.0)
            trough_lo_candidate = lo if lo is not None else cur_px
            trough_px = min(f(o['trough_px']) or entry, cur_px, trough_lo_candidate)

            # TP/SL-Hit-Check via high/low der Buckets seit last_check_at.
            # Bei Wick-Kollision (TP UND SL im selben Range): konservativ SL nehmen.
            status = None
            exit_px = cur_px
            if hi is not None and lo is not None:
                if side == 'long':
                    tp_hit = hi >= tp
                    sl_hit = lo <= sl
                    if sl_hit:
                        status = 'loss'; exit_px = sl
                    elif tp_hit:
                        status = 'win'; exit_px = tp
                else:
                    tp_hit = lo <= tp
                    sl_hit = hi >= sl
                    if sl_hit:
                        status = 'loss'; exit_px = sl
                    elif tp_hit:
                        status = 'win'; exit_px = tp
            else:
                # Fallback (kein last_check_at oder keine klines im Range)
                if side == 'long':
                    if cur_px >= tp: status = 'win'; exit_px = tp
                    elif cur_px <= sl: status = 'loss'; exit_px = sl
                else:
                    if cur_px <= tp: status = 'win'; exit_px = tp
                    elif cur_px >= sl: status = 'loss'; exit_px = sl

            age_h = (datetime.now(timezone.utc) - o['created_at']).total_seconds() / 3600
            if status is None and age_h >= timeout_h:
                status = 'timeout'; exit_px = cur_px

            with app.cursor() as cur_a:
                if status:
                    pnl_pct = ((exit_px - entry) / entry * 100) if side == 'long' else ((entry - exit_px) / entry * 100)
                    peak_pct_v = (peak_px - entry) / entry * 100.0
                    trough_pct_v = (entry - trough_px) / entry * 100.0
                    cur_a.execute("""
                        UPDATE open_predictions
                        SET status=%s, exit_px=%s, pnl_pct=%s, closed_at=now(),
                            last_px=%s, last_check_at=now(), peak_px=%s, trough_px=%s
                        WHERE id=%s AND status='open'
                    """, (status, exit_px, pnl_pct, cur_px, peak_px, trough_px, o['id']))
                    # Multi-Head-Learn nur fuer multi_head-Trades (keine resync_hl).
                    if o.get('source') == 'multi_head':
                        closes.append({
                            'id': o['id'], 'symbol': o['symbol'], 'side': side,
                            'entry': entry, 'exit': exit_px, 'pnl_pct': pnl_pct,
                            'peak_pct': peak_pct_v, 'trough_pct': trough_pct_v,
                            'status': status, 'features': o['features'],
                            'rule': o['rule_name'], 'score': f(o['score']),
                            'created_at': o['created_at'],
                        })
                    # Virtuelle Wallet aktualisieren — Bandit lernt Drawdown.
                    # Slippage (taker+maker, ca 0.1%) wird vom Brutto abgezogen
                    # damit Wallet-Verlauf realistisch ist.
                    eff_lev = int(o.get('effective_leverage') or 5)
                    notional = eff_lev * vwallet_margin
                    gross_pnl = pnl_pct / 100.0 * eff_lev * vwallet_margin
                    slip_dollar = notional * vwallet_slip / 100.0
                    pnl_dollar = gross_pnl - slip_dollar
                    new_dd = update_virtual_wallet(app, pnl_dollar)
                    log.info("WALLET %s gross=$%.2f slip=$%.2f net=$%.2f drawdown=%.2f%%",
                             o['symbol'], gross_pnl, slip_dollar, pnl_dollar, new_dd)
                    log.info("CLOSE %s %s status=%s pnl=%.3f%% peak=%.2f%% trough=%.2f%%",
                             o['symbol'], side, status, pnl_pct, peak_pct_v, trough_pct_v)
                else:
                    cur_a.execute("""
                        UPDATE open_predictions
                        SET last_px=%s, last_check_at=now(), peak_px=%s, trough_px=%s
                        WHERE id=%s AND status='open'
                    """, (cur_px, peak_px, trough_px, o['id']))
            app.commit()

    if not closes: return 0

    # Multi-Head-Learn: direkt aus realisierten peak/trough beim Close lernen.
    # Kein Queue mehr — Modell sieht echte realisierte Bewegung sofort.
    mh_cfg = s["predictor"].get("multi_head", {})
    flat_thresh = float(mh_cfg.get("timeout_flat_threshold_pct", 0.3))
    to_correct_w = float(mh_cfg.get("timeout_correct_weight", 0.3))
    to_wrong_w = float(mh_cfg.get("timeout_wrong_weight", 1.0))
    loss_w = float(mh_cfg.get("loss_weight", 1.0))
    win_w = float(mh_cfg.get("win_weight", 1.0))

    # Extended-Hindsight: Lernen nicht direkt beim Close, sondern delayed via Queue.
    # ready_at = created_at + timeout_h + extended_min — bis dahin sind alle relevanten
    # klines verfuegbar, das Modell sieht die volle Bewegungs-Range (auch nach early-TP-Hit).
    timeout_h_now = get_timeout_hours(s)
    extended_min = float(mh_cfg.get("hindsight_extended_minutes", 30))

    with db_learner(s) as ldb, db_app(s) as app:
        with ldb.cursor() as cur_l:
            for c in closes:
                feat = c.get('features') or {}
                if not isinstance(feat, dict): feat = {}
                ready_at = c['created_at'] + timedelta(hours=timeout_h_now, minutes=extended_min)
                try:
                    with app.cursor() as cur_q:
                        cur_q.execute("""
                            INSERT INTO predictor_hindsight_queue
                              (prediction_id, closed_at, ready_at)
                            VALUES (%s, now(), %s)
                            ON CONFLICT (prediction_id) DO NOTHING
                        """, (c['id'], ready_at))
                    app.commit()
                except Exception as e:
                    log.warning("hindsight-queue insert %s id=%s failed: %s",
                                c['symbol'], c['id'], e)

                duration = int((datetime.now(timezone.utc) - c['created_at']).total_seconds() / 60)
                won = c['status'] == 'win' or (c['status'] == 'timeout' and (c.get('pnl_pct') or 0) > 0)
                cur_l.execute("""
                    INSERT INTO prediction_feedback
                      (prediction_id, scanner_type, symbol, direction, entry_price,
                       detected_at, resolved_at, status, was_correct,
                       actual_result_pct, duration_minutes, time_result,
                       features, rule_name, score, peak_pct, trough_pct)
                    VALUES (%s, 'predictor', %s, %s, %s, %s, now(), %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id, scanner_type) DO NOTHING
                """, (c['id'], c['symbol'], c['side'], c['entry'], c['created_at'],
                       c['status'], won, c['pnl_pct'], duration,
                       'within_timeout' if c['status'] != 'timeout' else 'timeout',
                       json.dumps(c['features'] or {}), c['rule'], c['score'],
                       c.get('peak_pct'), c.get('trough_pct')))
        ldb.commit()

        with app.cursor() as cur_a:
            cur_a.execute("""
                UPDATE predictor_state SET closed_count = COALESCE(closed_count,0) + %s,
                       updated_at=now() WHERE id=1
            """, (len(closes),))
        app.commit()

    log.info("watch-pass: %d closes, alle in Hindsight-Queue (ready_at=open+%dh+%dmin)",
             len(closes), int(timeout_h_now), int(extended_min))
    return len(closes)


def process_due_mh_hindsight(s, mh_model, state_path, batch_size=20):
    """Verarbeitet faellige Eintraege aus predictor_hindsight_queue.

    Bei jedem fälligen Eintrag:
      1. Hole prediction-Row (features, side, status, created_at, entry_px)
      2. Hole klines im Range [created_at, created_at + timeout_h + extended_min]
      3. Berechne echtes peak_pct + trough_pct über VOLLE Range (nicht nur bis Close!)
      4. mh_model.learn_close mit echten Werten

    Damit lernt das Modell die WIRKLICHE Coin-Bewegung — auch wenn der Trade früh
    durch TP geschlossen wurde, sieht der Magnitude-Regressor das tatsaechliche
    Hoch/Tief der vollen Lookahead-Range.
    """
    cfg = s["predictor"]
    mh_cfg = cfg.get("multi_head", {})
    flat_thresh = float(mh_cfg.get("timeout_flat_threshold_pct", 0.3))
    to_correct_w = float(mh_cfg.get("timeout_correct_weight", 0.3))
    to_wrong_w = float(mh_cfg.get("timeout_wrong_weight", 1.0))
    loss_w = float(mh_cfg.get("loss_weight", 1.0))
    win_w = float(mh_cfg.get("win_weight", 1.0))

    n_learned = 0
    with db_app(s) as app, db_coins(s) as coins:
        while True:
            with app.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT q.id AS qid, q.prediction_id,
                           p.symbol, p.side, p.status, p.entry_px,
                           p.features, p.created_at
                    FROM predictor_hindsight_queue q
                    JOIN open_predictions p ON p.id = q.prediction_id
                    WHERE q.processed_at IS NULL AND q.ready_at <= now()
                      AND p.source = 'multi_head'
                    ORDER BY q.ready_at
                    LIMIT %s
                """, (batch_size,))
                due = cur.fetchall()
            if not due:
                break
            for d in due:
                feat = d['features'] if isinstance(d['features'], dict) else {}
                feat_clean = {k: v for k, v in feat.items() if not k.startswith('_')}
                entry = float(d['entry_px'])
                created = d['created_at']
                end_time = created + timedelta(hours=get_timeout_hours(s),
                                               minutes=float(mh_cfg.get("hindsight_extended_minutes", 30)))
                with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
                    cur_c.execute("""
                        SELECT MAX(high) AS hi, MIN(low) AS lo
                        FROM klines
                        WHERE symbol=%s AND interval='10s'
                          AND open_time BETWEEN %s AND %s
                    """, (d['symbol'], created, end_time))
                    row = cur_c.fetchone()
                if not row or row['hi'] is None or row['lo'] is None:
                    log.warning("process_due_mh_hindsight %s id=%s: keine klines im Range, skip",
                                d['symbol'], d['prediction_id'])
                    with app.cursor() as cur_u:
                        cur_u.execute("UPDATE predictor_hindsight_queue SET processed_at=now(), n_updates=0 WHERE id=%s",
                                      (d['qid'],))
                    app.commit()
                    continue
                peak_pct_full = (float(row['hi']) - entry) / entry * 100.0
                trough_pct_full = (entry - float(row['lo'])) / entry * 100.0

                try:
                    mh_model.learn_close(
                        features=feat_clean,
                        predicted_side=d['side'],
                        status=d['status'],
                        peak_pct=max(0.0, peak_pct_full),
                        trough_pct=max(0.0, trough_pct_full),
                        timeout_flat_threshold_pct=flat_thresh,
                        timeout_correct_weight=to_correct_w,
                        timeout_wrong_weight=to_wrong_w,
                        loss_weight=loss_w,
                        win_weight=win_w,
                    )
                    n_learned += 1
                    log.info("HINDSIGHT-LEARN %s id=%s side=%s status=%s "
                             "FULL-peak=%.3f%% FULL-trough=%.3f%% (vs closed-peak=?, n_obs=%d)",
                             d['symbol'], d['prediction_id'], d['side'], d['status'],
                             peak_pct_full, trough_pct_full, mh_model.direction.n_obs)
                except Exception as e:
                    log.exception("mh_model.learn_close failed for %s id=%s: %s",
                                  d['symbol'], d['prediction_id'], e)
                with app.cursor() as cur_u:
                    cur_u.execute("UPDATE predictor_hindsight_queue SET processed_at=now(), n_updates=1 WHERE id=%s",
                                  (d['qid'],))
                app.commit()
            if len(due) < batch_size:
                break

    if n_learned > 0:
        try:
            mh_model.save(state_path)
            log.info("mh-hindsight processed: %d trades learned (n_obs=%d)",
                     n_learned, mh_model.direction.n_obs)
        except Exception as e:
            log.exception("mh_model.save failed: %s", e)
    return n_learned


# =============================================================================
# MODIFY-BANDIT (Phase 1) — passt TP/SL laufender Trades alle 30s an
# Eigene LinTSBandit-Instanz, eigene Tabelle (modify_decisions),
# eigenes State-File, eigene Reward-Welt. Open-Bandit unberuehrt.
# =============================================================================

_MODIFY_POSITION_KEYS = [
    'time_in_trade_h', 'time_remaining_h',
    'pnl_now_pct', 'peak_pct_now', 'trough_pct_now',
    'dist_to_tp_pct', 'dist_to_sl_pct', 'original_action_idx',
    'leverage', 'margin_pnl_pct',
]
# modify_count entfernt 12.05.2026: seit Reform 10.05. nur noch 2 Actions
# (hold + close_now), keine TP/SL-Modifikationen mehr -> Feature war ALWAYS 0.


def build_modify_feature_keys(s):
    """Trader-Bandit Features = Open-Bandit-Features + 10 Position-Features."""
    return build_feature_keys(s) + _MODIFY_POSITION_KEYS


# Werden in main() konkret gesetzt.
MODIFY_FEATURE_KEYS = []
N_FEAT_MODIFY = 1


def vectorize_modify(feat_dict):
    v = np.zeros(N_FEAT_MODIFY)
    v[0] = 1.0
    for i, k in enumerate(MODIFY_FEATURE_KEYS, 1):
        try:
            v[i] = float(feat_dict.get(k, 0.0) or 0.0)
        except (TypeError, ValueError):
            v[i] = 0.0
    return v


def build_modify_actions(*_args, **_kwargs):
    """Action-Space: nur hold + close_now (Reform 10.05.2026).
    Trader entscheidet pro Tick: weiter halten oder jetzt schliessen.
    TP/SL bleiben auf Predictor-Werten — Trader greift nicht ein.
    Args werden ignoriert (rueckwaerts-kompatibel zu alten Callern)."""
    return [
        {'name': 'hold', 'special': 'hold', 'tp_delta': None, 'sl_delta': None},
        {'name': 'close_now', 'special': 'close', 'tp_delta': None, 'sl_delta': None},
    ]


def position_features(trader_pos_row, mark_px, current_tp_px, current_sl_px, timeout_h=2.0):
    """Position-Features fuer Trader-Bandit (Trade-State + Hebel + time_remaining).
    Eingabe: Row aus trader_positions (NICHT open_predictions!) — saubere Trennung.
    timeout_h kommt aus settings.predictor.bandit.timeout_hours (Predictor-Hard-Close).
    modify_count entfernt 12.05.2026 (seit 2-Action-Reform immer 0)."""
    entry = float(trader_pos_row['entry_px'])
    side = trader_pos_row['side']
    leverage = float(trader_pos_row.get('leverage') or 1)
    age_s = (datetime.now(timezone.utc) - trader_pos_row['opened_at']).total_seconds()
    time_in_h = age_s / 3600.0
    time_left_h = max(0.0, float(timeout_h) - time_in_h)
    if mark_px <= 0:
        log.error("FALLBACK_TRIGGERED position_features tid=%s: mark_px=%s ungueltig -> None", trader_pos_row.get('id'), mark_px)
        return None
    peak_raw = trader_pos_row.get('peak_px')
    trough_raw = trader_pos_row.get('trough_px')
    if peak_raw is None or trough_raw is None:
        log.error("FALLBACK_TRIGGERED position_features tid=%s: peak_px/trough_px NULL -> None", trader_pos_row.get('id'))
        return None
    peak = float(peak_raw); trough = float(trough_raw)
    if side == 'long':
        pnl_now = (mark_px - entry) / entry * 100.0
        peak_pct = (peak - entry) / entry * 100.0
        trough_pct = (trough - entry) / entry * 100.0
        dist_tp = (current_tp_px - mark_px) / mark_px * 100.0
        dist_sl = (mark_px - current_sl_px) / mark_px * 100.0
    else:
        pnl_now = (entry - mark_px) / entry * 100.0
        peak_pct = (entry - trough) / entry * 100.0
        trough_pct = (entry - peak) / entry * 100.0
        dist_tp = (mark_px - current_tp_px) / mark_px * 100.0
        dist_sl = (current_sl_px - mark_px) / mark_px * 100.0
    feat_orig = trader_pos_row.get('features_at_open')
    if not isinstance(feat_orig, dict) or '_action_idx' not in feat_orig:
        log.error("FALLBACK_TRIGGERED position_features tid=%s: features_at_open fehlt _action_idx -> None",
                  trader_pos_row.get('id'))
        return None
    try:
        orig_idx = float(feat_orig['_action_idx'])
    except (TypeError, ValueError):
        log.error("FALLBACK_TRIGGERED position_features tid=%s: _action_idx=%r nicht numerisch -> None",
                  trader_pos_row.get('id'), feat_orig.get('_action_idx'))
        return None
    return {
        'time_in_trade_h': time_in_h,
        'time_remaining_h': time_left_h,
        'pnl_now_pct': pnl_now,
        'peak_pct_now': peak_pct,
        'trough_pct_now': trough_pct,
        'dist_to_tp_pct': dist_tp,
        'dist_to_sl_pct': dist_sl,
        'leverage': leverage,
        'margin_pnl_pct': pnl_now * leverage,
        'original_action_idx': orig_idx,
    }


def get_current_tp_sl(app_conn, prediction_id, fallback_tp, fallback_sl):
    """Aktueller TP/SL = letzte modify_decisions.tp/sl_px_after, sonst original."""
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT tp_px_after, sl_px_after FROM modify_decisions
            WHERE prediction_id=%s AND tp_px_after IS NOT NULL AND sl_px_after IS NOT NULL
            ORDER BY decided_at DESC LIMIT 1
        """, (prediction_id,))
        r = cur.fetchone()
    if r and r[0] is not None and r[1] is not None:
        return float(r[0]), float(r[1])
    return float(fallback_tp), float(fallback_sl)


def get_modify_count(app_conn, prediction_id):
    with app_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM modify_decisions WHERE prediction_id=%s", (prediction_id,))
        r = cur.fetchone()
    return int(r[0]) if r else 0


def safe_close_position_hl(creds, coin, addr, max_retries=2):
    """Schliesst HL-Position + cancelt alle offenen Orders fuer den Coin.
    Verifiziert dass Position weg ist. Retry bei Fehler. Returns dict:
      {success: bool, position_closed: bool, orders_cancelled: int}.
    Garantiert: keine Phantom-Orders mehr nach erfolgreichem Close."""
    if '/opt/coin/backend' not in sys.path:
        sys.path.insert(0, '/opt/coin/backend')
    from rl_agent.trader import (
        close_position_hl, cancel_all_orders_for_coin_hl,
        get_hl_open_positions,
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            close_res = close_position_hl(creds, coin, addr)
        except Exception as e:
            close_res = {'success': False, 'error': str(e)}
            last_err = str(e)
        try:
            cancel_res = cancel_all_orders_for_coin_hl(creds, coin)
        except Exception as e:
            cancel_res = {'success': False, 'cancelled': 0, 'error': str(e)}
            last_err = str(e)
        time.sleep(0.5)
        try:
            positions = get_hl_open_positions(addr)
            still_open = next((p for p in positions if p.get('coin') == coin), None)
        except Exception:
            still_open = None
        if not still_open:
            return {'success': True, 'position_closed': True,
                    'orders_cancelled': cancel_res.get('cancelled', 0),
                    'avg_price': close_res.get('avg_price')}
        log.warning("safe_close_position_hl %s: position lebt noch (attempt %d), retry",
                    coin, attempt + 1)
    return {'success': False, 'position_closed': False,
            'orders_cancelled': 0, 'error': last_err or 'position survived close'}


def apply_modify_to_hl(prediction, action, current_tp, current_sl, mark_px):
    """Setzt Trader-Action auf HL um. 2 Actions: hold | close_now.
    Returns (success, new_tp, new_sl, close_executed).

    - hold:     keine HL-Aktion
    - close_now: safe_close (cancel + close + verify)"""
    if action.get('special') == 'hold':
        return True, current_tp, current_sl, False
    if action.get('special') != 'close':
        return False, current_tp, current_sl, False

    if '/opt/coin/backend' not in sys.path:
        sys.path.insert(0, '/opt/coin/backend')
    from rl_agent.trader import get_hl_credentials, get_hl_open_positions
    creds = get_hl_credentials()
    coin = prediction['symbol']
    positions = get_hl_open_positions(creds['wallet_address'])
    match = next((p for p in positions if p.get('coin') == coin), None)
    if not match:
        return False, current_tp, current_sl, False  # HL-Position weg
    r = safe_close_position_hl(creds, coin, creds['wallet_address'])
    return r['success'], current_tp, current_sl, r.get('position_closed', False)


def modify_pass(s, modify_bandit, modify_scaler, rng):
    """Trader-Tick — operiert AUSSCHLIESSLICH auf trader_positions / trader_decisions.
    KEIN Zugriff auf open_predictions (saubere Welt-Trennung).
    Action-Space: hold | close_now (Reform 10.05.2026)."""
    cfg = s["predictor"]
    mb_cfg = cfg.get("modify_bandit", {})
    if not mb_cfg.get("enabled"):
        return 0
    if not bool(cfg.get("trading", {}).get("auto_trade")):
        return 0

    timeout_h = get_timeout_hours(s)
    min_age = float(mb_cfg.get("min_open_age_seconds", 60))
    expl_floor = float(mb_cfg.get("exploration_floor", 0.05))
    expl_init = float(mb_cfg.get("exploration_init", 1.0))
    expl_decay = float(mb_cfg.get("exploration_decay_per_decision", 0.0001))
    n_total = sum(modify_bandit.n_obs)
    exploration = max(expl_floor, expl_init - expl_decay * n_total)
    n_modified = n_evaluated = 0

    with db_app(s) as app, db_coins(s) as coins:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, prediction_id, symbol, side, entry_px, qty, leverage,
                       original_tp_px, original_sl_px, current_tp_px, current_sl_px,
                       peak_px, trough_px, opened_at, features_at_open
                FROM trader_positions
                WHERE status='open'
                  AND opened_at <= now() - (%s || ' seconds')::interval
            """, (min_age,))
            opens = cur.fetchall()
        if not opens:
            return 0

        btc_moves = load_btc_moves(coins)
        if btc_moves is None:
            log.error("FALLBACK_TRIGGERED modify_pass: load_btc_moves None -> modify-pass abgebrochen")
            return 0
        symbols = list({o['symbol'] for o in opens})
        with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
            cur_c.execute("""
                SELECT DISTINCT ON (symbol) symbol, mid_px, close FROM klines
                WHERE symbol = ANY(%s) AND interval='10s' ORDER BY symbol, open_time DESC
            """, (symbols,))
            mark_pxs = {r['symbol']: (f(r['mid_px']) or f(r['close'])) for r in cur_c.fetchall()}

        for o in opens:
            mark_px = mark_pxs.get(o['symbol'])
            if mark_px is None: continue
            current_tp = float(o['current_tp_px'])
            current_sl = float(o['current_sl_px'])

            # peak/trough auf aktuellen Stand bringen
            new_peak = max(float(o.get('peak_px') or o['entry_px']), mark_px)
            new_trough = min(float(o.get('trough_px') or o['entry_px']), mark_px)
            if new_peak != float(o.get('peak_px') or 0) or new_trough != float(o.get('trough_px') or 0):
                with app.cursor() as cur_pt:
                    cur_pt.execute("UPDATE trader_positions SET peak_px=%s, trough_px=%s WHERE id=%s",
                                   (new_peak, new_trough, o['id']))
                app.commit()
                # update local row dict so position_features sieht aktuelle Werte
                o = dict(o); o['peak_px'] = new_peak; o['trough_px'] = new_trough

            base_feat = feature_snapshot_v2(coins, o['symbol'], rule_flags={}, btc_moves=btc_moves)
            if base_feat is None: continue
            pos_feat = position_features(o, mark_px, current_tp, current_sl, timeout_h=timeout_h)
            if pos_feat is None: continue
            full_feat = {**base_feat, **pos_feat}

            # Hard-Timeout-Check (Reform 10.05.2026): age >= timeout_h -> Trader close_now.
            # Wird in trader_decisions als reguläre Bandit-Action (close_now) eingetragen,
            # damit Hindsight den Reward wie für eine normale Trader-Entscheidung berechnet.
            age_h = (datetime.now(timezone.utc) - o['opened_at']).total_seconds() / 3600.0
            if age_h >= timeout_h:
                try:
                    if '/opt/coin/backend' not in sys.path:
                        sys.path.insert(0, '/opt/coin/backend')
                    from rl_agent.trader import get_hl_credentials
                    creds = get_hl_credentials()
                    sc = safe_close_position_hl(creds, o['symbol'], creds['wallet_address'])
                    if sc.get('success'):
                        entry = float(o['entry_px'])
                        exit_px = float(sc.get('avg_price') or mark_px)
                        pnl = ((exit_px - entry)/entry*100) if o['side']=='long' else ((entry - exit_px)/entry*100)
                        idx_close = next((i for i, a in enumerate(modify_bandit.actions)
                                          if a.get('special') == 'close'), 1)
                        with app.cursor() as cur_t:
                            cur_t.execute("""
                                UPDATE trader_positions
                                SET status='closed', closed_at=now(), exit_px=%s, pnl_pct=%s
                                WHERE id=%s AND status='open'
                            """, (exit_px, pnl, o['id']))
                        with app.cursor() as cur_d:
                            cur_d.execute("""
                                INSERT INTO trader_decisions
                                  (position_id, features, action_idx, action_name,
                                   tp_delta_pct, sl_delta_pct, expected_r,
                                   tp_px_before, sl_px_before, tp_px_after, sl_px_after,
                                   executed, close_triggered)
                                VALUES (%s, %s::jsonb, %s, 'close_now', NULL, NULL, NULL,
                                        %s, %s, NULL, NULL, true, true)
                            """, (o['id'], json.dumps(full_feat), idx_close,
                                  float(current_tp), float(current_sl)))
                        app.commit()
                        n_modified += 1
                        log.info("TRADER-TIMEOUT-CLOSE %s %s age=%.2fh pnl=%.3f%% trader_pid=%s",
                                 o['symbol'], o['side'], age_h, pnl, o['id'])
                        continue
                    else:
                        log.warning("TIMEOUT-CLOSE %s safe_close failed: %s", o['symbol'], sc.get('error'))
                except Exception as e:
                    log.warning("TIMEOUT-CLOSE %s exception: %s", o['symbol'], e)
                    try: app.rollback()
                    except Exception: pass

            x_raw = vectorize_modify(full_feat)
            modify_scaler.update(x_raw)
            x = modify_scaler.transform(x_raw)

            idx, expected_r, _ = modify_bandit.select(x, exploration=exploration, rng=rng)
            action = modify_bandit.actions[idx]
            n_evaluated += 1

            tp_after = sl_after = None
            close_triggered = False
            executed = False

            try:
                ok, new_tp, new_sl, close_done = apply_modify_to_hl(
                    o, action, current_tp, current_sl, mark_px)
                executed = ok
                if action.get('special') == 'close':
                    close_triggered = close_done
                    if close_done:
                        entry = float(o['entry_px'])
                        pnl_now = ((mark_px - entry)/entry*100) if o['side']=='long' else ((entry - mark_px)/entry*100)
                        # Trader-eigene Welt: trader_positions schliessen.
                        # open_predictions UNANGETASTET (Predictor weiss nichts).
                        with app.cursor() as cur_c2:
                            cur_c2.execute("""
                                UPDATE trader_positions
                                SET status='closed', exit_px=%s, pnl_pct=%s,
                                    closed_at=now()
                                WHERE id=%s AND status='open'
                            """, (mark_px, pnl_now, o['id']))
                        app.commit()
                        n_modified += 1
                        log.info("TRADER-CLOSE %s %s pnl=%.3f%% mark=%.6g trader_pid=%s",
                                 o['symbol'], o['side'], pnl_now, mark_px, o['id'])
            except Exception as e:
                log.warning("modify_pass %s/%s failed: %s", o['symbol'], action.get('name'), e)
                executed = False
                try: app.rollback()
                except Exception: pass

            try:
                with app.cursor() as cur2:
                    cur2.execute("""
                        INSERT INTO trader_decisions
                          (position_id, features, action_idx, action_name,
                           tp_delta_pct, sl_delta_pct, expected_r,
                           tp_px_before, sl_px_before, tp_px_after, sl_px_after,
                           executed, close_triggered)
                        VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (o['id'], json.dumps(full_feat), idx, action['name'],
                           action.get('tp_delta'), action.get('sl_delta'), float(expected_r),
                           float(current_tp), float(current_sl),
                           float(tp_after) if tp_after is not None else None,
                           float(sl_after) if sl_after is not None else None,
                           executed, close_triggered))
                app.commit()
            except Exception as e:
                log.warning("modify_pass insert decision %s/%s failed: %s",
                            o['symbol'], action.get('name'), e)
                try: app.rollback()
                except Exception: pass

    if n_evaluated > 0:
        log.info("modify-pass: %d evaluated, %d applied (expl=%.3f, n_obs=%d)",
                 n_evaluated, n_modified, exploration, sum(modify_bandit.n_obs))
    return n_modified


def replay_modify_action(klines_rows, side, entry, tp_px, sl_px):
    """Counterfactual: was wuerde mit diesen TP/SL-Levels passieren?
    Returns (status, exit_px, pnl_pct, dur_h_after_decision).
    dur_h ist Zeit vom ersten Klinen-Tick bis zum Hit/End.
    Bei TP+SL im selben Bucket: konservativ SL annehmen."""
    if not klines_rows:
        return 'no_data', None, 0.0, 0.0
    start_t = klines_rows[0]['open_time']
    for r in klines_rows:
        h = r.get('high'); l = r.get('low')
        if h is None or l is None: continue
        h = float(h); l = float(l)
        if side == 'long':
            sl_hit = l <= sl_px
            tp_hit = h >= tp_px
        else:
            sl_hit = h >= sl_px
            tp_hit = l <= tp_px
        elapsed_h = (r['open_time'] - start_t).total_seconds() / 3600.0
        if sl_hit:
            pnl = ((sl_px - entry)/entry*100) if side=='long' else ((entry - sl_px)/entry*100)
            return 'loss', sl_px, pnl, elapsed_h
        if tp_hit:
            pnl = ((tp_px - entry)/entry*100) if side=='long' else ((entry - tp_px)/entry*100)
            return 'win', tp_px, pnl, elapsed_h
    last = klines_rows[-1]
    last_close = float(last.get('close') or entry)
    elapsed_h = (last['open_time'] - start_t).total_seconds() / 3600.0
    pnl = ((last_close - entry)/entry*100) if side=='long' else ((entry - last_close)/entry*100)
    return 'incomplete', last_close, pnl, elapsed_h


def pick_efficiency_multiplier(eff_pct, stairs):
    """Stufen-Lookup: stairs sortiert nach threshold aufsteigend, returnt Mult der
    ersten Stufe die eff_pct <= threshold erfuellt. Bei eff_pct > letzter
    threshold -> letzter Mult."""
    eff_pct = max(0.0, min(100.0, float(eff_pct)))
    for thresh, mult in stairs:
        if eff_pct <= float(thresh):
            return float(mult)
    return float(stairs[-1][1])


def compute_efficiency_reward(captured_margin, max_margin_in_box, min_margin_in_box, cfg=None, leverage=None):
    """Stufen-Efficiency-Reward (Reform 12.05.2026, Loss-x2 + Missed-Win-x3 13.05.2026).

    captured_margin   = Margin-PnL der Action (% inkl. Hebel, vorzeichenbehaftet)
    max_margin_in_box = max(margin_pnl) im Fenster — bestmöglicher Profit
    min_margin_in_box = min(margin_pnl) im Fenster — schlimmstmöglicher Drawdown
    leverage          = Hebel der Position (fuer Coin-%-Schwelle bei Missed-Win-Strafe)
    cfg.efficiency_stairs        = Win-Stairs [[threshold_pct, multiplier], ...]
    cfg.efficiency_stairs_loss   = Loss-Stairs (optional, fallback auf efficiency_stairs)
    cfg.missed_win_threshold_coin_pct = Coin-%-Schwelle ab der max_box als verpasste Chance gilt
    cfg.missed_win_penalty_mult       = Extra-Multiplier auf Loss-Reward bei verpasster Chance
    """
    if cfg is None: cfg = {}
    stairs = cfg.get("efficiency_stairs")
    if not stairs or not isinstance(stairs, list) or len(stairs) == 0:
        log.error("FALLBACK_TRIGGERED compute_efficiency_reward: efficiency_stairs fehlt in modify_bandit.reward -> raise")
        raise RuntimeError("predictor.modify_bandit.reward.efficiency_stairs missing in settings.json")

    cap = float(captured_margin)
    mx = float(max_margin_in_box)
    mn = float(min_margin_in_box)

    if cap >= 0:
        if mx <= 0:
            return 0.0
        eff_pct = cap / mx * 100.0
        mult = pick_efficiency_multiplier(eff_pct, stairs)
        return cap * mult
    else:
        stairs_loss = cfg.get("efficiency_stairs_loss") or stairs
        missed_mult = 1.0
        if leverage is not None and float(leverage) > 0:
            max_coin_pct = mx / float(leverage)
            threshold = float(cfg.get("missed_win_threshold_coin_pct", 0.5))
            if max_coin_pct >= threshold:
                missed_mult = float(cfg.get("missed_win_penalty_mult", 3.0))
        if mn >= 0:
            mult = float(stairs_loss[-1][1])
            return cap * mult * missed_mult
        loss_eff_pct = cap / mn * 100.0
        mult = pick_efficiency_multiplier(loss_eff_pct, stairs_loss)
        return cap * mult * missed_mult


def hindsight_replay_modify_neural(coins_conn, app_conn, modify_bandit, modify_scaler,
                                     prediction, lookahead_minutes=120,
                                     reward_cfg=None, **_legacy_kwargs):
    """Trader-Hindsight — Stufen-Efficiency-Reward (Stand 12.05.2026), 2 Actions.

    Pro Decision werden BEIDE Actions (hold, close_now) counterfactual bewertet:
      max_box  = max(margin_pnl) im Fenster — bester moeglicher Profit
      min_box  = min(margin_pnl) im Fenster — schlimmster moeglicher Drawdown
      captured_hold  = margin_pnl am Fenster-Ende (was wuerde Halten bringen?)
      captured_close = margin_pnl am decided_at (was wuerde sofort schliessen bringen?)
      reward = compute_efficiency_reward(captured, max_box, min_box, reward_cfg)
              -> Stufen-Multiplier × captured (siehe Funktion).

    Predictor-DB (open_predictions) wird NICHT angefasst (Welt-Trennung).
    LinTSBandit.update(i, x, r) — kein Replay-Buffer, direkt-online.

    legacy_kwargs (penalty_cfg, sweet_cfg) werden ignoriert — nur fuer Aufruf-Kompatibilitaet."""
    pid = prediction['id']
    side = prediction['side']
    entry = float(prediction['entry_px'])
    closed_at = prediction['closed_at']
    opened_at = prediction.get('opened_at')

    with app_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, decided_at, features
            FROM trader_decisions WHERE position_id=%s ORDER BY decided_at
        """, (pid,))
        decisions = cur.fetchall()
    if not decisions:
        return 0

    sym = prediction['symbol']
    # Fenster-Ende: closed_at (echter Trade-Close), max +lookahead. Niemals in Zukunft.
    base_end = closed_at if closed_at else datetime.now(timezone.utc)
    end_at = base_end + timedelta(minutes=int(lookahead_minutes))
    end_at = min(end_at, datetime.now(timezone.utc))

    # Klines ab opened_at (gesamter Trade-Verlauf), nicht erst ab erster Decision
    box_start = opened_at if opened_at else decisions[0]['decided_at']

    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur_c:
        cur_c.execute("""
            SELECT open_time, high, low, close FROM klines
            WHERE symbol=%s AND interval='10s' AND open_time BETWEEN %s AND %s
            ORDER BY open_time
        """, (sym, box_start, end_at))
        klines = cur_c.fetchall()
    if not klines:
        return 0

    # Action-Index-Lookup (robust gegen Reihenfolgen-Aenderungen)
    name_to_idx = {a['name']: i for i, a in enumerate(modify_bandit.actions)}
    idx_hold = name_to_idx.get('hold')
    idx_close = name_to_idx.get('close_now')
    if idx_hold is None or idx_close is None:
        return 0

    # GLOBALE Box ueber gesamten Trade-Verlauf [opened_at, end_at] — jeder Decision-Tick
    # wird gegen DIESELBEN max/min bewertet. Verhindert dass spaete Ticks die verpasste
    # Peak-Chance aus der Vergangenheit "vergessen" und so der Missed-Win-Strafe entgehen.
    max_pnl_pct_g = -1e9
    min_pnl_pct_g = 1e9
    for k in klines:
        h = k.get('high'); l = k.get('low')
        if h is None or l is None: continue
        h = float(h); l = float(l)
        if side == 'long':
            pnl_high = (h - entry) / entry * 100.0  # bester Profit ueber Trade
            pnl_low  = (l - entry) / entry * 100.0  # schlimmster Drawdown
        else:
            pnl_high = (entry - l) / entry * 100.0
            pnl_low  = (entry - h) / entry * 100.0
        if pnl_high > max_pnl_pct_g: max_pnl_pct_g = pnl_high
        if pnl_low < min_pnl_pct_g: min_pnl_pct_g = pnl_low
    if max_pnl_pct_g == -1e9:
        return 0

    n_updates = 0
    for d in decisions:
        feat_dict = d['features'] if isinstance(d['features'], dict) else {}
        leverage = float(feat_dict.get('leverage', 1) or 1)
        x_raw = vectorize_modify(feat_dict)
        x = modify_scaler.transform(x_raw)

        klines_after = [k for k in klines if k['open_time'] >= d['decided_at']]
        if not klines_after: continue
        first_close = float(klines_after[0].get('close') or entry)
        last_close = float(klines_after[-1].get('close') or entry)

        # Globale Box in Margin umrechnen (leverage trade-konstant)
        max_margin = max_pnl_pct_g * leverage
        min_margin = min_pnl_pct_g * leverage

        # captured_close = jetzt schliessen zum first_close
        pnl_close_pct = ((first_close - entry)/entry*100) if side == 'long' else ((entry - first_close)/entry*100)
        captured_close = pnl_close_pct * leverage

        # captured_hold = warten bis Fenster-Ende, exit zum last_close
        pnl_hold_pct = ((last_close - entry)/entry*100) if side == 'long' else ((entry - last_close)/entry*100)
        captured_hold = pnl_hold_pct * leverage

        r_close = compute_efficiency_reward(captured_close, max_margin, min_margin, reward_cfg, leverage=leverage)
        r_hold = compute_efficiency_reward(captured_hold, max_margin, min_margin, reward_cfg, leverage=leverage)
        modify_bandit.update(idx_close, x, r_close)
        modify_bandit.update(idx_hold, x, r_hold)
        n_updates += 2
    return n_updates


def sync_hl_to_db(s):
    """Sync HL <-> trader_positions (NUR Trader-Welt!).

    Predictor-DB (open_predictions) wird NIE angefasst — saubere Welt-Trennung.

    Drei Richtungen:
    A) trader_positions.status='open' aber HL-Position weg -> status='failsafe'
       (manueller Close via Wallet-UI oder HL-Liquidation -> Trader merkt's hier)
    B) HL-Position vorhanden aber TP/SL-Orders fehlen -> Repair aus current_tp/sl_px
       (Schutz-Verifier — Position nie ungeschuetzt lassen)
    C) Order auf HL ohne Position -> cancel (Phantom-Order-Cleanup)

    Edge-Case: HL-Position OHNE trader_positions-Open-Eintrag (= manueller
    Volker-Trade ohne Auto-Trade-Hook): wird ignoriert. Trader managed nur was
    der Predictor durch try_auto_trade in trader_positions geschrieben hat."""
    if '/opt/coin/backend' not in sys.path:
        sys.path.insert(0, '/opt/coin/backend')
    try:
        from rl_agent.trader import (
            get_hl_credentials, get_hl_open_positions, get_hl_info,
            cancel_all_orders_for_coin_hl, place_tp_sl_hl,
        )
    except Exception as e:
        log.warning("sync_hl_to_db trader-import fehlgeschlagen: %s", e)
        return 0

    creds = get_hl_credentials()
    if not creds: return 0
    addr = creds.get('wallet_address')
    positions = get_hl_open_positions(addr) or []
    info = get_hl_info()
    try:
        fe_orders = info.frontend_open_orders(addr) or []
    except Exception:
        fe_orders = []

    hl_pos_by_coin = {p['coin']: p for p in positions}
    orders_by_coin = {}
    for o in fe_orders:
        orders_by_coin.setdefault(o['coin'], []).append(o)

    # HL user_fills laden — Quelle der Wahrheit fuer exit_px + pnl_pct bei
    # failsafe-Closes (TP-Hit, SL-Hit, manueller Close auf HL).
    try:
        hl_fills = info.user_fills(addr) or []
    except Exception as e:
        log.warning("sync_hl_to_db user_fills laden scheiterte: %s", e)
        hl_fills = []

    def _real_close_from_fills(coin, side, opened_at, qty):
        """Aggregiert HL-Close-Fills im Fenster [opened_at, jetzt] fuer (coin, side).
        Returns (exit_px, pnl_coin_pct) oder (None, None) falls keine Fills."""
        target_dir = 'Close Long' if side == 'long' else 'Close Short'
        opened_ms = int(opened_at.timestamp() * 1000)
        matching = [f for f in hl_fills
                    if f.get('coin') == coin
                    and target_dir in str(f.get('dir', ''))
                    and int(f.get('time', 0)) >= opened_ms
                    and float(f.get('closedPnl', 0) or 0) != 0]
        if not matching:
            return None, None
        total_sz = sum(float(f.get('sz', 0)) for f in matching)
        if total_sz <= 0: return None, None
        total_quote = sum(float(f.get('sz', 0)) * float(f.get('px', 0)) for f in matching)
        total_pnl_usd = sum(float(f.get('closedPnl', 0) or 0) for f in matching)
        avg_exit = total_quote / total_sz
        # closedPnl ist bereits side-aware (positiv=Gewinn). Caller rechnet daraus
        # pnl_coin_pct = total_pnl_usd / (qty_at_open * entry_px) * 100
        return avg_exit, total_pnl_usd

    n_failsafe = 0; n_repaired = 0; n_orphan_cancelled = 0
    with db_app(s) as app:
        # Richtung A: trader_positions.open ohne HL-Pos -> failsafe
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_px, qty, opened_at,
                       current_tp_px, current_sl_px
                FROM trader_positions WHERE status='open'
            """)
            tp_rows = cur.fetchall()
        for tr in tp_rows:
            coin = tr['symbol']
            if coin in hl_pos_by_coin: continue  # HL-Pos da, alles OK
            entry = float(tr['entry_px'])
            qty = float(tr['qty']) if tr['qty'] else 0.0
            side = tr['side']
            avg_exit, pnl_usd = _real_close_from_fills(coin, side, tr['opened_at'], qty)
            if avg_exit is None or qty <= 0 or entry <= 0:
                log.error("FALLBACK_TRIGGERED sync_hl_to_db.failsafe %s tid=%s: keine HL-fills/qty/entry -> Row bleibt 'open' fuer naechsten Sync-Tick", coin, tr['id'])
                continue
            pnl_coin_pct = pnl_usd / (qty * entry) * 100.0
            with app.cursor() as cur2:
                cur2.execute("""
                    UPDATE trader_positions
                    SET status='failsafe', closed_at=now(),
                        exit_px=%s, pnl_pct=%s
                    WHERE id=%s AND status='open'
                """, (avg_exit, round(pnl_coin_pct, 4), tr['id']))
                if cur2.rowcount > 0:
                    n_failsafe += 1
                    log.info("TRADER-FAILSAFE %s trader_pid=%s pnl_coin=%.3f%% exit=%.6g (HL-fills)",
                             coin, tr['id'], pnl_coin_pct, avg_exit)
            app.commit()

        # Richtung B: HL-Pos da aber TP/SL-Orders fehlen -> Repair aus trader_positions
        for pos in positions:
            coin = pos.get('coin'); direction = pos.get('direction')
            if not coin or not direction: continue
            qty = abs(float(pos.get('size', 0)))
            if qty <= 0: continue
            is_long = direction == 'long'
            target_side = 'A' if is_long else 'B'
            coin_orders = [o for o in orders_by_coin.get(coin, [])
                            if o.get('reduceOnly') and o.get('side') == target_side]
            has_tp = any(str(o.get('orderType','')).startswith('Limit') for o in coin_orders)
            has_sl = any('Trigger' in str(o.get('orderType','')) or 'Stop' in str(o.get('orderType',''))
                         for o in coin_orders)
            if has_tp and has_sl: continue
            # current_tp/sl aus trader_positions
            with app.cursor(cursor_factory=RealDictCursor) as cur_r:
                cur_r.execute("""
                    SELECT id, current_tp_px, current_sl_px FROM trader_positions
                    WHERE symbol=%s AND side=%s AND status='open'
                    ORDER BY id DESC LIMIT 1
                """, (coin, direction))
                row = cur_r.fetchone()
            if not row:
                log.warning("PROTECT %s %s ungeschuetzt + keine trader_positions-Row -> skip",
                            coin, direction)
                continue
            db_tp = float(row['current_tp_px']); db_sl = float(row['current_sl_px'])
            log.warning("PROTECT %s %s missing tp=%s sl=%s -> restore mit tp=%.6g sl=%.6g",
                        coin, direction, has_tp, has_sl, db_tp, db_sl)
            try:
                cancel_all_orders_for_coin_hl(creds, coin)
                pl = place_tp_sl_hl(creds, coin, is_long=is_long, quantity=qty,
                                     tp_price=db_tp, sl_price=db_sl)
                if pl.get('success'):
                    n_repaired += 1
                    log.info("PROTECT %s repaired", coin)
                else:
                    log.error("PROTECT %s repair scheitert (%s) -> safe_close", coin, pl.get('error'))
                    sc = safe_close_position_hl(creds, coin, addr)
                    if sc.get('success'):
                        # trader_positions auch schliessen
                        with app.cursor() as cur_f:
                            cur_f.execute("""
                                UPDATE trader_positions
                                SET status='failsafe', closed_at=now(),
                                    exit_px=current_tp_px, pnl_pct=0
                                WHERE id=%s AND status='open'
                            """, (row['id'],))
                        app.commit()
                        n_failsafe += 1
            except Exception as e:
                log.exception("PROTECT %s repair exception: %s", coin, e)

        # Richtung C: tote Orders ohne HL-Position -> cancel
        try:
            fe_after = info.frontend_open_orders(addr) or []
        except Exception:
            fe_after = fe_orders
        active_pos_coins = set(hl_pos_by_coin.keys())
        orphan_coins = sorted({o['coin'] for o in fe_after if o['coin'] not in active_pos_coins})
        for coin in orphan_coins:
            try:
                cr = cancel_all_orders_for_coin_hl(creds, coin)
                cancelled = int(cr.get('cancelled', 0)) if cr else 0
                if cancelled > 0:
                    n_orphan_cancelled += cancelled
                    log.info("ORPHAN-ORDER cleanup %s: %d cancelled", coin, cancelled)
            except Exception as e:
                log.warning("ORPHAN-ORDER cancel %s exception: %s", coin, e)

        # Richtung E (10.05.2026): Phantom-Orders bei AKTIVEN Positionen.
        # Pro offene Position: reduceOnly-Orders mit sz != position.sz sind Phantome
        # aus früheren Iterationen (TP/SL-Hit hat 1 Order genutzt, andere blieb +
        # neue Position bekam neue Orders). Sicheres Canceln: nur reduceOnly + sz-Mismatch.
        n_phantom_cancelled = 0
        try:
            from rl_agent.trader import cancel_order_hl
        except Exception:
            cancel_order_hl = None
        for coin, pos in hl_pos_by_coin.items():
            try:
                pos_sz = abs(float(pos.get('szi', 0)))
                if pos_sz <= 0: continue
                coin_orders = [o for o in fe_after if o.get('coin') == coin and o.get('reduceOnly')]
                # sz_mismatch = sz weicht > 0.1% von Position-sz ab
                phantoms = [o for o in coin_orders
                            if abs(float(o.get('sz', 0)) - pos_sz) / max(pos_sz, 1e-9) > 0.001]
                for ph in phantoms:
                    if cancel_order_hl is None: break
                    try:
                        oid = ph.get('oid')
                        if oid is None: continue
                        cr = cancel_order_hl(creds, coin, oid)
                        if cr.get('success'):
                            n_phantom_cancelled += 1
                            log.info("PHANTOM-ORDER cancel %s oid=%s (sz=%s pos_sz=%s)",
                                     coin, oid, ph.get('sz'), pos_sz)
                        else:
                            log.warning("PHANTOM-ORDER cancel %s oid=%s failed: %s",
                                        coin, oid, cr.get('error'))
                    except Exception as e:
                        log.warning("PHANTOM-ORDER cancel %s oid=%s exception: %s",
                                    coin, ph.get('oid'), e)
            except Exception as e:
                log.warning("PHANTOM-ORDER check %s exception: %s", coin, e)

    if n_failsafe or n_repaired or n_orphan_cancelled or n_phantom_cancelled:
        log.info("trader-sync: failsafe=%d repaired=%d orphan-orders=%d phantom-orders=%d",
                 n_failsafe, n_repaired, n_orphan_cancelled, n_phantom_cancelled)
    return n_failsafe + n_repaired


def timeout_watch(s):
    """Schliesst trader_positions mit timeout_enabled=true die laenger als
    predictor.bandit.timeout_hours offen sind. settings_reload zieht den
    Wert live nach, kein Hardcode. Trader-Welt only.

    Verhalten: HL-Position via close_position_hl schliessen, dann DB-Row
    status='closed' mit pnl_pct aus aktuellem mark-Price und exit_px=mark."""
    if '/opt/coin/backend' not in sys.path:
        sys.path.insert(0, '/opt/coin/backend')
    try:
        from rl_agent.trader import get_hl_credentials, close_position_hl, get_hl_info
    except Exception as e:
        log.warning("timeout_watch trader-import fehlgeschlagen: %s", e)
        return 0
    timeout_h = get_timeout_hours(s)
    creds = get_hl_credentials()
    if not creds: return 0
    addr = creds.get('wallet_address')
    n_closed = 0
    try:
        info = get_hl_info()
        mids = info.all_mids() or {}
    except Exception:
        mids = {}
    with db_app(s) as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_px, opened_at
                FROM trader_positions
                WHERE status='open' AND timeout_enabled=TRUE
                  AND opened_at < now() - (%s || ' hours')::interval
                ORDER BY opened_at
            """, (str(timeout_h),))
            due = cur.fetchall()
        for r in due:
            coin = r['symbol']; side = r['side']
            entry = float(r['entry_px'])
            mark_raw = mids.get(coin)
            if mark_raw is None:
                log.error("FALLBACK_TRIGGERED timeout_watch %s tid=%s: kein mark in HL.all_mids -> skip diesen Tick", coin, r['id'])
                continue
            mark = float(mark_raw)
            pnl_pct = (mark - entry) / entry * 100.0 if side == 'long' else (entry - mark) / entry * 100.0
            try:
                close_res = close_position_hl(creds, coin, addr)
                ok = close_res.get('success') if close_res else False
            except Exception as e:
                log.error("FALLBACK_TRIGGERED timeout_watch %s tid=%s: close_position_hl Exception '%s' -> Row bleibt open", coin, r['id'], e)
                ok = False
            if ok:
                with app.cursor() as cur2:
                    cur2.execute("""
                        UPDATE trader_positions
                        SET status='closed', closed_at=now(),
                            exit_px=%s, pnl_pct=%s
                        WHERE id=%s AND status='open'
                    """, (mark, round(pnl_pct, 4), r['id']))
                app.commit()
                n_closed += 1
                log.info("TIMEOUT-CLOSE %s %s trader_pid=%s pnl=%.3f%% (age>%.1fh)",
                         coin, side, r['id'], pnl_pct, timeout_h)
    return n_closed


def queue_pending_hindsight(s, lookahead_minutes=120):
    """Findet frisch geschlossene trader_positions, queued sie in
    trader_hindsight_queue mit ready_at = closed_at + lookahead. Trader-eigene Welt."""
    with db_app(s) as app:
        with app.cursor() as cur:
            cur.execute("""
                INSERT INTO trader_hindsight_queue (position_id, closed_at, ready_at)
                SELECT tp.id, tp.closed_at,
                       tp.closed_at + (%s || ' minutes')::interval
                FROM trader_positions tp
                WHERE tp.status IN ('closed','failsafe')
                  AND tp.closed_at IS NOT NULL
                  AND tp.closed_at >= now() - interval '6 hours'
                  AND EXISTS (SELECT 1 FROM trader_decisions WHERE position_id = tp.id)
                  AND NOT EXISTS (SELECT 1 FROM trader_hindsight_queue q WHERE q.position_id = tp.id)
                ON CONFLICT (position_id) DO NOTHING
            """, (str(int(lookahead_minutes)),))
            n = cur.rowcount
        app.commit()
    return n


def backfill_trader_from_closed_trades(s, modify_bandit, modify_scaler,
                                         days=7, tick_seconds=30, train_epochs=200,
                                         train_batch=64):
    """Trader-Cold-Start-Backfill aus historischen Closed Trades.

    Pro Trade alle 30s-Ticks im Range [created_at, closed_at + lookahead]:
    - Base-Features = open_predictions.features (Snapshot am Trade-Open)
    - Position-Features rekonstruiert aus Klines (exakt: pnl, peak, trough, dist_to_tp/sl)
    - Pro Tick × 102 Actions counterfactual replay -> Buffer

    Approximation: Base-Features sind nur am Open exakt; im Trade-Verlauf minimal
    verzerrt (z.B. RSI driftet). Akzeptabel fuer Cold-Start-Beschleunigung.

    Predictor-DB nur lesend genutzt, Trader-State + Buffer werden gefuellt."""
    log.info("Trader-Backfill: starte aus closed Trades letzte %dd", days)
    n_trades = 0; n_obs_added = 0
    # Hindsight-Lookahead = Trade-Timeout (Single Source of Truth, predictor.bandit.timeout_hours)
    lookahead_min = get_timeout_hours(s) * 60.0

    with db_app(s) as app, db_coins(s) as coins:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_px, sl_px, tp_px, features, created_at, closed_at
                FROM open_predictions
                WHERE status IN ('win','loss','timeout','auto_close_failsafe','auto_trade_failed')
                  AND closed_at IS NOT NULL AND features IS NOT NULL
                  AND created_at >= now() - (%s || ' days')::interval
                ORDER BY closed_at
            """, (str(days),))
            trades = cur.fetchall()
        if not trades:
            log.info("Trader-Backfill: keine closed Trades")
            return 0, 0

        for t in trades:
            sym = t['symbol']; side = t['side']
            entry = float(t['entry_px']); tp_orig = float(t['tp_px']); sl_orig = float(t['sl_px'])
            base_feat = dict(t['features']) if isinstance(t['features'], dict) else {}
            created = t['created_at']; closed = t['closed_at']
            end_at = closed + timedelta(minutes=lookahead_min)

            with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
                cur_c.execute("""
                    SELECT open_time, high, low, close FROM klines
                    WHERE symbol=%s AND interval='10s' AND open_time BETWEEN %s AND %s
                    ORDER BY open_time
                """, (sym, created, end_at))
                klines = cur_c.fetchall()
            if not klines:
                continue

            kl_by_time = {k['open_time']: k for k in klines}
            kl_times = sorted(kl_by_time.keys())

            cur_tick = created
            running_max = entry; running_min = entry
            while cur_tick <= closed:
                near = [kt for kt in kl_times if kt <= cur_tick]
                if not near:
                    cur_tick += timedelta(seconds=tick_seconds); continue
                nearest_kt = near[-1]
                kl = kl_by_time[nearest_kt]
                mark = float(kl.get('close') or entry)
                running_max = max(running_max, float(kl.get('high') or entry))
                running_min = min(running_min, float(kl.get('low') or entry))

                age_h = (cur_tick - created).total_seconds() / 3600.0
                if side == 'long':
                    pnl_now = (mark - entry) / entry * 100.0
                    peak_pct = (running_max - entry) / entry * 100.0
                    trough_pct = (running_min - entry) / entry * 100.0
                    dist_tp = (tp_orig - mark) / mark * 100.0 if mark > 0 else 0.0
                    dist_sl = (mark - sl_orig) / mark * 100.0 if mark > 0 else 0.0
                else:
                    pnl_now = (entry - mark) / entry * 100.0
                    peak_pct = (entry - running_min) / entry * 100.0
                    trough_pct = (entry - running_max) / entry * 100.0
                    dist_tp = (mark - tp_orig) / mark * 100.0 if mark > 0 else 0.0
                    dist_sl = (sl_orig - mark) / mark * 100.0 if mark > 0 else 0.0

                orig_idx_raw = base_feat.get('_action_idx', -1)
                try: orig_idx = float(orig_idx_raw)
                except (TypeError, ValueError): orig_idx = -1.0

                full_feat = dict(base_feat)
                full_feat.update({
                    'time_in_trade_h': age_h, 'pnl_now_pct': pnl_now,
                    'peak_pct_now': peak_pct, 'trough_pct_now': trough_pct,
                    'dist_to_tp_pct': dist_tp, 'dist_to_sl_pct': dist_sl,
                    'original_action_idx': orig_idx,
                })

                x_raw = vectorize_modify(full_feat)
                modify_scaler.update(x_raw)
                x = modify_scaler.transform(x_raw)

                klines_after = [kl_by_time[kt] for kt in kl_times if kt >= cur_tick]
                if not klines_after:
                    cur_tick += timedelta(seconds=tick_seconds); continue
                first_close = float(klines_after[0].get('close') or entry)

                for i, action in enumerate(modify_bandit.actions):
                    if action.get('special') == 'hold':
                        test_tp, test_sl = tp_orig, sl_orig
                    elif action.get('special') == 'close':
                        pnl = ((first_close - entry)/entry*100) if side=='long' else ((entry - first_close)/entry*100)
                        modify_bandit.add_observation(x, i, pnl)
                        n_obs_added += 1
                        continue
                    else:
                        tp_d = float(action.get('tp_delta') or 0.0)
                        sl_d = float(action.get('sl_delta') or 0.0)
                        if side == 'long':
                            test_tp = tp_orig + (first_close * tp_d / 100.0)
                            test_sl = sl_orig - (first_close * sl_d / 100.0)
                        else:
                            test_tp = tp_orig - (first_close * tp_d / 100.0)
                            test_sl = sl_orig + (first_close * sl_d / 100.0)
                    status, _, pnl, _dur = replay_modify_action(klines_after, side, entry, test_tp, test_sl)
                    if status == 'no_data':
                        continue
                    modify_bandit.add_observation(x, i, pnl)
                    n_obs_added += 1

                cur_tick += timedelta(seconds=tick_seconds)
            n_trades += 1

    log.info("Trader-Backfill: %d Trades, %d Obs gesammelt -> Training (%d epochs)",
             n_trades, n_obs_added, train_epochs)
    if n_obs_added > 0:
        steps, avg_loss = modify_bandit.train_steps(batch_size=train_batch, n_epochs=train_epochs)
        log.info("Trader-Backfill Training fertig: %d steps, avg_loss=%.4f, buffer=%d",
                 steps, avg_loss, len(modify_bandit.replay_buffer))
    return n_trades, n_obs_added


def process_due_hindsight(s, modify_bandit, modify_scaler, batch_size=64,
                           epochs=5, lookahead_minutes=120):
    """Holt due Pending-Hindsights aus Queue (ready_at <= now), faehrt
    hindsight_replay_modify_neural fuer jeden. LinTSBandit.update ist online —
    kein separater Train-Step noetig. Markiert processed_at + n_updates."""
    n_processed = 0
    n_total_updates = 0
    reward_cfg = s["predictor"]["modify_bandit"].get("reward", {})
    with db_app(s) as app, db_coins(s) as coins:
        while True:
            with app.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT q.id AS queue_id, q.position_id, q.closed_at,
                           tp.symbol, tp.side, tp.entry_px, tp.opened_at
                    FROM trader_hindsight_queue q
                    JOIN trader_positions tp ON q.position_id = tp.id
                    WHERE q.processed_at IS NULL AND q.ready_at <= now()
                    ORDER BY q.ready_at LIMIT 10
                """)
                due = cur.fetchall()
            if not due:
                break
            for d in due:
                pred_dict = {
                    'id': d['position_id'], 'symbol': d['symbol'],
                    'side': d['side'], 'entry_px': d['entry_px'],
                    'closed_at': d['closed_at'], 'opened_at': d['opened_at'],
                }
                try:
                    n_u = hindsight_replay_modify_neural(
                        coins, app, modify_bandit, modify_scaler,
                        pred_dict, lookahead_minutes=lookahead_minutes,
                        reward_cfg=reward_cfg)
                except Exception as e:
                    log.warning("hindsight position_id=%s fehlgeschlagen: %s",
                                d['position_id'], e)
                    n_u = -1
                with app.cursor() as cur2:
                    cur2.execute("""
                        UPDATE trader_hindsight_queue
                        SET processed_at = now(), n_updates = %s
                        WHERE id = %s
                    """, (n_u, d['queue_id']))
                app.commit()
                if n_u > 0:
                    n_total_updates += n_u
                n_processed += 1
            if len(due) < 10:
                break
    if n_processed > 0:
        log.info("trader-hindsight: %d positions, %d action-updates (n_obs total=%d)",
                 n_processed, n_total_updates, sum(modify_bandit.n_obs))
    return n_processed


# =============================================================================
# Observer (v1, 21.05.2026) — bei Marktshift offene Positionen neu bewerten
# =============================================================================

def detect_market_shift(coins, obs_cfg):
    """True wenn Referenz-Coin (BTC) sich in window_minutes mehr als
    btc_move_threshold_pct bewegt hat. Returnt (move_pct, direction) oder None."""
    ref = obs_cfg.get("reference_coin")
    win = obs_cfg.get("window_minutes")
    thr = obs_cfg.get("btc_move_threshold_pct")
    if ref is None or win is None or thr is None:
        log.error("FALLBACK_TRIGGERED detect_market_shift: observer-settings unvollständig")
        return None
    with coins.cursor() as cur:
        cur.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT %s", (ref, int(win)+1))
        rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    if len(rows) < int(win)+1:
        return None
    now_px = rows[0]; old_px = rows[-1]
    if not old_px:
        return None
    move = (now_px - old_px) / old_px * 100.0
    if abs(move) >= float(thr):
        return {"move_pct": move, "direction": "up" if move > 0 else "down"}
    return None


def _evaluate_coin_reeval(coins, app, sym, cfg, mh_model, mh_model_lock, btc_moves, uni_ctx):
    """Berechnet für EINEN Coin die aktuellen Features + Predictor-Bewertung.
    Identische compute_*-Pipeline wie scan_pass_mh (geteilte Feature-Module).
    Returnt {side, confidence, tp_pct, sl_pct} oder None."""
    mh_cfg = cfg.get("multi_head", {})
    rules = cfg.get("rules", [])
    lookback = cfg["lookback_minutes"]
    min_n = int(mh_cfg.get("min_n_for_predict", 30))
    tp_safety = float(mh_cfg.get("tp_safety_factor", 0.6))
    sl_safety = float(mh_cfg.get("sl_safety_factor", 1.3))
    max_tp_pct = float(mh_cfg.get("max_tp_pct"))
    stalker_cfg = cfg.get("stalker", {}) or {}
    stalker_on = bool(stalker_cfg.get("enabled")) and uni_ctx.get("stalker_btc_regime") is not None

    rule_flags = {}
    with coins.cursor(cursor_factory=RealDictCursor) as cur_w:
        for rule in rules:
            try: ok = evaluate_rule(cur_w, sym, rule, lookback)["ok"]
            except Exception: ok = False
            rule_flags[rule['name']] = 1 if ok else 0

    feat = feature_snapshot_v2(coins, sym, rule_flags=rule_flags, btc_moves=btc_moves)
    if feat is None: return None
    of = compute_orderflow_features(coins, sym)
    if of is None: return None
    feat.update(of)
    fb = compute_funding_features(coins, sym, uni_ctx["funding_median"])
    if fb is None: return None
    feat.update(fb)
    sec = compute_sector_features(sym, uni_ctx["sector_map"], uni_ctx["sector_stats"], uni_ctx["coin_pcts"])
    if sec is None: return None
    feat.update(sec)
    if uni_ctx["whale_enabled"]:
        wh = compute_whale_features(app, sym)
        if wh: feat.update(wh)
    if stalker_on:
        bl = compute_baseline_dev_features(coins, app, sym, stalker_cfg,
                                           btc_regime=uni_ctx["stalker_btc_regime"],
                                           effective_min_samples=uni_ctx["stalker_eff_min_samples"])
        if bl is None: return None
        feat.update(bl)
        feat.update(compute_cross_coin_features(coins, sym, stalker_cfg["cross_coin"], uni_ctx["stalker_ref_cache"]))
        feat.update(compute_identity_features(sym, app, stalker_cfg["coin_identity"]))

    with mh_model_lock:
        probs = mh_model.predict_proba(feat, min_n_for_predict=min_n)
        if probs is None:
            return None
        side = 'long' if probs['long'] >= probs['short'] else 'short'
        confidence = max(probs['long'], probs['short'])
        tp_raw = mh_model.predict_tp(feat, side, min_n_for_predict=min_n)
        sl_raw = mh_model.predict_sl(feat, side, min_n_for_predict=min_n)
    if tp_raw is None or sl_raw is None:
        return None
    tp_pct = min(tp_raw * tp_safety, max_tp_pct)
    sl_pct = max(sl_raw * sl_safety, 0.1)
    return {"side": side, "confidence": float(confidence), "tp_pct": tp_pct, "sl_pct": sl_pct}


def observer_reeval(s, mh_model, mh_model_lock):
    """Bei Marktshift: offene Paper-Positionen am Predictor neu bewerten.
    Wenn der Predictor die Richtung der Position noch bestätigt → TP/SL an die
    aktuellen Magnituden anpassen (Endlosrutsche kappen). Wenn der Predictor
    die GEGEN-Richtung sieht → SL enger ziehen (auf aktuelle Magnitude).
    v1: nur paper_positions (Live ist Paper). Lernen über finalen Close."""
    cfg = s["predictor"]
    obs_cfg = cfg.get("observer", {})
    if not obs_cfg.get("enabled"):
        return 0
    paper_mode = bool(cfg.get("trading", {}).get("paper_mode"))
    if not paper_mode:
        return 0  # v1: nur Paper-Welt
    min_age = float(obs_cfg.get("reeval_min_age_minutes", 3))

    with db_coins(s) as coins, db_app(s) as app:
        shift = detect_market_shift(coins, obs_cfg)
        if not shift:
            return 0
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id, symbol, side, entry_px, tp_px, sl_px,
                                  EXTRACT(EPOCH FROM (now()-opened_at))/60.0 AS age_min
                           FROM paper_positions WHERE status='open'
                             AND EXTRACT(EPOCH FROM (now()-opened_at))/60.0 >= %s""", (min_age,))
            opens = cur.fetchall()
        if not opens:
            return 0

        # Universe-Kontext 1× bauen
        symbols = list({o['symbol'] for o in opens})
        sector_priority = cfg.get("sector_priority", [])
        sector_map = build_coin_sector_map(app, symbols, sector_priority)
        sector_stats, coin_pcts = compute_sector_close_pcts(coins, sector_map)
        uni_ctx = {
            "sector_map": sector_map, "sector_stats": sector_stats, "coin_pcts": coin_pcts,
            "funding_median": compute_universe_funding_median(coins, symbols),
            "whale_enabled": bool(cfg.get("whale_tracker", {}).get("enabled", False)),
            "stalker_ref_cache": None, "stalker_btc_regime": None, "stalker_eff_min_samples": None,
        }
        stalker_cfg = cfg.get("stalker", {}) or {}
        if stalker_cfg.get("enabled"):
            try:
                uni_ctx["stalker_ref_cache"] = stalker_build_ref_cache(coins, stalker_cfg["cross_coin"])
                uni_ctx["stalker_btc_regime"] = stalker_classify_btc_regime(coins, stalker_cfg["btc_regime"])
                uni_ctx["stalker_eff_min_samples"] = stalker_effective_min_samples(stalker_cfg, stalker_data_days(coins))
            except Exception as e:
                log.warning("observer stalker-ctx failed: %s", e)
        btc_moves = load_btc_moves(coins)

        n_adj = 0
        for o in opens:
            try:
                ev = _evaluate_coin_reeval(coins, app, o['symbol'], cfg, mh_model, mh_model_lock, btc_moves, uni_ctx)
                if ev is None:
                    continue
                entry = float(o['entry_px']); side = o['side']
                # Neue TP/SL gegen ENTRY der bestehenden Position (Position bleibt offen)
                if side == 'long':
                    new_tp = entry * (1 + ev['tp_pct']/100.0)
                    new_sl = entry * (1 - ev['sl_pct']/100.0)
                else:
                    new_tp = entry * (1 - ev['tp_pct']/100.0)
                    new_sl = entry * (1 + ev['sl_pct']/100.0)
                # Nur anpassen wenn der Predictor die Richtung der Position noch hält.
                # Sieht er die Gegenrichtung → SL eng an aktuellen Preis (Reißleine).
                same_dir = (ev['side'] == side)
                if not same_dir:
                    # Gegenrichtung erkannt → SL auf aktuelle Magnitude (enger), TP lassen
                    with app.cursor() as cu:
                        cu.execute("UPDATE paper_positions SET sl_px=%s WHERE id=%s AND status='open'", (new_sl, o['id']))
                    app.commit()
                    log.info("OBSERVER %s ppid=%s GEGENRICHTUNG (%s→%s) SL→%.6g (shift %.2f%%)",
                             o['symbol'], o['id'], side, ev['side'], new_sl, shift['move_pct'])
                    n_adj += 1
                else:
                    with app.cursor() as cu:
                        cu.execute("UPDATE paper_positions SET tp_px=%s, sl_px=%s WHERE id=%s AND status='open'", (new_tp, new_sl, o['id']))
                    app.commit()
                    log.info("OBSERVER %s ppid=%s bestätigt %s TP→%.6g SL→%.6g (shift %.2f%%)",
                             o['symbol'], o['id'], side, new_tp, new_sl, shift['move_pct'])
                    n_adj += 1
            except Exception as e:
                log.warning("observer reeval %s failed: %s", o['symbol'], e)
        if n_adj > 0:
            log.info("OBSERVER market-shift %.2f%% (%s): %d Paper-Positionen neu bewertet",
                     shift['move_pct'], shift['direction'], n_adj)
        return n_adj


# =============================================================================
# Main
# =============================================================================

def main():
    global FEATURE_KEYS, N_FEAT, MODIFY_FEATURE_KEYS, N_FEAT_MODIFY
    s = load_settings()
    cfg = s["predictor"]
    if not cfg.get("enabled"):
        log.info("predictor disabled in settings, exit"); return

    # Feature-Keys dynamisch aus settings (rule-namen koennen sich aendern)
    FEATURE_KEYS = build_feature_keys(s)
    N_FEAT = len(FEATURE_KEYS) + 1

    # ============= Multi-Head Predictor (v5) =============
    mh_cfg = cfg.get("multi_head", {})
    state_path = mh_cfg.get("state_path",
                            "/opt/coin/database/data/models/predictor_v5_mh.pkl")
    seed = int(mh_cfg.get("seed", 42))
    n_models_direction = int(mh_cfg.get("n_models_direction", 10))
    grace_period = int(mh_cfg.get("grace_period", 50))

    mh_model = None
    if os.path.exists(state_path):
        try:
            mh_model = MultiHeadPredictor.load(state_path)
            if mh_model.version != MultiHeadPredictor.VERSION:
                log.warning("Multi-Head state version mismatch (%s != %s), fresh start",
                            mh_model.version, MultiHeadPredictor.VERSION)
                mh_model = None
            else:
                log.info("Multi-Head loaded: stats=%s", mh_model.stats())
        except Exception as e:
            log.warning("Multi-Head load failed (%s), fresh start", e)
            mh_model = None
    if mh_model is None:
        mh_model = MultiHeadPredictor(
            seed=seed,
            n_models_direction=n_models_direction,
            grace_period=grace_period,
        )
        mh_model.save(state_path)
        log.info("Multi-Head fresh start: seed=%d n_models=%d grace=%d",
                 seed, n_models_direction, grace_period)

    rng = np.random.default_rng(seed)
    log.info("Predictor v5 (Multi-Head) start. scan=%ss watch=%ss top_n=%d",
             cfg["scan_interval_seconds"], cfg["watch_interval_seconds"], cfg["universe_top_n"])

    # ============= Watch-Thread =============
    # Watch_pass läuft eigenständig (sekündlich) damit TP/SL-Hits nicht durch
    # langsame scan-passes (8-10min mit Stalker) verzögert werden.
    # mh_model-Schreibzugriffe (learn_close, save) sind via mh_model_lock serialisiert.
    import threading
    mh_model_lock = threading.Lock()

    def _watch_thread_loop():
        watch_interval = float(cfg["watch_interval_seconds"])
        last_timeout_watch_local = 0.0
        last_hl_sync_local = 0.0
        last_observer_local = 0.0
        while True:
            try:
                s_local = load_settings()
                with mh_model_lock:
                    c = watch_pass_mh(s_local, mh_model, state_path)
                if c > 0:
                    log.info("watch-pass: %d closed", c)
                # timeout_watch (Trader-Welt): 10s-Tick, entkoppelt vom scan_pass.
                # Verkauf max 10s verspaetet statt bis 9-min-scan_pass durch ist.
                now_inner = time.time()
                if now_inner - last_timeout_watch_local >= 10:
                    try:
                        n_to = timeout_watch(s_local)
                        if n_to > 0:
                            log.info("timeout-watch: %d Positionen via Timeout geschlossen", n_to)
                    except Exception as e:
                        log.exception("timeout-watch (in watch-thread) failed: %s", e)
                    last_timeout_watch_local = now_inner
                # HL-Sync: trader_positions ↔ echte HL-Realität abgleichen (auch hier
                # entkoppelt vom scan_pass — sonst weiß Wallet erst nach 9 min was HL
                # tatsächlich gemacht hat, z.B. SL-Trigger).
                if now_inner - last_hl_sync_local >= 30:
                    try:
                        n_sync = sync_hl_to_db(s_local)
                        if n_sync > 0:
                            log.info("hl-sync: %d Phantom-Positionen reaktiviert", n_sync)
                    except Exception as e:
                        log.exception("hl-sync (in watch-thread) failed: %s", e)
                    last_hl_sync_local = now_inner

                # Paper-Watcher: virtuelle paper_positions auf TP/SL/Timeout prüfen
                # (nur wenn paper_mode aktiv — sonst no-op). Sekündlich wie watch_pass.
                if bool(s_local.get("predictor", {}).get("trading", {}).get("paper_mode")):
                    try:
                        with db_coins(s_local) as _pc, db_app(s_local) as _pa:
                            n_pc = paper_watch(_pa, _pc, s_local["predictor"], get_timeout_hours(s_local))
                        if n_pc > 0:
                            log.info("paper-watch: %d Paper-Positionen geschlossen", n_pc)
                    except Exception as e:
                        log.exception("paper-watch failed: %s", e)

                # Observer: bei Marktshift offene Positionen neu bewerten + TP/SL anpassen.
                # Eigener Tick (check_interval_seconds), nutzt mh_model_lock via _evaluate_coin_reeval.
                obs_cfg = s_local.get("predictor", {}).get("observer", {})
                if obs_cfg.get("enabled"):
                    obs_iv = float(obs_cfg.get("check_interval_seconds", 30))
                    if now_inner - last_observer_local >= obs_iv:
                        try:
                            n_obs_adj = observer_reeval(s_local, mh_model, mh_model_lock)
                            if n_obs_adj > 0:
                                log.info("observer: %d Positionen neu bewertet", n_obs_adj)
                        except Exception as e:
                            log.exception("observer_reeval failed: %s", e)
                        last_observer_local = now_inner
                with mh_model_lock:
                    n_h = process_due_mh_hindsight(s_local, mh_model, state_path)
                if n_h > 0:
                    log.info("mh-hindsight: %d Trades extended-learned (n_obs=%d)",
                             n_h, mh_model.direction.n_obs)
            except Exception as e:
                log.exception("watch-thread error: %s", e)
            time.sleep(watch_interval)

    watch_thread = threading.Thread(target=_watch_thread_loop, name="watch_thread", daemon=True)
    watch_thread.start()
    log.info("watch-thread started (interval=%ss)", cfg["watch_interval_seconds"])

    # ============= Trader-Setup (LinTSBandit, 2 Actions, eigene Welt) =============
    # Reform 10.05.2026: NeuralBandit -> LinTSBandit, 102 Actions -> 2 Actions
    mb_cfg = cfg.get("modify_bandit", {})
    modify_bandit = None
    modify_scaler = None
    modify_state_path = None
    modify_interval = float(mb_cfg.get("interval_seconds", 30))
    # hindsight_delay_min wird pro Tick in der main-loop live aus get_timeout_hours(s) * 60 berechnet
    if mb_cfg.get("enabled"):
        MODIFY_FEATURE_KEYS = build_modify_feature_keys(s)
        N_FEAT_MODIFY = len(MODIFY_FEATURE_KEYS) + 1
        m_actions = build_modify_actions()
        m_action_names = [a['name'] for a in m_actions]
        modify_state_path = mb_cfg.get(
            "state_path",
            "/opt/coin/database/data/models/predictor_modify_lints.pkl")
        b_alpha = float(mb_cfg.get("prior_alpha", 1.0))
        b_sigma2 = float(mb_cfg.get("noise_sigma2", 1.0))

        m_state = None
        m_fresh = None
        if os.path.exists(modify_state_path):
            try:
                with open(modify_state_path, 'rb') as fp:
                    m_state = pickle.load(fp)
            except Exception as e:
                log.warning("modify-bandit state load failed: %s", e)
                m_state = None

        if m_state is None:
            m_fresh = "no state file"
        elif m_state.get('arch') != 'lints':
            m_fresh = f"arch changed ({m_state.get('arch')!r} -> lints)"
        elif m_state.get('action_names') != m_action_names:
            m_fresh = "action-space changed"
        elif m_state.get('feature_keys') != MODIFY_FEATURE_KEYS:
            m_fresh = "feature-keys changed"
        elif m_state.get('n_features') != N_FEAT_MODIFY:
            m_fresh = "n_features changed"

        if m_fresh:
            if m_state is not None:
                log.warning("Trader fresh start — %s", m_fresh)
            modify_bandit = LinTSBandit(N_FEAT_MODIFY, m_actions,
                                         alpha=b_alpha, sigma2=b_sigma2)
            modify_scaler = OnlineScaler(N_FEAT_MODIFY)
            log.info("Trader (LinTS) fresh start: %d actions, %d features",
                     len(m_actions), N_FEAT_MODIFY)
            with open(modify_state_path, 'wb') as fp:
                pickle.dump({'arch': 'lints', 'bandit': modify_bandit,
                             'scaler': modify_scaler,
                             'feature_keys': MODIFY_FEATURE_KEYS,
                             'n_features': N_FEAT_MODIFY,
                             'action_names': m_action_names}, fp)
        else:
            modify_bandit = m_state['bandit']
            modify_scaler = m_state['scaler']
            log.info("Trader (LinTS) loaded: %d actions, %d features, n_obs=%s, cum_reward=%s",
                     len(m_actions), N_FEAT_MODIFY,
                     modify_bandit.n_obs, [round(r, 2) for r in modify_bandit.cum_reward])
    else:
        log.info("Trader disabled in settings")

    # Hindsight-Backfill v4 entfällt — Multi-Head v5 lernt online beim Close,
    # kein Backfill auf historischen Closes (Cardinal Rule 4 — kein Pretrain).

    last_scan = 0.0; last_watch = 0.0; last_settings_reload = 0.0; last_modify = 0.0; last_modify_hindsight_id = 0; last_hl_sync = 0.0
    last_quality_check = 0.0; last_quality_telegram = 0.0; last_timeout_watch = 0.0

    while True:
        try:
            now = time.time()
            if now - last_settings_reload >= 60:
                s = load_settings(); cfg = s["predictor"]
                last_settings_reload = now
                # Multi-Head hat keine TP/SL-Buckets — Live-Bucket-Reload entfällt.

            # HL-Sync + Timeout-Watch laufen jetzt im watch-thread (entkoppelt
            # vom scan_pass). Im Main-Loop NUR noch scan_pass + modify_pass + quality_check.

            # Daten-Quality: alle 30 Min Quick-Log, alle 12h Telegram-Report
            if now - last_quality_check >= 1800:
                try:
                    send_tg = (now - last_quality_telegram >= 12 * 3600)
                    data_quality_check(s, send_telegram_report=send_tg)
                    if send_tg:
                        last_quality_telegram = now
                except Exception as e:
                    log.exception("data_quality_check failed: %s", e)
                last_quality_check = now

            if now - last_scan >= cfg["scan_interval_seconds"]:
                # scan_pass macht nur predict_* (read-only auf mh_model) -> kein Lock,
                # sonst blockt 8-min scan den sekündlichen watch-thread komplett.
                # watch_pass (writes via learn_close) hält den Lock alleine.
                m = scan_pass_mh(s, mh_model, rng, mh_model_lock)
                last_scan = now
                if m > 0:
                    log.info("scan-pass: %d new predictions", m)

            # watch_pass + extended-hindsight laufen in eigenem Thread (sekündlich) —
            # nicht mehr im main-loop.

            # Trader-Tick alle interval_seconds (live TP/SL-Modifikation)
            if modify_bandit is not None and now - last_modify >= modify_interval:
                try:
                    modify_pass(s, modify_bandit, modify_scaler, rng)
                except Exception as e:
                    log.exception("modify_pass failed: %s", e)
                last_modify = now

                # Hindsight-Lookahead = aktuelles Timeout (Single Source of Truth)
                hindsight_delay_min = get_timeout_hours(s) * 60.0
                # Hindsight-Queue: frisch geschlossene Trades fuer Hindsight einreihen
                try:
                    n_q = queue_pending_hindsight(s, lookahead_minutes=int(hindsight_delay_min))
                    if n_q > 0:
                        log.info("modify-hindsight: %d Trades fuer Lernen in %dmin queued",
                                 n_q, int(hindsight_delay_min))
                except Exception as e:
                    log.exception("queue_pending_hindsight failed: %s", e)

                # Due-Hindsights verarbeiten — LinTS lernt online, kein Train-Step.
                try:
                    n_p = process_due_hindsight(s, modify_bandit, modify_scaler,
                                                 lookahead_minutes=int(hindsight_delay_min))
                    if n_p > 0:
                        with open(modify_state_path, 'wb') as fp:
                            pickle.dump({'arch': 'lints', 'bandit': modify_bandit,
                                         'scaler': modify_scaler,
                                         'feature_keys': MODIFY_FEATURE_KEYS,
                                         'n_features': N_FEAT_MODIFY,
                                         'action_names': [a['name'] for a in modify_bandit.actions]},
                                        fp)
                except Exception as e:
                    log.exception("process_due_hindsight failed: %s", e)

            time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("interrupt"); break
        except Exception as e:
            log.exception("loop error: %s", e); time.sleep(5)


if __name__ == "__main__":
    main()
