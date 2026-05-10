"""Predictor Service v4 — Contextual Bandit (Linear Thompson Sampling).

Ein Modell, ein Ziel: maximize expected reward.
Action-Space: skip + (long|short) × tp_buckets × sl_buckets, alle aus settings.json.

Settings: settings.json -> 'predictor.bandit' (neu).
Tabellen unverändert: open_predictions, predictor_state, prediction_feedback.
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
    if len(clean) < period + 1: return 50.0
    gains = []; losses = []
    for i in range(1, len(clean)):
        diff = clean[i] - clean[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if len(gains) < period: return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes, fast=12, slow=26):
    clean = [c for c in closes if c is not None]
    if len(clean) < slow: return 0.0
    ef = _ema(clean, fast); es = _ema(clean, slow)
    if ef is None or es is None: return 0.0
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


def load_btc_moves(coins_conn):
    out = {'5m': 0.0, '15m': 0.0, '60m': 0.0}
    try:
        with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT bucket, close FROM agg_1m WHERE symbol='BTC'
                ORDER BY bucket DESC LIMIT 65
            """)
            rows = list(cur.fetchall())
        if len(rows) < 6: return out
        cl_now = f(rows[0]['close'])
        if not cl_now: return out
        for h in (5, 15, 60):
            if len(rows) > h:
                c0 = f(rows[h]['close'])
                if c0 and c0 > 0:
                    out[f'{h}m'] = (cl_now - c0) / c0 * 100.0
    except Exception as e:
        log.warning("load_btc_moves failed: %s", e)
    return out


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
            FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 240
        """, (symbol,))
        rows = list(reversed(cur.fetchall()))
        if not rows or len(rows) < 16: return None
        last = rows[-1]
        if last['close'] is None: return None
        cl = f(last['close'])

        for k in ("funding","open_interest","premium","spread_bps","book_imbalance_5","book_depth_5"):
            v = f(last[k])
            feat[k] = v if v is not None else 0.0

        bidsz = f(last['bbo_bid_sz']) or 0
        asksz = f(last['bbo_ask_sz']) or 0
        feat['bbo_size_ratio'] = (bidsz - asksz) / (bidsz + asksz + 1e-9)

        mark = f(last['mark_px']); mid = f(last['mid_px']); oracle = f(last['oracle_px'])
        feat['mark_vs_mid_bps'] = ((mark - mid) / mid * 10000) if (mark and mid) else 0.0
        feat['mark_vs_oracle_bps'] = ((mark - oracle) / oracle * 10000) if (mark and oracle) else 0.0

        def close_at(ago_min):
            idx = -1 - ago_min
            if abs(idx) > len(rows): return None
            v = f(rows[idx]['close'])
            return v if v else None

        for h in (1, 5, 15, 30, 60, 240):
            c0 = close_at(h)
            feat[f'close_pct_{h}m'] = ((cl - c0) / c0 * 100) if c0 else 0.0

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
        feat['rsi_14_1m'] = _rsi(closes_1m, 14)
        closes_5m = closes_1m[::-5][::-1] if len(closes_1m) >= 70 else closes_1m
        feat['rsi_14_5m'] = _rsi(closes_5m, 14)
        feat['macd_line'] = _macd(closes_1m, 12, 26)
        feat['macd_pct'] = (feat['macd_line'] / cl * 100.0) if cl else 0.0

        cur.execute("""
            SELECT bucket, funding FROM agg_1h WHERE symbol=%s ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        f24 = [f(r['funding']) for r in cur.fetchall()]
        clean_f = [x for x in f24 if x is not None]
        if len(clean_f) >= 6:
            f_mean = sum(clean_f) / len(clean_f)
            f_var = sum((x - f_mean) ** 2 for x in clean_f) / len(clean_f)
            f_std = math.sqrt(f_var)
            cur_f = clean_f[0]
            feat['funding_zscore_24h'] = ((cur_f - f_mean) / f_std) if f_std > 0 else 0.0
        else:
            feat['funding_zscore_24h'] = 0.0
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

    if btc_moves is not None and symbol != 'BTC':
        feat['btc_close_pct_5m'] = btc_moves.get('5m', 0.0)
        feat['btc_close_pct_15m'] = btc_moves.get('15m', 0.0)
        feat['btc_close_pct_60m'] = btc_moves.get('60m', 0.0)
        feat['rel_strength_5m'] = feat.get('close_pct_5m', 0.0) - btc_moves.get('5m', 0.0)
        feat['rel_strength_15m'] = feat.get('close_pct_15m', 0.0) - btc_moves.get('15m', 0.0)
        feat['rel_strength_60m'] = feat.get('close_pct_60m', 0.0) - btc_moves.get('60m', 0.0)
    else:
        for k in ('btc_close_pct_5m','btc_close_pct_15m','btc_close_pct_60m',
                   'rel_strength_5m','rel_strength_15m','rel_strength_60m'):
            feat[k] = 0.0

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
    """Liste der Feature-Keys = base + sortierte rule_<name>-Keys aus settings."""
    rule_keys = sorted([f"rule_{r['name']}" for r in s.get("predictor", {}).get("rules", [])])
    return _BASE_FEATURE_KEYS + rule_keys


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
# Scan-Pass v4 — Bandit entscheidet
# =============================================================================

def scan_pass_v4(s, bandit, scaler, rng):
    cfg = s["predictor"]
    bandit_cfg = cfg.get("bandit", {})
    rules = cfg.get("rules", [])
    lookback = cfg["lookback_minutes"]
    cooldown = cfg.get("cooldown_seconds_per_symbol", 300)
    expl_floor = float(bandit_cfg.get("exploration_floor", 0.05))
    expl_init = float(bandit_cfg.get("exploration_init", 1.0))
    expl_decay = float(bandit_cfg.get("exploration_decay_per_trade", 0.0005))
    n_total = sum(bandit.n_obs)
    exploration = max(expl_floor, expl_init - expl_decay * n_total)
    max_open = int(bandit_cfg.get("max_open_predictions", 20))
    min_reward = float(bandit_cfg.get("min_expected_reward", 0.3))
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

        # Quality-Gate: wieviele Slots frei?
        with app.cursor() as cur_a:
            cur_a.execute("SELECT COUNT(*) FROM open_predictions WHERE status='open'")
            n_open = cur_a.fetchone()[0]
        slots_free = max(0, max_open - n_open)
        if slots_free == 0:
            log.info("scan: %d/%d offen, keine neuen Trades", n_open, max_open)
            return 0

        # Cooldown: letzten Close pro Symbol bulk-fetchen, damit wir Mindestabstand
        # zwischen Trades am gleichen Symbol halten koennen.
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

        candidates = []
        with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
            for sym in uni:
                if has_open(app, sym): continue

                # Cooldown nach letztem Close
                last_close = last_closes.get(sym)
                if last_close is not None:
                    age = (datetime.now(timezone.utc) - last_close).total_seconds()
                    if age < cooldown:
                        continue

                rule_flags = {}
                for rule in rules:
                    try:
                        ok = evaluate_rule(cur_c, sym, rule, lookback)["ok"]
                    except Exception:
                        ok = False
                    rule_flags[rule['name']] = 1 if ok else 0

                feat = feature_snapshot_v2(coins, sym, rule_flags=rule_flags, btc_moves=btc_moves)
                if feat is None: continue

                x_raw = vectorize(feat)
                x = scaler.transform(x_raw)
                idx, expected_r, _ = bandit.select(x, exploration=exploration, rng=rng)
                action = bandit.actions[idx]

                if action['side'] is None:  # skip
                    continue

                if expected_r < min_reward:
                    continue

                cur_c.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
                p = cur_c.fetchone()
                if not p or p['close'] is None: continue
                entry = f(p['close'])
                tp_pct = float(action['tp_pct']); sl_pct = float(action['sl_pct'])
                if action['side'] == 'long':
                    tp = entry * (1 + tp_pct / 100.0)
                    sl = entry * (1 - sl_pct / 100.0)
                else:
                    tp = entry * (1 - tp_pct / 100.0)
                    sl = entry * (1 + sl_pct / 100.0)

                feat_with_action = dict(feat)
                feat_with_action['_action_idx'] = idx
                feat_with_action['_action_name'] = action['name']

                candidates.append({
                    'sym': sym, 'side': action['side'], 'entry': entry, 'sl': sl, 'tp': tp,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'expected_r': expected_r,
                    'action_name': action['name'], 'feat': feat_with_action,
                })

        # Sortieren nach expected_reward DESC, Top-slots_free oeffnen
        candidates.sort(key=lambda c: c['expected_r'], reverse=True)
        log.info("scan: %d/%d offen, %d Slots frei, %d Kandidaten >= min_reward=%.2f",
                 n_open, max_open, slots_free, len(candidates), min_reward)
        for c in candidates[:slots_free]:
            pid = open_prediction(app, c['sym'], c['side'], c['entry'], c['sl'], c['tp'],
                                   c['expected_r'], c['action_name'], 'bandit', c['feat'],
                                   pred_up=c['tp_pct'], pred_down=c['sl_pct'])
            if pid:
                matches += 1
                log.info("OPEN %s %s entry=%.6g tp=%.6g sl=%.6g action=%s exp_r=%.4f expl=%.3f",
                         c['sym'], c['side'], c['entry'], c['tp'], c['sl'], c['action_name'],
                         c['expected_r'], exploration)
                try_auto_trade(s, pid, c['sym'], c['side'], c['entry'], c['tp'], c['sl'],
                               c['tp_pct'], c['sl_pct'])
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

    try:
        # Live-Mid als Entry-Anker (nicht der DB-close, der ist evtl. paar Sekunden alt)
        from rl_agent.trader import get_current_prices_hl, get_hl_credentials
        from predictor.order_executor import execute_order_with_failsafe
        creds = get_hl_credentials(user_id=1)
        if not creds:
            log.warning("auto-trade %s: keine HL-Creds", symbol); return
        mids = get_current_prices_hl()
        live_px = mids.get(symbol) if mids else None
        if not live_px:
            log.warning("auto-trade %s: kein Live-Preis von HL", symbol); return
        entry_live = float(live_px)
        # tp_pct/sl_pct werden an execute_order_with_failsafe weitergegeben
        # — der rechnet TP/SL gegen den TATSAECHLICHEN Fill-Preis neu (verhindert
        # dass Slippage zwischen Mid und Order-Execution die TP/SL verfaelscht).
        # Bei use_predictor_targets=True keine Recompute (Bandit-Levels absolut).
        res = execute_order_with_failsafe(
            creds, symbol, is_long=(side == "long"),
            leverage=leverage, size_usd=size_usd,
            tp_px=tp_px, sl_px=sl_px,
            entry_px=entry_live, slippage_pct=slippage_pct,
            tp_pct=tp_pct, sl_pct=sl_pct,  # None wenn use_predictor_targets
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
                               peak_px, trough_px, status, features_at_open)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', %s::jsonb)
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
            log.error("AUTO-TRADE FAIL %s %s pid=%s lev=%dx: %s",
                      symbol, side, pid, eff_lev, res.get("error"))
            # Failsafe-Close: HL-Position weg, DB-Row sofort markieren damit
            # Watch-Pass nicht spaeter spurious schliesst.
            if res.get("failsafe_closed"):
                _mark_predict_closed(s, pid, 'auto_close_failsafe', entry_live)
                log.info("DB pid=%s -> auto_close_failsafe (HL-Position bereits zu)", pid)
            else:
                # Order failed (kein Failsafe) — HL hat keine Position -> Phantom verhindern
                _mark_predict_closed(s, pid, 'auto_trade_failed', entry_live)
                log.info("DB pid=%s -> auto_trade_failed (Order fehlgeschlagen, keine HL-Position)", pid)
    except Exception as e:
        # Aussere Exception (z.B. HL-API Rate-Limit 429 vor Order-Versuch).
        # HL hat keine Position -> DB-Row sofort schliessen damit Watch-Pass
        # sie nicht spaeter spurious schliesst (Phantom-Trade-Bug).
        log.exception("auto-trade %s scheiterte: %s", symbol, e)
        try:
            _mark_predict_closed(s, pid, 'auto_trade_failed', predicted_entry)
            log.info("DB pid=%s -> auto_trade_failed (Exception, Phantom verhindert)", pid)
        except Exception as e2:
            log.warning("DB-Phantom-Mark pid=%s fehlgeschlagen: %s", pid, e2)


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

def replay_action(klines_rows, entry, side, tp_pct, sl_pct, timeout_h,
                   time_penalty_per_h, reward_timeout_pen):
    """Simuliert einen hypothetischen Trade durch die echte 10s-Klines-Trajectory.

    klines_rows: list of dict mit open_time, high, low, close (sortiert ASC)
    side: 'long' / 'short' / None (skip — wird vom Caller separat berechnet)
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

    for r in klines_rows:
        h = r.get('high'); l = r.get('low'); c = r.get('close')
        if h is None or l is None: continue
        h = float(h); l = float(l)

        elapsed_h = (r['open_time'] - start_time).total_seconds() / 3600.0
        if elapsed_h >= timeout_h:
            # Echter Timeout: Trade haengt timeout_h ohne TP/SL-Hit -> Penalty.
            close_v = float(c) if c is not None else entry
            pnl = ((close_v - entry) / entry * 100) if side == 'long' else ((entry - close_v) / entry * 100)
            reward = pnl - elapsed_h * time_penalty_per_h + reward_timeout_pen
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
            reward = pnl - elapsed_h * time_penalty_per_h
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
    timeout_h = float(bandit_cfg.get("timeout_hours", 6))
    time_penalty = float(bandit_cfg.get("reward_time_penalty_per_hour", 0.15))
    reward_skip = float(bandit_cfg.get("reward_skip", 0.1))
    reward_timeout_pen = float(bandit_cfg.get("reward_timeout_penalty", -0.3))

    feat = prediction.get('features') or {}
    if not isinstance(feat, dict) or not feat:
        return 0, {}, None
    entry = float(prediction['entry_px'])

    # Volle Bewertungsspanne. min(now()) damit wir nicht in die Zukunft fragen,
    # wenn der Trade noch jung ist.
    full_range_end = prediction['created_at'] + timedelta(hours=timeout_h)
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
        replay = replay_action(
            klines_rows, entry,
            action['side'], action['tp_pct'] or 0.0, action['sl_pct'] or 0.0,
            timeout_h, time_penalty, reward_timeout_pen,
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
            cur.execute("""
                SELECT id, symbol, side, entry_px, sl_px, tp_px, features,
                       created_at, closed_at, rule_name, status, pnl_pct,
                       effective_leverage
                FROM open_predictions
                WHERE status IN ('win','loss','timeout')
                  AND closed_at IS NOT NULL AND features IS NOT NULL
                ORDER BY closed_at
            """)
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


# =============================================================================
# Watch-Pass v4 — TP/SL/Timeout-Check + Hindsight-Update fuer ALLE Actions
# =============================================================================

def watch_pass_v4(s, bandit, scaler, state_path):
    cfg = s["predictor"]
    bandit_cfg = cfg.get("bandit", {})
    timeout_h = float(bandit_cfg.get("timeout_hours", cfg.get("timeout_hours", 12)))
    reward_skip = float(bandit_cfg.get("reward_skip", 0.1))
    reward_timeout_pen = float(bandit_cfg.get("reward_timeout_penalty", -0.3))
    reward_time_penalty_h = float(bandit_cfg.get("reward_time_penalty_per_hour", 0.15))
    # Virtuelle Wallet — fuer Drawdown-Feature des Bandits
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
                    # Predictor-Hindsight nur fuer Bandit-Trades (keine resync_hl).
                    if o.get('source') != 'resync_hl':
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
                    log.info("CLOSE %s %s status=%s pnl=%.3f%% action=%s",
                             o['symbol'], side, status, pnl_pct, o['rule_name'])
                else:
                    cur_a.execute("""
                        UPDATE open_predictions
                        SET last_px=%s, last_check_at=now(), peak_px=%s, trough_px=%s
                        WHERE id=%s AND status='open'
                    """, (cur_px, peak_px, trough_px, o['id']))
            app.commit()

    if not closes: return 0

    # Hindsight-Replay: pro Close werden ALLE 31 Actions aus echter Klines-Trajectory bewertet.
    with db_learner(s) as ldb, db_app(s) as app, db_coins(s) as coins:
        with ldb.cursor() as cur_l:
            for c in closes:
                feat = c.get('features') or {}
                if not isinstance(feat, dict): feat = {}
                pred_dict = {
                    'symbol': c['symbol'], 'side': c['side'],
                    'entry_px': c['entry'], 'features': feat,
                    'created_at': c['created_at'],
                    'closed_at': datetime.now(timezone.utc),
                }
                try:
                    n, rewards_map, chosen_replay = hindsight_replay_for_prediction(
                        coins, bandit, scaler, pred_dict, s["predictor"])
                    if n > 0:
                        chosen_name = feat.get('_action_name', '?')
                        chosen_r = chosen_replay.get('reward') if chosen_replay else None
                        # Top-3 Actions nach reward — nur Actions mit gueltigem Reward,
                        # incomplete (reward=None) wird im Log uebersprungen.
                        rated = [(k, v) for k, v in rewards_map.items()
                                 if v.get('reward') is not None]
                        top = sorted(rated, key=lambda kv: -kv[1]['reward'])[:3]
                        log.info("HINDSIGHT %s chosen=%s r=%s | top3: %s | n_updates=%d",
                                 c['symbol'], chosen_name,
                                 f"{chosen_r:+.3f}" if chosen_r is not None else "?",
                                 ", ".join(f"{k}:{v['reward']:+.2f}" for k, v in top),
                                 n)
                except Exception as e:
                    log.warning("hindsight replay %s: %s", c['symbol'], e)

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

        # State-Counters
        with app.cursor() as cur_a:
            cur_a.execute("""
                UPDATE predictor_state SET closed_count = COALESCE(closed_count,0) + %s,
                       updated_at=now() WHERE id=1
            """, (len(closes),))
        app.commit()

    # Bandit-State persistieren
    save_state({'bandit': bandit, 'scaler': scaler, 'feature_keys': FEATURE_KEYS}, state_path)
    return len(closes)


# =============================================================================
# MODIFY-BANDIT (Phase 1) — passt TP/SL laufender Trades alle 30s an
# Eigene LinTSBandit-Instanz, eigene Tabelle (modify_decisions),
# eigenes State-File, eigene Reward-Welt. Open-Bandit unberuehrt.
# =============================================================================

_MODIFY_POSITION_KEYS = [
    'time_in_trade_h', 'time_remaining_h',  # 10.05.2026: time_remaining ergaenzt
    'pnl_now_pct', 'peak_pct_now', 'trough_pct_now',
    'dist_to_tp_pct', 'dist_to_sl_pct', 'modify_count', 'original_action_idx',
    'leverage', 'margin_pnl_pct',
]


def build_modify_feature_keys(s):
    """Modify-Bandit Features = Open-Bandit-Features + 10 Position-Features."""
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


def position_features(trader_pos_row, mark_px, current_tp_px, current_sl_px, modify_count, timeout_h=2.0):
    """Position-Features fuer Modify-Bandit (Trade-State + Hebel + time_remaining).
    Eingabe: Row aus trader_positions (NICHT open_predictions!) — saubere Trennung.
    timeout_h kommt aus settings.predictor.bandit.timeout_hours (Predictor-Hard-Close)."""
    entry = float(trader_pos_row['entry_px'])
    side = trader_pos_row['side']
    leverage = float(trader_pos_row.get('leverage') or 1)
    age_s = (datetime.now(timezone.utc) - trader_pos_row['opened_at']).total_seconds()
    time_in_h = age_s / 3600.0
    time_left_h = max(0.0, float(timeout_h) - time_in_h)
    if side == 'long':
        pnl_now = (mark_px - entry) / entry * 100.0
        peak = float(trader_pos_row.get('peak_px') or entry)
        trough = float(trader_pos_row.get('trough_px') or entry)
        peak_pct = (peak - entry) / entry * 100.0
        trough_pct = (trough - entry) / entry * 100.0
        dist_tp = (current_tp_px - mark_px) / mark_px * 100.0 if mark_px > 0 else 0.0
        dist_sl = (mark_px - current_sl_px) / mark_px * 100.0 if mark_px > 0 else 0.0
    else:
        pnl_now = (entry - mark_px) / entry * 100.0
        peak = float(trader_pos_row.get('peak_px') or entry)
        trough = float(trader_pos_row.get('trough_px') or entry)
        peak_pct = (entry - trough) / entry * 100.0
        trough_pct = (entry - peak) / entry * 100.0
        dist_tp = (mark_px - current_tp_px) / mark_px * 100.0 if mark_px > 0 else 0.0
        dist_sl = (current_sl_px - mark_px) / mark_px * 100.0 if mark_px > 0 else 0.0
    feat_orig = trader_pos_row.get('features_at_open') or {}
    if not isinstance(feat_orig, dict): feat_orig = {}
    orig_idx = feat_orig.get('_action_idx', -1)
    try: orig_idx = float(orig_idx)
    except (TypeError, ValueError): orig_idx = -1.0
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
        'modify_count': float(modify_count),
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

    timeout_h = float(cfg.get("bandit", {}).get("timeout_hours", 2.0))
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
                       peak_px, trough_px, opened_at, features_at_open, modify_count
                FROM trader_positions
                WHERE status='open'
                  AND opened_at <= now() - (%s || ' seconds')::interval
            """, (min_age,))
            opens = cur.fetchall()
        if not opens:
            return 0

        btc_moves = load_btc_moves(coins)
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
            mod_count = int(o.get('modify_count') or 0)

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
            pos_feat = position_features(o, mark_px, current_tp, current_sl, mod_count, timeout_h=timeout_h)
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


def compute_efficiency_reward(captured_margin, max_margin_in_box, cfg=None):
    """Efficiency-Reward (Reform 10.05.2026).

    captured_margin    = realisierter PnL der Action × Hebel (Margin-PnL in %)
    max_margin_in_box  = max(margin_pnl) im Fenster [decision .. min(close, open+timeout)]

    Logik:
    - Verlust  -> reward = captured * loss_scale  (negativer wird linear bestraft)
    - max <= 0 -> reward = 0  (Trade konnte nicht profitabel werden, neutral)
    - sonst    -> efficiency = captured/max in [0..1], reward = (eff-0.5)*2*profit_scale

    Der Bandit bekommt damit:
    - Nahe Maximum erwischt = +profit_scale
    - Halbwegs erwischt    = ~0
    - Wenig vom Potential  = -profit_scale
    - Verlust              = captured * loss_scale (negativ)
    """
    if cfg is None: cfg = {}
    profit_scale = float(cfg.get("profit_scale", 2.0))
    loss_scale = float(cfg.get("loss_scale", 1.5))
    if captured_margin < 0:
        return float(captured_margin) * loss_scale
    if max_margin_in_box <= 0:
        return 0.0
    efficiency = float(captured_margin) / float(max_margin_in_box)
    efficiency = max(0.0, min(1.0, efficiency))
    return (efficiency - 0.5) * 2.0 * profit_scale


def hindsight_replay_modify_neural(coins_conn, app_conn, modify_bandit, modify_scaler,
                                     prediction, lookahead_minutes=120,
                                     reward_cfg=None, **_legacy_kwargs):
    """Trader-Hindsight (Reform 10.05.2026) — Efficiency-Reward, 2 Actions.

    Pro Decision werden BEIDE Actions (hold, close_now) counterfactual bewertet:
      max_box      = max(margin_pnl) im Fenster [decided_at .. min(closed_at, open+timeout)]
      captured_hold  = margin_pnl am Fenster-Ende (was wuerde Halten bringen?)
      captured_close = margin_pnl am decided_at (was wuerde sofort schliessen bringen?)
      reward = compute_efficiency_reward(captured, max_box, reward_cfg)

    Predictor-DB (open_predictions) wird NICHT angefasst (Welt-Trennung).
    LinTSBandit.update(i, x, r) — kein Replay-Buffer, direkt-online.

    legacy_kwargs (penalty_cfg, sweet_cfg) werden ignoriert — nur fuer Aufruf-Kompatibilitaet."""
    pid = prediction['id']
    side = prediction['side']
    entry = float(prediction['entry_px'])
    closed_at = prediction['closed_at']

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

    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur_c:
        cur_c.execute("""
            SELECT open_time, high, low, close FROM klines
            WHERE symbol=%s AND interval='10s' AND open_time BETWEEN %s AND %s
            ORDER BY open_time
        """, (sym, decisions[0]['decided_at'], end_at))
        klines = cur_c.fetchall()
    if not klines:
        return 0

    # Action-Index-Lookup (robust gegen Reihenfolgen-Aenderungen)
    name_to_idx = {a['name']: i for i, a in enumerate(modify_bandit.actions)}
    idx_hold = name_to_idx.get('hold')
    idx_close = name_to_idx.get('close_now')
    if idx_hold is None or idx_close is None:
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

        # Margin-PnL der einzelnen Klines im Fenster (high+low fuer max_box)
        max_pnl_pct = -1e9
        for k in klines_after:
            h = k.get('high'); l = k.get('low')
            if h is None or l is None: continue
            h = float(h); l = float(l)
            best = h if side == 'long' else l
            if side == 'long':
                pnl_pct = (best - entry) / entry * 100.0
            else:
                pnl_pct = (entry - best) / entry * 100.0
            if pnl_pct > max_pnl_pct: max_pnl_pct = pnl_pct
        if max_pnl_pct == -1e9:
            continue
        max_margin = max_pnl_pct * leverage

        # captured_close = jetzt schliessen zum first_close
        pnl_close_pct = ((first_close - entry)/entry*100) if side == 'long' else ((entry - first_close)/entry*100)
        captured_close = pnl_close_pct * leverage

        # captured_hold = warten bis Fenster-Ende, exit zum last_close
        pnl_hold_pct = ((last_close - entry)/entry*100) if side == 'long' else ((entry - last_close)/entry*100)
        captured_hold = pnl_hold_pct * leverage

        r_close = compute_efficiency_reward(captured_close, max_margin, reward_cfg)
        r_hold = compute_efficiency_reward(captured_hold, max_margin, reward_cfg)
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

    n_failsafe = 0; n_repaired = 0; n_orphan_cancelled = 0
    with db_app(s) as app:
        # Richtung A: trader_positions.open ohne HL-Pos -> failsafe
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, symbol, side, entry_px, current_tp_px, current_sl_px
                FROM trader_positions WHERE status='open'
            """)
            tp_rows = cur.fetchall()
        for tr in tp_rows:
            coin = tr['symbol']
            if coin in hl_pos_by_coin: continue  # HL-Pos da, alles OK
            with app.cursor() as cur2:
                cur2.execute("""
                    UPDATE trader_positions
                    SET status='failsafe', closed_at=now(),
                        exit_px=entry_px, pnl_pct=0
                    WHERE id=%s AND status='open'
                """, (tr['id'],))
                if cur2.rowcount > 0:
                    n_failsafe += 1
                    log.info("TRADER-FAILSAFE %s trader_pid=%s (HL-Pos weg, status=failsafe)",
                             coin, tr['id'])
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
    lookahead_min = float(s["predictor"]["modify_bandit"].get("hindsight_delay_minutes", 120))

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
                    'modify_count': 0.0, 'original_action_idx': orig_idx,
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
                           tp.symbol, tp.side, tp.entry_px
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
                    'closed_at': d['closed_at'],
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

    bandit_cfg = cfg.get("bandit", {})
    state_path = bandit_cfg.get("state_path", "/opt/coin/database/data/models/predictor_v4_bandit.pkl")
    tp_buckets = bandit_cfg.get("tp_buckets_pct", [1.0, 2.0, 3.0, 5.0, 8.0])
    sl_buckets = bandit_cfg.get("sl_buckets_pct", [1.0, 2.0, 3.0])
    alpha = float(bandit_cfg.get("prior_alpha", 1.0))
    sigma2 = float(bandit_cfg.get("noise_sigma2", 1.0))
    seed = int(bandit_cfg.get("seed", 42))

    actions = build_actions(tp_buckets, sl_buckets)
    action_names = [a['name'] for a in actions]

    # Bandit + Scaler laden oder frisch erstellen.
    # Fresh-Start triggert wenn: kein State, Action-Names veraendert, oder
    # Feature-Keys (z.B. neue/umbenannte Regel) veraendert.
    state = load_state(state_path)
    fresh_reason = None
    if state is None:
        fresh_reason = "no state file"
    else:
        st_action_names = [a['name'] for a in state.get('bandit').actions]
        st_feature_keys = state.get('feature_keys', [])
        if st_action_names != action_names:
            fresh_reason = f"action-space changed ({len(st_action_names)} -> {len(action_names)} or names differ)"
        elif st_feature_keys != FEATURE_KEYS:
            fresh_reason = f"feature-keys changed ({len(st_feature_keys)} -> {len(FEATURE_KEYS)} or names differ)"

    if fresh_reason:
        if state is not None:
            log.warning("Bandit fresh start — %s", fresh_reason)
        bandit = LinTSBandit(N_FEAT, actions, alpha=alpha, sigma2=sigma2)
        scaler = OnlineScaler(N_FEAT)
        save_state({'bandit': bandit, 'scaler': scaler, 'feature_keys': FEATURE_KEYS}, state_path)
        log.info("Bandit fresh start: %d actions, %d features", len(actions), N_FEAT)
        # Bei Fresh-Start: Backfill-Flag-File loeschen damit Backfill automatisch
        # laeuft (sonst startet Bandit komplett cold mit den ersten Trades).
        backfill_flag_pre = state_path + '.backfill_done'
        if os.path.exists(backfill_flag_pre):
            try:
                os.remove(backfill_flag_pre)
                log.info("Backfill-Flag-File geloescht — Backfill laeuft beim Start neu durch")
            except Exception as e:
                log.warning("Backfill-Flag-File loeschen scheiterte: %s", e)
    else:
        bandit = state['bandit']; scaler = state['scaler']
        log.info("Bandit loaded: %d actions, %d features, %d total observations",
                 len(actions), N_FEAT, sum(bandit.n_obs))

    rng = np.random.default_rng(seed)
    log.info("Predictor %s start. tp_buckets=%s sl_buckets=%s scan=%ss watch=%ss top_n=%d",
             ACTIVE_VERSION, tp_buckets, sl_buckets,
             cfg["scan_interval_seconds"], cfg["watch_interval_seconds"], cfg["universe_top_n"])

    # ============= Trader-Setup (LinTSBandit, 2 Actions, eigene Welt) =============
    # Reform 10.05.2026: NeuralBandit -> LinTSBandit, 102 Actions -> 2 Actions
    mb_cfg = cfg.get("modify_bandit", {})
    modify_bandit = None
    modify_scaler = None
    modify_state_path = None
    modify_interval = float(mb_cfg.get("interval_seconds", 30))
    hindsight_delay_min = float(mb_cfg.get("hindsight_delay_minutes", 120))
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

    # Hindsight-Backfill (einmalig). Marker ist eine flag-Datei neben dem state-file
    # — damit modifiziert der Service NICHT settings.json. Re-run = flag-File loeschen.
    backfill_flag = state_path + '.backfill_done'
    if bandit_cfg.get("hindsight_backfill_on_start", False):
        if os.path.exists(backfill_flag):
            log.info("Backfill-Flag in settings ist true, aber %s existiert -> skip. "
                     "Zum erneuten Ausfuehren Flag-File loeschen.", backfill_flag)
        else:
            try:
                n_preds, n_updates = backfill_hindsight(s, bandit, scaler)
                save_state({'bandit': bandit, 'scaler': scaler, 'feature_keys': FEATURE_KEYS}, state_path)
                Path(backfill_flag).touch()
                log.info("Hindsight-Backfill erledigt. Marker: %s", backfill_flag)
            except Exception as e:
                log.exception("backfill_hindsight scheiterte: %s", e)

    last_scan = 0.0; last_watch = 0.0; last_settings_reload = 0.0; last_modify = 0.0; last_modify_hindsight_id = 0; last_hl_sync = 0.0

    while True:
        try:
            now = time.time()
            if now - last_settings_reload >= 60:
                s = load_settings(); cfg = s["predictor"]
                last_settings_reload = now

                # Live-Bucket-Reload: wenn settings.json tp/sl_buckets geaendert hat
                # → Bandit Fresh-Start + Hindsight-Backfill (ohne Service-Restart).
                bcfg = cfg.get("bandit", {})
                new_tp = bcfg.get("tp_buckets_pct")
                new_sl = bcfg.get("sl_buckets_pct")
                if new_tp and new_sl:
                    new_actions = build_actions(new_tp, new_sl)
                    new_names = [a["name"] for a in new_actions]
                    cur_names = [a["name"] for a in bandit.actions]
                    if new_names != cur_names:
                        log.warning("Live Bucket-Aenderung erkannt: %d -> %d Actions, Fresh-Start + Backfill",
                                    len(cur_names), len(new_names))
                        alpha = float(bcfg.get("prior_alpha", 1.0))
                        sigma2 = float(bcfg.get("noise_sigma2", 1.0))
                        bandit = LinTSBandit(N_FEAT, new_actions, alpha=alpha, sigma2=sigma2)
                        scaler = OnlineScaler(N_FEAT)
                        save_state({'bandit': bandit, 'scaler': scaler,
                                    'feature_keys': FEATURE_KEYS}, state_path)
                        try:
                            n_p, n_u = backfill_hindsight(s, bandit, scaler)
                            save_state({'bandit': bandit, 'scaler': scaler,
                                        'feature_keys': FEATURE_KEYS}, state_path)
                            log.info("Live-Backfill nach Bucket-Aenderung: %d Predictions, %d Action-Updates",
                                     n_p, n_u)
                        except Exception as e:
                            log.exception("Live-Backfill scheiterte: %s", e)

            # HL-Sync: alle 60s pruefen ob HL-Phantom-Positionen ohne DB-Row sind
            if now - last_hl_sync >= 60:
                try:
                    n_sync = sync_hl_to_db(s)
                    if n_sync > 0:
                        log.info("hl-sync: %d Phantom-Positionen reaktiviert", n_sync)
                except Exception as e:
                    log.exception("hl-sync failed: %s", e)
                last_hl_sync = now

            if now - last_scan >= cfg["scan_interval_seconds"]:
                m = scan_pass_v4(s, bandit, scaler, rng)
                last_scan = now
                if m > 0:
                    log.info("scan-pass: %d new predictions", m)

            if now - last_watch >= cfg["watch_interval_seconds"]:
                c = watch_pass_v4(s, bandit, scaler, state_path)
                last_watch = now
                if c > 0:
                    log.info("watch-pass: %d closed", c)

            # Trader-Tick alle interval_seconds (live TP/SL-Modifikation)
            if modify_bandit is not None and now - last_modify >= modify_interval:
                try:
                    modify_pass(s, modify_bandit, modify_scaler, rng)
                except Exception as e:
                    log.exception("modify_pass failed: %s", e)
                last_modify = now

                # Hindsight-Queue: frisch geschlossene Trades fuer +120min-Hindsight einreihen
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
