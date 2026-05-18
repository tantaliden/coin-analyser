"""Predictor Service v2 — KI-getrieben (River-Modell entscheidet).

Settings: settings.json -> 'predictor'.
Tabellen:
  analyser_app.open_predictions   - offene+abgeschlossene Predictions
  analyser_app.predictor_state    - Threshold + Universe + Counts
  learner.prediction_feedback     - Closed Outcomes mit Feature-Snapshot

Zwei River-Modelle:
  long_model  - P(long-Trade profitabel)
  short_model - P(short-Trade profitabel)

Decision pro Coin pro Scan:
  rules-mode:  Regeln triggern (alt)
  ml-mode:     max(P_long, P_short) > ml_threshold -> Prediction
  blend-mode:  beide laufen, Modell wird durch Cold-Start-Regeln ergaenzt
"""

import json
import logging
import math
import os
import pickle
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from river import tree, ensemble, linear_model, optim, preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("predictor")

SETTINGS_PATH = '/opt/coin/settings.json'


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
# Operatoren
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
    """Spiegel zu rising_pct: change <= -threshold_pct (also Fall um mindestens X%)."""
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
    "rising_pct": op_rising_pct,
    "falling_pct": op_falling_pct,
    "abs_change_pct_max": op_abs_change_pct_max,
    "z_score": op_z_score,
    "bollinger_pos": op_bollinger_pos,
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
            details.append({"cond": cond, "ok": ok, "value": last})
        elif op == "bollinger_pos":
            period = cond.get("window_min", 20)
            ok = op_bollinger_pos(series, val, period)
            details.append({"cond": cond, "ok": ok})
        else:
            ok = fn(series_window, val)
            details.append({"cond": cond, "ok": ok, "n": len(series_window)})
        if not ok:
            return {"ok": False, "details": details}
    return {"ok": True, "details": details}


# =============================================================================
# ATR
# =============================================================================

def calc_atr(coins_conn, symbol, period, source_view):
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT high, low, close FROM {source_view}
            WHERE symbol=%s ORDER BY bucket DESC LIMIT %s
        """, (symbol, period + 1))
        rows = list(reversed(cur.fetchall()))
    if len(rows) < period + 1: return None
    trs = []
    prev_close = f(rows[0]['close'])
    for r in rows[1:]:
        h, l, c = f(r['high']), f(r['low']), f(r['close'])
        if h is None or l is None or c is None or prev_close is None:
            prev_close = c
            continue
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if not trs: return None
    return sum(trs) / len(trs)


# =============================================================================
# Helper fuer technische Indikatoren (RSI, MACD, EMA, CVD, BTC-Cross)
# =============================================================================

def _ema(values, period):
    """Standard-EMA, returns None bei zu kurzer Reihe."""
    clean = [v for v in values if v is not None]
    if len(clean) < period:
        return None
    k = 2.0 / (period + 1)
    e = clean[0]
    for v in clean[1:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes, period=14):
    """Wilder-RSI. Returns 50 (neutral) bei zu kurzer Reihe."""
    clean = [c for c in closes if c is not None]
    if len(clean) < period + 1:
        return 50.0
    gains = []; losses = []
    for i in range(1, len(clean)):
        diff = clean[i] - clean[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if len(gains) < period:
        return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes, fast=12, slow=26):
    """MACD-Line (EMA-fast minus EMA-slow). Signal-Line vereinfacht."""
    clean = [c for c in closes if c is not None]
    if len(clean) < slow:
        return 0.0
    ef = _ema(clean, fast)
    es = _ema(clean, slow)
    if ef is None or es is None:
        return 0.0
    return ef - es


def _cvd_horizon(rows_window):
    """Cumulative Volume Delta = sum(taker_buy - taker_sell) ueber Fenster."""
    cvd = 0.0
    for r in rows_window:
        v = f(r['volume']) or 0.0
        tb = f(r['taker_buy_base']) or 0.0
        ts = v - tb
        cvd += (tb - ts)
    return cvd


def _cvd_ratio(rows_window):
    """CVD normalisiert auf Gesamtvolumen, [-1, 1]."""
    total_v = sum((f(r['volume']) or 0.0) for r in rows_window)
    if total_v <= 0:
        return 0.0
    return _cvd_horizon(rows_window) / total_v


def load_btc_moves(coins_conn):
    """BTC-Returns ueber Multi-Horizon, einmal pro scan_pass laden."""
    out = {'5m': 0.0, '15m': 0.0, '60m': 0.0}
    try:
        with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT bucket, close FROM agg_1m WHERE symbol='BTC'
                ORDER BY bucket DESC LIMIT 65
            """)
            rows = list(cur.fetchall())
        if len(rows) < 6:
            return out
        cl_now = f(rows[0]['close'])
        if not cl_now:
            return out
        for h in (5, 15, 60):
            if len(rows) > h:
                c0 = f(rows[h]['close'])
                if c0 and c0 > 0:
                    out[f'{h}m'] = (cl_now - c0) / c0 * 100.0
    except Exception as e:
        log.warning("load_btc_moves failed: %s", e)
    return out


# =============================================================================
# Feature-Snapshot v2 — alle verfuegbaren Daten
# =============================================================================

def feature_snapshot_v2(coins_conn, symbol, rule_flags=None, btc_moves=None):
    """Erweiterte Features. ~30 Werte aus klines/agg_*/hl_asset_ctx + Rule-Flags."""
    feat = {}
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1m-Reihe ueber 4h (240 Buckets) als Basis
        cur.execute("""
            SELECT bucket, open, high, low, close, volume, trades,
                   taker_buy_base, taker_buy_quote,
                   funding, open_interest, premium,
                   spread_bps, book_imbalance_5, book_depth_5,
                   mark_px, mid_px, oracle_px,
                   bbo_bid_sz, bbo_ask_sz
            FROM agg_1m WHERE symbol=%s
            ORDER BY bucket DESC LIMIT 240
        """, (symbol,))
        rows = list(reversed(cur.fetchall()))
        if not rows or len(rows) < 16: return None
        last = rows[-1]
        if last['close'] is None: return None
        cl = f(last['close'])

        # === Snapshot Direkt-Werte ===
        for k in ("funding","open_interest","premium","spread_bps",
                  "book_imbalance_5","book_depth_5"):
            v = f(last[k])
            feat[k] = v if v is not None else 0.0

        bidsz = f(last['bbo_bid_sz']) or 0
        asksz = f(last['bbo_ask_sz']) or 0
        feat['bbo_size_ratio'] = (bidsz - asksz) / (bidsz + asksz + 1e-9)

        mark = f(last['mark_px']); mid = f(last['mid_px']); oracle = f(last['oracle_px'])
        feat['mark_vs_mid_bps'] = ((mark - mid) / mid * 10000) if (mark and mid) else 0.0
        feat['mark_vs_oracle_bps'] = ((mark - oracle) / oracle * 10000) if (mark and oracle) else 0.0

        # === Multi-Horizon Returns ===
        def close_at(ago_min):
            idx = -1 - ago_min
            if abs(idx) > len(rows): return None
            v = f(rows[idx]['close'])
            return v if v else None

        for h in (1, 5, 15, 30, 60, 240):
            c0 = close_at(h)
            feat[f'close_pct_{h}m'] = ((cl - c0) / c0 * 100) if c0 else 0.0

        # === Volumen / Taker-Ratio ===
        vols_15m = [f(r['volume']) or 0 for r in rows[-15:]]
        vols_1h = [f(r['volume']) or 0 for r in rows[-60:]] if len(rows) >= 60 else vols_15m
        feat['vol_15m_sum'] = sum(vols_15m)
        feat['vol_spike_ratio'] = (sum(vols_15m) / max(sum(vols_1h) / 4, 1e-9)) if sum(vols_1h) else 1.0
        # vol z-score 1h
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

        # 15m taker-Ratio
        tb_15 = sum(f(r['taker_buy_base']) or 0 for r in rows[-15:])
        v_15 = sum(f(r['volume']) or 0 for r in rows[-15:])
        feat['taker_buy_ratio_15m'] = (tb_15 / v_15) if v_15 > 0 else 0.5

        # === Trends/Deltas ===
        def at(ago_min, key):
            idx = -1 - ago_min
            if abs(idx) > len(rows): return None
            return f(rows[idx][key])

        # OI delta
        for h in (5, 15, 60):
            oi_then = at(h, 'open_interest') or 0
            oi_now = feat['open_interest']
            feat[f'oi_pct_{h}m'] = ((oi_now - oi_then) / oi_then * 100) if oi_then else 0.0

        # Funding delta 30m
        f30 = at(30, 'funding')
        feat['funding_delta_30m'] = (feat['funding'] - f30) if f30 is not None else 0.0

        # Spread delta 5m
        sp5 = at(5, 'spread_bps')
        feat['spread_delta_5m'] = (feat['spread_bps'] - sp5) if sp5 is not None else 0.0

        # Premium delta 5m
        pr5 = at(5, 'premium')
        feat['premium_delta_5m'] = (feat['premium'] - pr5) if pr5 is not None else 0.0

        # === Liquidation-Proxies (Heuristik aus OI-Drop x Preis-Move) ===
        # HL hat keinen public Liquidations-Stream. Stattdessen:
        #   OI faellt + Preis faellt  -> wahrscheinl. Long-Liquidations
        #   OI faellt + Preis steigt  -> wahrscheinl. Short-Liquidations
        oi_n = feat.get('open_interest') or 0.0

        def _oi_drop_pct(ago_min):
            oi_then = at(ago_min, 'open_interest')
            if not oi_then or oi_then <= 0 or oi_n is None:
                return 0.0
            return (oi_then - oi_n) / oi_then * 100.0

        oi_drop_1m = max(0.0, _oi_drop_pct(1))
        oi_drop_5m = max(0.0, _oi_drop_pct(5))
        oi_drop_15m = max(0.0, _oi_drop_pct(15))
        feat['oi_drop_1m'] = oi_drop_1m
        feat['oi_drop_5m'] = oi_drop_5m
        feat['oi_drop_15m'] = oi_drop_15m

        # Preis-Moves haben wir noch nicht — erst nach close_at-Schleife. Fallback ueber lokale close-Reihe:
        cl_1m_ago = at(1, 'close')
        cl_5m_ago = at(5, 'close')
        move_1m_pct = ((cl - cl_1m_ago) / cl_1m_ago * 100.0) if (cl_1m_ago and cl_1m_ago > 0) else 0.0
        move_5m_pct = ((cl - cl_5m_ago) / cl_5m_ago * 100.0) if (cl_5m_ago and cl_5m_ago > 0) else 0.0

        feat['liq_long_proxy_1m']  = oi_drop_1m * max(0.0, -move_1m_pct)
        feat['liq_long_proxy_5m']  = oi_drop_5m * max(0.0, -move_5m_pct)
        feat['liq_short_proxy_1m'] = oi_drop_1m * max(0.0, move_1m_pct)
        feat['liq_short_proxy_5m'] = oi_drop_5m * max(0.0, move_5m_pct)

        # === Volatilitaet / ATR ===
        def atr_local(window_rows):
            trs = []; prev_c = None
            for r in window_rows:
                h, l, c = f(r['high']), f(r['low']), f(r['close'])
                if h is None or l is None or c is None: continue
                if prev_c is None:
                    prev_c = c; continue
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                trs.append(tr)
                prev_c = c
            if not trs or not cl: return 0.0
            return (sum(trs) / len(trs)) / cl * 100  # in % vom aktuellen Preis

        feat['atr_pct_15m'] = atr_local(rows[-15:])
        feat['atr_pct_1h'] = atr_local(rows[-60:]) if len(rows) >= 60 else feat['atr_pct_15m']
        feat['atr_pct_4h'] = atr_local(rows[-240:]) if len(rows) >= 240 else feat['atr_pct_1h']

        # === Asset-Ctx (latest) ===
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

        # === L2-Tiefe Level 10/20 (aus klines_10s direkt - agg_1m hat nur _5) ===
        cur.execute("""
            SELECT book_imbalance_10, book_depth_10, book_imbalance_20, book_depth_20
            FROM klines WHERE symbol=%s AND interval='10s'
            ORDER BY open_time DESC LIMIT 1
        """, (symbol,))
        l2ext = cur.fetchone()
        if l2ext:
            for k in ('book_imbalance_10', 'book_depth_10', 'book_imbalance_20', 'book_depth_20'):
                v = f(l2ext[k])
                feat[k] = v if v is not None else 0.0
        else:
            feat['book_imbalance_10'] = 0.0
            feat['book_depth_10'] = 0.0
            feat['book_imbalance_20'] = 0.0
            feat['book_depth_20'] = 0.0

        # === CVD (Cumulative Volume Delta) ===
        feat['cvd_5m'] = _cvd_horizon(rows[-5:])
        feat['cvd_15m'] = _cvd_horizon(rows[-15:])
        feat['cvd_60m'] = _cvd_horizon(rows[-60:]) if len(rows) >= 60 else feat['cvd_15m']
        feat['cvd_ratio_15m'] = _cvd_ratio(rows[-15:])
        feat['cvd_ratio_60m'] = _cvd_ratio(rows[-60:]) if len(rows) >= 60 else feat['cvd_ratio_15m']

        # === RSI / MACD aus close-Reihe (1m, 5m via Sampling) ===
        closes_1m = [f(r['close']) for r in rows]
        feat['rsi_14_1m'] = _rsi(closes_1m, 14)
        # 5m-Sampling: jeden 5. Wert (vom Ende rückwärts: -1, -6, -11, ...)
        closes_5m = closes_1m[::-5][::-1] if len(closes_1m) >= 70 else closes_1m
        feat['rsi_14_5m'] = _rsi(closes_5m, 14)
        feat['macd_line'] = _macd(closes_1m, 12, 26)
        feat['macd_pct'] = (feat['macd_line'] / cl * 100.0) if cl else 0.0

        # === Funding-Z-Score und Streak (24h via agg_1h) ===
        cur.execute("""
            SELECT bucket, funding FROM agg_1h WHERE symbol=%s
            ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        f24 = [f(r['funding']) for r in cur.fetchall()]
        clean_f = [x for x in f24 if x is not None]
        if len(clean_f) >= 6:
            f_mean = sum(clean_f) / len(clean_f)
            f_var = sum((x - f_mean) ** 2 for x in clean_f) / len(clean_f)
            f_std = math.sqrt(f_var)
            cur_f = clean_f[0]  # newest first wegen DESC
            feat['funding_zscore_24h'] = ((cur_f - f_mean) / f_std) if f_std > 0 else 0.0
        else:
            feat['funding_zscore_24h'] = 0.0
        # Streak: wie viele Stunden in Folge gleiches Vorzeichen (newest first)
        streak = 0; sign = None
        for v in f24:
            if v is None:
                continue
            s = 1 if v > 0 else (-1 if v < 0 else 0)
            if sign is None:
                sign = s; streak = 1
            elif s == sign:
                streak += 1
            else:
                break
        feat['funding_streak_signed'] = streak * (sign or 0)

        # === Realized Volatility multi-horizon (24h via agg_1h) ===
        cur.execute("""
            SELECT bucket, high, low, close FROM agg_1h WHERE symbol=%s
            ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        hl_rows = cur.fetchall()
        ranges = []
        for r in hl_rows:
            h_, l_, c_ = f(r['high']), f(r['low']), f(r['close'])
            if h_ and l_ and c_ and c_ > 0:
                ranges.append((h_ - l_) / c_ * 100.0)
        if len(ranges) >= 4:
            last_1h = ranges[0]
            avg_24h = sum(ranges) / len(ranges)
            feat['vol_1h_vs_24h_ratio'] = (last_1h / avg_24h) if avg_24h > 0 else 1.0
            # Z-Score der aktuellen 1h-Range gegen 24h-Verteilung
            r_mean = avg_24h
            r_var = sum((x - r_mean) ** 2 for x in ranges) / len(ranges)
            r_std = math.sqrt(r_var)
            feat['vol_zscore_24h'] = ((last_1h - r_mean) / r_std) if r_std > 0 else 0.0
        else:
            feat['vol_1h_vs_24h_ratio'] = 1.0
            feat['vol_zscore_24h'] = 0.0

    # === BTC-Cross-Move (vom scan_pass uebergeben) ===
    if btc_moves is not None and symbol != 'BTC':
        feat['btc_close_pct_5m'] = btc_moves.get('5m', 0.0)
        feat['btc_close_pct_15m'] = btc_moves.get('15m', 0.0)
        feat['btc_close_pct_60m'] = btc_moves.get('60m', 0.0)
        feat['rel_strength_5m'] = feat.get('close_pct_5m', 0.0) - btc_moves.get('5m', 0.0)
        feat['rel_strength_15m'] = feat.get('close_pct_15m', 0.0) - btc_moves.get('15m', 0.0)
        feat['rel_strength_60m'] = feat.get('close_pct_60m', 0.0) - btc_moves.get('60m', 0.0)
    else:
        feat['btc_close_pct_5m'] = 0.0
        feat['btc_close_pct_15m'] = 0.0
        feat['btc_close_pct_60m'] = 0.0
        feat['rel_strength_5m'] = 0.0
        feat['rel_strength_15m'] = 0.0
        feat['rel_strength_60m'] = 0.0

    # === Tageszeit (cyclical) ===
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0
    feat['hour_sin'] = math.sin(2 * math.pi * hour / 24)
    feat['hour_cos'] = math.cos(2 * math.pi * hour / 24)
    feat['weekday'] = float(now_utc.weekday())

    # === Rule-Flags als Features (kein direkter Decision-Pfad mehr) ===
    if rule_flags:
        for name, val in rule_flags.items():
            feat[f'rule_{name}'] = float(1 if val else 0)

    return feat


# =============================================================================
# River Models — Long + Short
# =============================================================================

def make_clf():
    """Online Random Forest: ADWIN-Bagging ueber 10 Hoeffding-Adaptive-Trees mit Drift-Detector."""
    return ensemble.ADWINBaggingClassifier(
        model=tree.HoeffdingAdaptiveTreeClassifier(grace_period=50, seed=42),
        n_models=10, seed=42,
    )


def make_reg():
    """Online Hoeffding-Adaptive-Tree-Regressor fuer peak/trough-Bewegung."""
    return tree.HoeffdingAdaptiveTreeRegressor(grace_period=50, seed=42)


def load_model(path, factory):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as fp:
                return pickle.load(fp)
        except Exception as e:
            log.warning("load_model %s failed: %s -> fresh", path, e)
    return factory()


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fp:
        pickle.dump(model, fp)
    os.replace(tmp, path)


def predict_proba(model, features):
    """Returns P(class=1). Robust gegen verschiedene River-Varianten."""
    if model is None or not features: return 0.5
    try:
        p = model.predict_proba_one(features)
        if not p: return 0.5
        return float(p.get(True, p.get(1, 0.5)))
    except Exception:
        return 0.5


def predict_value(model, features, default=0.0):
    """Returns regression value. Default fuer unbekannte/leere features."""
    if model is None or not features: return default
    try:
        v = model.predict_one(features)
        return float(v) if v is not None else default
    except Exception:
        return default


# =============================================================================
# Bootstrap aus prediction_feedback
# =============================================================================

def bootstrap_from_feedback(s, long_path, short_path, up_path, down_path):
    """Replay aller closed predictions: trainiere 4 Modelle (Klassifikatoren + Peak/Trough-Regressoren).
    Peak/Trough werden aus klines berechnet (echte historische Bewegung)."""
    long_m = make_clf(); short_m = make_clf()
    up_m = make_reg(); down_m = make_reg()
    n_long = n_short = 0
    n_with_peak = n_without_peak = 0
    with db_learner(s) as ldb, db_coins(s) as coins:
        with ldb.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT direction, was_correct, status, features, symbol,
                       detected_at, resolved_at, entry_price, peak_pct, trough_pct
                FROM prediction_feedback
                WHERE scanner_type='predictor' AND features IS NOT NULL
                ORDER BY resolved_at
            """)
            rows = cur.fetchall()

        for r in rows:
            feat = r['features']
            if not feat or not isinstance(feat, dict): continue
            feat = {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in feat.items()}
            won = bool(r['was_correct'])

            # peak/trough mit Lookahead-Fenster: groesseres von resolved_at oder detected_at + lookahead_h.
            # Damit lernen die Regressoren echte "potential moves" auch wenn die alte Prediction
            # nach 30s wegen Mini-SL geschlossen wurde.
            lookahead_h = float(s["predictor"].get("bootstrap_lookahead_hours", 1.0))
            try:
                with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
                    cur_c.execute("""
                        SELECT MAX(high) AS mh, MIN(low) AS ml
                        FROM klines
                        WHERE symbol=%s AND interval='10s'
                          AND open_time BETWEEN %s
                            AND LEAST(now(), GREATEST(%s, %s + (%s || ' hours')::interval))
                    """, (r['symbol'], r['detected_at'], r['resolved_at'],
                           r['detected_at'], lookahead_h))
                    bnd = cur_c.fetchone()
                peak_pct_v = trough_pct_v = None
                if bnd and bnd['mh'] and bnd['ml'] and r['entry_price']:
                    entry = float(r['entry_price'])
                    peak_pct_v = (float(bnd['mh']) - entry) / entry * 100.0
                    trough_pct_v = (entry - float(bnd['ml'])) / entry * 100.0
                    n_with_peak += 1
            except Exception as e:
                log.warning("bootstrap klines lookup %s failed: %s", r['symbol'], e)
                peak_pct_v = trough_pct_v = None

            if peak_pct_v is None or trough_pct_v is None:
                n_without_peak += 1
            else:
                up_m.learn_one(feat, max(0.0, abs(peak_pct_v)))
                down_m.learn_one(feat, max(0.0, abs(trough_pct_v)))

            # Klassifikatoren — beide Modelle pro Close (long-win = short-loss-hint und umgekehrt)
            if r['direction'] == 'long':
                long_m.learn_one(feat, 1 if won else 0)
                short_m.learn_one(feat, 0 if won else 1)
                n_long += 1
            else:
                short_m.learn_one(feat, 1 if won else 0)
                long_m.learn_one(feat, 0 if won else 1)
                n_short += 1

    save_model(long_m, long_path); save_model(short_m, short_path)
    save_model(up_m, up_path); save_model(down_m, down_path)
    log.info("bootstrap done: clf long=%d short=%d, regr trained=%d skipped=%d",
             n_long, n_short, n_with_peak, n_without_peak)
    return long_m, short_m, up_m, down_m, n_long + n_short


# =============================================================================
# DB-Helpers
# =============================================================================

def has_open(app_conn, symbol):
    with app_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM open_predictions WHERE symbol=%s AND status='open' LIMIT 1", (symbol,))
        return cur.fetchone() is not None


def open_prediction(app_conn, symbol, side, entry, sl, tp, score, rule, source, features,
                     pred_up=None, pred_down=None):
    with app_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO open_predictions
              (symbol, side, entry_px, sl_px, tp_px, score, rule_name, source, features,
               last_px, last_check_at, peak_px, trough_px, predicted_up_pct, predicted_down_pct)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,now(),%s,%s,%s,%s)
            ON CONFLICT DO NOTHING RETURNING id
        """, (symbol, side, entry, sl, tp, score, rule, source, json.dumps(features),
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
# Scan-Pass v2 — Modell entscheidet
# =============================================================================

def make_sl_tp(side, entry, atr, sl_mult, tp_mult):
    if side == "long":
        return entry - sl_mult * atr, entry + tp_mult * atr
    return entry + sl_mult * atr, entry - tp_mult * atr


def scan_pass_v2(s, long_m, short_m, up_m, down_m, threshold, last_alert):
    cfg = s["predictor"]
    rules = cfg["rules"]
    cooldown = cfg["cooldown_seconds_per_symbol"]
    lookback = cfg["lookback_minutes"]
    mode = cfg.get("model_mode", "ml")
    ml_thr = float(cfg.get("ml_threshold", 0.55))
    min_closed_for_ml = int(cfg.get("min_closed_for_ml", 100))
    min_move = float(cfg.get("min_move_pct", 0.5))

    matches = 0
    with db_coins(s) as coins, db_app(s) as app:
        # Universe ggf. refreshen
        with app.cursor(cursor_factory=RealDictCursor) as cur_a:
            cur_a.execute("SELECT universe, universe_refreshed_at, closed_count FROM predictor_state WHERE id=1")
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
        closed_count = (st['closed_count'] if st else 0) or 0
        ml_active = closed_count >= min_closed_for_ml and mode in ("ml", "blend")

        # BTC-Bewegungen einmal pro scan_pass laden (Cross-Feature fuer alle Coins)
        btc_moves = load_btc_moves(coins)

        with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
            for sym in uni:
                if has_open(app, sym):
                    continue

                # === Regeln zuerst auswerten -> als Features ans Modell ===
                rule_flags = {}
                matched_long_rules = []
                matched_short_rules = []
                for rule in rules:
                    try:
                        result = evaluate_rule(cur_c, sym, rule, lookback)
                        ok = result["ok"]
                    except Exception as e:
                        log.warning("rule eval %s/%s failed: %s", sym, rule['name'], e)
                        ok = False
                    rule_flags[rule['name']] = 1 if ok else 0
                    if ok:
                        if rule['direction'] == 'long': matched_long_rules.append(rule['name'])
                        else: matched_short_rules.append(rule['name'])

                feat = feature_snapshot_v2(coins, sym, rule_flags=rule_flags, btc_moves=btc_moves)
                if feat is None: continue

                # === ML-Pfad: das Modell entscheidet ===
                ml_decision = None
                if ml_active and (long_m is not None or short_m is not None):
                    p_long = predict_proba(long_m, feat) if long_m else 0.5
                    p_short = predict_proba(short_m, feat) if short_m else 0.5
                    if max(p_long, p_short) >= ml_thr:
                        if p_long >= p_short:
                            ml_decision = ('long', p_long)
                        else:
                            ml_decision = ('short', p_short)

                # === Rule-Pfad nur als Cold-Start-Notbremse (mode=rules oder ML noch nicht aktiv) ===
                rule_decision = None
                if mode == "rules" or (mode != "ml" and not ml_active):
                    if matched_long_rules:
                        rule_decision = ('long', matched_long_rules[0], 1.0)
                    elif matched_short_rules:
                        rule_decision = ('short', matched_short_rules[0], 1.0)
                    if rule_decision:
                        now_s = time.time()
                        key = (sym, rule_decision[1])
                        if key in last_alert and now_s - last_alert[key] < cooldown:
                            rule_decision = None
                        else:
                            last_alert[key] = now_s

                # === Decision ===
                final = None
                if mode == "ml":
                    if ml_decision:
                        # Source-Tag: zeigt ob Regeln in gleiche Richtung gestützt haben
                        side = ml_decision[0]
                        if side == 'long' and matched_long_rules: src = 'ml+rule'
                        elif side == 'short' and matched_short_rules: src = 'ml+rule'
                        else: src = 'ml'
                        final = (side, 'ml', ml_decision[1], src)
                elif mode == "rules":
                    if rule_decision:
                        final = (rule_decision[0], rule_decision[1], rule_decision[2], 'rule')
                else:  # blend
                    if ml_decision and rule_decision and ml_decision[0] == rule_decision[0]:
                        final = (ml_decision[0], rule_decision[1], (ml_decision[1] + 1.0) / 2, 'blend')
                    elif ml_decision:
                        final = (ml_decision[0], 'ml', ml_decision[1], 'ml')
                    elif rule_decision:
                        final = (rule_decision[0], rule_decision[1], rule_decision[2], 'rule')

                if not final: continue
                side, rule_or_ml, score, source = final

                # Threshold (selbstkalibriert)
                if score < threshold: continue

                # === Modell predicted erwartete Bewegung ===
                pred_up = predict_value(up_m, feat, default=0.0)
                pred_down = predict_value(down_m, feat, default=0.0)
                # Sanity: niemals negativ
                pred_up = max(0.0, pred_up); pred_down = max(0.0, pred_down)

                # Min-Move-Filter: in der gewaehlten Richtung muss erwartete Bewegung >= min_move sein
                move_in_dir = pred_up if side == 'long' else pred_down
                if move_in_dir < min_move:
                    continue

                # Aktueller Preis fuer entry
                cur_c.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
                p = cur_c.fetchone()
                if not p or p['close'] is None: continue
                entry = f(p['close'])

                # SL/TP komplett aus Modell-Predictions (kein ATR mehr)
                if side == 'long':
                    tp = entry * (1 + pred_up / 100.0)
                    sl = entry * (1 - pred_down / 100.0)
                else:
                    tp = entry * (1 - pred_down / 100.0)
                    sl = entry * (1 + pred_up / 100.0)

                pid = open_prediction(app, sym, side, entry, sl, tp, score,
                                       rule_or_ml, source, feat,
                                       pred_up=pred_up, pred_down=pred_down)
                if pid:
                    matches += 1
                    log.info("OPEN %s %s entry=%.6g sl=%.6g tp=%.6g score=%.3f up=%.2f%% down=%.2f%% source=%s",
                             sym, side, entry, sl, tp, score, pred_up, pred_down, source)
                    if cfg.get("alert_on_open"):
                        send_telegram(s, f"[Predictor] OPEN {sym} {side} score={score:.3f} ({source})")
    return matches


# =============================================================================
# Watch-Pass — gleich wie v1, plus Lerner trainiert beide Modelle
# =============================================================================

def watch_pass_v2(s, long_m, short_m, up_m, down_m, paths):
    cfg = s["predictor"]
    timeout_h = float(cfg["timeout_hours"])
    closes = []
    with db_coins(s) as coins, db_app(s) as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur_a:
            cur_a.execute("""
                SELECT id, symbol, side, entry_px, sl_px, tp_px, score, rule_name, features,
                       created_at, peak_px, trough_px
                FROM open_predictions WHERE status='open'
            """)
            opens = cur_a.fetchall()
        if not opens: return 0

        symbols = list({o['symbol'] for o in opens})
        with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
            cur_c.execute("""
                SELECT DISTINCT ON (symbol) symbol, mid_px, close
                FROM klines
                WHERE symbol = ANY(%s) AND interval='10s'
                ORDER BY symbol, open_time DESC
            """, (symbols,))
            prices = {r['symbol']: (f(r['mid_px']) or f(r['close'])) for r in cur_c.fetchall()}

        for o in opens:
            cur_px = prices.get(o['symbol'])
            if cur_px is None: continue
            entry = f(o['entry_px']); sl = f(o['sl_px']); tp = f(o['tp_px'])
            side = o['side']
            # peak/trough mit aktuellem Preis updaten
            peak_px = max(f(o['peak_px']) or entry, cur_px)
            trough_px = min(f(o['trough_px']) or entry, cur_px)

            status = None
            if side == 'long':
                if cur_px >= tp: status = 'win'
                elif cur_px <= sl: status = 'loss'
            else:
                if cur_px <= tp: status = 'win'
                elif cur_px >= sl: status = 'loss'
            age_h = (datetime.now(timezone.utc) - o['created_at']).total_seconds() / 3600
            if status is None and age_h >= timeout_h:
                status = 'timeout'

            with app.cursor() as cur_a:
                if status:
                    pnl_pct = ((cur_px - entry) / entry * 100) if side == 'long' else ((entry - cur_px) / entry * 100)
                    # Peak/Trough relativ zu entry, IMMER positiv (Bewegungs-Magnitude)
                    peak_pct_v = (peak_px - entry) / entry * 100.0
                    trough_pct_v = (entry - trough_px) / entry * 100.0
                    cur_a.execute("""
                        UPDATE open_predictions
                        SET status=%s, exit_px=%s, pnl_pct=%s, closed_at=now(),
                            last_px=%s, last_check_at=now(), peak_px=%s, trough_px=%s
                        WHERE id=%s AND status='open'
                    """, (status, cur_px, pnl_pct, cur_px, peak_px, trough_px, o['id']))
                    closes.append({
                        'id': o['id'], 'symbol': o['symbol'], 'side': side,
                        'entry': entry, 'exit': cur_px, 'pnl_pct': pnl_pct,
                        'peak_pct': peak_pct_v, 'trough_pct': trough_pct_v,
                        'status': status, 'features': o['features'],
                        'rule': o['rule_name'], 'score': f(o['score']),
                        'created_at': o['created_at'],
                    })
                    log.info("CLOSE %s %s status=%s pnl=%.3f%% peak=%.2f%% trough=%.2f%%",
                             o['symbol'], side, status, pnl_pct, peak_pct_v, trough_pct_v)
                else:
                    cur_a.execute("""
                        UPDATE open_predictions
                        SET last_px=%s, last_check_at=now(), peak_px=%s, trough_px=%s
                        WHERE id=%s AND status='open'
                    """, (cur_px, peak_px, trough_px, o['id']))
            app.commit()

    if closes:
        learn_and_calibrate_v2(s, long_m, short_m, up_m, down_m, paths, closes)
    return len(closes)


def learn_and_calibrate_v2(s, long_m, short_m, up_m, down_m, paths, closes):
    cfg = s["predictor"]
    learner_cfg = cfg["online_learner"]
    cal_cfg = cfg["self_calibration"]
    treat_to_loss = bool(cfg.get("treat_timeout_as_loss", False))
    long_path, short_path, up_path, down_path = paths

    with db_learner(s) as ldb, db_app(s) as app:
        with ldb.cursor() as cur_l:
            for c in closes:
                # D-Light: bei Timeout NICHT mehr skippen, sondern Lernsignal
                # aus realisierter pnl ableiten (won = pnl > 0). Verschwendet kein
                # Trainingssignal, klassifiziert profitable Timeouts korrekt als win.
                if c['status'] == 'timeout':
                    won = (c.get('pnl_pct') or 0.0) > 0.0
                else:
                    won = c['status'] == 'win'
                if learner_cfg.get("enabled") and c.get('features'):
                    try:
                        feat = c['features']
                        if isinstance(feat, dict):
                            # Klassifikatoren: beide Modelle pro Close (long-win = short-loss-hint)
                            if c['side'] == 'long':
                                if long_m is not None:  long_m.learn_one(feat, 1 if won else 0)
                                if short_m is not None: short_m.learn_one(feat, 0 if won else 1)
                            else:
                                if short_m is not None: short_m.learn_one(feat, 1 if won else 0)
                                if long_m is not None:  long_m.learn_one(feat, 0 if won else 1)
                            # Regressoren: peak/trough sind richtungs-unabhaengige Magnituden
                            pp = c.get('peak_pct'); tp_v = c.get('trough_pct')
                            if pp is not None and up_m is not None:
                                up_m.learn_one(feat, max(0.0, abs(float(pp))))
                            if tp_v is not None and down_m is not None:
                                down_m.learn_one(feat, max(0.0, abs(float(tp_v))))
                    except Exception as e:
                        log.warning("learn_one %s: %s", c['symbol'], e)

                duration = int((datetime.now(timezone.utc) - c['created_at']).total_seconds() / 60)
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

        if learner_cfg.get("enabled"):
            try:
                if long_m is not None: save_model(long_m, long_path)
                if short_m is not None: save_model(short_m, short_path)
                if up_m is not None: save_model(up_m, up_path)
                if down_m is not None: save_model(down_m, down_path)
            except Exception as e:
                log.warning("save_model: %s", e)

        # Kalibrierung
        if cal_cfg.get("enabled"):
            with app.cursor() as cur_a:
                cur_a.execute("UPDATE predictor_state SET closed_count = closed_count + %s WHERE id=1",
                              (len(closes),))
            app.commit()
            with app.cursor() as cur_a:
                cur_a.execute("SELECT closed_count, threshold FROM predictor_state WHERE id=1")
                row = cur_a.fetchone()
            cnt, thr = row[0], f(row[1])
            review_n = int(cal_cfg.get("review_after_n_closes", 10))
            if review_n > 0 and cnt % review_n == 0:
                win = int(cal_cfg["rolling_window"])
                with ldb.cursor() as cur_l:
                    cur_l.execute("""
                        SELECT was_correct FROM prediction_feedback
                        WHERE scanner_type='predictor'
                        ORDER BY resolved_at DESC LIMIT %s
                    """, (win,))
                    rows = cur_l.fetchall()
                if len(rows) >= 20:
                    wr = sum(1 for r in rows if r[0]) / len(rows)
                    target = float(cal_cfg["target_winrate"])
                    step = float(cal_cfg["adjust_step"])
                    if wr < target: thr += step
                    elif wr > target + 0.05: thr -= step
                    thr = max(float(cal_cfg.get("min_threshold", 0.0)),
                              min(float(cal_cfg.get("max_threshold", 0.95)), thr))
                    with app.cursor() as cur_a:
                        cur_a.execute("""
                            UPDATE predictor_state SET threshold=%s, rolling_winrate=%s,
                                last_calibration_at=now(), updated_at=now() WHERE id=1
                        """, (thr, wr))
                    app.commit()
                    log.info("calibrate: WR=%.3f target=%.3f -> threshold=%.3f", wr, target, thr)


# =============================================================================
# Main loop
# =============================================================================

def main():
    s = load_settings()
    cfg = s["predictor"]
    if not cfg.get("enabled"):
        log.info("predictor disabled in settings, exit")
        return

    learner_cfg = cfg["online_learner"]
    long_path = learner_cfg.get("long_model_path", "/opt/coin/database/data/models/predictor_long.pkl")
    short_path = learner_cfg.get("short_model_path", "/opt/coin/database/data/models/predictor_short.pkl")
    up_path = learner_cfg.get("up_model_path", "/opt/coin/database/data/models/predictor_up.pkl")
    down_path = learner_cfg.get("down_model_path", "/opt/coin/database/data/models/predictor_down.pkl")
    paths = (long_path, short_path, up_path, down_path)

    # Bootstrap wenn irgendein Modell fehlt
    if not all(os.path.exists(p) for p in paths):
        log.info("bootstrapping models from prediction_feedback (fetching peak/trough from klines)...")
        long_m, short_m, up_m, down_m, n = bootstrap_from_feedback(s, long_path, short_path, up_path, down_path)
        log.info("bootstrap: %d historical closes replayed", n)
    else:
        long_m = load_model(long_path, make_clf)
        short_m = load_model(short_path, make_clf)
        up_m = load_model(up_path, make_reg)
        down_m = load_model(down_path, make_reg)
        log.info("4 models loaded")

    with db_app(s) as app:
        with app.cursor() as cur:
            cur.execute("SELECT threshold FROM predictor_state WHERE id=1")
            r = cur.fetchone()
            threshold = f(r[0]) if r else 0.0

    log.info("Predictor v2 start. mode=%s threshold=%.3f scan=%ss watch=%ss top_n=%d ml_thr=%s",
             cfg.get("model_mode", "blend"), threshold,
             cfg["scan_interval_seconds"], cfg["watch_interval_seconds"],
             cfg["universe_top_n"], cfg.get("ml_threshold", 0.55))

    last_alert = {}; last_scan = 0.0; last_watch = 0.0; last_settings_reload = 0.0

    while True:
        try:
            now = time.time()
            if now - last_settings_reload >= 60:
                s = load_settings()
                cfg = s["predictor"]
                last_settings_reload = now
                with db_app(s) as app:
                    with app.cursor() as cur:
                        cur.execute("SELECT threshold FROM predictor_state WHERE id=1")
                        r = cur.fetchone()
                        threshold = f(r[0]) if r else 0.0

            if now - last_scan >= cfg["scan_interval_seconds"]:
                m = scan_pass_v2(s, long_m, short_m, up_m, down_m, threshold, last_alert)
                last_scan = now
                if m > 0:
                    log.info("scan-pass: %d new predictions, threshold=%.3f", m, threshold)

            if now - last_watch >= cfg["watch_interval_seconds"]:
                c = watch_pass_v2(s, long_m, short_m, up_m, down_m, paths)
                last_watch = now
                if c > 0:
                    log.info("watch-pass: %d closed", c)

            time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("interrupt"); break
        except Exception as e:
            log.exception("loop error: %s", e); time.sleep(5)


if __name__ == "__main__":
    main()
