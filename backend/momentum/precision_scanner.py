#!/usr/bin/env python3
"""
PRECISION SCANNER v2
Findet Indikator-Sets die ≥5% Events vorhersagen + Gegenprobe

Phase 1: Event-Erkennung via kline_metrics (alle pct-Spalten 60m-600m)
Phase 2: Pre-Event Indikatoren (5min-Dynamik + 1h/4h/1d State)
Phase 3: Greedy Filter-Suche (Long/Short Trennung)
Phase 4: TP/SL-Simulation (5min candle walk, Entry am Anfang des Moves)
Phase 5: Gegenprobe (Filter auf ALLE Zeitpunkte → False Positive Rate)

Usage: python3 precision_scanner.py [days] [min_move_pct]
"""

import sys, os, logging, time
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

LOG_FILE = '/opt/coin/logs/precision_scanner.log'
REPORT_DIR = '/opt/coin/database/data'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('precision')

# === CONFIG ===
PCT_COLS = ['pct_60m','pct_90m','pct_120m','pct_180m','pct_240m',
            'pct_300m','pct_360m','pct_420m','pct_480m','pct_540m','pct_600m']
DUR_MAP = {'pct_60m':60,'pct_90m':90,'pct_120m':120,'pct_180m':180,'pct_240m':240,
           'pct_300m':300,'pct_360m':360,'pct_420m':420,'pct_480m':480,'pct_540m':540,'pct_600m':600}

TP_SL_COMBOS = [
    (2,2),(2,3),(2,4),(2,5),
    (3,3),(3,4),(3,5),
    (4,4),(4,5),
    (5,5),
]
DURATION_MINUTES = [120, 240, 360, 480, 600, 720]
MIN_MOVE_PCT = 5.0
LOOKBACK_DAYS = 30


def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


# ========================================================================
# PHASE 1: EVENT DETECTION
# ========================================================================

def find_events(conn, days, min_pct):
    cur = conn.cursor()
    logger.info("=" * 70)
    logger.info("PHASE 1: EVENT DETECTION")
    logger.info(f"  kline_metrics, alle pct-Spalten 60m-600m, threshold ≥{min_pct}%")
    logger.info("=" * 70)

    where = ' OR '.join(f'ABS({col}) >= {min_pct}' for col in PCT_COLS)
    cols = ', '.join(PCT_COLS)

    t0 = time.time()
    cur.execute(f"""
        SELECT symbol, open_time, {cols}
        FROM kline_metrics
        WHERE open_time >= NOW() - INTERVAL '{days} days'
          AND ({where})
        ORDER BY symbol, open_time
    """)
    rows = cur.fetchall()
    logger.info(f"  Query done: {len(rows)} raw rows in {time.time()-t0:.1f}s")

    all_events = []
    for row in rows:
        best_col = None
        best_pct = 0
        for col in PCT_COLS:  # kürzestes Zeitfenster zuerst
            val = row[col]
            if val is not None and abs(val) >= min_pct:
                best_col = col
                best_pct = val
                break
        if best_col is None:
            continue

        duration = DUR_MAP[best_col]
        direction = 'long' if best_pct > 0 else 'short'
        entry_time = row['open_time'] - timedelta(minutes=duration)

        all_events.append({
            'symbol': row['symbol'],
            'time': entry_time,
            'event_time': row['open_time'],
            'duration_min': duration,
            'direction': direction,
            'best_pct': round(best_pct, 2),
            'best_tf': best_col,
        })

    # Dedup: max 1 Event pro Symbol pro 60 Min
    all_events.sort(key=lambda e: (e['symbol'], e['time']))
    deduped = []
    last_by_sym = {}
    for e in all_events:
        key = e['symbol']
        if key in last_by_sym:
            diff = (e['time'] - last_by_sym[key]).total_seconds() / 60
            if diff < 60:
                continue
        last_by_sym[key] = e['time']
        deduped.append(e)

    longs = sum(1 for e in deduped if e['direction'] == 'long')
    shorts = sum(1 for e in deduped if e['direction'] == 'short')
    logger.info(f"  Events: {len(deduped)} (LONG={longs}, SHORT={shorts})")

    tf_dist = defaultdict(int)
    for e in deduped:
        tf_dist[e['best_tf']] += 1
    for tf in PCT_COLS:
        if tf in tf_dist:
            logger.info(f"    {tf}: {tf_dist[tf]}")

    return deduped


# ========================================================================
# PHASE 2: PRE-EVENT INDIKATOREN
# ========================================================================

# --- Hilfsfunktionen ---
def _rsi(closes, period=14):
    if len(closes) < period + 1: return None
    d = np.diff(closes)
    g = np.mean(np.where(d > 0, d, 0)[-period:])
    l = np.mean(np.where(d < 0, -d, 0)[-period:])
    return 100 - (100 / (1 + g / l)) if l > 0 else 100.0

def _ema(vals, p):
    if len(vals) < p: return None
    m = 2 / (p + 1)
    e = vals[0]
    for v in vals[1:]: e = (v - e) * m + e
    return e

def _bb_pos(closes, p=20):
    if len(closes) < p: return None
    s = np.mean(closes[-p:])
    d = np.std(closes[-p:])
    if d == 0: return 0.5
    return (closes[-1] - (s - 2*d)) / (4*d) if d > 0 else 0.5

def _range_pos(highs, lows, closes, n):
    if len(highs) < n: return None
    hi = max(highs[-n:])
    lo = min(lows[-n:])
    return (closes[-1] - lo) / (hi - lo) if hi != lo else 0.5

def _trend(closes, n):
    if len(closes) < n: return 0
    r = closes[-n:]
    return (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r)-1)) * 2 - 1

def _hh_hl(highs, lows, n):
    if len(highs) < n: return 0
    rh, rl = highs[-n:], lows[-n:]
    return (sum(1 for i in range(1,n) if rh[i]>rh[i-1]) + sum(1 for i in range(1,n) if rl[i]>rl[i-1])) / ((n-1)*2)

def _obv_trend(closes, volumes, n=10):
    if len(closes) < n+1: return 0
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
        else: obv.append(obv[-1])
    x = np.arange(n)
    slope = np.polyfit(x, obv[-n:], 1)[0]
    av = np.mean(volumes[-n:])
    return slope / av if av > 0 else 0

def _vwap_dist(closes, volumes, n=20):
    if len(closes) < n: return None
    c, v = closes[-n:], volumes[-n:]
    tv = sum(v)
    if tv == 0: return 0
    vwap = sum(c[i]*v[i] for i in range(n)) / tv
    return ((closes[-1] - vwap) / vwap * 100) if vwap > 0 else 0

def _atr(highs, lows, closes, p=14):
    if len(closes) < p+1: return None
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
    return np.mean(trs[-p:])


def compute_indicators(conn, symbol, entry_time):
    """
    Berechnet ~80+ Indikatoren VOR entry_time:
    A) 5min-Dynamik (Volumen, Trades, Taker, Momentum in verschiedenen Zeitfenstern)
    B) 1h-State (RSI, BB, EMA, Trends, Candle-Patterns)
    C) 4h/1d-Kontext (Range, Trends, ATR)
    """
    cur = conn.cursor()
    ind = {}

    # --- A) 5min Dynamik: letzte 5h vor Entry ---
    cur.execute("""
        SELECT bucket, open, high, low, close, volume, number_of_trades,
               taker_buy_base_asset_volume
        FROM agg_5m
        WHERE symbol = %s AND bucket >= %s - INTERVAL '300 minutes' AND bucket < %s
        ORDER BY bucket
    """, (symbol, entry_time, entry_time))
    r5m = cur.fetchall()

    if len(r5m) >= 6:
        c5 = [float(r['close']) for r in r5m]
        h5 = [float(r['high']) for r in r5m]
        l5 = [float(r['low']) for r in r5m]
        v5 = [float(r['volume']) for r in r5m]
        t5 = [float(r['number_of_trades'] or 0) for r in r5m]
        tb5 = [float(r['taker_buy_base_asset_volume'] or 0) for r in r5m]

        n = len(r5m)
        entry_price = c5[-1] if c5[-1] > 0 else None
        if entry_price is None:
            return None

        # Volumen-Dynamik: letzte X min vs Durchschnitt der gesamten 3h
        avg_vol_5m = np.mean(v5) if v5 else 1
        avg_trades_5m = np.mean(t5) if t5 else 1

        for minutes, label in [(15,3),(30,6),(60,12),(120,24)]:
            count = min(minutes // 5, n)
            if count < 1: continue
            recent_v = v5[-count:]
            older_v = v5[:-count] if len(v5) > count else v5
            recent_t = t5[-count:]
            older_t = t5[:-count] if len(t5) > count else t5

            # Volume ratio: recent vs average
            ind[f'vol_ratio_{minutes}m'] = round(np.mean(recent_v) / avg_vol_5m, 3) if avg_vol_5m > 0 else 1
            # Volume trend: recent vs older period
            if older_v and np.mean(older_v) > 0:
                ind[f'vol_trend_{minutes}m'] = round(np.mean(recent_v) / np.mean(older_v), 3)
            # Trades ratio
            ind[f'trades_ratio_{minutes}m'] = round(np.mean(recent_t) / avg_trades_5m, 3) if avg_trades_5m > 0 else 1
            # Trades trend
            if older_t and np.mean(older_t) > 0:
                ind[f'trades_trend_{minutes}m'] = round(np.mean(recent_t) / np.mean(older_t), 3)

        # Taker Buy Ratio (verschiedene Fenster)
        for minutes, label in [(15,3),(30,6),(60,12)]:
            count = min(minutes // 5, n)
            if count < 1: continue
            rv = v5[-count:]
            rtb = tb5[-count:]
            total_v = sum(rv)
            total_tb = sum(rtb)
            if total_v > 0:
                ind[f'taker_ratio_{minutes}m'] = round(total_tb / total_v * 100, 2)

        # Taker Shift: letzte 30min vs vorherige 30min
        if n >= 12:
            v_recent = sum(v5[-6:])
            tb_recent = sum(tb5[-6:])
            v_older = sum(v5[-12:-6])
            tb_older = sum(tb5[-12:-6])
            if v_recent > 0 and v_older > 0:
                ind['taker_shift_30m'] = round((tb_recent/v_recent - tb_older/v_older) * 100, 2)

        # Price Momentum aus 5min (verschiedene Fenster)
        for minutes in [5, 10, 15, 30, 60, 120]:
            count = minutes // 5
            if n > count and c5[-(count+1)] > 0:
                ind[f'mom_{minutes}m'] = round((c5[-1] - c5[-(count+1)]) / c5[-(count+1)] * 100, 3)

        # Range (Volatilität) in verschiedenen Fenstern
        for minutes in [15, 30, 60]:
            count = min(minutes // 5, n)
            if count >= 2:
                window_h = h5[-count:]
                window_l = l5[-count:]
                range_pct = (max(window_h) - min(window_l)) / entry_price * 100
                ind[f'range_{minutes}m'] = round(range_pct, 3)

        # 5min Candle Patterns (letzte 6 Candles)
        o5 = [float(r['open']) for r in r5m]
        bodies, uwicks, lwicks, bulls = [], [], [], 0
        for i in range(-min(6, n), 0):
            full = h5[i] - l5[i]
            if full <= 0: continue
            bodies.append(abs(c5[i] - o5[i]) / full)
            if c5[i] >= o5[i]:
                uwicks.append((h5[i] - c5[i]) / full)
                lwicks.append((o5[i] - l5[i]) / full)
                bulls += 1
            else:
                uwicks.append((h5[i] - o5[i]) / full)
                lwicks.append((c5[i] - l5[i]) / full)
        if bodies:
            ind['body_ratio_5m'] = round(np.mean(bodies), 3)
            ind['upper_wick_5m'] = round(np.mean(uwicks), 3)
            ind['lower_wick_5m'] = round(np.mean(lwicks), 3)
            ind['bullish_ratio_5m'] = round(bulls / len(bodies), 2)

        # 5min Einzelcandle-Patterns (letzte Candle)
        if n >= 2:
            full_last = h5[-1] - l5[-1]
            if full_last > 0:
                br = abs(c5[-1] - o5[-1]) / full_last
                uw = (h5[-1] - max(c5[-1], o5[-1])) / full_last
                lw = (min(c5[-1], o5[-1]) - l5[-1]) / full_last
                ind['doji_5m'] = 1 if br < 0.1 else 0
                ind['hammer_5m'] = 1 if lw > 2 * br and uw < br + 0.05 else 0
                ind['shooting_star_5m'] = 1 if uw > 2 * br and lw < br + 0.05 else 0
                ind['marubozu_5m'] = 1 if br > 0.8 else 0

        # 5min Engulfing (letzte 2 Candles)
        if n >= 3:
            full_prev = h5[-2] - l5[-2]
            full_last = h5[-1] - l5[-1]
            if full_prev > 0 and full_last > 0:
                body_prev = c5[-2] - o5[-2]
                body_last = c5[-1] - o5[-1]
                ind['engulfing_bull_5m'] = 1 if body_prev < 0 and body_last > 0 and \
                    abs(body_last) > abs(body_prev) else 0
                ind['engulfing_bear_5m'] = 1 if body_prev > 0 and body_last < 0 and \
                    abs(body_last) > abs(body_prev) else 0
                ind['inside_bar_5m'] = 1 if h5[-1] <= h5[-2] and l5[-1] >= l5[-2] else 0

        # 5min Max consecutive bull/bear (letzte 12)
        if n >= 12:
            max_bull_run, max_bear_run, cur_bull, cur_bear = 0, 0, 0, 0
            for i in range(-12, 0):
                if c5[i] >= o5[i]:
                    cur_bull += 1; cur_bear = 0
                else:
                    cur_bear += 1; cur_bull = 0
                max_bull_run = max(max_bull_run, cur_bull)
                max_bear_run = max(max_bear_run, cur_bear)
            ind['max_bull_run_5m'] = max_bull_run
            ind['max_bear_run_5m'] = max_bear_run

        # 5min Body-Expansion (letzte vs Durchschnitt der 5 davor)
        if n >= 7:
            full_last = h5[-1] - l5[-1]
            prev_fulls = [h5[i] - l5[i] for i in range(-6, -1)]
            avg_prev = np.mean([f for f in prev_fulls if f > 0]) if any(f > 0 for f in prev_fulls) else 1
            if avg_prev > 0 and full_last > 0:
                ind['body_expansion_5m'] = round(full_last / avg_prev, 2)

        # Consecutive ups/downs in 5min
        if n >= 10:
            ups = sum(1 for i in range(-10, 0) if c5[i] > c5[i-1])
            ind['consec_ups_5m'] = ups
            ind['consec_downs_5m'] = 10 - ups

        # ================================================================
        # PRE-EVENT SPIKE DETECTION (Sliding Windows ueber 5h vor Entry)
        # Sucht kurze Kurs-/Volumen-Spikes in verschiedenen Fenstergroessen
        # ================================================================
        avg_vol_all = np.mean(v5) if np.mean(v5) > 0 else 1
        avg_trades_all = np.mean(t5) if np.mean(t5) > 0 else 1

        for win_candles, win_label in [(2, '10m'), (3, '15m'), (4, '20m'), (6, '30m')]:
            if n < win_candles + 1:
                continue

            best_up, best_down = 0, 0
            best_up_time, best_down_time = 0, 0
            best_vol_ratio = 0
            best_vol_time = 0
            best_trades_ratio = 0
            best_taker_at_vol_spike = 50  # neutral

            for i in range(n - win_candles):
                start_price = c5[i]
                end_price = c5[i + win_candles]
                minutes_before = (n - 1 - i - win_candles) * 5

                # Kurs-Spike: Veraenderung ueber das Window
                if start_price > 0:
                    pct_change = (end_price - start_price) / start_price * 100
                    if pct_change > best_up:
                        best_up = pct_change
                        best_up_time = minutes_before
                    if pct_change < best_down:
                        best_down = pct_change
                        best_down_time = minutes_before

                # Volumen-Spike: Window-Durchschnitt vs Gesamtdurchschnitt
                win_vol = np.mean(v5[i:i + win_candles])
                vol_ratio = win_vol / avg_vol_all if avg_vol_all > 0 else 1
                if vol_ratio > best_vol_ratio:
                    best_vol_ratio = vol_ratio
                    best_vol_time = minutes_before
                    # Taker-Ratio waehrend dieses Volumen-Spikes
                    win_tb = sum(tb5[i:i + win_candles])
                    win_v = sum(v5[i:i + win_candles])
                    best_taker_at_vol_spike = (win_tb / win_v * 100) if win_v > 0 else 50

                # Trades-Spike
                win_trades = np.mean(t5[i:i + win_candles])
                trades_ratio = win_trades / avg_trades_all if avg_trades_all > 0 else 1
                if trades_ratio > best_trades_ratio:
                    best_trades_ratio = trades_ratio

            # Intra-Window Max-Range (groesste Einzelcandle im Window-Bereich)
            max_single_range = 0
            if entry_price and entry_price > 0:
                for i in range(n):
                    candle_range = (h5[i] - l5[i]) / entry_price * 100
                    if candle_range > max_single_range:
                        max_single_range = candle_range

            ind['spike_up_%s' % win_label] = round(best_up, 3)
            ind['spike_down_%s' % win_label] = round(abs(best_down), 3)
            ind['spike_up_time_%s' % win_label] = best_up_time
            ind['spike_down_time_%s' % win_label] = best_down_time
            ind['vol_spike_%s' % win_label] = round(best_vol_ratio, 2)
            ind['vol_spike_time_%s' % win_label] = best_vol_time
            ind['trades_spike_%s' % win_label] = round(best_trades_ratio, 2)
            ind['taker_at_vol_spike_%s' % win_label] = round(best_taker_at_vol_spike, 2)

        # Max Einzelcandle-Range in den 3h (groesster 5min-Ausreisser)
        if entry_price and entry_price > 0:
            candle_ranges = [(h5[i] - l5[i]) / entry_price * 100 for i in range(n)]
            ind['max_candle_range_5h'] = round(max(candle_ranges), 3) if candle_ranges else 0
            # Wo war die groesste Candle (Minuten vor Entry)
            max_idx = candle_ranges.index(max(candle_ranges))
            ind['max_candle_time_5h'] = (n - 1 - max_idx) * 5

    else:
        return None  # Nicht genug 5min-Daten

    # --- B) 1h State Indikatoren ---
    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_1h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 50
    """, (symbol, entry_time))
    r1h = list(reversed(cur.fetchall()))
    if len(r1h) < 24: return None

    c1h = [float(r['close']) for r in r1h]
    h1h = [float(r['high']) for r in r1h]
    l1h = [float(r['low']) for r in r1h]
    o1h = [float(r['open']) for r in r1h]
    v1h = [float(r['volume']) for r in r1h]

    rsi = _rsi(c1h, 14)
    if rsi is None: return None
    ind['rsi_1h'] = round(rsi, 1)

    ema9 = _ema(c1h, 9)
    ema21 = _ema(c1h, 21)
    ema50 = _ema(c1h[-50:], 50) if len(c1h) >= 50 else None
    if ema9 and ema21 and ema21 > 0:
        ind['ema_9_21_pct'] = round((ema9-ema21)/ema21*100, 3)
    if ema50 and ema50 > 0:
        ind['ema_price_50_pct'] = round((c1h[-1]-ema50)/ema50*100, 3)

    atr = _atr(h1h, l1h, c1h, 14)
    if atr and c1h[-1] > 0:
        ind['atr_1h_pct'] = round(atr/c1h[-1]*100, 3)

    ind['bb_position_1h'] = round(_bb_pos(c1h), 3)

    ind['trend_1h_5'] = round(_trend(c1h, 5), 2)
    ind['trend_1h_10'] = round(_trend(c1h, 10), 2)
    if len(c1h) >= 20:
        ind['trend_1h_20'] = round(_trend(c1h, 20), 2)

    ind['hh_hl_6'] = round(_hh_hl(h1h, l1h, 6), 2)
    if len(h1h) >= 12:
        ind['hh_hl_12'] = round(_hh_hl(h1h, l1h, 12), 2)

    cu = sum(1 for i in range(-10, 0) if c1h[i] > c1h[i-1]) if len(c1h) > 10 else 0
    ind['consec_ups_1h'] = cu
    ind['consec_downs_1h'] = 10 - cu

    avg_v1h = np.mean(v1h[-20:])
    if avg_v1h > 0:
        ind['vol_ratio_1h'] = round(np.mean(v1h[-3:]) / avg_v1h, 2)

    ind['obv_trend_1h'] = round(_obv_trend(c1h, v1h, 10), 4)
    vd = _vwap_dist(c1h, v1h, 20)
    if vd is not None:
        ind['vwap_dist_1h'] = round(vd, 3)

    ind['range_pos_24h'] = round(_range_pos(h1h, l1h, c1h, 24), 3)

    # 1h Candle Patterns (Durchschnitte letzte 6)
    cp_bodies, cp_uw, cp_lw, cp_bulls = [], [], [], 0
    for i in range(-6, 0):
        full = h1h[i] - l1h[i]
        if full <= 0: continue
        cp_bodies.append(abs(c1h[i] - o1h[i]) / full)
        if c1h[i] >= o1h[i]:
            cp_uw.append((h1h[i] - c1h[i]) / full)
            cp_lw.append((o1h[i] - l1h[i]) / full)
            cp_bulls += 1
        else:
            cp_uw.append((h1h[i] - o1h[i]) / full)
            cp_lw.append((c1h[i] - l1h[i]) / full)
    if cp_bodies:
        ind['body_ratio_1h'] = round(np.mean(cp_bodies), 3)
        ind['upper_wick_1h'] = round(np.mean(cp_uw), 3)
        ind['lower_wick_1h'] = round(np.mean(cp_lw), 3)
        ind['bullish_candles_1h'] = cp_bulls

    # 1h Einzelcandle-Patterns (letzte Candle)
    full_last_1h = h1h[-1] - l1h[-1]
    if full_last_1h > 0:
        br = abs(c1h[-1] - o1h[-1]) / full_last_1h
        uw = (h1h[-1] - max(c1h[-1], o1h[-1])) / full_last_1h
        lw = (min(c1h[-1], o1h[-1]) - l1h[-1]) / full_last_1h
        ind['doji_1h'] = 1 if br < 0.1 else 0
        ind['hammer_1h'] = 1 if lw > 2 * br and uw < br + 0.05 else 0
        ind['shooting_star_1h'] = 1 if uw > 2 * br and lw < br + 0.05 else 0
        ind['marubozu_1h'] = 1 if br > 0.8 else 0
        ind['wick_ratio_1h'] = round((uw + lw) / (br + 0.001), 2)

    # 1h Engulfing + Inside Bar (letzte 2 Candles)
    if len(c1h) >= 2:
        full_prev_1h = h1h[-2] - l1h[-2]
        if full_last_1h > 0 and full_prev_1h > 0:
            body_prev = c1h[-2] - o1h[-2]
            body_last = c1h[-1] - o1h[-1]
            ind['engulfing_bull_1h'] = 1 if body_prev < 0 and body_last > 0 and \
                abs(body_last) > abs(body_prev) else 0
            ind['engulfing_bear_1h'] = 1 if body_prev > 0 and body_last < 0 and \
                abs(body_last) > abs(body_prev) else 0
            ind['inside_bar_1h'] = 1 if h1h[-1] <= h1h[-2] and l1h[-1] >= l1h[-2] else 0

    # 1h Max consecutive bull/bear runs (letzte 12)
    if len(c1h) >= 12:
        max_bull_run_1h, max_bear_run_1h, cb, cbr = 0, 0, 0, 0
        for i in range(-12, 0):
            if c1h[i] >= o1h[i]:
                cb += 1; cbr = 0
            else:
                cbr += 1; cb = 0
            max_bull_run_1h = max(max_bull_run_1h, cb)
            max_bear_run_1h = max(max_bear_run_1h, cbr)
        ind['max_bull_run_1h'] = max_bull_run_1h
        ind['max_bear_run_1h'] = max_bear_run_1h

    # 1h Body-Expansion (letzte Candle vs Durchschnitt der 5 davor)
    if len(c1h) >= 7:
        prev_fulls_1h = [h1h[i] - l1h[i] for i in range(-6, -1)]
        avg_prev_1h = np.mean([f for f in prev_fulls_1h if f > 0]) if any(f > 0 for f in prev_fulls_1h) else 1
        if avg_prev_1h > 0 and full_last_1h > 0:
            ind['body_expansion_1h'] = round(full_last_1h / avg_prev_1h, 2)

    # 1h Largest body position (welche der letzten 6 war die groesste)
    if len(c1h) >= 6:
        body_sizes = [abs(c1h[i] - o1h[i]) for i in range(-6, 0)]
        if max(body_sizes) > 0:
            ind['largest_body_pos_1h'] = body_sizes.index(max(body_sizes)) + 1

    # 1h Gap (Open vs Previous Close)
    if len(c1h) >= 2 and c1h[-2] > 0:
        ind['gap_1h'] = round((o1h[-1] - c1h[-2]) / c1h[-2] * 100, 3)

    # --- C) 4h / 1d Kontext ---
    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_4h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 30
    """, (symbol, entry_time))
    r4h = list(reversed(cur.fetchall()))
    if len(r4h) >= 10:
        c4h = [float(r['close']) for r in r4h]
        h4h = [float(r['high']) for r in r4h]
        l4h = [float(r['low']) for r in r4h]
        v4h = [float(r['volume']) for r in r4h]

        rsi4 = _rsi(c4h, 14)
        if rsi4: ind['rsi_4h'] = round(rsi4, 1)
        if len(c4h) >= 20:
            ind['bb_position_4h'] = round(_bb_pos(c4h), 3)
        ind['trend_4h_5'] = round(_trend(c4h, 5), 2)
        if len(c4h) >= 10:
            ind['trend_4h_10'] = round(_trend(c4h, 10), 2)
        if len(h4h) >= 6:
            ind['hh_hl_4h'] = round(_hh_hl(h4h, l4h, 6), 2)
        atr4 = _atr(h4h, l4h, c4h, 14)
        if atr4 and c4h[-1] > 0:
            ind['atr_4h_pct'] = round(atr4/c4h[-1]*100, 3)
        avg_v4h = np.mean(v4h[-10:])
        if avg_v4h > 0:
            ind['vol_ratio_4h'] = round(np.mean(v4h[-2:]) / avg_v4h, 2)
        ind['obv_trend_4h'] = round(_obv_trend(c4h, v4h, 6), 4)
        vd4 = _vwap_dist(c4h, v4h, 10)
        if vd4 is not None:
            ind['vwap_dist_4h'] = round(vd4, 3)

    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_1d WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 15
    """, (symbol, entry_time))
    r1d = list(reversed(cur.fetchall()))
    if len(r1d) >= 5:
        c1d = [float(r['close']) for r in r1d]
        h1d = [float(r['high']) for r in r1d]
        l1d = [float(r['low']) for r in r1d]

        ind['trend_1d_5'] = round(_trend(c1d, 5), 2)
        if len(c1d) >= 10:
            ind['trend_1d_10'] = round(_trend(c1d, 10), 2)
        if len(h1d) >= 7:
            ind['range_pos_7d'] = round(_range_pos(h1d, l1d, c1d, 7), 3)

        # Pct changes
        for n, label in [(3,'3d'),(7,'7d')]:
            if len(c1d) > n and c1d[-(n+1)] > 0:
                ind[f'pct_{label}'] = round((c1d[-1]-c1d[-(n+1)])/c1d[-(n+1)]*100, 2)

    return ind


# ========================================================================
# PHASE 3: GREEDY FILTER ENGINE
# ========================================================================

def build_rules(events):
    indicators = set()
    for e in events:
        for k, v in e.get('indicators', {}).items():
            if v is not None: indicators.add(k)

    rules = []
    for ind in sorted(indicators):
        vals = [e['indicators'][ind] for e in events if e.get('indicators', {}).get(ind) is not None]
        if len(vals) < 50: continue
        for p in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            t = float(np.percentile(vals, p))
            rules.append({'ind': ind, 'op': '>=', 'thr': round(t, 4), 'label': f'{ind} >= {t:.4f}'})
            rules.append({'ind': ind, 'op': '<', 'thr': round(t, 4), 'label': f'{ind} < {t:.4f}'})
    return rules


def match_rule(event, rule):
    v = event.get('indicators', {}).get(rule['ind'])
    if v is None: return False
    return v >= rule['thr'] if rule['op'] == '>=' else v < rule['thr']


def eval_rules(events, rules, direction):
    matching = [e for e in events if all(match_rule(e, r) for r in rules)]
    if len(matching) < 15: return None
    correct = sum(1 for e in matching if e['direction'] == direction)
    wins = sum(1 for e in matching if e['direction'] == direction and e.get('outcome') == 'win')
    losses = sum(1 for e in matching if e.get('outcome') == 'loss')
    prec = correct / len(matching) * 100
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    return {
        'precision': round(prec, 1), 'win_rate': round(wr, 1),
        'n': len(matching), 'correct': correct, 'wrong': len(matching) - correct,
        'wins': wins, 'losses': losses,
        'coverage': round(len(matching) / len(events) * 100, 1),
    }


def greedy_search(events, direction, min_n=15, max_filters=7):
    all_rules = build_rules(events)
    logger.info(f"      {len(all_rules)} possible rules")

    best = None
    best_prec = 0
    for r in all_rules:
        res = eval_rules(events, [r], direction)
        if res and res['precision'] > best_prec and res['n'] >= min_n:
            best_prec = res['precision']
            best = r
    if not best:
        return [], None, []

    chosen = [best]
    cur_res = eval_rules(events, chosen, direction)
    steps = [(dict(best), dict(cur_res))]
    used = {best['ind']}

    for step in range(2, max_filters + 1):
        best_next, best_next_prec, best_next_res = None, cur_res['precision'], None
        for r in all_rules:
            if r['ind'] in used: continue
            res = eval_rules(events, chosen + [r], direction)
            if res and res['precision'] > best_next_prec and res['n'] >= min_n:
                best_next_prec = res['precision']
                best_next = r
                best_next_res = res
        if not best_next or best_next_prec <= cur_res['precision'] + 0.2:
            break
        chosen.append(best_next)
        used.add(best_next['ind'])
        cur_res = best_next_res
        steps.append((dict(best_next), dict(cur_res)))

    return chosen, cur_res, steps


# ========================================================================
# PHASE 4: TP/SL SIMULATION
# ========================================================================

def simulate_tp_sl(conn, events, tp_pct, sl_pct, max_dur_min):
    cur = conn.cursor()
    wins, losses, neutral = 0, 0, 0
    results = []

    for event in events:
        symbol = event['symbol']
        entry_time = event['time']
        direction = event['direction']

        cur.execute("""
            SELECT bucket, open, high, low, close FROM agg_5m
            WHERE symbol=%s AND bucket >= %s AND bucket < %s + INTERVAL '%s minutes'
            ORDER BY bucket
        """, (symbol, entry_time, entry_time, max_dur_min))
        candles = cur.fetchall()
        if len(candles) < 2: continue

        ep = float(candles[0]['open'])
        if ep <= 0: continue

        outcome = 'neutral'
        for c in candles:
            h, l = float(c['high']), float(c['low'])
            if direction == 'long':
                if l <= ep * (1 - sl_pct/100): outcome = 'loss'; break
                if h >= ep * (1 + tp_pct/100): outcome = 'win'; break
            else:
                if h >= ep * (1 + sl_pct/100): outcome = 'loss'; break
                if l <= ep * (1 - tp_pct/100): outcome = 'win'; break

        r = dict(event)
        r['outcome'] = outcome
        results.append(r)
        if outcome == 'win': wins += 1
        elif outcome == 'loss': losses += 1
        else: neutral += 1

    return results, wins, losses, neutral


# ========================================================================
# PHASE 5: GEGENPROBE
# ========================================================================

def counter_check(conn, rules, direction, days, min_pct):
    """
    Nimmt das Filterset und prüft: Wenn die Indikatoren auf BELIEBIGE
    Zeitpunkte angewandt werden (nicht nur Events), wie oft folgt
    tatsächlich ein ≥min_pct% Event in der richtigen Richtung?
    """
    cur = conn.cursor()
    logger.info(f"\n      [COUNTER-CHECK] {direction.upper()}: Sampling random non-event timepoints...")

    # Zufällige Symbole + Zeitpunkte samplen
    cur.execute("SELECT DISTINCT symbol FROM agg_1h WHERE bucket >= NOW() - INTERVAL '%s days'", (days,))
    symbols = [r['symbol'] for r in cur.fetchall()]

    # Systematisch alle 2h samplen (statt random, für Reproduzierbarkeit)
    test_points = []
    t0 = time.time()
    sample_count = 0
    matches = 0
    true_positives = 0
    false_positives = 0

    for si, symbol in enumerate(symbols):
        if si % 50 == 0:
            logger.info(f"      Counter-check: {si}/{len(symbols)} symbols, "
                        f"{matches} matches, {true_positives} TP, {false_positives} FP "
                        f"({time.time()-t0:.0f}s)")

        # Alle 2h einen Testpunkt
        cur.execute("""
            SELECT bucket FROM agg_2h
            WHERE symbol = %s AND bucket >= NOW() - INTERVAL '%s days'
            ORDER BY bucket
        """, (symbol, days))
        timepoints = [r['bucket'] for r in cur.fetchall()]

        for tp_time in timepoints:
            sample_count += 1
            # Indikatoren berechnen
            ind = compute_indicators(conn, symbol, tp_time)
            if ind is None:
                continue

            fake_event = {'indicators': ind, 'direction': direction}

            # Prüfe ob alle Filter-Regeln matchen
            if not all(match_rule(fake_event, r) for r in rules):
                continue

            matches += 1

            # Filter matcht! Prüfe ob tatsächlich ein ≥min_pct% Event folgt
            cur.execute("""
                SELECT bucket, open, high, low, close FROM agg_5m
                WHERE symbol = %s AND bucket >= %s AND bucket < %s + INTERVAL '720 minutes'
                ORDER BY bucket
            """, (symbol, tp_time, tp_time))
            candles = cur.fetchall()
            if len(candles) < 2:
                false_positives += 1
                continue

            ep = float(candles[0]['open'])
            if ep <= 0:
                false_positives += 1
                continue

            max_up = max((float(c['high']) - ep) / ep * 100 for c in candles)
            max_down = max((ep - float(c['low'])) / ep * 100 for c in candles)

            if direction == 'long' and max_up >= min_pct:
                true_positives += 1
            elif direction == 'short' and max_down >= min_pct:
                true_positives += 1
            else:
                false_positives += 1

    precision = true_positives / matches * 100 if matches > 0 else 0
    logger.info(f"      Counter-check DONE: {sample_count} samples, {matches} filter matches, "
                f"{true_positives} TP, {false_positives} FP → Real Precision: {precision:.1f}%")

    return {
        'samples': sample_count,
        'matches': matches,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'real_precision': round(precision, 1),
    }


# ========================================================================
# MAIN
# ========================================================================

def run(days=LOOKBACK_DAYS, min_move=MIN_MOVE_PCT):
    T0 = time.time()
    logger.info("=" * 70)
    logger.info("PRECISION SCANNER v2")
    logger.info(f"Period: {days} days | Min move: ±{min_move}%")
    logger.info(f"TP/SL: {TP_SL_COMBOS}")
    logger.info(f"Durations: {DURATION_MINUTES}")
    logger.info("=" * 70)

    conn = coins_db()

    # PHASE 1
    events = find_events(conn, days, min_move)
    if not events:
        logger.error("No events found!")
        return

    # PHASE 2
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: INDICATORS")
    logger.info(f"  Computing ~80 indicators for {len(events)} events...")
    logger.info("=" * 70)

    t0 = time.time()
    good = []
    for i, ev in enumerate(events):
        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(events) - i) / rate if rate > 0 else 0
            logger.info(f"  [{i}/{len(events)}] {len(good)} with indicators | "
                        f"{rate:.0f}/s | ETA {remaining/60:.1f}min")
        ind = compute_indicators(conn, ev['symbol'], ev['time'])
        if ind:
            ev['indicators'] = ind
            good.append(ev)

    events = good
    n_ind = len(events[0]['indicators']) if events else 0
    logger.info(f"  Done: {len(events)} events with {n_ind} indicators ({time.time()-t0:.0f}s)")

    # PHASE 3 + 4: Pro TP/SL + Duration Combo
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3+4: FILTER SEARCH + TP/SL SIMULATION")
    logger.info(f"  {len(TP_SL_COMBOS)} TP/SL × {len(DURATION_MINUTES)} durations = "
                f"{len(TP_SL_COMBOS) * len(DURATION_MINUTES)} combinations")
    logger.info("=" * 70)

    all_configs = []
    combo_num = 0
    total_combos = len(TP_SL_COMBOS) * len(DURATION_MINUTES)

    for tp, sl in TP_SL_COMBOS:
        for dur_min in DURATION_MINUTES:
            combo_num += 1
            dur_h = dur_min / 60
            logger.info(f"\n  [{combo_num}/{total_combos}] TP={tp}%/SL={sl}%/Dur={dur_h:.0f}h")

            dur_events = [e for e in events if e['duration_min'] <= dur_min]
            if len(dur_events) < 50:
                logger.info(f"    Skip: nur {len(dur_events)} events")
                continue

            # TP/SL Simulation
            t0 = time.time()
            results, wins, losses, neutral = simulate_tp_sl(conn, dur_events, tp, sl, dur_min)
            total_resolved = wins + losses
            if total_resolved < 30:
                logger.info(f"    Skip: nur {total_resolved} resolved")
                continue

            base_wr = wins / total_resolved * 100
            logger.info(f"    Sim: {wins}W/{losses}L/{neutral}N = {base_wr:.1f}% WR ({time.time()-t0:.0f}s)")

            resolved = [r for r in results if r['outcome'] in ('win', 'loss')]

            # Filter search LONG
            logger.info(f"    [LONG filter]")
            l_rules, l_res, l_steps = greedy_search(resolved, 'long')
            if l_res:
                logger.info(f"    → LONG: Prec={l_res['precision']}% WR={l_res['win_rate']}% n={l_res['n']}")
                for rule, res in l_steps:
                    logger.info(f"      {rule['label']} → prec={res['precision']}% n={res['n']}")

            # Filter search SHORT
            logger.info(f"    [SHORT filter]")
            s_rules, s_res, s_steps = greedy_search(resolved, 'short')
            if s_res:
                logger.info(f"    → SHORT: Prec={s_res['precision']}% WR={s_res['win_rate']}% n={s_res['n']}")
                for rule, res in s_steps:
                    logger.info(f"      {rule['label']} → prec={res['precision']}% n={res['n']}")

            # Cross-Validation
            mid = len(resolved) // 2
            l_cv1 = eval_rules(resolved[:mid], l_rules, 'long') if l_rules else None
            l_cv2 = eval_rules(resolved[mid:], l_rules, 'long') if l_rules else None
            s_cv1 = eval_rules(resolved[:mid], s_rules, 'short') if s_rules else None
            s_cv2 = eval_rules(resolved[mid:], s_rules, 'short') if s_rules else None

            if l_cv1 and l_cv2:
                logger.info(f"    LONG CV: {l_cv1['precision']}% (n={l_cv1['n']}) / {l_cv2['precision']}% (n={l_cv2['n']})")
            if s_cv1 and s_cv2:
                logger.info(f"    SHORT CV: {s_cv1['precision']}% (n={s_cv1['n']}) / {s_cv2['precision']}% (n={s_cv2['n']})")

            # EV
            l_ev = 0
            if l_res and (l_res['wins'] + l_res['losses']) > 0:
                wr = l_res['wins'] / (l_res['wins'] + l_res['losses'])
                l_ev = wr * tp - (1-wr) * sl
            s_ev = 0
            if s_res and (s_res['wins'] + s_res['losses']) > 0:
                wr = s_res['wins'] / (s_res['wins'] + s_res['losses'])
                s_ev = wr * tp - (1-wr) * sl

            # Score
            l_score = l_res['precision'] * np.sqrt(l_res['n']) if l_res and l_res['n'] >= 15 else 0
            s_score = s_res['precision'] * np.sqrt(s_res['n']) if s_res and s_res['n'] >= 15 else 0

            all_configs.append({
                'tp': tp, 'sl': sl, 'dur_min': dur_min,
                'base_wr': round(base_wr, 1), 'n_events': len(dur_events), 'n_resolved': len(resolved),
                'long': {'rules': l_rules, 'res': l_res, 'cv1': l_cv1, 'cv2': l_cv2,
                         'ev': round(l_ev, 3), 'score': round(l_score, 1), 'steps': l_steps},
                'short': {'rules': s_rules, 'res': s_res, 'cv1': s_cv1, 'cv2': s_cv2,
                          'ev': round(s_ev, 3), 'score': round(s_score, 1), 'steps': s_steps},
                'combined_score': round(l_score + s_score, 1),
            })

    if not all_configs:
        logger.error("No valid configurations!")
        conn.close()
        return

    # Sortieren
    all_configs.sort(key=lambda c: -c['combined_score'])

    # PHASE 5: GEGENPROBE für Top 3
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: GEGENPROBE (Counter-Check)")
    logger.info("  Top Filtersets auf ALLE Zeitpunkte anwenden → False Positive Rate")
    logger.info("=" * 70)

    for rank, cfg in enumerate(all_configs[:3], 1):
        dur_h = cfg['dur_min'] / 60
        logger.info(f"\n  #{rank} TP={cfg['tp']}%/SL={cfg['sl']}%/Dur={dur_h:.0f}h")

        for direction in ['long', 'short']:
            d = cfg[direction]
            if not d['rules']:
                continue
            logger.info(f"    {direction.upper()} filters: {[r['label'] for r in d['rules']]}")
            cc = counter_check(conn, d['rules'], direction, days, min_move)
            d['counter_check'] = cc
            logger.info(f"    → Real Precision: {cc['real_precision']}% "
                        f"({cc['true_positives']}/{cc['matches']} matches)")

    conn.close()

    # ========================================================================
    # REPORT
    # ========================================================================
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORT_DIR, f'precision_report_{ts}.txt')
    lines = []
    def w(s=''):
        lines.append(s)
        logger.info(s)

    w("=" * 80)
    w("PRECISION SCANNER v2 REPORT")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Period: {days} days | Min move: ±{min_move}%")
    w(f"Runtime: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)")
    w(f"Events: {len(events)} | Indicators per event: {n_ind}")
    w("=" * 80)

    w(f"\nTOP 10 CONFIGURATIONS")
    w("=" * 80)

    for rank, cfg in enumerate(all_configs[:10], 1):
        dur_h = cfg['dur_min'] / 60
        w(f"\n{'─'*80}")
        w(f"#{rank}  TP={cfg['tp']}% / SL={cfg['sl']}% / Duration={dur_h:.0f}h  "
          f"(Score={cfg['combined_score']})")
        w(f"    Base WR={cfg['base_wr']}% | Events={cfg['n_events']} | Resolved={cfg['n_resolved']}")

        for direction in ['long', 'short']:
            d = cfg[direction]
            if not d['res']:
                w(f"\n  {direction.upper()}: No filter found")
                continue
            r = d['res']
            w(f"\n  {direction.upper()}: Prec={r['precision']}% | WR={r['win_rate']}% | "
              f"n={r['n']} | EV={d['ev']:+.3f}%")
            for rule, res in d['steps']:
                w(f"    → {rule['label']}  (prec={res['precision']}%, n={res['n']})")
            if d['cv1'] and d['cv2']:
                w(f"    CV: {d['cv1']['precision']}% (n={d['cv1']['n']}) / "
                  f"{d['cv2']['precision']}% (n={d['cv2']['n']})")
            cc = d.get('counter_check')
            if cc:
                w(f"    COUNTER-CHECK: {cc['real_precision']}% real precision "
                  f"({cc['true_positives']} TP / {cc['false_positives']} FP aus {cc['matches']} matches, "
                  f"{cc['samples']} samples)")

    # Best EV
    w(f"\n{'='*80}")
    w("BEST EXPECTED VALUE")
    w("=" * 80)
    bl = max(all_configs, key=lambda c: c['long']['ev'])
    bs = max(all_configs, key=lambda c: c['short']['ev'])
    if bl['long']['res']:
        r = bl['long']['res']
        w(f"LONG:  TP={bl['tp']}%/SL={bl['sl']}%/Dur={bl['dur_min']//60}h "
          f"→ EV={bl['long']['ev']:+.3f}% | Prec={r['precision']}% | WR={r['win_rate']}% | n={r['n']}")
    if bs['short']['res']:
        r = bs['short']['res']
        w(f"SHORT: TP={bs['tp']}%/SL={bs['sl']}%/Dur={bs['dur_min']//60}h "
          f"→ EV={bs['short']['ev']:+.3f}% | Prec={r['precision']}% | WR={r['win_rate']}% | n={r['n']}")

    w(f"\n{'='*80}")
    w("END OF REPORT")
    w("=" * 80)

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"\nReport: {report_path}")


if __name__ == '__main__':
    d = int(sys.argv[1]) if len(sys.argv) > 1 else LOOKBACK_DAYS
    m = float(sys.argv[2]) if len(sys.argv) > 2 else MIN_MOVE_PCT
    run(d, m)
