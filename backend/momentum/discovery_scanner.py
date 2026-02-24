#!/usr/bin/env python3
"""
DISCOVERY SCANNER
Findet alle ≥4% Moves, berechnet Indikatoren VOR dem Move,
und sucht die beste Kombination um Long von Short zu trennen.
"""

import sys, os, json, logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

LOG_FILE = '/opt/coin/logs/discovery_scanner.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger('discovery')

def db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)

# === INDICATOR FUNCTIONS ===
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_ema(values, period):
    if len(values) < period: return None
    m = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]: ema = (v - ema) * m + ema
    return ema

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return None
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
    return np.mean(trs[-period:])

def calc_bb_width(closes, period=20):
    """Bollinger Band Breite - misst Volatilität"""
    if len(closes) < period: return None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    if sma == 0: return None
    return (std * 2 / sma) * 100  # In Prozent

def calc_macd(closes):
    """MACD Histogram"""
    if len(closes) < 26: return None, None
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    if ema12 is None or ema26 is None: return None, None
    macd_line = ema12 - ema26
    # Signal braucht eigentlich 9er EMA über MACD History - vereinfacht
    return macd_line, None

def calc_volume_profile(volumes, n_recent=3, n_avg=20):
    """Volume Ratio + Volume Trend"""
    if len(volumes) < n_avg: return 1.0, 0
    avg = np.mean(volumes[-n_avg:])
    recent = np.mean(volumes[-n_recent:])
    ratio = recent / avg if avg > 0 else 1.0
    # Trend: steigen Volumes?
    if len(volumes) >= 6:
        v_first = np.mean(volumes[-6:-3])
        v_last = np.mean(volumes[-3:])
        v_trend = ((v_last - v_first) / v_first * 100) if v_first > 0 else 0
    else:
        v_trend = 0
    return ratio, v_trend

def calc_candle_body_ratio(opens, highs, lows, closes, n=6):
    """Durchschnittliches Body/Wick Verhältnis der letzten n Candles"""
    if len(opens) < n: return None
    ratios = []
    for i in range(-n, 0):
        full = highs[i] - lows[i]
        if full <= 0: continue
        body = abs(closes[i] - opens[i])
        ratios.append(body / full)
    return np.mean(ratios) if ratios else None

def calc_consecutive_direction(closes, n=10):
    """Wie viele der letzten n Candles in gleiche Richtung?"""
    if len(closes) < n + 1: return 0, 0
    ups = sum(1 for i in range(-n, 0) if closes[i] > closes[i-1])
    downs = n - ups
    return ups, downs


# === SCHRITT 1: EVENTS FINDEN ===
def find_moves(conn, min_pct=4.0, days=30):
    """Findet alle ≥min_pct% Moves in 4h Fenstern"""
    cur = conn.cursor()
    logger.info(f"Finding ≥{min_pct}% moves in last {days} days...")
    
    cur.execute("""
        SELECT symbol, bucket, open, close, high, low, volume,
            ((close - open) / NULLIF(open, 0)) * 100 as pct_change
        FROM agg_4h
        WHERE bucket >= NOW() - INTERVAL '%s days'
          AND open > 0
          AND ABS((close - open) / NULLIF(open, 0)) * 100 >= %s
        ORDER BY bucket
    """, (days, min_pct))
    
    moves = cur.fetchall()
    up = [m for m in moves if m['pct_change'] > 0]
    down = [m for m in moves if m['pct_change'] < 0]
    logger.info(f"Found {len(moves)} moves: {len(up)} UP, {len(down)} DOWN")
    return moves


# === SCHRITT 2: INDIKATOREN VOR DEM MOVE BERECHNEN ===
def compute_indicators_before(conn, symbol, move_time):
    """Berechnet ALLE Indikatoren zum Zeitpunkt VOR dem Move"""
    cur = conn.cursor()
    
    # 1h Candles laden (50 vor dem Move)
    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_1h WHERE symbol = %s AND bucket < %s
        ORDER BY bucket DESC LIMIT 50
    """, (symbol, move_time))
    rows_1h = cur.fetchall()
    rows_1h.reverse()
    if len(rows_1h) < 30: return None
    
    # 4h Candles
    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_4h WHERE symbol = %s AND bucket < %s
        ORDER BY bucket DESC LIMIT 20
    """, (symbol, move_time))
    rows_4h = cur.fetchall()
    rows_4h.reverse()
    if len(rows_4h) < 10: return None
    
    # 1d Candles
    cur.execute("""
        SELECT bucket, open, high, low, close, volume
        FROM agg_1d WHERE symbol = %s AND bucket < %s
        ORDER BY bucket DESC LIMIT 15
    """, (symbol, move_time))
    rows_1d = cur.fetchall()
    rows_1d.reverse()
    if len(rows_1d) < 5: return None
    
    closes_1h = [r['close'] for r in rows_1h]
    highs_1h = [r['high'] for r in rows_1h]
    lows_1h = [r['low'] for r in rows_1h]
    opens_1h = [r['open'] for r in rows_1h]
    volumes_1h = [r['volume'] for r in rows_1h]
    closes_4h = [r['close'] for r in rows_4h]
    closes_1d = [r['close'] for r in rows_1d]
    volumes_1d = [r['volume'] for r in rows_1d]
    
    current = closes_1h[-1]
    if not current or current <= 0: return None
    
    # --- ALLE INDIKATOREN ---
    rsi_1h = calc_rsi(closes_1h, 14)
    if rsi_1h is None: return None
    
    ema_9 = calc_ema(closes_1h, 9)
    ema_21 = calc_ema(closes_1h, 21)
    ema_50 = calc_ema(closes_1h[-50:], 50) if len(closes_1h) >= 50 else None
    
    # EMA Abstand in %
    ema_9_21_pct = ((ema_9 - ema_21) / ema_21 * 100) if ema_9 and ema_21 and ema_21 > 0 else 0
    ema_price_50_pct = ((current - ema_50) / ema_50 * 100) if ema_50 and ema_50 > 0 else 0
    
    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    atr_pct = (atr / current * 100) if atr and current > 0 else 0
    
    vol_ratio, vol_trend = calc_volume_profile(volumes_1h)
    
    # Trend 4h
    trend_4h = 0
    if len(closes_4h) >= 5:
        r5 = closes_4h[-5:]
        rising = sum(1 for i in range(1, len(r5)) if r5[i] > r5[i-1])
        trend_4h = (rising / 4) * 2 - 1
    
    # Trend 1d
    trend_1d = 0
    if len(closes_1d) >= 5:
        r5 = closes_1d[-5:]
        rising = sum(1 for i in range(1, len(r5)) if r5[i] > r5[i-1])
        trend_1d = (rising / 4) * 2 - 1
    
    # HH/HL
    hh_hl = 0
    if len(highs_1h) >= 6:
        rh = highs_1h[-6:]
        rl = lows_1h[-6:]
        hh = sum(1 for i in range(1, 6) if rh[i] > rh[i-1])
        hl = sum(1 for i in range(1, 6) if rl[i] > rl[i-1])
        hh_hl = (hh + hl) / 10
    
    # Momentum
    pct_6h = ((closes_1h[-1] - closes_1h[-6]) / closes_1h[-6] * 100) if closes_1h[-6] > 0 else 0
    pct_12h = ((closes_1h[-1] - closes_1h[-12]) / closes_1h[-12] * 100) if len(closes_1h) >= 12 and closes_1h[-12] > 0 else 0
    pct_24h = ((closes_1h[-1] - closes_1h[-24]) / closes_1h[-24] * 100) if len(closes_1h) >= 24 and closes_1h[-24] > 0 else 0
    
    # Bollinger Breite
    bb_width = calc_bb_width(closes_1h) or 0
    
    # MACD
    macd_line, _ = calc_macd(closes_1h)
    macd_pct = (macd_line / current * 100) if macd_line and current > 0 else 0
    
    # Candle Body Ratio
    body_ratio = calc_candle_body_ratio(opens_1h, highs_1h, lows_1h, closes_1h) or 0
    
    # Consecutive Direction
    ups, downs = calc_consecutive_direction(closes_1h)
    
    # kline_metrics
    cur.execute("""
        SELECT pct_30m, pct_60m, pct_120m, pct_360m
        FROM kline_metrics WHERE symbol = %s AND open_time <= %s
        ORDER BY open_time DESC LIMIT 1
    """, (symbol, move_time))
    km = cur.fetchone()
    
    return {
        'rsi_1h': round(rsi_1h, 1),
        'ema_9_21_pct': round(ema_9_21_pct, 3),
        'ema_price_50_pct': round(ema_price_50_pct, 3),
        'atr_pct': round(atr_pct, 3),
        'vol_ratio': round(vol_ratio, 2),
        'vol_trend': round(vol_trend, 1),
        'trend_4h': round(trend_4h, 2),
        'trend_1d': round(trend_1d, 2),
        'hh_hl': round(hh_hl, 2),
        'pct_6h': round(pct_6h, 2),
        'pct_12h': round(pct_12h, 2),
        'pct_24h': round(pct_24h, 2),
        'bb_width': round(bb_width, 3),
        'macd_pct': round(macd_pct, 4),
        'body_ratio': round(body_ratio, 3),
        'consecutive_ups': ups,
        'consecutive_downs': downs,
        'pct_30m': float(km['pct_30m']) if km and km['pct_30m'] is not None else None,
        'pct_60m': float(km['pct_60m']) if km and km['pct_60m'] is not None else None,
        'pct_120m': float(km['pct_120m']) if km and km['pct_120m'] is not None else None,
        'pct_360m': float(km['pct_360m']) if km and km['pct_360m'] is not None else None,
    }


# === SCHRITT 3: FILTER SUCHEN ===
def find_best_filters(events):
    """Systematisch alle Indikator-Schwellen testen um Long von Short zu trennen"""
    
    longs = [e for e in events if e['actual'] == 'long']
    shorts = [e for e in events if e['actual'] == 'short']
    
    logger.info(f"\nSearching best filters to separate {len(longs)} LONG from {len(shorts)} SHORT moves...")
    
    # Alle numerischen Indikatoren
    indicators = [
        'rsi_1h', 'ema_9_21_pct', 'ema_price_50_pct', 'atr_pct',
        'vol_ratio', 'vol_trend', 'trend_4h', 'trend_1d', 'hh_hl',
        'pct_6h', 'pct_12h', 'pct_24h', 'bb_width', 'macd_pct',
        'body_ratio', 'consecutive_ups', 'consecutive_downs',
        'pct_30m', 'pct_60m', 'pct_120m', 'pct_360m',
    ]
    
    # === SINGLE INDICATOR ANALYSIS ===
    logger.info(f"\n{'='*70}")
    logger.info(f"SINGLE INDICATOR ANALYSIS")
    logger.info(f"{'='*70}")
    
    single_results = []
    
    for ind in indicators:
        l_vals = [e['indicators'][ind] for e in longs if e['indicators'].get(ind) is not None]
        s_vals = [e['indicators'][ind] for e in shorts if e['indicators'].get(ind) is not None]
        
        if not l_vals or not s_vals:
            continue
        
        l_avg = np.mean(l_vals)
        s_avg = np.mean(s_vals)
        l_med = np.median(l_vals)
        s_med = np.median(s_vals)
        
        # Separation Score: Wie gut trennt dieser Indikator?
        combined = l_vals + s_vals
        combined_std = np.std(combined) if len(combined) > 1 else 1
        separation = abs(l_avg - s_avg) / combined_std if combined_std > 0 else 0
        
        single_results.append((ind, l_avg, s_avg, l_med, s_med, separation, len(l_vals), len(s_vals)))
    
    single_results.sort(key=lambda x: -x[5])
    
    logger.info(f"\n  {'Indicator':22s} {'Long avg':>10s} {'Short avg':>10s} {'Long med':>10s} {'Short med':>10s} {'Separation':>10s}")
    logger.info(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for ind, la, sa, lm, sm, sep, ln, sn in single_results:
        logger.info(f"  {ind:22s} {la:>+10.3f} {sa:>+10.3f} {lm:>+10.3f} {sm:>+10.3f} {sep:>10.3f}  (n={ln}/{sn})")
    
    # === THRESHOLD SEARCH: Jeder Indikator, viele Schwellen ===
    logger.info(f"\n{'='*70}")
    logger.info(f"THRESHOLD SEARCH: Best single-indicator filters")
    logger.info(f"{'='*70}")
    
    best_filters = []
    
    for ind in indicators:
        all_vals = []
        for e in events:
            v = e['indicators'].get(ind)
            if v is not None:
                all_vals.append((v, e['actual']))
        
        if len(all_vals) < 50:
            continue
        
        vals_only = [v for v, _ in all_vals]
        # Test Percentile-basierte Schwellen
        for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
            thresh = np.percentile(vals_only, pct)
            
            # Filter: ind >= thresh → predict Long
            above_long = len([v for v, a in all_vals if v >= thresh and a == 'long'])
            above_short = len([v for v, a in all_vals if v >= thresh and a == 'short'])
            below_long = len([v for v, a in all_vals if v < thresh and a == 'long'])
            below_short = len([v for v, a in all_vals if v < thresh and a == 'short'])
            
            above_total = above_long + above_short
            below_total = below_long + below_short
            
            if above_total >= 20 and below_total >= 20:
                long_precision = above_long / above_total * 100 if above_total > 0 else 0
                short_precision = below_short / below_total * 100 if below_total > 0 else 0
                avg_precision = (long_precision + short_precision) / 2
                
                if avg_precision >= 60:
                    best_filters.append({
                        'indicator': ind,
                        'threshold': round(thresh, 4),
                        'rule': f'{ind} >= {thresh:.4f} → Long',
                        'long_precision': round(long_precision, 1),
                        'short_precision': round(short_precision, 1),
                        'avg_precision': round(avg_precision, 1),
                        'long_n': above_total,
                        'short_n': below_total,
                    })
    
    best_filters.sort(key=lambda x: -x['avg_precision'])
    
    logger.info(f"\n  Top single filters (avg precision ≥ 60%):")
    logger.info(f"  {'Rule':45s} {'Long%':>6s} {'Short%':>7s} {'Avg%':>5s} {'nL':>4s} {'nS':>4s}")
    logger.info(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*5} {'-'*4} {'-'*4}")
    for f in best_filters[:30]:
        logger.info(f"  {f['rule']:45s} {f['long_precision']:>5.1f}% {f['short_precision']:>6.1f}% {f['avg_precision']:>4.1f}% {f['long_n']:>4d} {f['short_n']:>4d}")
    
    # === MULTI-INDICATOR COMBINATIONS ===
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-INDICATOR COMBINATIONS (Top 2-3 filters)")
    logger.info(f"{'='*70}")
    
    # Nimm die top 10 Single-Filters und kombiniere sie
    top_singles = best_filters[:15]
    
    combo_results = []
    
    for i, f1 in enumerate(top_singles):
        for j, f2 in enumerate(top_singles):
            if j <= i: continue
            if f1['indicator'] == f2['indicator']: continue
            
            # Beide Filter zusammen anwenden
            ind1, t1 = f1['indicator'], f1['threshold']
            ind2, t2 = f2['indicator'], f2['threshold']
            
            # Long: beide >= threshold, Short: beide < threshold
            correct_long = 0
            correct_short = 0
            total_long_pred = 0
            total_short_pred = 0
            
            for e in events:
                v1 = e['indicators'].get(ind1)
                v2 = e['indicators'].get(ind2)
                if v1 is None or v2 is None: continue
                
                if v1 >= t1 and v2 >= t2:
                    total_long_pred += 1
                    if e['actual'] == 'long': correct_long += 1
                elif v1 < t1 and v2 < t2:
                    total_short_pred += 1
                    if e['actual'] == 'short': correct_short += 1
            
            if total_long_pred >= 15 and total_short_pred >= 15:
                lp = correct_long / total_long_pred * 100
                sp = correct_short / total_short_pred * 100
                ap = (lp + sp) / 2
                coverage = (total_long_pred + total_short_pred) / len(events) * 100
                
                if ap >= 65:
                    combo_results.append({
                        'filters': f'{ind1}>={t1:.3f} AND {ind2}>={t2:.3f}',
                        'long_pct': round(lp, 1),
                        'short_pct': round(sp, 1),
                        'avg_pct': round(ap, 1),
                        'coverage': round(coverage, 1),
                        'n_long': total_long_pred,
                        'n_short': total_short_pred,
                    })
    
    combo_results.sort(key=lambda x: -x['avg_pct'])
    
    logger.info(f"\n  Top combinations (avg ≥ 65%):")
    logger.info(f"  {'Filters':60s} {'L%':>5s} {'S%':>5s} {'Avg':>5s} {'Cov':>5s} {'nL':>4s} {'nS':>4s}")
    logger.info(f"  {'-'*60} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*4}")
    for c in combo_results[:20]:
        logger.info(f"  {c['filters']:60s} {c['long_pct']:>4.1f}% {c['short_pct']:>4.1f}% {c['avg_pct']:>4.1f}% {c['coverage']:>4.1f}% {c['n_long']:>4d} {c['n_short']:>4d}")
    
    return best_filters, combo_results


# === MAIN ===
def run_discovery(days=30, min_pct=4.0):
    logger.info(f"{'='*70}")
    logger.info(f"DISCOVERY SCANNER")
    logger.info(f"Finding indicators that predict ≥{min_pct}% moves")
    logger.info(f"Period: last {days} days")
    logger.info(f"{'='*70}")
    
    conn = db()
    
    # 1. Alle Moves finden
    moves = find_moves(conn, min_pct, days)
    
    # 2. Für jeden Move: Indikatoren VOR dem Move berechnen
    events = []
    skipped = 0
    
    for i, m in enumerate(moves):
        if i % 500 == 0 and i > 0:
            logger.info(f"[PROGRESS] {i}/{len(moves)} moves processed, {len(events)} with indicators")
        
        indicators = compute_indicators_before(conn, m['symbol'], m['bucket'])
        if indicators is None:
            skipped += 1
            continue
        
        events.append({
            'symbol': m['symbol'],
            'time': m['bucket'],
            'pct_change': float(m['pct_change']),
            'actual': 'long' if m['pct_change'] > 0 else 'short',
            'indicators': indicators,
        })
    
    logger.info(f"\nProcessed {len(moves)} moves → {len(events)} with indicators ({skipped} skipped)")
    
    longs = [e for e in events if e['actual'] == 'long']
    shorts = [e for e in events if e['actual'] == 'short']
    logger.info(f"LONG moves: {len(longs)}, SHORT moves: {len(shorts)}")
    
    # 3. Beste Filter finden
    singles, combos = find_best_filters(events)
    
    conn.close()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DISCOVERY COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    run_discovery(days, min_pct)
