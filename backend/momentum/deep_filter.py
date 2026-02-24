#!/usr/bin/env python3
"""
DEEP FILTER SCANNER
1. Sammelt alle ≥4% Moves (Long+Short) der letzten N Tage
2. Berechnet alle Indikatoren VOR dem Move
3. Sucht die beste Multi-Indikator-Kombination die Long von Short trennt
4. Gegenprobe: Erfasst der Long-Filter auch negative Moves? → schlecht
5. Iterativ: Fügt solange Filter hinzu bis Precision nicht mehr steigt
"""

import sys, os, json, logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from itertools import combinations
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

LOG_FILE = '/opt/coin/logs/deep_filter.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger('deepfilter')

def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


# === INDIKATOR-FUNKTIONEN ===
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

def calc_bb_position(closes, period=20):
    """Wo steht der Preis innerhalb der Bollinger Bänder? 0=unteres Band, 1=oberes"""
    if len(closes) < period: return None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    if std == 0: return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5

def calc_bb_width(closes, period=20):
    if len(closes) < period: return None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return (std * 4 / sma * 100) if sma > 0 else None

def calc_stoch_rsi(closes, period=14, smooth_k=3, smooth_d=3):
    """Stochastic RSI"""
    if len(closes) < period + smooth_k + smooth_d: return None, None
    rsis = []
    for i in range(period + smooth_k + smooth_d, 0, -1):
        r = calc_rsi(closes[:-i] if i > 0 else closes, period)
        if r is not None: rsis.append(r)
    if len(rsis) < period: return None, None
    min_rsi = min(rsis[-period:])
    max_rsi = max(rsis[-period:])
    if max_rsi == min_rsi: return 50, 50
    stoch_k = (rsis[-1] - min_rsi) / (max_rsi - min_rsi) * 100
    return stoch_k, None

def calc_obv_trend(closes, volumes, n=10):
    """On-Balance Volume Trend: steigt OBV oder fällt es?"""
    if len(closes) < n + 1 or len(volumes) < n + 1: return 0
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
        else: obv.append(obv[-1])
    if len(obv) < n: return 0
    obv_recent = obv[-n:]
    # Normalisierte Steigung
    x = np.arange(n)
    slope = np.polyfit(x, obv_recent, 1)[0]
    avg_vol = np.mean(volumes[-n:])
    return slope / avg_vol if avg_vol > 0 else 0

def calc_vwap_distance(closes, volumes, n=20):
    """Abstand zum VWAP in %"""
    if len(closes) < n or len(volumes) < n: return None
    c = closes[-n:]
    v = volumes[-n:]
    total_vol = sum(v)
    if total_vol == 0: return 0
    vwap = sum(c[i] * v[i] for i in range(n)) / total_vol
    return ((closes[-1] - vwap) / vwap * 100) if vwap > 0 else 0

def calc_range_position(highs, lows, closes, n=24):
    """Wo steht der Preis im High-Low Range der letzten n Candles? 0=Low, 1=High"""
    if len(highs) < n: return None
    highest = max(highs[-n:])
    lowest = min(lows[-n:])
    if highest == lowest: return 0.5
    return (closes[-1] - lowest) / (highest - lowest)

def calc_candle_patterns(opens, highs, lows, closes, n=6):
    """Diverse Candle-Pattern Metriken"""
    if len(opens) < n: return {}
    
    body_ratios = []
    upper_wicks = []
    lower_wicks = []
    directions = []  # 1=bullish, -1=bearish
    
    for i in range(-n, 0):
        full = highs[i] - lows[i]
        if full <= 0: continue
        body = abs(closes[i] - opens[i])
        body_ratios.append(body / full)
        
        if closes[i] >= opens[i]:  # bullish
            upper_wicks.append((highs[i] - closes[i]) / full)
            lower_wicks.append((opens[i] - lows[i]) / full)
            directions.append(1)
        else:  # bearish
            upper_wicks.append((highs[i] - opens[i]) / full)
            lower_wicks.append((closes[i] - lows[i]) / full)
            directions.append(-1)
    
    if not body_ratios: return {}
    
    return {
        'body_ratio': float(np.mean(body_ratios)),
        'upper_wick_avg': float(np.mean(upper_wicks)),
        'lower_wick_avg': float(np.mean(lower_wicks)),
        'bullish_count': sum(1 for d in directions if d > 0),
        'bearish_count': sum(1 for d in directions if d < 0),
    }


def compute_all_indicators(conn, symbol, move_time):
    """Berechnet ALLE erdenklichen Indikatoren vor einem Move"""
    cur = conn.cursor()
    
    cur.execute("SELECT bucket, open, high, low, close, volume FROM agg_1h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 50", (symbol, move_time))
    r1h = list(reversed(cur.fetchall()))
    if len(r1h) < 30: return None
    
    cur.execute("SELECT bucket, open, high, low, close, volume FROM agg_4h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 30", (symbol, move_time))
    r4h = list(reversed(cur.fetchall()))
    if len(r4h) < 10: return None
    
    cur.execute("SELECT bucket, open, high, low, close, volume FROM agg_1d WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 15", (symbol, move_time))
    r1d = list(reversed(cur.fetchall()))
    if len(r1d) < 5: return None
    
    c1h = [r['close'] for r in r1h]
    h1h = [r['high'] for r in r1h]
    l1h = [r['low'] for r in r1h]
    o1h = [r['open'] for r in r1h]
    v1h = [r['volume'] for r in r1h]
    c4h = [r['close'] for r in r4h]
    h4h = [r['high'] for r in r4h]
    l4h = [r['low'] for r in r4h]
    o4h = [r['open'] for r in r4h]
    v4h = [r['volume'] for r in r4h]
    c1d = [r['close'] for r in r1d]
    h1d = [r['high'] for r in r1d]
    l1d = [r['low'] for r in r1d]
    v1d = [r['volume'] for r in r1d]
    
    current = c1h[-1]
    if not current or current <= 0: return None
    
    rsi_1h = calc_rsi(c1h, 14)
    if rsi_1h is None: return None
    rsi_4h = calc_rsi(c4h, 14)
    
    ema_9 = calc_ema(c1h, 9)
    ema_21 = calc_ema(c1h, 21)
    ema_50 = calc_ema(c1h[-50:], 50) if len(c1h) >= 50 else None
    
    atr_1h = calc_atr(h1h, l1h, c1h, 14)
    atr_4h = calc_atr(h4h, l4h, c4h, 14) if len(c4h) >= 15 else None
    
    # Trends (verschiedene Perioden)
    def trend(closes, n=5):
        if len(closes) < n: return 0
        r = closes[-n:]
        return (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r)-1)) * 2 - 1
    
    # HH/HL
    def hh_hl(highs, lows, n=6):
        if len(highs) < n: return 0
        rh, rl = highs[-n:], lows[-n:]
        return (sum(1 for i in range(1,n) if rh[i]>rh[i-1]) + sum(1 for i in range(1,n) if rl[i]>rl[i-1])) / ((n-1)*2)
    
    # Consecutive
    def consec(closes, n=10):
        if len(closes) < n+1: return 0, 0
        ups = sum(1 for i in range(-n, 0) if closes[i] > closes[i-1])
        return ups, n - ups
    
    cu, cd = consec(c1h)
    cu4h, cd4h = consec(c4h)
    
    # Volume
    avg_vol_1h = np.mean(v1h[-20:]) if len(v1h) >= 20 else np.mean(v1h)
    recent_vol_1h = np.mean(v1h[-3:])
    vol_ratio_1h = recent_vol_1h / avg_vol_1h if avg_vol_1h > 0 else 1
    
    avg_vol_4h = np.mean(v4h[-10:]) if len(v4h) >= 10 else np.mean(v4h)
    recent_vol_4h = np.mean(v4h[-2:])
    vol_ratio_4h = recent_vol_4h / avg_vol_4h if avg_vol_4h > 0 else 1
    
    # Candle Patterns
    cp_1h = calc_candle_patterns(o1h, h1h, l1h, c1h, 6)
    cp_4h = calc_candle_patterns(o4h, h4h, l4h, c4h, 4)
    
    # Momentum verschiedene Zeiträume
    def pct(closes, n):
        if len(closes) < n+1 or closes[-n-1] == 0: return 0
        return ((closes[-1] - closes[-n-1]) / closes[-n-1]) * 100
    
    # kline_metrics
    cur.execute("SELECT pct_30m, pct_60m, pct_120m, pct_360m FROM kline_metrics WHERE symbol=%s AND open_time<=%s ORDER BY open_time DESC LIMIT 1", (symbol, move_time))
    km = cur.fetchone()
    
    stoch_k, _ = calc_stoch_rsi(c1h)
    
    ind = {
        # RSI
        'rsi_1h': round(rsi_1h, 1),
        'rsi_4h': round(rsi_4h, 1) if rsi_4h else None,
        'stoch_rsi_k': round(stoch_k, 1) if stoch_k else None,
        
        # EMA distances
        'ema_9_21_pct': round(((ema_9-ema_21)/ema_21*100), 3) if ema_9 and ema_21 and ema_21>0 else None,
        'ema_price_50_pct': round(((current-ema_50)/ema_50*100), 3) if ema_50 and ema_50>0 else None,
        
        # ATR
        'atr_1h_pct': round(atr_1h/current*100, 3) if atr_1h else None,
        'atr_4h_pct': round(atr_4h/current*100, 3) if atr_4h else None,
        
        # Trends
        'trend_4h_5': round(trend(c4h, 5), 2),
        'trend_4h_10': round(trend(c4h, 10), 2) if len(c4h) >= 10 else None,
        'trend_1d_5': round(trend(c1d, 5), 2),
        'trend_1d_10': round(trend(c1d, 10), 2) if len(c1d) >= 10 else None,
        'trend_1h_10': round(trend(c1h, 10), 2),
        'trend_1h_20': round(trend(c1h, 20), 2) if len(c1h) >= 20 else None,
        
        # HH/HL
        'hh_hl_6': round(hh_hl(h1h, l1h, 6), 2),
        'hh_hl_12': round(hh_hl(h1h, l1h, 12), 2) if len(h1h) >= 12 else None,
        'hh_hl_4h': round(hh_hl(h4h, l4h, 6), 2) if len(h4h) >= 6 else None,
        
        # Consecutive
        'consec_ups_1h': cu,
        'consec_downs_1h': cd,
        'consec_ups_4h': cu4h,
        'consec_downs_4h': cd4h,
        
        # Volume
        'vol_ratio_1h': round(vol_ratio_1h, 2),
        'vol_ratio_4h': round(vol_ratio_4h, 2),
        'obv_trend_1h': round(calc_obv_trend(c1h, v1h, 10), 4),
        'obv_trend_4h': round(calc_obv_trend(c4h, v4h, 6), 4) if len(c4h) >= 7 else None,
        
        # Bollinger
        'bb_position_1h': round(calc_bb_position(c1h), 3),
        'bb_width_1h': round(calc_bb_width(c1h), 3) if calc_bb_width(c1h) else None,
        'bb_position_4h': round(calc_bb_position(c4h), 3) if len(c4h) >= 20 else None,
        
        # VWAP
        'vwap_dist_1h': round(calc_vwap_distance(c1h, v1h, 20), 3),
        'vwap_dist_4h': round(calc_vwap_distance(c4h, v4h, 10), 3) if len(c4h) >= 10 else None,
        
        # Range Position
        'range_pos_24h': round(calc_range_position(h1h, l1h, c1h, 24), 3),
        'range_pos_7d': round(calc_range_position(h1d, l1d, c1d, 7), 3) if len(h1d) >= 7 else None,
        
        # Candle Patterns 1h
        'body_ratio_1h': round(cp_1h.get('body_ratio', 0.5), 3),
        'upper_wick_1h': round(cp_1h.get('upper_wick_avg', 0), 3),
        'lower_wick_1h': round(cp_1h.get('lower_wick_avg', 0), 3),
        'bullish_candles_1h': cp_1h.get('bullish_count', 0),
        'bearish_candles_1h': cp_1h.get('bearish_count', 0),
        
        # Candle Patterns 4h
        'body_ratio_4h': round(cp_4h.get('body_ratio', 0.5), 3),
        'upper_wick_4h': round(cp_4h.get('upper_wick_avg', 0), 3),
        'lower_wick_4h': round(cp_4h.get('lower_wick_avg', 0), 3),
        
        # Momentum
        'pct_3h': round(pct(c1h, 3), 2),
        'pct_6h': round(pct(c1h, 6), 2),
        'pct_12h': round(pct(c1h, 12), 2),
        'pct_24h': round(pct(c1h, 24), 2) if len(c1h) >= 25 else None,
        'pct_3d': round(pct(c1d, 3), 2) if len(c1d) >= 4 else None,
        'pct_7d': round(pct(c1d, 7), 2) if len(c1d) >= 8 else None,
        
        # kline_metrics
        'pct_30m': float(km['pct_30m']) if km and km['pct_30m'] is not None else None,
        'pct_60m': float(km['pct_60m']) if km and km['pct_60m'] is not None else None,
        'pct_120m': float(km['pct_120m']) if km and km['pct_120m'] is not None else None,
        'pct_360m': float(km['pct_360m']) if km and km['pct_360m'] is not None else None,
    }
    
    return ind


# === FILTER-ENGINE ===
def build_filter_rules(events):
    """Erzeugt alle möglichen Filter-Regeln für jeden Indikator"""
    indicators = set()
    for e in events:
        for k, v in e['indicators'].items():
            if v is not None: indicators.add(k)
    
    rules = []
    for ind in sorted(indicators):
        vals = [e['indicators'][ind] for e in events if e['indicators'].get(ind) is not None]
        if len(vals) < 50: continue
        
        for pct in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
            thresh = np.percentile(vals, pct)
            rules.append({
                'indicator': ind,
                'op': '>=',
                'threshold': round(float(thresh), 4),
                'label': f'{ind} >= {thresh:.4f}',
            })
            rules.append({
                'indicator': ind,
                'op': '<',
                'threshold': round(float(thresh), 4),
                'label': f'{ind} < {thresh:.4f}',
            })
    
    return rules


def apply_rule(event, rule):
    """Prüft ob ein Event eine Regel erfüllt"""
    val = event['indicators'].get(rule['indicator'])
    if val is None: return False
    if rule['op'] == '>=': return val >= rule['threshold']
    if rule['op'] == '<': return val < rule['threshold']
    return False


def evaluate_ruleset(events, rules, direction):
    """Bewertet ein Set von Regeln für eine Richtung"""
    matching = [e for e in events if all(apply_rule(e, r) for r in rules)]
    if len(matching) < 10: return None
    
    correct = sum(1 for e in matching if e['actual'] == direction)
    precision = correct / len(matching) * 100
    
    # Gegenprobe: Wie viele falsche Richtung werden auch erfasst?
    wrong = len(matching) - correct
    wrong_pct_avg = 0
    if wrong > 0:
        wrong_events = [e for e in matching if e['actual'] != direction]
        wrong_pct_avg = np.mean([abs(e['pct_change']) for e in wrong_events])
    
    return {
        'precision': round(precision, 1),
        'n_total': len(matching),
        'n_correct': correct,
        'n_wrong': wrong,
        'wrong_avg_pct': round(wrong_pct_avg, 1),
        'coverage': round(len(matching) / len(events) * 100, 1),
    }


def greedy_filter_search(events, direction, max_filters=7):
    """
    Greedy: Startet mit dem besten Einzelfilter, fügt iterativ den nächsten 
    hinzu der die Precision am meisten verbessert.
    """
    all_rules = build_filter_rules(events)
    logger.info(f"\n[{direction.upper()}] Testing {len(all_rules)} possible rules...")
    
    # Schritt 1: Besten Einzelfilter finden
    best_single = None
    best_single_prec = 0
    
    for rule in all_rules:
        result = evaluate_ruleset(events, [rule], direction)
        if result and result['precision'] > best_single_prec and result['n_total'] >= 20:
            best_single_prec = result['precision']
            best_single = rule
    
    if not best_single:
        logger.info(f"  No useful single filter found")
        return [], None
    
    chosen = [best_single]
    current_result = evaluate_ruleset(events, chosen, direction)
    logger.info(f"  Filter 1: {best_single['label']} → {current_result['precision']}% "
                f"(n={current_result['n_total']}, wrong={current_result['n_wrong']}, "
                f"wrong_avg={current_result['wrong_avg_pct']}%)")
    
    # Schritt 2-N: Iterativ beste Erweiterung finden
    used_indicators = {best_single['indicator']}
    
    for step in range(2, max_filters + 1):
        best_next = None
        best_next_prec = current_result['precision']
        best_next_result = None
        
        for rule in all_rules:
            if rule['indicator'] in used_indicators: continue
            
            candidate = chosen + [rule]
            result = evaluate_ruleset(events, candidate, direction)
            if result is None: continue
            
            # Muss besser sein UND genug Samples haben
            min_n = max(15, len(events) * 0.005)  # Min 0.5% der Events
            if result['precision'] > best_next_prec and result['n_total'] >= min_n:
                best_next_prec = result['precision']
                best_next = rule
                best_next_result = result
        
        if best_next is None or best_next_prec <= current_result['precision'] + 0.5:
            logger.info(f"  No improvement possible at step {step}. Stopping.")
            break
        
        chosen.append(best_next)
        used_indicators.add(best_next['indicator'])
        current_result = best_next_result
        logger.info(f"  Filter {step}: +{best_next['label']} → {current_result['precision']}% "
                    f"(n={current_result['n_total']}, wrong={current_result['n_wrong']}, "
                    f"wrong_avg={current_result['wrong_avg_pct']}%)")
    
    return chosen, current_result


def cross_validate(events, rules, direction):
    """Gegenprobe: Filterset auf erste und zweite Hälfte getrennt testen"""
    mid = len(events) // 2
    first_half = events[:mid]
    second_half = events[mid:]
    
    r1 = evaluate_ruleset(first_half, rules, direction)
    r2 = evaluate_ruleset(second_half, rules, direction)
    
    return r1, r2


# === MAIN ===
def run_deep_filter(days=30, min_pct=4.0):
    logger.info(f"{'='*70}")
    logger.info(f"DEEP FILTER SCANNER")
    logger.info(f"Period: {days} days, min move: ±{min_pct}%")
    logger.info(f"Indicators: ~50 features across 1h/4h/1d timeframes")
    logger.info(f"{'='*70}")
    
    conn = coins_db()
    cur = conn.cursor()
    
    # 1. Alle Moves sammeln
    logger.info(f"\n[STEP 1] Finding all ≥{min_pct}% moves...")
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
    logger.info(f"  Found {len(moves)} moves")
    
    # 2. Indikatoren berechnen
    logger.info(f"\n[STEP 2] Computing indicators for each move...")
    events = []
    for i, m in enumerate(moves):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Progress: {i}/{len(moves)} ({len(events)} with indicators)")
        
        ind = compute_all_indicators(conn, m['symbol'], m['bucket'])
        if ind is None: continue
        
        events.append({
            'symbol': m['symbol'],
            'time': m['bucket'],
            'pct_change': float(m['pct_change']),
            'actual': 'long' if m['pct_change'] > 0 else 'short',
            'indicators': ind,
        })
    
    conn.close()
    
    longs = [e for e in events if e['actual'] == 'long']
    shorts = [e for e in events if e['actual'] == 'short']
    logger.info(f"  Total: {len(events)} events ({len(longs)} long, {len(shorts)} short)")
    
    # 3. Indikator-Statistiken
    logger.info(f"\n[STEP 3] Indicator separation analysis...")
    indicators = sorted(set(k for e in events for k in e['indicators'].keys()))
    
    sep_scores = []
    for ind in indicators:
        l_vals = [e['indicators'][ind] for e in longs if e['indicators'].get(ind) is not None]
        s_vals = [e['indicators'][ind] for e in shorts if e['indicators'].get(ind) is not None]
        if not l_vals or not s_vals: continue
        
        combined_std = np.std(l_vals + s_vals)
        if combined_std == 0: continue
        sep = abs(np.mean(l_vals) - np.mean(s_vals)) / combined_std
        sep_scores.append((ind, sep, np.mean(l_vals), np.mean(s_vals), np.median(l_vals), np.median(s_vals)))
    
    sep_scores.sort(key=lambda x: -x[1])
    logger.info(f"\n  {'Indicator':25s} {'Sep':>6s} {'L_avg':>8s} {'S_avg':>8s} {'L_med':>8s} {'S_med':>8s}")
    logger.info(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ind, sep, la, sa, lm, sm in sep_scores:
        logger.info(f"  {ind:25s} {sep:>6.3f} {la:>+8.3f} {sa:>+8.3f} {lm:>+8.3f} {sm:>+8.3f}")
    
    # 4. Greedy Filter Search
    logger.info(f"\n[STEP 4] Greedy filter search...")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"LONG FILTER SEARCH")
    logger.info(f"{'='*70}")
    long_rules, long_result = greedy_filter_search(events, 'long', max_filters=7)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SHORT FILTER SEARCH")
    logger.info(f"{'='*70}")
    short_rules, short_result = greedy_filter_search(events, 'short', max_filters=7)
    
    # 5. Gegenprobe
    logger.info(f"\n[STEP 5] Cross-validation...")
    
    if long_rules:
        r1, r2 = cross_validate(events, long_rules, 'long')
        logger.info(f"\n  LONG cross-validation:")
        if r1: logger.info(f"    First half:  {r1['precision']}% (n={r1['n_total']}, wrong={r1['n_wrong']})")
        if r2: logger.info(f"    Second half: {r2['precision']}% (n={r2['n_total']}, wrong={r2['n_wrong']})")
        
        # Gegenprobe: Was fängt der Long-Filter an Shorts?
        long_caught_shorts = [e for e in events if all(apply_rule(e, r) for r in long_rules) and e['actual'] == 'short']
        if long_caught_shorts:
            avg_wrong = np.mean([e['pct_change'] for e in long_caught_shorts])
            logger.info(f"    ⚠ Long filter catches {len(long_caught_shorts)} shorts (avg {avg_wrong:+.1f}%)")
    
    if short_rules:
        r1, r2 = cross_validate(events, short_rules, 'short')
        logger.info(f"\n  SHORT cross-validation:")
        if r1: logger.info(f"    First half:  {r1['precision']}% (n={r1['n_total']}, wrong={r1['n_wrong']})")
        if r2: logger.info(f"    Second half: {r2['precision']}% (n={r2['n_total']}, wrong={r2['n_wrong']})")
        
        short_caught_longs = [e for e in events if all(apply_rule(e, r) for r in short_rules) and e['actual'] == 'long']
        if short_caught_longs:
            avg_wrong = np.mean([e['pct_change'] for e in short_caught_longs])
            logger.info(f"    ⚠ Short filter catches {len(short_caught_longs)} longs (avg {avg_wrong:+.1f}%)")
    
    # 6. Zusammenfassung
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n  LONG FILTERS ({len(long_rules)} rules):")
    for r in long_rules:
        logger.info(f"    {r['label']}")
    if long_result:
        logger.info(f"    → Precision: {long_result['precision']}%, n={long_result['n_total']}, "
                    f"coverage={long_result['coverage']}%")
    
    logger.info(f"\n  SHORT FILTERS ({len(short_rules)} rules):")
    for r in short_rules:
        logger.info(f"    {r['label']}")
    if short_result:
        logger.info(f"    → Precision: {short_result['precision']}%, n={short_result['n_total']}, "
                    f"coverage={short_result['coverage']}%")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DEEP FILTER COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    run_deep_filter(days, min_pct)
