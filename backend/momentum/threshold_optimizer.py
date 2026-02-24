#!/usr/bin/env python3
"""
THRESHOLD OPTIMIZER v2 - Deep Filter
Wöchentlich: Analysiert letzte 14 Tage, optimiert die 6 Deep Filter Thresholds.
"""

import logging
import numpy as np
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger('momentum')

DRIFT_THRESHOLD = 0.05  # 5% Abweichung = Update

def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)

def app_db():
    return psycopg2.connect(host='localhost', dbname='analyser_app', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


def calc_ema(values, period):
    if len(values) < period: return None
    m = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]: ema = (v - ema) * m + ema
    return ema


def compute_indicators(conn, symbol, move_time):
    """Berechnet die 6 Deep Filter Indikatoren VOR einem Move"""
    cur = conn.cursor()

    cur.execute("SELECT bucket, open, high, low, close, volume FROM agg_1h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 50", (symbol, move_time))
    r1h = list(reversed(cur.fetchall()))
    if len(r1h) < 30: return None

    cur.execute("SELECT bucket, high, low, close FROM agg_1d WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 10", (symbol, move_time))
    r1d = list(reversed(cur.fetchall()))
    if len(r1d) < 7: return None

    closes_1h = [r['close'] for r in r1h]
    highs_1h  = [r['high'] for r in r1h]
    lows_1h   = [r['low'] for r in r1h]
    opens_1h  = [r['open'] for r in r1h]
    highs_1d  = [r['high'] for r in r1d]
    lows_1d   = [r['low'] for r in r1d]
    closes_1d = [r['close'] for r in r1d]

    current = closes_1h[-1]
    if not current or current <= 0: return None

    # range_pos_7d
    h7 = highs_1d[-7:]
    l7 = lows_1d[-7:]
    highest = max(h7)
    lowest = min(l7)
    range_pos_7d = (current - lowest) / (highest - lowest) if highest != lowest else 0.5

    # ema_price_50_pct
    ema_50 = calc_ema(closes_1h[-50:], 50) if len(closes_1h) >= 50 else None
    ema_price_50_pct = ((current - ema_50) / ema_50 * 100) if ema_50 and ema_50 > 0 else 0

    # bb_position_1h
    sma = np.mean(closes_1h[-20:])
    std = np.std(closes_1h[-20:])
    if std > 0:
        bb_position_1h = (current - (sma - 2*std)) / (4*std) if 4*std != 0 else 0.5
    else:
        bb_position_1h = 0.5

    # upper_wick / lower_wick (avg 6 candles)
    upper_wicks, lower_wicks = [], []
    for i in range(-6, 0):
        full = highs_1h[i] - lows_1h[i]
        if full <= 0: continue
        if closes_1h[i] >= opens_1h[i]:
            upper_wicks.append((highs_1h[i] - closes_1h[i]) / full)
            lower_wicks.append((opens_1h[i] - lows_1h[i]) / full)
        else:
            upper_wicks.append((highs_1h[i] - opens_1h[i]) / full)
            lower_wicks.append((closes_1h[i] - lows_1h[i]) / full)

    upper_wick_1h = np.mean(upper_wicks) if upper_wicks else 0
    lower_wick_1h = np.mean(lower_wicks) if lower_wicks else 0

    return {
        'range_pos_7d': round(float(range_pos_7d), 4),
        'ema_price_50_pct': round(float(ema_price_50_pct), 4),
        'bb_position_1h': round(float(bb_position_1h), 4),
        'upper_wick_1h': round(float(upper_wick_1h), 4),
        'lower_wick_1h': round(float(lower_wick_1h), 4),
    }


def find_optimal_threshold(events, indicator, direction):
    """
    Findet den optimalen Schwellenwert per Percentil-Suche.
    direction='long': Sucht thresh wo >= thresh möglichst viele Longs sind
    direction='short': Sucht thresh wo < thresh möglichst viele Shorts sind
    """
    vals = [(e['indicators'][indicator], e['actual']) for e in events if e['indicators'].get(indicator) is not None]
    if len(vals) < 50: return None, 0, 0

    best_thresh, best_score, best_n = None, 0, 0

    all_vals = [v[0] for v in vals]
    for pct in range(5, 96, 5):
        thresh = np.percentile(all_vals, pct)

        if direction == 'long':
            matching = [(v, a) for v, a in vals if v >= thresh]
        else:
            matching = [(v, a) for v, a in vals if v < thresh]

        if len(matching) < max(20, len(vals) * 0.05):
            continue

        correct = sum(1 for _, a in matching if a == direction)
        precision = correct / len(matching) * 100
        # Penalize low coverage
        score = precision - max(0, (50 - len(matching)) * 0.5)

        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_n = len(matching)

    precision = 0
    if best_thresh is not None and best_n > 0:
        if direction == 'long':
            matching = [(v, a) for v, a in vals if v >= best_thresh]
        else:
            matching = [(v, a) for v, a in vals if v < best_thresh]
        correct = sum(1 for _, a in matching if a == direction)
        precision = correct / len(matching) * 100

    return best_thresh, precision, best_n


def run_threshold_optimization(days=14, min_pct=4.0):
    """Wöchentliche Deep Filter Threshold Optimierung"""
    logger.info(f"[OPTIMIZER] Starting Deep Filter threshold optimization ({days}d, ±{min_pct}%)")

    cconn = coins_db()
    ccur = cconn.cursor()

    # 1. Alle Moves sammeln
    ccur.execute("""
        SELECT symbol, bucket, open, close,
            ((close - open) / NULLIF(open, 0)) * 100 as pct_change
        FROM agg_4h
        WHERE bucket >= NOW() - INTERVAL '%s days'
          AND open > 0
          AND ABS((close - open) / NULLIF(open, 0)) * 100 >= %s
        ORDER BY bucket
    """, (days, min_pct))
    moves = ccur.fetchall()
    logger.info(f"[OPTIMIZER] Found {len(moves)} moves")

    # 2. Indikatoren berechnen
    events = []
    for m in moves:
        ind = compute_indicators(cconn, m['symbol'], m['bucket'])
        if ind is None: continue
        events.append({
            'indicators': ind,
            'actual': 'long' if m['pct_change'] > 0 else 'short',
        })

    longs = [e for e in events if e['actual'] == 'long']
    shorts = [e for e in events if e['actual'] == 'short']
    logger.info(f"[OPTIMIZER] Events with indicators: {len(events)} ({len(longs)} long, {len(shorts)} short)")

    if len(events) < 100:
        logger.warning("[OPTIMIZER] Too few events. Skipping.")
        cconn.close()
        return

    # 3. Optimale Thresholds für die 6 Deep Filter Parameter
    # Long: range_pos_7d >=, ema_price_50_pct <, lower_wick_1h <
    # Short: range_pos_7d <, upper_wick_1h >=, bb_position_1h <
    threshold_map = {
        'df_long_range_pos_7d_min':  ('range_pos_7d', 'long'),      # >= thresh
        'df_long_ema_price_50_max':  ('ema_price_50_pct', 'short'),  # < thresh (inverted: Long wants SMALL ema dist)
        'df_long_lower_wick_max':    ('lower_wick_1h', 'short'),     # < thresh (inverted: Long wants SMALL lower wick)
        'df_short_range_pos_7d_max': ('range_pos_7d', 'short'),     # < thresh
        'df_short_upper_wick_min':   ('upper_wick_1h', 'long'),     # >= thresh (inverted: Short wants LARGE upper wick)
        'df_short_bb_position_max':  ('bb_position_1h', 'short'),   # < thresh
    }

    optimal = {}
    for col, (indicator, opt_direction) in threshold_map.items():
        thresh, precision, n = find_optimal_threshold(events, indicator, opt_direction)
        if thresh is not None:
            optimal[col] = {
                'value': round(float(thresh), 4),
                'precision': round(precision, 1),
                'n_samples': n,
            }
            logger.info(f"[OPTIMIZER]   {col}: optimal={thresh:.4f} (precision={precision:.1f}%, n={n})")
        else:
            logger.info(f"[OPTIMIZER]   {col}: insufficient data, keeping current")

    cconn.close()

    # 4. Mit DB vergleichen und updaten
    aconn = app_db()
    acur = aconn.cursor()
    acur.execute("SELECT * FROM momentum_scan_config WHERE user_id = 1")
    current_config = acur.fetchone()

    if not current_config:
        logger.warning("[OPTIMIZER] No config row found. Skipping.")
        aconn.close()
        return

    updates = []
    for col, opt_data in optimal.items():
        current_val = float(current_config.get(col) or 0)
        new_val = opt_data['value']

        if current_val == 0:
            drift = 1.0
        else:
            drift = abs(new_val - current_val) / abs(current_val)

        if drift >= DRIFT_THRESHOLD:
            updates.append((col, new_val, current_val, drift, opt_data['precision'], opt_data['n_samples']))
            logger.info(f"[OPTIMIZER] UPDATE {col}: {current_val} → {new_val} (drift {drift*100:.1f}%, prec={opt_data['precision']:.1f}%)")
        else:
            logger.info(f"[OPTIMIZER] KEEP {col}: {current_val} (drift {drift*100:.1f}% < {DRIFT_THRESHOLD*100}%)")

    if updates:
        set_clauses = ', '.join(f"{col} = %s" for col, _, _, _, _, _ in updates)
        values = [v for _, v, _, _, _, _ in updates]
        acur.execute(f"UPDATE momentum_scan_config SET {set_clauses} WHERE user_id = 1", values)

        for col, new_val, old_val, drift, prec, n in updates:
            acur.execute("""
                INSERT INTO momentum_optimization_log
                (user_id, parameter, old_value, new_value, recommendation, reason)
                VALUES (1, %s, %s, %s, %s, %s)
            """, (col, str(old_val), str(new_val), 'adjusted',
                  f'drift={drift*100:.1f}% precision={prec:.1f}% n={n}'))

        aconn.commit()
        logger.info(f"[OPTIMIZER] Updated {len(updates)} thresholds")
    else:
        logger.info("[OPTIMIZER] No updates needed (all within drift threshold)")

    aconn.close()
    logger.info("[OPTIMIZER] Done")
