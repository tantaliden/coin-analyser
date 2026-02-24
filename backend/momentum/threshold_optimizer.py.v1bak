#!/usr/bin/env python3
"""
THRESHOLD OPTIMIZER
Basiert auf Discovery Scanner Ergebnisse.
Läuft 1x täglich, berechnet optimale Schwellwerte und passt Scanner-Config an.
"""

import sys, os, json, logging
from datetime import datetime, timedelta, timezone
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json

LOG_FILE = '/opt/coin/logs/threshold_optimizer.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger('optimizer')

def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)

def app_db():
    return psycopg2.connect(host='localhost', dbname='analyser_app', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


# === INDIKATOR-BERECHNUNG (identisch mit Discovery Scanner) ===
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_body_ratio(opens, highs, lows, closes, n=6):
    if len(opens) < n: return None
    ratios = []
    for i in range(-n, 0):
        r = highs[i] - lows[i]
        if r <= 0: continue
        ratios.append(abs(closes[i] - opens[i]) / r)
    return float(np.mean(ratios)) if ratios else None


def compute_indicators(conn, symbol, move_time):
    """Berechnet die relevanten Indikatoren VOR einem Move"""
    cur = conn.cursor()

    cur.execute("SELECT bucket, open, high, low, close, volume FROM agg_1h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 50", (symbol, move_time))
    r1h = list(reversed(cur.fetchall()))
    if len(r1h) < 30: return None

    cur.execute("SELECT bucket, open, high, low, close FROM agg_4h WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 20", (symbol, move_time))
    r4h = list(reversed(cur.fetchall()))
    if len(r4h) < 10: return None

    cur.execute("SELECT bucket, close FROM agg_1d WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 15", (symbol, move_time))
    r1d = list(reversed(cur.fetchall()))
    if len(r1d) < 5: return None

    closes_1h = [r['close'] for r in r1h]
    highs_1h  = [r['high'] for r in r1h]
    lows_1h   = [r['low'] for r in r1h]
    opens_1h  = [r['open'] for r in r1h]
    closes_4h = [r['close'] for r in r4h]
    closes_1d = [r['close'] for r in r1d]

    rsi = calc_rsi(closes_1h, 14)
    if rsi is None: return None

    # trend_4h
    t4h = 0
    if len(closes_4h) >= 5:
        r = closes_4h[-5:]
        t4h = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r)-1)) * 2 - 1

    # trend_1d
    t1d = 0
    if len(closes_1d) >= 5:
        r = closes_1d[-5:]
        t1d = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r)-1)) * 2 - 1

    # body_ratio
    br = calc_body_ratio(opens_1h, highs_1h, lows_1h, closes_1h, 6) or 0.5

    # hh_hl
    hh_hl = 0
    if len(highs_1h) >= 6:
        rh, rl = highs_1h[-6:], lows_1h[-6:]
        hh_hl = (sum(1 for i in range(1,6) if rh[i]>rh[i-1]) + sum(1 for i in range(1,6) if rl[i]>rl[i-1])) / 10

    # consecutive
    cu, cd = 0, 0
    if len(closes_1h) >= 11:
        cu = sum(1 for i in range(-10, 0) if closes_1h[i] > closes_1h[i-1])
        cd = 10 - cu

    return {
        'rsi_1h': round(rsi, 1),
        'trend_4h': round(t4h, 2),
        'trend_1d': round(t1d, 2),
        'body_ratio': round(br, 3),
        'hh_hl': round(hh_hl, 2),
        'consec_ups': cu,
        'consec_downs': cd,
    }


def find_optimal_threshold(events, indicator, direction):
    """
    Findet den optimalen Schwellwert für einen Indikator.
    Für Long: threshold wo precision (= echte Longs / predicted Longs) am besten ist.
    Für Short: threshold wo precision (= echte Shorts / predicted Shorts) am besten ist.
    """
    vals = [(e['indicators'][indicator], e['actual']) for e in events
            if e['indicators'].get(indicator) is not None]
    if len(vals) < 50:
        return None, None, None

    vals_only = [v for v, _ in vals]
    best_thresh = None
    best_precision = 0
    best_n = 0

    for pct in range(5, 96, 5):
        thresh = np.percentile(vals_only, pct)

        if direction == 'long':
            # Long: indicator >= thresh → predict Long
            matched = [(v, a) for v, a in vals if v >= thresh]
            correct = sum(1 for _, a in matched if a == 'long')
        else:
            # Short: indicator < thresh → predict Short
            matched = [(v, a) for v, a in vals if v < thresh]
            correct = sum(1 for _, a in matched if a == 'short')

        n = len(matched)
        if n < 20:
            continue
        precision = correct / n * 100

        # Optimal = hohe Precision bei vernünftiger Coverage
        score = precision - max(0, (50 - n) * 0.5)  # Penalty für zu wenig Samples
        if score > best_precision:
            best_precision = score
            best_thresh = thresh
            best_n = n

    return best_thresh, best_precision, best_n


def run_optimization(days=14, min_pct=4.0):
    """Hauptfunktion: Discovery + Threshold-Optimierung + DB-Update"""
    logger.info(f"{'='*60}")
    logger.info(f"THRESHOLD OPTIMIZER START")
    logger.info(f"Period: last {days} days, min move: {min_pct}%")
    logger.info(f"{'='*60}")

    cconn = coins_db()
    ccur = cconn.cursor()

    # === 1. Moves finden ===
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
    logger.info(f"Found {len(moves)} moves (≥{min_pct}%)")

    # === 2. Indikatoren berechnen ===
    events = []
    for m in moves:
        ind = compute_indicators(cconn, m['symbol'], m['bucket'])
        if ind is None: continue
        events.append({
            'symbol': m['symbol'],
            'time': m['bucket'],
            'pct_change': float(m['pct_change']),
            'actual': 'long' if m['pct_change'] > 0 else 'short',
            'indicators': ind,
        })

    longs = [e for e in events if e['actual'] == 'long']
    shorts = [e for e in events if e['actual'] == 'short']
    logger.info(f"Events with indicators: {len(events)} ({len(longs)} long, {len(shorts)} short)")

    if len(events) < 100:
        logger.warning("Too few events for reliable optimization. Skipping.")
        cconn.close()
        return

    # === 3. Optimale Thresholds berechnen ===
    # Mapping: config_spalte → (indikator, richtung, modus)
    # modus: 'gte' = long wenn >= thresh, 'lt' = short wenn < thresh
    threshold_map = {
        'scan_trend_1d_long_min':    ('trend_1d', 'long'),
        'scan_trend_1d_short_max':   ('trend_1d', 'short'),
        'scan_trend_4h_long_min':    ('trend_4h', 'long'),
        'scan_trend_4h_short_max':   ('trend_4h', 'short'),
        'scan_body_ratio_long_min':  ('body_ratio', 'long'),
        'scan_body_ratio_short_max': ('body_ratio', 'short'),
        'scan_rsi_long_min':         ('rsi_1h', 'long'),
        'scan_rsi_short_max':        ('rsi_1h', 'short'),
        'scan_hh_hl_long_min':       ('hh_hl', 'long'),
        'scan_hh_hl_short_max':      ('hh_hl', 'short'),
    }

    optimal = {}
    for col, (indicator, direction) in threshold_map.items():
        thresh, precision, n = find_optimal_threshold(events, indicator, direction)
        if thresh is not None:
            optimal[col] = {
                'value': round(float(thresh), 4),
                'precision': round(precision, 1),
                'n_samples': n,
            }
            logger.info(f"  {col}: optimal={thresh:.4f} (precision={precision:.1f}%, n={n})")
        else:
            logger.info(f"  {col}: insufficient data, keeping current")

    # Consecutive: Median-basiert statt Percentil (ganzzahlig)
    l_ups = [e['indicators']['consec_ups'] for e in longs if e['indicators'].get('consec_ups') is not None]
    s_downs = [e['indicators']['consec_downs'] for e in shorts if e['indicators'].get('consec_downs') is not None]
    if l_ups:
        optimal['scan_consec_ups_long_min'] = {
            'value': int(np.percentile(l_ups, 40)),
            'precision': 0, 'n_samples': len(l_ups),
        }
        logger.info(f"  scan_consec_ups_long_min: optimal={optimal['scan_consec_ups_long_min']['value']}")
    if s_downs:
        optimal['scan_consec_downs_short_min'] = {
            'value': int(np.percentile(s_downs, 40)),
            'precision': 0, 'n_samples': len(s_downs),
        }
        logger.info(f"  scan_consec_downs_short_min: optimal={optimal['scan_consec_downs_short_min']['value']}")

    cconn.close()

    # === 4. Mit aktueller DB-Config vergleichen und updaten ===
    aconn = app_db()
    acur = aconn.cursor()

    acur.execute("SELECT * FROM momentum_scan_config WHERE is_active = true LIMIT 1")
    config = acur.fetchone()
    if not config:
        logger.error("No active scan config found!")
        aconn.close()
        return

    changes = {}
    DRIFT_THRESHOLD = 0.05  # 5% relative Abweichung

    for col, opt_data in optimal.items():
        new_val = opt_data['value']
        current_val = config.get(col)

        if current_val is None:
            # Neuer Wert, noch nie gesetzt
            changes[col] = {'old': None, 'new': new_val, 'reason': 'initial_set', 'precision': opt_data['precision']}
            continue

        current_val = float(current_val)
        if current_val == 0 and new_val == 0:
            continue

        # Relative Abweichung berechnen
        if current_val == 0:
            rel_diff = abs(new_val)
        else:
            rel_diff = abs(new_val - current_val) / abs(current_val)

        if rel_diff > DRIFT_THRESHOLD:
            changes[col] = {
                'old': round(current_val, 4),
                'new': new_val,
                'drift_pct': round(rel_diff * 100, 1),
                'precision': opt_data['precision'],
                'reason': f'drift {rel_diff*100:.1f}% > {DRIFT_THRESHOLD*100}% threshold',
            }

    # === 5. Änderungen anwenden ===
    if changes:
        logger.info(f"\n{'='*60}")
        logger.info(f"APPLYING {len(changes)} THRESHOLD CHANGES:")
        logger.info(f"{'='*60}")

        set_parts = []
        set_vals = []
        for col, info in changes.items():
            set_parts.append(f"{col} = %s")
            set_vals.append(info['new'])
            logger.info(f"  {col}: {info['old']} → {info['new']} ({info['reason']})")

        set_parts.append("updated_at = NOW()")
        set_vals.append(config['config_id'])

        acur.execute(
            f"UPDATE momentum_scan_config SET {', '.join(set_parts)} WHERE config_id = %s",
            set_vals
        )

        # Änderungslog in die DB
        acur.execute("""
            INSERT INTO momentum_optimization_log
            (user_id, period_start, period_end, total_predictions, total_tp, total_sl,
             total_expired, current_hit_rate, simulations_run, recommendation, applied, reason, changes_applied)
            VALUES (%s, NOW() - INTERVAL '%s days', NOW(), %s, %s, %s, 0, 0, %s,
                    'threshold_drift', true, %s, %s)
        """, (config['user_id'], days, len(events), len(longs), len(shorts),
              len(optimal), f"{len(changes)} thresholds adjusted", Json(changes)))

        aconn.commit()
        logger.info(f"\n✓ {len(changes)} thresholds updated in DB")
    else:
        logger.info(f"\n✓ All thresholds within {DRIFT_THRESHOLD*100}% tolerance. No changes needed.")

    aconn.close()
    logger.info(f"\n{'='*60}")
    logger.info(f"THRESHOLD OPTIMIZER COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    run_optimization(days, min_pct)
