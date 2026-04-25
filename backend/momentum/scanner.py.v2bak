#!/usr/bin/env python3
"""
Momentum Scanner - Separates Autoscan-Modul
Scannt Coins auf Long/Short Momentum, schreibt Predictions,
überwacht aktive Predictions, führt Autokorrektur durch.

Liest: coins DB (agg_1h, agg_4h, agg_1d)
Schreibt: analyser_app DB (momentum_predictions, momentum_stats, momentum_corrections)
"""

import os
import sys
import time
import json
import signal
import logging
import traceback
# Threshold Optimizer (direkt importiert, gleiche Ebene)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threshold_optimizer import run_optimization as run_threshold_optimization
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/coin/logs/momentum_scanner.log')
    ]
)
logger = logging.getLogger('momentum_scanner')

# ============================================
# CONFIG aus settings.json
# ============================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

COINS_DB_CFG = SETTINGS['databases']['coins']
APP_DB_CFG = SETTINGS['databases']['app']

# ============================================
# GLOBALS
# ============================================
running = True

def signal_handler(signum, frame):
    global running
    logger.info(f"Signal {signum} received, shutting down...")
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================
# DB CONNECTIONS
# ============================================
@contextmanager
def coins_db():
    conn = psycopg2.connect(
        host=COINS_DB_CFG['host'], port=COINS_DB_CFG['port'],
        dbname=COINS_DB_CFG['name'], user=COINS_DB_CFG['user'],
        password=COINS_DB_CFG['password'], cursor_factory=RealDictCursor
    )
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def app_db():
    conn = psycopg2.connect(
        host=APP_DB_CFG['host'], port=APP_DB_CFG['port'],
        dbname=APP_DB_CFG['name'], user=APP_DB_CFG['user'],
        password=APP_DB_CFG['password'], cursor_factory=RealDictCursor
    )
    try:
        yield conn
    finally:
        conn.close()


# ============================================
# SIGNAL DETECTION LOGIC
# ============================================

def calc_rsi(closes, period=14):
    """RSI berechnen aus Close-Preisen"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(values, period):
    """EMA berechnen"""
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = (v - ema) * multiplier + ema
    return ema


def calc_atr(highs, lows, closes, period=14):
    """Average True Range"""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])


def calc_body_ratio(opens, highs, lows, closes, n=6):
    """Durchschnittliches Body/Wick Verhältnis der letzten n Candles"""
    if len(opens) < n:
        return None
    ratios = []
    for i in range(-n, 0):
        full_range = highs[i] - lows[i]
        if full_range <= 0:
            continue
        body = abs(closes[i] - opens[i])
        ratios.append(body / full_range)
    return float(np.mean(ratios)) if ratios else None


def calc_consecutive(closes, n=10):
    """Wie viele der letzten n Candles steigen vs fallen"""
    if len(closes) < n + 1:
        return 0, 0
    ups = sum(1 for i in range(-n, 0) if closes[i] > closes[i-1])
    return ups, n - ups


def analyze_symbol(candles_1h, candles_4h, candles_1d, current_price, market_context=None, scan_config=None):
    """
    v2: Datengetriebene Gewichtung basierend auf Discovery Scanner.
    Thresholds kommen aus scan_config (DB) und sind vom Optimizer anpassbar.
    """
    if len(candles_1h) < 30 or len(candles_4h) < 15 or len(candles_1d) < 10:
        return None

    closes_1h  = [c['close'] for c in candles_1h]
    highs_1h   = [c['high'] for c in candles_1h]
    lows_1h    = [c['low'] for c in candles_1h]
    opens_1h   = [c['open'] for c in candles_1h]
    volumes_1h = [c['volume'] for c in candles_1h]
    closes_4h  = [c['close'] for c in candles_4h]
    closes_1d  = [c['close'] for c in candles_1d]

    # === CONFIG THRESHOLDS (DB, Optimizer-anpassbar) ===
    cfg = scan_config or {}
    t4h_long_min   = float(cfg.get('scan_trend_4h_long_min') or -0.5)
    t4h_short_max  = float(cfg.get('scan_trend_4h_short_max') or -0.5)
    t1d_long_min   = float(cfg.get('scan_trend_1d_long_min') or -0.5)
    t1d_short_max  = float(cfg.get('scan_trend_1d_short_max') or -0.5)
    br_long_min    = float(cfg.get('scan_body_ratio_long_min') or 0.55)
    br_short_max   = float(cfg.get('scan_body_ratio_short_max') or 0.55)
    rsi_long_min   = float(cfg.get('scan_rsi_long_min') or 40.0)
    rsi_short_max  = float(cfg.get('scan_rsi_short_max') or 40.0)
    hhhl_long_min  = float(cfg.get('scan_hh_hl_long_min') or 0.4)
    hhhl_short_max = float(cfg.get('scan_hh_hl_short_max') or 0.4)
    cu_long_min    = int(cfg.get('scan_consec_ups_long_min') or 4)
    cd_short_min   = int(cfg.get('scan_consec_downs_short_min') or 5)
    min_score      = int(cfg.get('scan_min_score') or 40)

    # === INDIKATOREN ===
    rsi_1h = calc_rsi(closes_1h, 14)
    if rsi_1h is None:
        return None

    trend_4h = 0
    if len(closes_4h) >= 5:
        r = closes_4h[-5:]
        trend_4h = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r) - 1)) * 2 - 1

    trend_1d = 0
    if len(closes_1d) >= 5:
        r = closes_1d[-5:]
        trend_1d = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r) - 1)) * 2 - 1

    body_ratio = calc_body_ratio(opens_1h, highs_1h, lows_1h, closes_1h, 6) or 0.5

    hh_hl = 0
    if len(highs_1h) >= 6:
        rh, rl = highs_1h[-6:], lows_1h[-6:]
        hh_hl = (sum(1 for i in range(1, 6) if rh[i] > rh[i-1]) + sum(1 for i in range(1, 6) if rl[i] > rl[i-1])) / 10

    consec_ups, consec_downs = calc_consecutive(closes_1h)

    avg_vol = np.mean(volumes_1h[-20:]) if len(volumes_1h) >= 20 else np.mean(volumes_1h)
    vol_ratio = np.mean(volumes_1h[-3:]) / avg_vol if avg_vol > 0 else 1

    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    ema_9  = calc_ema(closes_1h, 9)
    ema_21 = calc_ema(closes_1h, 21)

    # === SCORING (Discovery-gewichtet) ===
    long_score = 0
    short_score = 0
    signals = []

    # TIER 1: trend_1d (Sep 0.475) → 25
    if trend_1d >= t1d_long_min:
        long_score += 25
        signals.append({'name': f'1d trend bullish ({trend_1d:+.2f})', 'type': 'long', 'weight': 25})
    if trend_1d <= t1d_short_max:
        short_score += 25
        signals.append({'name': f'1d trend bearish ({trend_1d:+.2f})', 'type': 'short', 'weight': 25})

    # TIER 1: trend_4h (Sep 0.471) → 25
    if trend_4h >= t4h_long_min:
        long_score += 25
        signals.append({'name': f'4h trend bullish ({trend_4h:+.2f})', 'type': 'long', 'weight': 25})
    if trend_4h <= t4h_short_max:
        short_score += 25
        signals.append({'name': f'4h trend bearish ({trend_4h:+.2f})', 'type': 'short', 'weight': 25})

    # TIER 1: body_ratio (Sep 0.430) → 20
    if body_ratio >= br_long_min:
        long_score += 20
        signals.append({'name': f'Strong bodies ({body_ratio:.2f})', 'type': 'long', 'weight': 20})
    if body_ratio < br_short_max:
        short_score += 20
        signals.append({'name': f'Weak bodies ({body_ratio:.2f})', 'type': 'short', 'weight': 20})

    # TIER 2: RSI (Sep 0.378) → 15
    if rsi_1h >= rsi_long_min:
        long_score += 15
        signals.append({'name': f'RSI bullish ({rsi_1h:.0f})', 'type': 'long', 'weight': 15})
    if rsi_1h < rsi_short_max:
        short_score += 15
        signals.append({'name': f'RSI bearish ({rsi_1h:.0f})', 'type': 'short', 'weight': 15})

    # TIER 2: consecutive (Sep 0.348) → 15
    if consec_ups >= cu_long_min:
        long_score += 15
        signals.append({'name': f'Consec ups ({consec_ups}/10)', 'type': 'long', 'weight': 15})
    if consec_downs >= cd_short_min:
        short_score += 15
        signals.append({'name': f'Consec downs ({consec_downs}/10)', 'type': 'short', 'weight': 15})

    # TIER 2: hh_hl (Sep 0.314) → 10
    if hh_hl >= hhhl_long_min:
        long_score += 10
        signals.append({'name': f'HH/HL ({hh_hl:.2f})', 'type': 'long', 'weight': 10})
    if hh_hl < hhhl_short_max:
        short_score += 10
        signals.append({'name': f'LL/LH ({hh_hl:.2f})', 'type': 'short', 'weight': 10})

    # TIER 3: Volume (Sep 0.236) → 5
    if vol_ratio > 1.5:
        pct_6h = ((closes_1h[-1] - closes_1h[-6]) / closes_1h[-6] * 100) if closes_1h[-6] > 0 else 0
        if pct_6h > 0:
            long_score += 5
            signals.append({'name': f'Vol +buy ({vol_ratio:.1f}x)', 'type': 'long', 'weight': 5})
        else:
            short_score += 5
            signals.append({'name': f'Vol +sell ({vol_ratio:.1f}x)', 'type': 'short', 'weight': 5})

    # === MARKTKONTEXT ===
    if market_context:
        trend      = market_context['market_trend']
        avg_4h     = market_context['avg_4h']
        breadth_4h = market_context['breadth_4h']
        breadth_1h = market_context.get('breadth_1h', 0.5)

        if avg_4h < -1.5 and breadth_4h < 0.25:
            short_score = 0
            signals.append({'name': 'Market oversold → short blocked', 'type': 'market_context', 'weight': -99})
        elif breadth_1h > 0.75:
            short_score = 0
            signals.append({'name': f'Market bouncing (b1h={breadth_1h:.2f}) → short blocked', 'type': 'market_context', 'weight': -99})
        else:
            if trend > 0.6: short_score -= 20; signals.append({'name': f'Mkt bull ({trend:+.2f}) → short -20', 'type': 'market_context', 'weight': -20})
            elif trend > 0.25: short_score -= 10; signals.append({'name': f'Mkt bull ({trend:+.2f}) → short -10', 'type': 'market_context', 'weight': -10})
            if breadth_1h > 0.60: short_score -= 15; signals.append({'name': f'Mkt recover (b1h={breadth_1h:.2f}) → short -15', 'type': 'market_context', 'weight': -15})
        short_score = max(short_score, 0)

        if breadth_1h < 0.25:
            long_score = 0
            signals.append({'name': f'Market dump (b1h={breadth_1h:.2f}) → long blocked', 'type': 'market_context', 'weight': -99})
        elif breadth_1h < 0.40:
            long_score -= 15; signals.append({'name': f'Mkt weak (b1h={breadth_1h:.2f}) → long -15', 'type': 'market_context', 'weight': -15})
        if trend < -0.6: long_score -= 20; signals.append({'name': f'Mkt bear ({trend:+.2f}) → long -20', 'type': 'market_context', 'weight': -20})
        elif trend < -0.25: long_score -= 10; signals.append({'name': f'Mkt bear ({trend:+.2f}) → long -10', 'type': 'market_context', 'weight': -10})
        long_score = max(long_score, 0)

    # === ENTSCHEIDUNG ===
    max_score = max(long_score, short_score)
    if max_score < min_score:
        return None

    direction = 'long' if long_score > short_score else 'short'
    confidence = min(max_score, 100)

    expected_move_pct = (atr * 2.5 / current_price * 100) if atr and current_price > 0 else 0

    top_signals = sorted([s for s in signals if s['type'] == direction], key=lambda x: -x['weight'])[:3]
    reason = ', '.join(s['name'] for s in top_signals)

    return {
        'direction': direction,
        'confidence': confidence,
        'entry_price': current_price,
        'expected_move_pct': round(expected_move_pct, 2),
        'reason': reason,
        'signals': signals,
        'indicators': {
            'rsi_1h': round(rsi_1h, 1),
            'ema_9': round(ema_9, 8) if ema_9 else None,
            'ema_21': round(ema_21, 8) if ema_21 else None,
            'vol_ratio': round(vol_ratio, 2),
            'trend_4h': round(trend_4h, 2),
            'trend_1d': round(trend_1d, 2),
            'atr': round(atr, 8) if atr else None,
            'hh_hl': round(hh_hl, 2),
            'body_ratio': round(body_ratio, 3),
            'consec_ups': consec_ups,
            'consec_downs': consec_downs,
        }
    }

# ============================================
# PREDICTION MONITORING
# ============================================

def check_active_predictions():
    """Prüft alle aktiven Predictions gegen aktuelle Preise"""
    with app_db() as aconn, coins_db() as cconn:
        acur = aconn.cursor()
        ccur = cconn.cursor()

        acur.execute("SELECT * FROM momentum_predictions WHERE status = 'active'")
        active = acur.fetchall()

        if not active:
            return 0

        resolved_count = 0
        for pred in active:
            symbol = pred['symbol']
            
            # Aktuellen Preis holen (letzter 1m Kurs + 1h High/Low für Peak/Trough)
            ccur.execute("""
                SELECT close FROM klines
                WHERE symbol = %s AND interval = '1m'
                ORDER BY open_time DESC LIMIT 1
            """, (symbol,))
            row_1m = ccur.fetchone()
            if not row_1m:
                continue

            # High/Low seit Detection für Peak/Trough
            ccur.execute("""
                SELECT MAX(high) as high, MIN(low) as low FROM klines
                WHERE symbol = %s AND interval = '1m' AND open_time >= %s
            """, (symbol, pred['detected_at']))
            row_hl = ccur.fetchone()

            current = row_1m['close']
            entry = pred['entry_price']
            row = {'close': current, 'high': row_hl['high'] or current, 'low': row_hl['low'] or current}
            
            # Peak/Trough tracken
            if pred['direction'] == 'long':
                pct_change = ((current - entry) / entry) * 100
                peak = max(pred['peak_pct'] or 0, ((row['high'] - entry) / entry) * 100)
                trough = min(pred['trough_pct'] or 0, ((row['low'] - entry) / entry) * 100)
            else:
                pct_change = ((entry - current) / entry) * 100
                peak = max(pred['peak_pct'] or 0, ((entry - row['low']) / entry) * 100)
                trough = min(pred['trough_pct'] or 0, ((entry - row['high']) / entry) * 100)

            # Status prüfen
            new_status = None
            duration = int((datetime.now(timezone.utc) - pred['detected_at']).total_seconds() / 60)

            if pred['direction'] == 'long':
                if current >= pred['take_profit_price']:
                    new_status = 'hit_tp'
                elif current <= pred['stop_loss_price']:
                    new_status = 'hit_sl'
            else:
                if current <= pred['take_profit_price']:
                    new_status = 'hit_tp'
                elif current >= pred['stop_loss_price']:
                    new_status = 'hit_sl'

            # Expiry: max 72h
            if not new_status and pred['expires_at'] and datetime.now(timezone.utc) >= pred['expires_at']:
                new_status = 'expired'

            # Invalidierung: Trend komplett gedreht (Confidence < 20 bei Neuberechnung)
            if not new_status:
                # Einfacher Check: wenn pct_change < -3% bei Long oder > +3% bei Short → invalidieren
                if (pred['direction'] == 'long' and pct_change < -3) or \
                   (pred['direction'] == 'short' and pct_change < -3):
                    new_status = 'invalidated'

            if new_status:
                was_correct = new_status == 'hit_tp'
                acur.execute("""
                    UPDATE momentum_predictions
                    SET status = %s, resolved_at = NOW(), actual_result_pct = %s,
                        peak_pct = %s, trough_pct = %s, duration_minutes = %s,
                        was_correct = %s, max_favorable_pct = %s
                    WHERE prediction_id = %s
                """, (new_status, round(pct_change, 4), round(peak, 4), round(trough, 4),
                      duration, was_correct, round(peak, 4), pred['prediction_id']))
                
                resolved_count += 1
                logger.info(f"[RESOLVE] {symbol} {pred['direction']} → {new_status} ({pct_change:+.2f}%)")

                # Autokorrektur starten
                if not was_correct:
                    run_autocorrection(aconn, cconn, pred, pct_change)
            else:
                # Peak/Trough updaten
                acur.execute("""
                    UPDATE momentum_predictions SET peak_pct = %s, trough_pct = %s
                    WHERE prediction_id = %s
                """, (round(peak, 4), round(trough, 4), pred['prediction_id']))

        aconn.commit()
        return resolved_count


# ============================================
# AUTOKORREKTUR
# ============================================



def track_resolved_predictions():
    """Post-Resolve Tracking: Candle-by-Candle Analyse seit Entry.
    Berechnet realistisches max_favorable (mit Drawdown-Beruecksichtigung)
    und optimales SL fuer maximalen Gewinn."""
    REVERSAL_THRESHOLD = 1.5  # 1.5% Ruecklauf vom Peak = Trendwende
    MAX_TRACK_HOURS = 48
    
    with app_db() as aconn, coins_db() as cconn:
        acur = aconn.cursor()
        ccur = cconn.cursor()

        acur.execute("""
            SELECT prediction_id, symbol, direction, entry_price, 
                   take_profit_pct, stop_loss_pct, detected_at,
                   peak_pct, trough_pct, resolved_at, max_favorable_pct
            FROM momentum_predictions
            WHERE status IN ('hit_tp', 'hit_sl')
              AND trend_reversal_at IS NULL
              AND resolved_at >= NOW() - INTERVAL '%s hours'
        """ % MAX_TRACK_HOURS)
        preds = acur.fetchall()

        if not preds:
            return 0

        tracked = 0
        for pred in preds:
            symbol = pred['symbol']
            entry = pred['entry_price']
            direction = pred['direction']
            
            # Alle 1m Candles seit Entry holen
            ccur.execute("""
                SELECT open_time, high, low, close FROM klines
                WHERE symbol = %s AND interval = '1m' AND open_time >= %s
                ORDER BY open_time ASC
            """, (symbol, pred['detected_at']))
            candles = ccur.fetchall()
            
            if len(candles) < 5:
                continue

            # Candle-by-Candle: Peak und Trough berechnen
            running_peak = 0.0  # Bester Punkt in richtige Richtung
            running_adverse = 0.0  # Schlechtester Punkt (Drawdown)
            peak_before_reversal = 0.0
            reversal_pct = None
            
            # Fuer jedes SL-Level simulieren: was waere der max Gewinn?
            sl_simulations = {}  # sl_pct -> max_gain_before_stopped
            for sl_test in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
                sl_simulations[sl_test] = {'stopped': False, 'max_gain': 0.0}
            
            for candle in candles:
                if direction == 'long':
                    high_pct = ((candle['high'] - entry) / entry) * 100
                    low_pct = ((candle['low'] - entry) / entry) * 100
                    close_pct = ((candle['close'] - entry) / entry) * 100
                    favorable = high_pct
                    adverse = low_pct
                else:
                    high_pct = ((entry - candle['low']) / entry) * 100  # guenstig bei short
                    low_pct = ((entry - candle['high']) / entry) * 100  # ungünstig bei short
                    close_pct = ((entry - candle['close']) / entry) * 100
                    favorable = high_pct
                    adverse = low_pct
                
                running_peak = max(running_peak, favorable)
                running_adverse = min(running_adverse, adverse)
                
                # SL Simulationen updaten
                for sl_pct, sim in sl_simulations.items():
                    if not sim['stopped']:
                        if adverse <= -sl_pct:
                            sim['stopped'] = True
                        else:
                            sim['max_gain'] = max(sim['max_gain'], favorable)
                
                # Trendwende erkennen: Peak erreicht und dann REVERSAL_THRESHOLD% zurueck
                if running_peak > 0 and reversal_pct is None:
                    if (running_peak - close_pct) >= REVERSAL_THRESHOLD:
                        reversal_pct = close_pct
                        peak_before_reversal = running_peak

            # Bestes SL finden (das mit hoechstem max_gain das nicht gestoppt wurde, 
            # oder falls gestoppt: hoechster Gain vor Stop)
            best_sl = None
            best_gain = 0.0
            for sl_pct in sorted(sl_simulations.keys()):
                sim = sl_simulations[sl_pct]
                if sim['max_gain'] > best_gain:
                    best_gain = sim['max_gain']
                    best_sl = sl_pct

            # Optimal SL als JSON in correction_data speichern
            import json
            sl_analysis = {
                'sl_simulations': {str(k): {'max_gain': round(v['max_gain'], 2), 'stopped': v['stopped']} for k, v in sl_simulations.items()},
                'optimal_sl': best_sl,
                'optimal_gain': round(best_gain, 2),
                'max_drawdown': round(running_adverse, 2)
            }

            if reversal_pct is not None:
                # Trendwende erkannt
                acur.execute("""
                    UPDATE momentum_predictions 
                    SET max_favorable_pct = %s, max_adverse_pct = %s,
                        trend_reversal_at = NOW(), trend_reversal_pct = %s,
                        correction_data = COALESCE(correction_data, '{}'::jsonb) || %s::jsonb
                    WHERE prediction_id = %s
                """, (round(running_peak, 4), round(running_adverse, 4),
                      round(reversal_pct, 4), json.dumps(sl_analysis),
                      pred['prediction_id']))
                tracked += 1
                opt_info = f"optimal SL={best_sl}% -> {best_gain:+.1f}%" if best_sl else ""
                logger.info(f"[POST-TRACK] {symbol} {direction}: peak={running_peak:+.2f}% drawdown={running_adverse:+.2f}% reversal={reversal_pct:+.2f}% {opt_info}")
            else:
                # Noch keine Trendwende - nur updaten
                acur.execute("""
                    UPDATE momentum_predictions 
                    SET max_favorable_pct = %s, max_adverse_pct = %s
                    WHERE prediction_id = %s
                """, (round(running_peak, 4), round(running_adverse, 4),
                      pred['prediction_id']))

        aconn.commit()
        return tracked


def run_autocorrection(aconn, cconn, pred, actual_pct):
    """
    Simuliert: was wäre gewesen mit leicht veränderten TP/SL?
    Prüft Varianten und speichert die beste Alternative.
    """
    symbol = pred['symbol']
    entry = pred['entry_price']
    detected = pred['detected_at']
    direction = pred['direction']

    # Historische Preise seit Detection holen
    ccur = cconn.cursor()
    ccur.execute("""
        SELECT bucket, high, low, close FROM agg_1h
        WHERE symbol = %s AND bucket >= %s AND bucket <= %s
        ORDER BY bucket
    """, (symbol, detected, detected + timedelta(hours=72)))
    candles = ccur.fetchall()

    if len(candles) < 2:
        return

    # Varianten simulieren
    tp_variants = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    sl_variants = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    best_improvement = None
    acur = aconn.cursor()

    for tp_pct in tp_variants:
        for sl_pct in sl_variants:
            if direction == 'long':
                tp_price = entry * (1 + tp_pct / 100)
                sl_price = entry * (1 - sl_pct / 100)
            else:
                tp_price = entry * (1 - tp_pct / 100)
                sl_price = entry * (1 + sl_pct / 100)

            sim_result = None
            sim_pct = 0

            for c in candles:
                if direction == 'long':
                    if c['high'] >= tp_price:
                        sim_result = 'hit_tp'
                        sim_pct = tp_pct
                        break
                    if c['low'] <= sl_price:
                        sim_result = 'hit_sl'
                        sim_pct = -sl_pct
                        break
                else:
                    if c['low'] <= tp_price:
                        sim_result = 'hit_tp'
                        sim_pct = tp_pct
                        break
                    if c['high'] >= sl_price:
                        sim_result = 'hit_sl'
                        sim_pct = -sl_pct
                        break

            if sim_result is None:
                sim_result = 'expired'
                last_close = candles[-1]['close']
                if direction == 'long':
                    sim_pct = ((last_close - entry) / entry) * 100
                else:
                    sim_pct = ((entry - last_close) / entry) * 100

            improvement = sim_pct - actual_pct

            if best_improvement is None or improvement > best_improvement:
                best_improvement = improvement
                best_variant = {
                    'tp_pct': tp_pct, 'sl_pct': sl_pct,
                    'sim_result': sim_result, 'sim_pct': round(sim_pct, 4),
                    'improvement': round(improvement, 4)
                }

    # Beste Variante speichern + globalen Impact prüfen
    if best_variant and best_variant['improvement'] > 0:
        new_tp = best_variant['tp_pct']
        new_sl = best_variant['sl_pct']

        # === GLOBALER IMPACT-TEST ===
        # Alle bisherigen resolved predictions holen (gleicher User)
        acur.execute("""
            SELECT prediction_id, symbol, direction, entry_price, status,
                   take_profit_pct, stop_loss_pct, detected_at, actual_result_pct
            FROM momentum_predictions
            WHERE user_id = %s AND status IN ('hit_tp', 'hit_sl', 'expired', 'invalidated')
              AND prediction_id != %s
        """, (pred['user_id'], pred['prediction_id']))
        past_predictions = acur.fetchall()

        tp_kept = 0
        tp_lost = 0
        total_tested = len(past_predictions)
        ccur2 = cconn.cursor()

        for pp in past_predictions:
            if pp['status'] != 'hit_tp':
                continue  # Nur prüfen ob TP-Treffer erhalten bleiben

            # Historische Candles für diese Prediction holen
            ccur2.execute("""
                SELECT high, low, close FROM agg_1h
                WHERE symbol = %s AND bucket >= %s AND bucket <= %s + INTERVAL '72 hours'
                ORDER BY bucket
            """, (pp['symbol'], pp['detected_at'], pp['detected_at']))
            pp_candles = ccur2.fetchall()

            if not pp_candles:
                tp_kept += 1
                continue

            # Simulation mit neuen TP/SL Werten
            pp_entry = pp['entry_price']
            if pp['direction'] == 'long':
                sim_tp_price = pp_entry * (1 + new_tp / 100)
                sim_sl_price = pp_entry * (1 - new_sl / 100)
            else:
                sim_tp_price = pp_entry * (1 - new_tp / 100)
                sim_sl_price = pp_entry * (1 + new_sl / 100)

            sim_hit = None
            for c in pp_candles:
                if pp['direction'] == 'long':
                    if c['high'] >= sim_tp_price:
                        sim_hit = 'hit_tp'
                        break
                    if c['low'] <= sim_sl_price:
                        sim_hit = 'hit_sl'
                        break
                else:
                    if c['low'] <= sim_tp_price:
                        sim_hit = 'hit_tp'
                        break
                    if c['high'] >= sim_sl_price:
                        sim_hit = 'hit_sl'
                        break

            if sim_hit == 'hit_tp':
                tp_kept += 1
            else:
                tp_lost += 1

        # Empfehlung: apply nur wenn max 1 bisheriger TP verloren geht
        recommendation = 'apply' if tp_lost <= 1 else 'reject'

        acur.execute("""
            INSERT INTO momentum_corrections 
            (prediction_id, original_tp_pct, original_sl_pct,
             simulated_tp_pct, simulated_sl_pct, simulated_result,
             simulated_result_pct, improvement_pct,
             global_tp_kept, global_tp_lost, global_total_tested, recommendation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (pred['prediction_id'], pred['take_profit_pct'], pred['stop_loss_pct'],
              new_tp, new_sl, best_variant['sim_result'], best_variant['sim_pct'],
              best_variant['improvement'],
              tp_kept, tp_lost, total_tested, recommendation))
        
        best_variant['global_tp_kept'] = tp_kept
        best_variant['global_tp_lost'] = tp_lost
        best_variant['recommendation'] = recommendation

        acur.execute("""
            UPDATE momentum_predictions SET correction_data = %s
            WHERE prediction_id = %s
        """, (Json(best_variant), pred['prediction_id']))
        
        logger.info(f"[CORRECTION] {symbol}: TP {pred['take_profit_pct']}->{new_tp}%, "
                     f"SL {pred['stop_loss_pct']}->{new_sl}% -> {best_variant['sim_result']} "
                     f"(+{best_variant['improvement']:.2f}%) | "
                     f"Global: {tp_kept} kept, {tp_lost} lost -> {recommendation}")



# ============================================
# STATS UPDATE
# ============================================

def update_stats(user_id):
    """Aktualisiert die Statistik-Tabelle für einen User (gesamt + long/short getrennt)"""
    with app_db() as conn:
        cur = conn.cursor()
        
        time_periods = {
            '24h': "detected_at >= NOW() - INTERVAL '24 hours'",
            '7d': "detected_at >= NOW() - INTERVAL '7 days'",
            '30d': "detected_at >= NOW() - INTERVAL '30 days'",
            'all': "TRUE"
        }

        # Alle Kombinationen: gesamt + long + short
        combos = []
        for tp, tw in time_periods.items():
            combos.append((tp, tw, None))  # gesamt
            combos.append((f'long_{tp}', tw, 'long'))
            combos.append((f'short_{tp}', tw, 'short'))

        for period_key, time_where, direction in combos:
            dir_where = f" AND direction = '{direction}'" if direction else ""
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE was_correct = true) as correct,
                    COUNT(*) FILTER (WHERE was_correct = false) as incorrect,
                    COUNT(*) FILTER (WHERE status = 'expired') as expired,
                    AVG(confidence) as avg_conf,
                    AVG(actual_result_pct) as avg_result,
                    MAX(actual_result_pct) as best,
                    MIN(actual_result_pct) as worst,
                    AVG(duration_minutes) as avg_dur
                FROM momentum_predictions
                WHERE user_id = %s AND status != 'active' AND {time_where}{dir_where}
            """, (user_id,))
            
            row = cur.fetchone()
            total = row['total'] or 0
            correct = row['correct'] or 0
            hit_rate = (correct / total * 100) if total > 0 else 0

            cur.execute("""
                INSERT INTO momentum_stats (user_id, period, total_predictions, correct_predictions,
                    incorrect_predictions, expired_predictions, avg_confidence, avg_result_pct,
                    best_result_pct, worst_result_pct, hit_rate_pct, avg_duration_minutes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, period) DO UPDATE SET
                    total_predictions = EXCLUDED.total_predictions,
                    correct_predictions = EXCLUDED.correct_predictions,
                    incorrect_predictions = EXCLUDED.incorrect_predictions,
                    expired_predictions = EXCLUDED.expired_predictions,
                    avg_confidence = EXCLUDED.avg_confidence,
                    avg_result_pct = EXCLUDED.avg_result_pct,
                    best_result_pct = EXCLUDED.best_result_pct,
                    worst_result_pct = EXCLUDED.worst_result_pct,
                    hit_rate_pct = EXCLUDED.hit_rate_pct,
                    avg_duration_minutes = EXCLUDED.avg_duration_minutes,
                    updated_at = NOW()
            """, (user_id, period_key, total, correct, row['incorrect'] or 0,
                  row['expired'] or 0, row['avg_conf'], row['avg_result'],
                  row['best'], row['worst'], round(hit_rate, 2),
                  int(row['avg_dur']) if row['avg_dur'] else None))

        conn.commit()




# ============================================
# DAILY OPTIMIZER - 2x täglich
# ============================================

_last_optimization_run = None

def should_run_optimization():
    """Prüft ob Optimierung laufen soll (08:00 und 20:00 Berlin)"""
    global _last_optimization_run
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Europe/Berlin'))
    run_hours = [8]  # 1x täglich für Threshold Optimizer
    
    if now.hour in run_hours:
        today_key = f"{now.date()}-{now.hour}"
        if _last_optimization_run != today_key:
            return today_key
    return None


def _optimize_direction(user_id, direction, resolved, config, acur, ccur):
    """Optimiert eine einzelne Richtung (long oder short)"""
    prefix = f"{direction}_" if direction else ""
    
    preds = [p for p in resolved if p['direction'] == direction] if direction else resolved
    
    if len(preds) < 5:
        logger.info(f"[OPTIMIZER] {direction}: Nur {len(preds)} predictions, brauche min. 5. Skip.")
        return None
    
    total = len(preds)
    tp_hits = [p for p in preds if p['status'] == 'hit_tp']
    sl_hits = [p for p in preds if p['status'] == 'hit_sl']
    current_hit_rate = (len(tp_hits) / total * 100) if total > 0 else 0
    
    logger.info(f"[OPTIMIZER] {direction.upper()}: {total} predictions: {len(tp_hits)} TP, "
                f"{len(sl_hits)} SL = {current_hit_rate:.1f}% hit rate")
    
    # Aktuelle direction-spezifische Config (Fallback auf global)
    current_conf = config.get(f'{prefix}min_confidence') or config.get('min_confidence', 60)
    current_mode = config.get(f'{prefix}tp_sl_mode') or config.get('tp_sl_mode', 'dynamic')
    
    # Varianten generieren
    variants = []
    
    for conf in [60, 65, 70, 75, 80, 85, 90]:
        if conf != current_conf:
            variants.append({'min_confidence': conf, 'label': f'conf={conf}'})
    
    for tp in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for sl in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            if sl < tp:
                variants.append({
                    'fixed_tp_pct': tp, 'fixed_sl_pct': sl, 'tp_sl_mode': 'fixed',
                    'label': f'fixed tp={tp} sl={sl}'
                })
    
    for conf in [65, 70, 75, 80]:
        for tp in [3.0, 5.0, 7.0]:
            for sl in [1.5, 2.0, 3.0]:
                if sl < tp:
                    variants.append({
                        'min_confidence': conf, 'fixed_tp_pct': tp, 'fixed_sl_pct': sl,
                        'tp_sl_mode': 'fixed',
                        'label': f'conf={conf} tp={tp} sl={sl}'
                    })
    
    # pct_30m zum Zeitpunkt jeder Prediction einmalig laden
    for pred in preds:
        ccur.execute("""
            SELECT pct_30m FROM kline_metrics
            WHERE symbol = %s AND open_time <= %s
            ORDER BY open_time DESC LIMIT 1
        """, (pred['symbol'], pred['detected_at']))
        row = ccur.fetchone()
        pred['_pct_30m'] = float(row['pct_30m']) if row and row['pct_30m'] is not None else None

    # pct_30m Varianten hinzufügen (richtungsabhängig)
    for pct_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        variants.append({
            'pct_30m_min': pct_thresh,
            'label': f'pct_30m>={pct_thresh}%'
        })
        # Kombis mit TP/SL
        for tp in [2.0, 3.0, 5.0]:
            for sl in [1.5, 2.0, 3.0]:
                if sl < tp:
                    variants.append({
                        'pct_30m_min': pct_thresh,
                        'fixed_tp_pct': tp, 'fixed_sl_pct': sl, 'tp_sl_mode': 'fixed',
                        'label': f'pct_30m>={pct_thresh} tp={tp} sl={sl}'
                    })

    logger.info(f"[OPTIMIZER] {direction.upper()}: Testing {len(variants)} variants")

    best_variant = None
    best_score = current_hit_rate
    best_details = None
    
    for variant in variants:
        sim_tp = 0
        sim_sl = 0
        sim_eliminated = 0
        sim_tp_lost = 0
        sim_total = 0
        
        v_conf = variant.get('min_confidence', current_conf)
        v_tp_pct = variant.get('fixed_tp_pct', None)
        v_sl_pct = variant.get('fixed_sl_pct', None)
        
        v_pct_30m_min = variant.get('pct_30m_min', None)

        for pred in preds:
            if pred['confidence'] < v_conf:
                sim_eliminated += 1
                if pred['status'] == 'hit_tp':
                    sim_tp_lost += 1
                continue

            # pct_30m Filter: Prediction hätte nicht ausgelöst werden dürfen
            if v_pct_30m_min is not None and pred.get('_pct_30m') is not None:
                pct_val = pred['_pct_30m']
                if direction == 'long' and pct_val < v_pct_30m_min:
                    sim_eliminated += 1
                    if pred['status'] == 'hit_tp':
                        sim_tp_lost += 1
                    continue
                if direction == 'short' and pct_val > -v_pct_30m_min:
                    sim_eliminated += 1
                    if pred['status'] == 'hit_tp':
                        sim_tp_lost += 1
                    continue
            
            sim_total += 1
            
            if v_tp_pct and v_sl_pct:
                ccur.execute("""
                    SELECT high, low, close FROM klines
                    WHERE symbol = %s AND interval = '1m'
                      AND open_time >= %s AND open_time <= %s + INTERVAL '72 hours'
                    ORDER BY open_time
                """, (pred['symbol'], pred['detected_at'], pred['detected_at']))
                candles = ccur.fetchall()
                
                if not candles:
                    if pred['status'] == 'hit_tp':
                        sim_tp += 1
                    else:
                        sim_sl += 1
                    continue
                
                entry = pred['entry_price']
                if pred['direction'] == 'long':
                    sim_tp_price = entry * (1 + v_tp_pct / 100)
                    sim_sl_price = entry * (1 - v_sl_pct / 100)
                else:
                    sim_tp_price = entry * (1 - v_tp_pct / 100)
                    sim_sl_price = entry * (1 + v_sl_pct / 100)
                
                hit = None
                for c in candles:
                    if pred['direction'] == 'long':
                        if c['high'] >= sim_tp_price:
                            hit = 'tp'
                            break
                        if c['low'] <= sim_sl_price:
                            hit = 'sl'
                            break
                    else:
                        if c['low'] <= sim_tp_price:
                            hit = 'tp'
                            break
                        if c['high'] >= sim_sl_price:
                            hit = 'sl'
                            break
                
                if hit == 'tp':
                    sim_tp += 1
                else:
                    sim_sl += 1
            else:
                if pred['status'] == 'hit_tp':
                    sim_tp += 1
                else:
                    sim_sl += 1
        
        if sim_total == 0:
            continue
        
        sim_hit_rate = (sim_tp / sim_total * 100)
        score = sim_hit_rate - (sim_tp_lost * 5)
        
        if score > best_score and sim_tp_lost <= 1:
            best_score = score
            best_variant = variant
            best_details = {
                'sim_total': sim_total,
                'sim_tp': sim_tp,
                'sim_sl': sim_sl,
                'sim_eliminated': sim_eliminated,
                'sim_tp_lost': sim_tp_lost,
                'sim_hit_rate': round(sim_hit_rate, 2),
            }
    
    return {
        'direction': direction,
        'total': total,
        'tp_hits': len(tp_hits),
        'sl_hits': len(sl_hits),
        'current_hit_rate': current_hit_rate,
        'variants_tested': len(variants),
        'best_variant': best_variant,
        'best_details': best_details or {'sim_total': 0, 'sim_tp': 0, 'sim_sl': 0,
                                          'sim_eliminated': 0, 'sim_tp_lost': 0, 'sim_hit_rate': 0},
    }


def run_daily_optimization(user_id, config):
    """
    Tägliche Optimierung: Long und Short GETRENNT optimieren.
    Jede Richtung bekommt eigene Parameter.
    """
    global _last_optimization_run
    
    logger.info(f"[OPTIMIZER] Starting daily optimization for user {user_id}...")
    
    with app_db() as aconn, coins_db() as cconn:
        acur = aconn.cursor()
        ccur = cconn.cursor()
        
        acur.execute("""
            SELECT p.*, p.signals::text as signals_text
            FROM momentum_predictions p
            WHERE p.user_id = %s 
              AND p.status IN ('hit_tp', 'hit_sl', 'expired', 'invalidated')
              AND p.detected_at >= NOW() - INTERVAL '7 days'
            ORDER BY p.detected_at
        """, (user_id,))
        resolved = acur.fetchall()
        
        if len(resolved) < 10:
            logger.info(f"[OPTIMIZER] Nur {len(resolved)} resolved predictions, brauche min. 10. Skip.")
            return
        
        total = len(resolved)
        tp_hits = [p for p in resolved if p['status'] == 'hit_tp']
        sl_hits = [p for p in resolved if p['status'] == 'hit_sl']
        expired = [p for p in resolved if p['status'] in ('expired', 'invalidated')]
        current_hit_rate = (len(tp_hits) / total * 100) if total > 0 else 0
        
        logger.info(f"[OPTIMIZER] {total} predictions total: {len(tp_hits)} TP, {len(sl_hits)} SL, "
                     f"{len(expired)} exp/inv = {current_hit_rate:.1f}% hit rate")
        
        # === GETRENNTE OPTIMIERUNG ===
        for direction in ['long', 'short']:
            result = _optimize_direction(user_id, direction, resolved, config, acur, ccur)
            
            if result is None:
                continue
            
            prefix = f"{direction}_"
            best_variant = result['best_variant']
            best_details = result['best_details']
            
            if best_variant and best_details:
                improvement = best_details['sim_hit_rate'] - result['current_hit_rate']
                
                if improvement >= 3.0 and best_details['sim_tp_lost'] <= 2:
                    recommendation = 'apply'
                    reason = (f"{direction.upper()} hit rate {result['current_hit_rate']:.1f}% -> "
                              f"{best_details['sim_hit_rate']:.1f}% (+{improvement:.1f}%), "
                              f"{best_details['sim_eliminated']} eliminated, "
                              f"{best_details['sim_tp_lost']} TP lost")
                    
                    logger.info(f"[OPTIMIZER] APPLY {direction.upper()}: {best_variant['label']} -> {reason}")
                    
                    # Direction-spezifische Config anwenden
                    changes = {}
                    if 'min_confidence' in best_variant:
                        field = f'{prefix}min_confidence'
                        old_val = config.get(field) or config.get('min_confidence', 60)
                        new_val = best_variant['min_confidence']
                        if old_val != new_val:
                            changes[field] = {'old': old_val, 'new': new_val}
                    
                    if 'tp_sl_mode' in best_variant:
                        field = f'{prefix}tp_sl_mode'
                        old_val = config.get(field) or config.get('tp_sl_mode', 'dynamic')
                        new_val = best_variant['tp_sl_mode']
                        if old_val != new_val:
                            changes[field] = {'old': old_val, 'new': new_val}
                    
                    if 'fixed_tp_pct' in best_variant:
                        field = f'{prefix}fixed_tp_pct'
                        old_val = float(config.get(field) or config.get('fixed_tp_pct', 5.0))
                        new_val = best_variant['fixed_tp_pct']
                        if old_val != new_val:
                            changes[field] = {'old': old_val, 'new': new_val}
                    
                    if 'fixed_sl_pct' in best_variant:
                        field = f'{prefix}fixed_sl_pct'
                        old_val = float(config.get(field) or config.get('fixed_sl_pct', 2.0))
                        new_val = best_variant['fixed_sl_pct']
                        if old_val != new_val:
                            changes[field] = {'old': old_val, 'new': new_val}
                    
                    if 'pct_30m_min' in best_variant:
                        field = f'{prefix}pct_30m_min'
                        old_val = float(config.get(field) or 1.5)
                        new_val = best_variant['pct_30m_min']
                        if old_val != new_val:
                            changes[field] = {'old': old_val, 'new': new_val}
                    
                    if changes:
                        set_parts = []
                        set_vals = []
                        for field, vals in changes.items():
                            set_parts.append(f"{field} = %s")
                            set_vals.append(vals['new'])
                        set_vals.append(user_id)
                        
                        acur.execute(
                            f"UPDATE momentum_scan_config SET {', '.join(set_parts)}, updated_at = NOW() "
                            f"WHERE user_id = %s",
                            set_vals
                        )
                        logger.info(f"[OPTIMIZER] {direction.upper()} config updated: {changes}")
                    
                    applied = True
                else:
                    recommendation = 'no_change'
                    reason = (f"{direction.upper()} improvement {improvement:.1f}% too small (need >=3%) or "
                              f"too many TP lost ({best_details['sim_tp_lost']})")
                    applied = False
                    changes = {}
                    logger.info(f"[OPTIMIZER] {direction.upper()} NO CHANGE: {reason}")
            else:
                recommendation = 'no_change'
                reason = f"{direction.upper()}: No variant improves current performance"
                applied = False
                changes = {}
                best_details = {'sim_total': 0, 'sim_tp': 0, 'sim_sl': 0,
                               'sim_eliminated': 0, 'sim_tp_lost': 0, 'sim_hit_rate': 0}
                logger.info(f"[OPTIMIZER] {direction.upper()} NO CHANGE: {reason}")
            
            # Log speichern (pro Richtung)
            acur.execute("""
                INSERT INTO momentum_optimization_log 
                (user_id, period_start, period_end, total_predictions, total_tp, total_sl,
                 total_expired, current_hit_rate, simulations_run, best_variant,
                 best_sim_hit_rate, best_sim_tp, best_sim_sl, best_sim_eliminated,
                 best_sim_tp_lost, recommendation, applied, reason, changes_applied)
                VALUES (%s, NOW() - INTERVAL '7 days', NOW(), %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, result['total'], result['tp_hits'], result['sl_hits'],
                  0,  # expired for this direction
                  round(result['current_hit_rate'], 2), result['variants_tested'],
                  Json(best_variant) if best_variant else None,
                  best_details['sim_hit_rate'], best_details['sim_tp'],
                  best_details['sim_sl'], best_details['sim_eliminated'],
                  best_details['sim_tp_lost'], recommendation, applied, reason,
                  Json(changes) if changes else None))
        
        aconn.commit()
        logger.info(f"[OPTIMIZER] Done. Long and Short optimized separately.")

# ============================================

def get_symbols_for_config(config):
    """Holt Symbole basierend auf Scan-Config (Coingruppe oder alle)"""
    if config['scan_all_symbols']:
        with coins_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT symbol FROM agg_1h ORDER BY symbol")
            return [r['symbol'] for r in cur.fetchall()]
    elif config['coin_group_id']:
        with app_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol FROM coin_group_members WHERE group_id = %s", 
                       (config['coin_group_id'],))
            return [r['symbol'] for r in cur.fetchall()]
    return []



def get_market_context(ccur):
    """
    Marktbreite über alle USDC-Symbole aus kline_metrics.
    Einmal pro Loop-Durchgang — eine Query, kein externes API.
    market_trend = avg_4h * 0.5 + (breadth_4h - 0.5) * 2 * 0.5  → -1..+1
    """
    try:
        ccur.execute("""
            SELECT
                AVG(pct_240m)                                          AS avg_4h,
                COUNT(*) FILTER (WHERE pct_240m > 0)::float / COUNT(*) AS breadth_4h,
                AVG(pct_60m)                                           AS avg_1h,
                COUNT(*) FILTER (WHERE pct_60m  > 0)::float / COUNT(*) AS breadth_1h
            FROM kline_metrics
            WHERE open_time = (
                SELECT MAX(open_time) FROM kline_metrics
                WHERE open_time <= date_trunc('hour', NOW())
            )
            AND pct_240m IS NOT NULL
        """)
        row = ccur.fetchone()
        if not row or row['avg_4h'] is None:
            return None

        avg_4h     = float(row['avg_4h'])
        breadth_4h = float(row['breadth_4h'])
        avg_1h     = float(row['avg_1h'] or 0)
        breadth_1h = float(row['breadth_1h'] or 0.5)

        # Normalisieren: avg_4h Sättigung bei ±2%
        norm_avg    = max(-1.0, min(1.0, avg_4h / 2.0))
        # breadth zentriert um 0.5 (neutral=0), Sättigung bei ±0.5
        norm_breadth = max(-1.0, min(1.0, (breadth_4h - 0.5) * 2.0))

        # Gleichgewichtet 50/50
        market_trend = round(norm_avg * 0.5 + norm_breadth * 0.5, 3)

        return {
            'avg_4h':       round(avg_4h, 3),
            'breadth_4h':   round(breadth_4h, 3),
            'avg_1h':       round(avg_1h, 3),
            'breadth_1h':   round(breadth_1h, 3),
            'market_trend': market_trend,
        }
    except Exception as e:
        logger.warning(f"[MKT_CTX] Fehler: {e}")
        return None


def scan_symbols(config, symbols):
    """Scannt eine Liste von Symbolen und erstellt Predictions"""
    user_id = config['user_id']
    min_conf = config['min_confidence'] or 60
    new_predictions = 0

    # TP/SL Config - direction-spezifisch (Fallback auf global)
    dir_config = {
        'long': {
            'tp_sl_mode': config.get('long_tp_sl_mode') or config.get('tp_sl_mode', 'dynamic'),
            'tp_pct': float(config.get('long_fixed_tp_pct') or config.get('fixed_tp_pct', 5.0)),
            'sl_pct': float(config.get('long_fixed_sl_pct') or config.get('fixed_sl_pct', 2.0)),
            'min_conf': config.get('long_min_confidence') or config.get('min_confidence', 60),
        },
        'short': {
            'tp_sl_mode': config.get('short_tp_sl_mode') or config.get('tp_sl_mode', 'dynamic'),
            'tp_pct': float(config.get('short_fixed_tp_pct') or config.get('fixed_tp_pct', 5.0)),
            'sl_pct': float(config.get('short_fixed_sl_pct') or config.get('fixed_sl_pct', 2.0)),
            'min_conf': config.get('short_min_confidence') or config.get('min_confidence', 60),
        },
    }
    cfg_range_tp_min = float(config.get('range_tp_min', 5.0))
    cfg_range_tp_max = float(config.get('range_tp_max', 15.0))
    cfg_range_sl_min = float(config.get('range_sl_min', 2.0))
    cfg_range_sl_max = float(config.get('range_sl_max', 6.0))

    with coins_db() as cconn, app_db() as aconn:
        ccur = cconn.cursor()
        acur = aconn.cursor()

        # Aktive Predictions holen (kein Doppel-Signal pro Symbol)
        acur.execute("""
            SELECT symbol FROM momentum_predictions 
            WHERE user_id = %s AND status = 'active'
        """, (user_id,))
        active_symbols = {r['symbol'] for r in acur.fetchall()}

        # Cooldown: Symbole die in den letzten 60 Min resolved wurden nicht nochmal scannen
        acur.execute("""
            SELECT DISTINCT symbol FROM momentum_predictions
            WHERE user_id = %s AND status != 'active'
              AND resolved_at >= NOW() - INTERVAL '60 minutes'
        """, (user_id,))
        cooldown_symbols = {r['symbol'] for r in acur.fetchall()}
        active_symbols = active_symbols | cooldown_symbols

        # Marktkontext einmal pro Loop holen (alle USDC-Symbole aus kline_metrics)
        market_context = get_market_context(ccur)
        if market_context:
            logger.info(f"[MKT_CTX] avg_4h={market_context['avg_4h']:+.2f}% breadth_4h={market_context['breadth_4h']:.2f} trend={market_context['market_trend']:+.3f}")
        else:
            logger.warning("[MKT_CTX] Kein Marktkontext verfügbar, kein Score-Modifier")

        cycle_short_count = 0  # Batch-Limit: max Shorts pro Scan-Zyklus

        for symbol in symbols:
            if not running:
                break
                
            if symbol in active_symbols:
                continue  # Schon aktive Prediction

            try:
                # 1h Candles (letzte 50)
                ccur.execute("""
                    SELECT bucket, open, high, low, close, volume 
                    FROM agg_1h WHERE symbol = %s 
                    ORDER BY bucket DESC LIMIT 50
                """, (symbol,))
                candles_1h = list(reversed(ccur.fetchall()))

                # 4h Candles (letzte 20)
                ccur.execute("""
                    SELECT bucket, open, high, low, close, volume 
                    FROM agg_4h WHERE symbol = %s 
                    ORDER BY bucket DESC LIMIT 20
                """, (symbol,))
                candles_4h = list(reversed(ccur.fetchall()))

                # 1d Candles (letzte 14)
                ccur.execute("""
                    SELECT bucket, open, high, low, close, volume 
                    FROM agg_1d WHERE symbol = %s 
                    ORDER BY bucket DESC LIMIT 14
                """, (symbol,))
                candles_1d = list(reversed(ccur.fetchall()))

                if not candles_1h:
                    continue

                # Aktuellen Preis aus 1m-Klines holen (nicht agg_1h!)
                ccur.execute("""
                    SELECT close FROM klines
                    WHERE symbol = %s AND interval = '1m'
                    ORDER BY open_time DESC LIMIT 1
                """, (symbol,))
                row_1m = ccur.fetchone()
                current_price = row_1m['close'] if row_1m else candles_1h[-1]['close']
                if not current_price or current_price == 0:
                    continue

                # Analyse (liefert Signal + expected_move, OHNE TP/SL)
                signal = analyze_symbol(candles_1h, candles_4h, candles_1d, current_price, market_context=market_context, scan_config=config)

                if signal and signal['confidence'] >= dir_config[signal['direction']]['min_conf']:
                    # Filter: erwartete Bewegung muss min_target_pct überschreiten
                    if signal['expected_move_pct'] < (config['min_target_pct'] or 5.0):
                        continue

                    # Momentum-Check: pct_30m muss in Signal-Richtung laufen
                    ccur.execute("""
                        SELECT pct_30m, pct_60m FROM kline_metrics
                        WHERE symbol = %s
                        ORDER BY open_time DESC LIMIT 1
                    """, (symbol,))
                    metrics_row = ccur.fetchone()
                    if metrics_row and metrics_row['pct_30m'] is not None:
                        pct_30m = float(metrics_row['pct_30m'])
                        long_pct_min = float(config.get('long_pct_30m_min') or 2.5)
                        short_pct_min = float(config.get('short_pct_30m_min') or 1.0)
                        if signal['direction'] == 'long' and pct_30m < long_pct_min:
                            continue  # Kein Long ohne ausreichend Momentum
                        if signal['direction'] == 'short' and pct_30m > -short_pct_min:
                            continue  # Kein Short ohne ausreichend Momentum

                        # Short-Blocker: Move schon zu weit gelaufen → Bounce wahrscheinlich
                        if signal['direction'] == 'short':
                            pct_60m = float(metrics_row.get('pct_60m') or 0) if metrics_row.get('pct_60m') is not None else 0
                            if pct_60m < -3.0:
                                logger.debug(f"[LATE_SHORT] {symbol} skipped — pct_60m={pct_60m:.2f}% (move already ran)")
                                continue

                    # Batch-Limit: Nicht mehr als 15 Shorts pro Scan-Zyklus
                    if signal['direction'] == 'short' and cycle_short_count >= 15:
                        logger.info(f"[BATCH_LIMIT] {symbol} short skipped — already {cycle_short_count} shorts this cycle")
                        continue

                    # TP/SL aus User-Config berechnen (direction-spezifisch)
                    entry = signal['entry_price']
                    d_cfg = dir_config[signal['direction']]
                    d_mode = d_cfg['tp_sl_mode']
                    if d_mode == 'fixed':
                        tp_pct = d_cfg['tp_pct']
                        sl_pct = d_cfg['sl_pct']
                    elif d_mode == 'range':
                        atr_val = signal['indicators'].get('atr')
                        if atr_val and atr_val > 0 and entry > 0:
                            atr_pct = (atr_val / entry) * 100
                            ratio = min(max((atr_pct - 0.5) / 4.5, 0), 1)
                            tp_pct = cfg_range_tp_min + ratio * (cfg_range_tp_max - cfg_range_tp_min)
                            sl_pct = cfg_range_sl_min + ratio * (cfg_range_sl_max - cfg_range_sl_min)
                        else:
                            tp_pct = cfg_range_tp_min
                            sl_pct = cfg_range_sl_min
                    else:  # dynamic
                        atr_val = signal['indicators'].get('atr')
                        if atr_val and atr_val > 0:
                            sl_dist = max(atr_val * 1.5, entry * 0.02)
                            tp_dist = sl_dist * 2.5
                            tp_pct = (tp_dist / entry) * 100
                            sl_pct = (sl_dist / entry) * 100
                            tp_pct = max(2.0, min(tp_pct, 25.0))
                            sl_pct = max(1.0, min(sl_pct, 10.0))
                        else:
                            tp_pct = 5.0
                            sl_pct = 2.0

                    # TP/SL Preise
                    if signal['direction'] == 'long':
                        tp_price = entry * (1 + tp_pct / 100)
                        sl_price = entry * (1 - sl_pct / 100)
                    else:
                        tp_price = entry * (1 - tp_pct / 100)
                        sl_price = entry * (1 + sl_pct / 100)

                    acur.execute("""
                        INSERT INTO momentum_predictions 
                        (user_id, symbol, direction, entry_price, take_profit_price,
                         stop_loss_price, take_profit_pct, stop_loss_pct,
                         confidence, reason, signals, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW() + INTERVAL '72 hours')
                        RETURNING prediction_id
                    """, (user_id, symbol, signal['direction'], entry,
                          float(tp_price), float(sl_price),
                          float(tp_pct), float(sl_pct),
                          int(signal["confidence"]), signal["reason"],
                          Json(signal['signals'])))
                    
                    pred_id = acur.fetchone()['prediction_id']
                    new_predictions += 1
                    if signal['direction'] == 'short':
                        cycle_short_count += 1
                    logger.info(f"[NEW] #{pred_id} {symbol} {signal['direction'].upper()} "
                               f"@ {current_price:.4f} (expect {signal['expected_move_pct']:.1f}%) "
                               f"→ TP {tp_pct:.1f}% SL {sl_pct:.1f}% (conf: {signal['confidence']})")

            except Exception as e:
                logger.error(f"[SCAN] {symbol} error: {e}")
                continue

        aconn.commit()
    return new_predictions


def main():
    """Main Loop"""
    logger.info("=" * 60)
    logger.info("Momentum Scanner starting...")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)

    while running:
        try:
            # Aktive Configs laden
            with app_db() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM momentum_scan_config WHERE is_active = true")
                configs = cur.fetchall()

            if not configs:
                time.sleep(10)
                continue

            for config in configs:
                if not running:
                    break

                user_id = config['user_id']
                idle = config['idle_seconds'] or 60

                # 0. Daily Optimizer (08:00 + 20:00 Berlin)
                opt_key = should_run_optimization()
                if opt_key:
                    try:
                        global _last_optimization_run
                        # Threshold Optimizer: Discovery-basiert, passt Scanner-Thresholds an
                        try:
                            run_threshold_optimization(14, 4.0)
                            logger.info("[OPTIMIZER] Threshold optimization complete")
                        except Exception as te:
                            logger.error(f"[OPTIMIZER] Threshold optimization failed: {te}")
                        # Legacy TP/SL Optimizer danach
                        run_daily_optimization(user_id, config)
                        _last_optimization_run = opt_key
                    except Exception as oe:
                        logger.error(f"[OPTIMIZER] Error: {oe}")
                        logger.error(traceback.format_exc())

                # 1. Aktive Predictions prüfen
                logger.info(f"[LOOP] Checking active predictions...")
                resolved = check_active_predictions()
                logger.info(f"[LOOP] Resolved: {resolved}")
                if resolved > 0:
                    update_stats(user_id)
                    logger.info(f"[MONITOR] Resolved {resolved} predictions for user {user_id}")

                # 1b. Post-Resolve Tracking (max favorable + Trendwende)
                try:
                    tracked = track_resolved_predictions()
                    if tracked > 0:
                        logger.info(f"[POST-TRACK] {tracked} predictions trend reversal detected")
                except Exception as te:
                    logger.error(f"[POST-TRACK] Error: {te}")

                # 2. Neue Symbole scannen
                symbols = get_symbols_for_config(config)
                logger.info(f"[LOOP] Scanning {len(symbols) if symbols else 0} symbols...")
                if symbols:
                    new = scan_symbols(config, symbols)
                    if new > 0:
                        logger.info(f"[SCAN] Created {new} new predictions for user {user_id}")

                # 3. Stats updaten
                update_stats(user_id)

                # Idle
                logger.info(f"[LOOP] Cycle done, sleeping {idle}s...")
                for _ in range(idle):
                    if not running:
                        break
                    time.sleep(1)

        except Exception as e:
            logger.error(f"[MAIN] Error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(30)

    logger.info("Momentum Scanner stopped.")


if __name__ == '__main__':
    main()
