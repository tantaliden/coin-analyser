#!/usr/bin/env python3
"""
Momentum Scanner 2h — CNN-basiert, eigenständiger Scanner für 2%-Moves in ≤2h.
Eigenes Modell: /opt/CNN/models/best_cnn_2h.pth
Eigene Config: momentum_scan_config_2h
Eigene Predictions: scanner_type = 'cnn_2h'
Eigenes Heartbeat: /opt/coin/logs/.scanner_2h_heartbeat
"""

import os
import sys
import time
import json
import signal
import logging
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta, timezone
from simclock import clock
from contextlib import contextmanager

import threading
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/coin/logs/momentum_scanner_2h.log')
    ]
)
logger = logging.getLogger('momentum_scanner_2h')

# ============================================
# CONFIG aus settings.json
# ============================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

COINS_DB_CFG = SETTINGS['databases']['coins']
APP_DB_CFG = SETTINGS['databases']['app']
LEARNER_DB_CFG = SETTINGS['databases']['learner']

# ============================================
# SCANNER-SPEZIFISCHE KONSTANTEN
# ============================================
SCANNER_TYPE = 'cnn_2h'
MODEL_PATH = '/opt/CNN/models/best_cnn_2h.pth'
MODEL_INFO_PATH = '/opt/CNN/models/model_info.json'
MODEL_CANDIDATE_PATH = '/opt/CNN/models/cnn_2h_candidate.pth'
MODEL_BACKUP_DIR = '/opt/CNN/models'
HEARTBEAT_FILE = '/opt/coin/logs/.scanner_2h_heartbeat'
CONFIG_TABLE = 'momentum_scan_config_2h'
EXPIRES_INTERVAL = '72 hours'
LEARNER_MIN_PCT = 2
LEARNER_PCT_COLS = ['pct_60m', 'pct_90m', 'pct_120m']
LEARNER_DUR_MAP = {'pct_60m': 60, 'pct_90m': 90, 'pct_120m': 120}

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

@contextmanager
def learner_db():
    conn = psycopg2.connect(
        host=LEARNER_DB_CFG['host'], port=LEARNER_DB_CFG['port'],
        dbname=LEARNER_DB_CFG['name'], user=LEARNER_DB_CFG['user'],
        password=LEARNER_DB_CFG['password'], cursor_factory=RealDictCursor
    )
    try:
        yield conn
    finally:
        conn.close()


def write_prediction_feedback(pred, new_status, was_correct, pct_change, duration, time_result, peak, trough):
    """Schreibt Feedback einer resolved Prediction in die Learner-DB."""
    try:
        with learner_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO prediction_feedback
                (prediction_id, scanner_type, symbol, direction, entry_price,
                 detected_at, resolved_at, status, was_correct, actual_result_pct,
                 duration_minutes, time_result, peak_pct, trough_pct,
                 confidence, take_profit_pct, stop_loss_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id, scanner_type) DO UPDATE SET
                    status = EXCLUDED.status,
                    was_correct = EXCLUDED.was_correct,
                    actual_result_pct = EXCLUDED.actual_result_pct,
                    duration_minutes = EXCLUDED.duration_minutes,
                    time_result = EXCLUDED.time_result,
                    peak_pct = EXCLUDED.peak_pct,
                    trough_pct = EXCLUDED.trough_pct,
                    resolved_at = EXCLUDED.resolved_at
            """, (pred['prediction_id'], SCANNER_TYPE, pred['symbol'], pred['direction'],
                  pred['entry_price'], pred['detected_at'], clock.now(), new_status, was_correct,
                  round(pct_change, 4), duration, time_result, round(peak, 4), round(trough, 4),
                  pred['confidence'], float(pred['take_profit_pct']), float(pred['stop_loss_pct'])))
            conn.commit()
            logger.info(f"[FEEDBACK] #{pred['prediction_id']} → learner DB ({time_result})")
    except Exception as e:
        logger.error(f"[FEEDBACK] Fehler: {e}")


# ============================================
# CNN MODEL DEFINITION (identisch zum Hauptscanner)
# ============================================

class MultiTimeframeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_ch = 7
        self.branch_5m = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=5, padding=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=7, padding=3), nn.BatchNorm1d(48), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(48, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=9, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.branch_1h = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.branch_4h = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.branch_1d = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        fusion_size = 64*8 + 64*4 + 64*4 + 64*4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x5m, x1h, x4h, x1d):
        f5m = self.branch_5m(x5m).flatten(1)
        f1h = self.branch_1h(x1h).flatten(1)
        f4h = self.branch_4h(x4h).flatten(1)
        f1d = self.branch_1d(x1d).flatten(1)
        fused = torch.cat([f5m, f1h, f4h, f1d], dim=1)
        return self.fusion(fused).squeeze(-1)


# CNN Model Singleton
_cnn_model = None

def get_cnn_model():
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"[CNN] Modell nicht gefunden: {MODEL_PATH}")
        return None
    try:
        model = MultiTimeframeCNN()
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location='cpu'))
        model.eval()
        _cnn_model = model
        logger.info(f"[CNN] Modell geladen: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"[CNN] Modell laden fehlgeschlagen: {e}")
        return None


def normalize_candles_for_cnn(candles, expected_len):
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    n = len(candles)
    closes = np.array([float(c['close']) for c in candles])
    opens  = np.array([float(c['open']) for c in candles])
    highs  = np.array([float(c['high']) for c in candles])
    lows   = np.array([float(c['low']) for c in candles])
    volumes = np.array([float(c['volume']) for c in candles])
    trades  = np.array([float(c.get('number_of_trades') or 0) for c in candles])
    taker   = np.array([float(c.get('taker_buy_base_asset_volume') or 0) for c in candles])

    price_ret = np.zeros(n)
    for j in range(1, n):
        if closes[j-1] > 0:
            price_ret[j] = (closes[j] / closes[j-1] - 1) * 100

    med_vol = np.median(volumes)
    volume_rel = volumes / med_vol if med_vol > 0 else np.ones(n)
    med_trades = np.median(trades)
    trades_rel = trades / med_trades if med_trades > 0 else np.ones(n)
    taker_ratio = np.where(volumes > 0, taker / volumes, 0.5)
    range_pct = np.where(closes > 0, (highs - lows) / closes * 100, 0)
    full_range = highs - lows
    body_dir = np.where(full_range > 0, (closes - opens) / full_range, 0)
    hl_pos = np.where(full_range > 0, (closes - lows) / full_range, 0.5)

    channels = np.stack([price_ret, volume_rel, trades_rel, taker_ratio,
                         range_pct, body_dir, hl_pos])
    if n >= expected_len:
        channels = channels[:, -expected_len:]
    else:
        pad = np.zeros((7, expected_len - n))
        channels = np.concatenate([pad, channels], axis=1)
    channels = np.nan_to_num(channels, nan=0.0, posinf=20.0, neginf=-20.0)
    channels = np.clip(channels, -50, 50)
    return channels.astype(np.float32)


def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])


def analyze_symbol_cnn(ccur, symbol, current_price, market_context=None, scan_config=None):
    model = get_cnn_model()
    if model is None:
        return None

    try:
        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_5m WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 144""", (symbol, clock.now()))
        candles_5m = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1h WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 24""", (symbol, clock.now()))
        candles_1h = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_4h WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 12""", (symbol, clock.now()))
        candles_4h = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1d WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 14""", (symbol, clock.now()))
        candles_1d = list(reversed(ccur.fetchall()))
    except Exception as e:
        logger.debug(f"[CNN] {symbol} Daten-Fehler: {e}")
        return None

    if len(candles_5m) < 72 or len(candles_1h) < 12 or len(candles_4h) < 6 or len(candles_1d) < 7:
        return None

    x5m = normalize_candles_for_cnn(candles_5m, 144)
    x1h = normalize_candles_for_cnn(candles_1h, 24)
    x4h = normalize_candles_for_cnn(candles_4h, 12)
    x1d = normalize_candles_for_cnn(candles_1d, 14)

    with torch.no_grad():
        t5 = torch.FloatTensor(x5m).unsqueeze(0)
        t1 = torch.FloatTensor(x1h).unsqueeze(0)
        t4 = torch.FloatTensor(x4h).unsqueeze(0)
        td = torch.FloatTensor(x1d).unsqueeze(0)
        logit = model(t5, t1, t4, td).item()
        prob = 1 / (1 + np.exp(-logit))

    if prob >= 0.5:
        direction = 'long'
        confidence = int(prob * 100)
    else:
        direction = 'short'
        confidence = int((1 - prob) * 100)

    # Marktkontext-Sicherheitsnetz
    if market_context:
        avg_4h     = market_context['avg_4h']
        breadth_4h = market_context['breadth_4h']
        breadth_1h = market_context.get('breadth_1h', 0.5)

        if avg_4h < -1.5 and breadth_4h < 0.25 and direction == 'short':
            return None
        if breadth_1h > 0.75 and direction == 'short':
            return None
        if breadth_1h < 0.25 and direction == 'long':
            return None

    closes_1h = [float(c['close']) for c in candles_1h]
    highs_1h = [float(c['high']) for c in candles_1h]
    lows_1h = [float(c['low']) for c in candles_1h]
    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    expected_move_pct = (atr * 2.5 / current_price * 100) if atr and current_price > 0 else 2.0

    reason = f"CNN 2h: {direction.upper()} {prob*100:.1f}%" if direction == 'long' else f"CNN 2h: {direction.upper()} {(1-prob)*100:.1f}%"

    return {
        'direction': direction,
        'confidence': confidence,
        'entry_price': current_price,
        'expected_move_pct': round(expected_move_pct, 2),
        'reason': reason,
        'signals': [{'name': reason, 'type': direction, 'weight': confidence}],
        'indicators': {
            'cnn_prob': round(prob, 4),
            'atr': round(atr, 8) if atr else None,
        }
    }


# ============================================
# PREDICTION MONITORING
# ============================================

def check_active_predictions():
    with app_db() as aconn, coins_db() as cconn:
        acur = aconn.cursor()
        ccur = cconn.cursor()

        acur.execute("SELECT * FROM momentum_predictions WHERE status = 'active' AND scanner_type = %s", (SCANNER_TYPE,))
        active = acur.fetchall()

        if not active:
            return 0

        # Config laden für live TP/SL
        acur.execute(f"SELECT * FROM {CONFIG_TABLE} WHERE user_id = 1")
        scan_cfg = acur.fetchone() or {}
        long_tp = float(scan_cfg.get('long_fixed_tp_pct') or 2.0)
        long_sl = float(scan_cfg.get('long_fixed_sl_pct') or 2.0)
        short_tp = float(scan_cfg.get('short_fixed_tp_pct') or 2.0)
        short_sl = float(scan_cfg.get('short_fixed_sl_pct') or 2.0)

        # Aktive Predictions updaten wenn Config-TP/SL abweicht
        for pred in active:
            entry = pred['entry_price']
            if pred['direction'] == 'long':
                new_tp_price = entry * (1 + long_tp / 100)
                new_sl_price = entry * (1 - long_sl / 100)
                new_tp_pct, new_sl_pct = long_tp, long_sl
            else:
                new_tp_price = entry * (1 - short_tp / 100)
                new_sl_price = entry * (1 + short_sl / 100)
                new_tp_pct, new_sl_pct = short_tp, short_sl

            if (abs(float(pred['take_profit_pct']) - new_tp_pct) > 0.01 or
                abs(float(pred['stop_loss_pct']) - new_sl_pct) > 0.01):
                acur.execute("""
                    UPDATE momentum_predictions
                    SET take_profit_price = %s, stop_loss_price = %s,
                        take_profit_pct = %s, stop_loss_pct = %s
                    WHERE prediction_id = %s
                """, (new_tp_price, new_sl_price, new_tp_pct, new_sl_pct, pred['prediction_id']))
                pred['take_profit_pct'] = new_tp_pct
                pred['stop_loss_pct'] = new_sl_pct
        aconn.commit()

        resolved_count = 0
        for pred in active:
            symbol = pred['symbol']

            ccur.execute("""
                SELECT close FROM klines
                WHERE symbol = %s AND interval = '1m' AND open_time <= %s
                ORDER BY open_time DESC LIMIT 1
            """, (symbol, clock.now()))
            row_1m = ccur.fetchone()
            if not row_1m:
                continue

            ccur.execute("""
                SELECT MAX(high) as high, MIN(low) as low FROM klines
                WHERE symbol = %s AND interval = '1m' AND open_time >= %s AND open_time <= %s
            """, (symbol, pred['detected_at'], clock.now()))
            row_hl = ccur.fetchone()

            current = row_1m['close']
            entry = pred['entry_price']
            row = {'close': current, 'high': row_hl['high'] or current, 'low': row_hl['low'] or current}

            if pred['direction'] == 'long':
                pct_change = ((current - entry) / entry) * 100
                peak = max(pred['peak_pct'] or 0, ((row['high'] - entry) / entry) * 100)
                trough = min(pred['trough_pct'] or 0, ((row['low'] - entry) / entry) * 100)
            else:
                pct_change = ((entry - current) / entry) * 100
                peak = max(pred['peak_pct'] or 0, ((entry - row['low']) / entry) * 100)
                trough = min(pred['trough_pct'] or 0, ((entry - row['high']) / entry) * 100)

            if pred['direction'] == 'long':
                live_tp = entry * (1 + long_tp / 100)
                live_sl = entry * (1 - long_sl / 100)
            else:
                live_tp = entry * (1 - short_tp / 100)
                live_sl = entry * (1 + short_sl / 100)

            new_status = None
            duration = int((clock.now() - pred['detected_at']).total_seconds() / 60)

            if pred['direction'] == 'long':
                if current >= live_tp:
                    new_status = 'hit_tp'
                elif current <= live_sl:
                    new_status = 'hit_sl'
            else:
                if current <= live_tp:
                    new_status = 'hit_tp'
                elif current >= live_sl:
                    new_status = 'hit_sl'

            # Expiry: 72h
            if not new_status and pred['expires_at'] and clock.now() >= pred['expires_at']:
                new_status = 'expired'

            # time_result: Feedback innerhalb Zeitfenster (2h = 120min für 2h-Scanner)
            TIME_RESULT_THRESHOLD = 120  # 2 Stunden in Minuten
            time_result = pred.get('time_result')
            if time_result is None and duration >= TIME_RESULT_THRESHOLD:
                tp_pct = float(pred['take_profit_pct'])
                sl_pct = float(pred['stop_loss_pct'])
                if peak >= tp_pct:
                    time_result = 'in_time_tp'
                elif trough <= -sl_pct:
                    time_result = 'in_time_sl'
                # Sonst bleibt NULL — wird bei Resolution gesetzt

            if new_status:
                was_correct = new_status == 'hit_tp'

                # time_result bei Resolution setzen wenn noch NULL
                if time_result is None:
                    if new_status == 'hit_tp':
                        time_result = 'expired_positive'
                    elif new_status == 'hit_sl':
                        time_result = 'expired_negative'
                    elif new_status == 'expired':
                        time_result = 'expired_neutral'

                acur.execute("""
                    UPDATE momentum_predictions
                    SET status = %s, resolved_at = %s, actual_result_pct = %s,
                        peak_pct = %s, trough_pct = %s, duration_minutes = %s,
                        was_correct = %s, max_favorable_pct = %s, time_result = %s
                    WHERE prediction_id = %s
                """, (new_status, clock.now(), round(pct_change, 4), round(peak, 4), round(trough, 4),
                      duration, was_correct, round(peak, 4), time_result, pred['prediction_id']))
                resolved_count += 1
                logger.info(f"[RESOLVE] {symbol} {pred['direction']} → {new_status} ({pct_change:+.2f}%) time_result={time_result}")

                # Feedback in Learner-DB schreiben
                write_prediction_feedback(pred, new_status, was_correct, pct_change, duration, time_result, peak, trough)
            else:
                # Peak/Trough updaten + time_result wenn gerade gesetzt
                if time_result is not None and pred.get('time_result') is None:
                    acur.execute("""
                        UPDATE momentum_predictions SET peak_pct = %s, trough_pct = %s, time_result = %s
                        WHERE prediction_id = %s
                    """, (round(peak, 4), round(trough, 4), time_result, pred['prediction_id']))
                    logger.info(f"[TIME_RESULT] {symbol} {pred['direction']} → {time_result} (duration={duration}min)")
                else:
                    acur.execute("""
                        UPDATE momentum_predictions SET peak_pct = %s, trough_pct = %s
                        WHERE prediction_id = %s
                    """, (round(peak, 4), round(trough, 4), pred['prediction_id']))

        aconn.commit()
        return resolved_count


# ============================================
# STATS UPDATE
# ============================================

def update_stats(user_id):
    with app_db() as conn:
        cur = conn.cursor()

        now = clock.now()
        time_periods = {
            '24h': ("detected_at >= %s - INTERVAL '24 hours'", (now,)),
            '7d': ("detected_at >= %s - INTERVAL '7 days'", (now,)),
            '30d': ("detected_at >= %s - INTERVAL '30 days'", (now,)),
            'all': ("TRUE", ())
        }

        combos = []
        for tp, tw in time_periods.items():
            combos.append((tp, tw, None))
            combos.append((f'long_{tp}', tw, 'long'))
            combos.append((f'short_{tp}', tw, 'short'))

        for period_key, (time_where, time_params), direction in combos:
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
                WHERE user_id = %s AND status != 'active' AND scanner_type = %s AND {time_where}{dir_where}
            """, (user_id, SCANNER_TYPE) + time_params)

            row = cur.fetchone()
            total = row['total'] or 0
            correct = row['correct'] or 0
            hit_rate = (correct / total * 100) if total > 0 else 0

            cur.execute("""
                INSERT INTO momentum_stats (user_id, period, scanner_type, total_predictions, correct_predictions,
                    incorrect_predictions, expired_predictions, avg_confidence, avg_result_pct,
                    best_result_pct, worst_result_pct, hit_rate_pct, avg_duration_minutes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, period, scanner_type) DO UPDATE SET
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
                    updated_at = %s
            """, (user_id, period_key, SCANNER_TYPE, total, correct, row['incorrect'] or 0,
                  row['expired'] or 0, row['avg_conf'], row['avg_result'],
                  row['best'], row['worst'], round(hit_rate, 2),
                  int(row['avg_dur']) if row['avg_dur'] else None, now, now))

        conn.commit()


# ============================================
# HELPERS
# ============================================

def get_symbols_for_config(config):
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
    try:
        ccur.execute("""
            SELECT
                AVG(pct_240m) AS avg_4h,
                COUNT(*) FILTER (WHERE pct_240m > 0)::float / COUNT(*) AS breadth_4h,
                AVG(pct_60m) AS avg_1h,
                COUNT(*) FILTER (WHERE pct_60m  > 0)::float / COUNT(*) AS breadth_1h
            FROM kline_metrics
            WHERE open_time = (
                SELECT MAX(open_time) FROM kline_metrics
                WHERE open_time <= date_trunc('hour', %s::timestamptz)
            )
            AND pct_240m IS NOT NULL
        """, (clock.now(),))
        row = ccur.fetchone()
        if not row or row['avg_4h'] is None:
            return None

        avg_4h     = float(row['avg_4h'])
        breadth_4h = float(row['breadth_4h'])
        avg_1h     = float(row['avg_1h'] or 0)
        breadth_1h = float(row['breadth_1h'] or 0.5)

        norm_avg    = max(-1.0, min(1.0, avg_4h / 2.0))
        norm_breadth = max(-1.0, min(1.0, (breadth_4h - 0.5) * 2.0))
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


# ============================================
# HEARTBEAT
# ============================================

def write_heartbeat(status='ok', extra=None):
    try:
        data = {
            'status': status,
            'pid': os.getpid(),
            'timestamp': clock.now().isoformat(),
            'scanner_type': SCANNER_TYPE,
        }
        if extra:
            data.update(extra)
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


# ============================================
# CNN LEARNING THREAD (2h-spezifisch: 2% Events, ≤120min)
# ============================================

class MultiTFDataset(Dataset):
    def __init__(self, x5m, x1h, x4h, x1d, y):
        self.x5m = torch.FloatTensor(x5m)
        self.x1h = torch.FloatTensor(x1h)
        self.x4h = torch.FloatTensor(x4h)
        self.x1d = torch.FloatTensor(x1d)
        self.y   = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x5m[idx], self.x1h[idx], self.x4h[idx], self.x1d[idx], self.y[idx]


def _load_recent_events(days=30, min_pct=None):
    """Lädt Events: ≥2% Moves in ≤120min aus kline_metrics + Gegenextrem-Check."""
    if min_pct is None:
        min_pct = LEARNER_MIN_PCT

    with coins_db() as conn:
        cur = conn.cursor()

        where = ' OR '.join(f'ABS({col}) >= {min_pct}' for col in LEARNER_PCT_COLS)
        cols = ', '.join(LEARNER_PCT_COLS)

        t0 = time.time()
        cur.execute(f"""
            SELECT symbol, open_time, {cols}
            FROM kline_metrics
            WHERE open_time >= %s - INTERVAL '{days} days'
              AND ({where})
            ORDER BY symbol, open_time
        """, (clock.now(),))
        rows = cur.fetchall()
        logger.info(f"[LEARNER] kline_metrics Query: {len(rows)} Kandidaten in {time.time()-t0:.1f}s")

        if not rows:
            return []

        candidates = []
        for row in rows:
            best_col = None
            best_pct = 0
            for col in LEARNER_PCT_COLS:
                val = row[col]
                if val is not None and abs(float(val)) >= min_pct:
                    best_col = col
                    best_pct = float(val)
                    break
            if best_col is None:
                continue
            duration = LEARNER_DUR_MAP[best_col]
            direction = 'long' if best_pct > 0 else 'short'
            entry_time = row['open_time'] - timedelta(minutes=duration)
            candidates.append({
                'symbol': row['symbol'],
                'time': entry_time,
                'event_time': row['open_time'],
                'duration_min': duration,
                'direction': direction,
                'best_pct': round(best_pct, 2),
                'best_tf': best_col,
            })

        # Dedup: max 1 Event pro Symbol pro 60 Min
        candidates.sort(key=lambda e: (e['symbol'], e['time']))
        deduped = []
        last_by_sym = {}
        for e in candidates:
            key = e['symbol']
            if key in last_by_sym:
                diff = (e['time'] - last_by_sym[key]).total_seconds() / 60
                if diff < 60:
                    continue
            last_by_sym[key] = e['time']
            deduped.append(e)

        logger.info(f"[LEARNER] Nach Dedup: {len(deduped)} Events")

        # Gegenextrem-Prüfung via agg_5m
        valid_events = []
        rejected = 0
        threshold = min_pct / 100.0

        for i, event in enumerate(deduped):
            if not running:
                return []
            if i > 0 and i % 5000 == 0:
                logger.info(f"[LEARNER] Gegenprüfung: {i}/{len(deduped)}, valid={len(valid_events)}, rejected={rejected}")

            try:
                cur.execute("""
                    SELECT bucket, open, high, low, close
                    FROM agg_5m
                    WHERE symbol = %s AND bucket >= %s AND bucket < %s
                    ORDER BY bucket
                """, (event['symbol'], event['time'], event['event_time']))
                candles = cur.fetchall()
            except Exception:
                conn.rollback()
                continue

            if len(candles) < 2:
                valid_events.append(event)
                continue

            ref_price = float(candles[0]['open'])
            if ref_price <= 0:
                valid_events.append(event)
                continue

            if event['direction'] == 'long':
                for c in candles:
                    low_pct = (float(c['low']) - ref_price) / ref_price
                    high_pct = (float(c['high']) - ref_price) / ref_price
                    if low_pct <= -threshold:
                        rejected += 1
                        break
                    if high_pct >= threshold:
                        valid_events.append(event)
                        break
                else:
                    valid_events.append(event)
            else:
                for c in candles:
                    high_pct = (float(c['high']) - ref_price) / ref_price
                    low_pct = (float(c['low']) - ref_price) / ref_price
                    if high_pct >= threshold:
                        rejected += 1
                        break
                    if low_pct <= -threshold:
                        valid_events.append(event)
                        break
                else:
                    valid_events.append(event)

        longs = sum(1 for e in valid_events if e['direction'] == 'long')
        shorts = sum(1 for e in valid_events if e['direction'] == 'short')
        logger.info(f"[LEARNER] Events: {len(valid_events)} valid ({longs} Long, {shorts} Short), {rejected} rejected (letzte {days}d)")
        return valid_events


def _load_timeframes_for_events(events):
    timeframes = {
        '5m':  {'table': 'agg_5m',  'hours_back': 12, 'expected': 144, 'min': 72},
        '1h':  {'table': 'agg_1h',  'hours_back': 24, 'expected': 24,  'min': 12},
        '4h':  {'table': 'agg_4h',  'hours_back': 48, 'expected': 12,  'min': 6},
        '1d':  {'table': 'agg_1d',  'hours_back': 336, 'expected': 14, 'min': 7},
    }

    results = []
    with coins_db() as conn:
        cur = conn.cursor()
        for i, event in enumerate(events):
            entry_time = event['time']
            symbol = event['symbol']
            tf_data = {}
            valid = True

            for tf_name, tf_cfg in timeframes.items():
                window_start = entry_time - timedelta(hours=tf_cfg['hours_back'])
                try:
                    cur.execute(f"""
                        SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM {tf_cfg['table']}
                        WHERE symbol = %s AND bucket >= %s AND bucket < %s
                        ORDER BY bucket
                    """, (symbol, window_start, entry_time))
                    candles = cur.fetchall()
                except Exception:
                    conn.rollback()
                    valid = False
                    break

                if len(candles) < tf_cfg['min']:
                    valid = False
                    break

                tf_data[tf_name] = candles

            if valid:
                results.append({'event': event, 'tf': tf_data})

            if (i + 1) % 2000 == 0:
                logger.info(f"[LEARNER] TF-Laden: {i+1}/{len(events)}, valid={len(results)}")

    logger.info(f"[LEARNER] {len(results)} Events mit allen TFs geladen")
    return results


def _normalize_tf_candles(candles, expected_len):
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    n = len(candles)
    closes  = np.array([float(c['close']) for c in candles])
    opens   = np.array([float(c['open']) for c in candles])
    highs   = np.array([float(c['high']) for c in candles])
    lows    = np.array([float(c['low']) for c in candles])
    volumes = np.array([float(c['volume']) for c in candles])
    trades  = np.array([float(c.get('number_of_trades') or 0) for c in candles])
    taker   = np.array([float(c.get('taker_buy_base_asset_volume') or 0) for c in candles])

    price_ret = np.zeros(n)
    for j in range(1, n):
        if closes[j-1] > 0:
            price_ret[j] = (closes[j] / closes[j-1] - 1) * 100

    med_vol = np.median(volumes)
    volume_rel = volumes / med_vol if med_vol > 0 else np.ones(n)
    med_trades = np.median(trades)
    trades_rel = trades / med_trades if med_trades > 0 else np.ones(n)
    taker_ratio = np.where(volumes > 0, taker / volumes, 0.5)
    range_pct = np.where(closes > 0, (highs - lows) / closes * 100, 0)
    full_range = highs - lows
    body_dir = np.where(full_range > 0, (closes - opens) / full_range, 0)
    hl_pos = np.where(full_range > 0, (closes - lows) / full_range, 0.5)

    channels = np.stack([price_ret, volume_rel, trades_rel, taker_ratio,
                         range_pct, body_dir, hl_pos])
    if n >= expected_len:
        channels = channels[:, -expected_len:]
    else:
        pad = np.zeros((7, expected_len - n))
        channels = np.concatenate([pad, channels], axis=1)
    channels = np.nan_to_num(channels, nan=0.0, posinf=20.0, neginf=-20.0)
    channels = np.clip(channels, -50, 50)
    return channels.astype(np.float32)


def _train_new_model(events_with_tf):
    TF_EXPECTED = {'5m': 144, '1h': 24, '4h': 12, '1d': 14}

    X_5m, X_1h, X_4h, X_1d, y_all, times = [], [], [], [], [], []
    for ed in events_with_tf:
        X_5m.append(_normalize_tf_candles(ed['tf']['5m'], TF_EXPECTED['5m']))
        X_1h.append(_normalize_tf_candles(ed['tf']['1h'], TF_EXPECTED['1h']))
        X_4h.append(_normalize_tf_candles(ed['tf']['4h'], TF_EXPECTED['4h']))
        X_1d.append(_normalize_tf_candles(ed['tf']['1d'], TF_EXPECTED['1d']))
        y_all.append(1.0 if ed['event']['direction'] == 'long' else 0.0)
        times.append(ed['event']['time'])

    X_5m = np.array(X_5m)
    X_1h = np.array(X_1h)
    X_4h = np.array(X_4h)
    X_1d = np.array(X_1d)
    y = np.array(y_all, dtype=np.float32)

    indices = np.argsort([t.timestamp() if hasattr(t, 'timestamp') else 0 for t in times])
    X_5m, X_1h, X_4h, X_1d, y = X_5m[indices], X_1h[indices], X_4h[indices], X_1d[indices], y[indices]
    s = int(len(y) * 0.7)

    train_ds = MultiTFDataset(X_5m[:s], X_1h[:s], X_4h[:s], X_1d[:s], y[:s])
    test_ds  = MultiTFDataset(X_5m[s:], X_1h[s:], X_4h[s:], X_1d[s:], y[s:])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    n_long = y[:s].sum()
    n_short = s - n_long
    baseline = max(y[s:].mean(), 1 - y[s:].mean()) * 100

    logger.info(f"[LEARNER] Training: {s} train, {len(y)-s} test, "
                f"Long={int(n_long)}, Short={int(n_short)}, Baseline={baseline:.1f}%")

    model = MultiTimeframeCNN()
    pos_weight = torch.FloatTensor([n_short / max(n_long, 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6)

    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience = 30

    for epoch in range(1000):
        if not running:
            logger.info("[LEARNER] Scanner stopping, abort training")
            return None, 0

        model.train()
        train_correct = 0
        train_total = 0
        for x5, x1, x4, xd, yb in train_loader:
            optimizer.zero_grad()
            out = model(x5, x1, x4, xd)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(out) >= 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x5, x1, x4, xd, yb in test_loader:
                out = model(x5, x1, x4, xd)
                preds = (torch.sigmoid(out) >= 0.5).float()
                test_correct += (preds == yb).sum().item()
                test_total += len(yb)

        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        scheduler.step(test_acc)

        if (epoch + 1) % 10 == 0:
            logger.info(f"[LEARNER] Epoch {epoch+1}: Train {train_acc:.1f}% Test {test_acc:.1f}% (Best {best_acc:.1f}%@{best_epoch})")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_CANDIDATE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[LEARNER] Early Stop: Epoch {epoch+1}, best {best_acc:.1f}% @ Epoch {best_epoch}")
                break

    model.load_state_dict(torch.load(MODEL_CANDIDATE_PATH, weights_only=True))
    model.eval()
    return model, best_acc


def _get_current_model_accuracy():
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)
        return info.get('test_accuracy', 0)
    return 89.1  # Initiale Accuracy vom ersten Training


def run_learning_cycle():
    logger.info("=" * 60)
    logger.info("[LEARNER] === LERNZYKLUS START (2h CNN) ===")
    logger.info("=" * 60)
    t0 = time.time()

    try:
        events = _load_recent_events(days=30)
        if len(events) < 500:
            logger.info(f"[LEARNER] Nur {len(events)} Events, brauche min. 500. Skip.")
            return

        events_with_tf = _load_timeframes_for_events(events)
        if len(events_with_tf) < 400:
            logger.info(f"[LEARNER] Nur {len(events_with_tf)} Events mit TF-Daten. Skip.")
            return

        new_model, new_acc = _train_new_model(events_with_tf)
        if new_model is None:
            return

        current_acc = _get_current_model_accuracy()
        logger.info(f"[LEARNER] Neues Modell: {new_acc:.1f}% vs Aktuelles: {current_acc:.1f}%")

        if new_acc > current_acc:
            global _cnn_model
            import shutil

            backup_path = f'{MODEL_BACKUP_DIR}/cnn_2h_backup_{clock.now().strftime("%Y%m%d_%H%M%S")}.pth'
            if os.path.exists(MODEL_PATH):
                shutil.copy2(MODEL_PATH, backup_path)
                logger.info(f"[LEARNER] Backup: {backup_path}")

            torch.save(new_model.state_dict(), MODEL_PATH)
            _cnn_model = new_model

            with open(MODEL_INFO_PATH, 'w') as f:
                json.dump({
                    'test_accuracy': round(new_acc, 2),
                    'trained_at': clock.now().isoformat(),
                    'events_used': len(events_with_tf),
                    'previous_accuracy': round(current_acc, 2),
                }, f, indent=2)

            elapsed = (time.time() - t0) / 60
            logger.info(f"[LEARNER] HOT-SWAP: {current_acc:.1f}% → {new_acc:.1f}% (+{new_acc-current_acc:.1f}pp) in {elapsed:.1f}min")
        else:
            elapsed = (time.time() - t0) / 60
            logger.info(f"[LEARNER] KEIN SWAP: Neues {new_acc:.1f}% ≤ Aktuelles {current_acc:.1f}% ({elapsed:.1f}min)")
            if os.path.exists(MODEL_CANDIDATE_PATH):
                os.remove(MODEL_CANDIDATE_PATH)

    except Exception as e:
        logger.error(f"[LEARNER] Fehler: {e}")
        logger.error(traceback.format_exc())


class LearnerThread(threading.Thread):
    def __init__(self, interval_hours=12):
        super().__init__(daemon=True, name='CNN-2h-Learner')
        self.interval = interval_hours * 3600
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        logger.info(f"[LEARNER] Thread gestartet — Intervall: alle {self.interval // 3600}h")
        self._stop_event.wait(3600 * 6)  # 6h Delay — versetzt zum Default-Scanner (der nach 1h startet)

        while not self._stop_event.is_set() and running:
            try:
                run_learning_cycle()
            except Exception as e:
                logger.error(f"[LEARNER] Thread-Fehler: {e}")
                logger.error(traceback.format_exc())
            self._stop_event.wait(self.interval)

        logger.info("[LEARNER] Thread beendet")


# ============================================
# SCAN SYMBOLS
# ============================================

def scan_symbols(config, symbols):
    user_id = config['user_id']
    new_predictions = 0

    dir_config = {
        'long': {
            'tp_sl_mode': config.get('long_tp_sl_mode') or config.get('tp_sl_mode', 'fixed'),
            'tp_pct': float(config.get('long_fixed_tp_pct') or 2.0),
            'sl_pct': float(config.get('long_fixed_sl_pct') or 2.0),
            'min_conf': config.get('long_min_confidence') or config.get('min_confidence', 60),
        },
        'short': {
            'tp_sl_mode': config.get('short_tp_sl_mode') or config.get('tp_sl_mode', 'fixed'),
            'tp_pct': float(config.get('short_fixed_tp_pct') or 2.0),
            'sl_pct': float(config.get('short_fixed_sl_pct') or 2.0),
            'min_conf': config.get('short_min_confidence') or config.get('min_confidence', 60),
        },
    }

    with coins_db() as cconn, app_db() as aconn:
        ccur = cconn.cursor()
        acur = aconn.cursor()

        # Aktive Predictions NUR von diesem Scanner
        acur.execute("""
            SELECT symbol FROM momentum_predictions
            WHERE user_id = %s AND status = 'active' AND scanner_type = %s
        """, (user_id, SCANNER_TYPE))
        active_symbols = {r['symbol'] for r in acur.fetchall()}

        # Cooldown: 30 Min für 2h-Scanner (statt 60 Min)
        acur.execute("""
            SELECT DISTINCT symbol FROM momentum_predictions
            WHERE user_id = %s AND status != 'active' AND scanner_type = %s
              AND resolved_at >= %s - INTERVAL '30 minutes'
        """, (user_id, SCANNER_TYPE, clock.now()))
        cooldown_symbols = {r['symbol'] for r in acur.fetchall()}
        active_symbols = active_symbols | cooldown_symbols

        # Hyperliquid-Coins laden (Short nur für diese)
        acur.execute("SELECT symbol FROM coin_info WHERE 'hyperliquid' = ANY(exchanges)")
        hl_symbols = {r['symbol'] for r in acur.fetchall()}

        market_context = get_market_context(ccur)
        if market_context:
            logger.info(f"[MKT_CTX] avg_4h={market_context['avg_4h']:+.2f}% breadth_4h={market_context['breadth_4h']:.2f}")

        cycle_short_count = 0

        for symbol in symbols:
            if not running:
                break

            if symbol in active_symbols:
                continue

            try:
                ccur.execute("""
                    SELECT close FROM klines
                    WHERE symbol = %s AND interval = '1m' AND open_time <= %s
                    ORDER BY open_time DESC LIMIT 1
                """, (symbol, clock.now()))
                row_1m = ccur.fetchone()
                if not row_1m or not row_1m['close'] or row_1m['close'] == 0:
                    continue
                current_price = row_1m['close']

                signal = analyze_symbol_cnn(ccur, symbol, current_price, market_context=market_context, scan_config=config)

                if signal and signal['confidence'] >= dir_config[signal['direction']]['min_conf']:
                    # Short nur für Hyperliquid-Coins
                    if signal['direction'] == 'short' and symbol not in hl_symbols:
                        continue

                    # Momentum-Check
                    ccur.execute("""
                        SELECT pct_30m, pct_60m FROM kline_metrics
                        WHERE symbol = %s AND open_time <= %s
                        ORDER BY open_time DESC LIMIT 1
                    """, (symbol, clock.now()))
                    metrics_row = ccur.fetchone()
                    if metrics_row and metrics_row['pct_30m'] is not None:
                        pct_30m = float(metrics_row['pct_30m'])
                        long_pct_min = float(config.get('long_pct_30m_min') or 1.0)
                        short_pct_min = float(config.get('short_pct_30m_min') or 0.5)
                        if signal['direction'] == 'long' and pct_30m < long_pct_min:
                            continue
                        if signal['direction'] == 'short' and pct_30m > -short_pct_min:
                            continue

                        if signal['direction'] == 'short':
                            pct_60m = float(metrics_row.get('pct_60m') or 0) if metrics_row.get('pct_60m') is not None else 0
                            if pct_60m < -3.0:
                                continue

                    if signal['direction'] == 'short' and cycle_short_count >= 15:
                        continue

                    # TP/SL
                    entry = signal['entry_price']
                    d_cfg = dir_config[signal['direction']]
                    tp_pct = d_cfg['tp_pct']
                    sl_pct = d_cfg['sl_pct']

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
                         confidence, reason, signals, detected_at, expires_at, scanner_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s + INTERVAL '72 hours', %s)
                        RETURNING prediction_id
                    """, (user_id, symbol, signal['direction'], entry,
                          float(tp_price), float(sl_price),
                          float(tp_pct), float(sl_pct),
                          int(signal["confidence"]), signal["reason"],
                          Json(signal['signals']), clock.now(), clock.now(), SCANNER_TYPE))

                    pred_id = acur.fetchone()['prediction_id']
                    new_predictions += 1
                    if signal['direction'] == 'short':
                        cycle_short_count += 1
                    logger.info(f"[NEW] #{pred_id} {symbol} {signal['direction'].upper()} "
                               f"@ {current_price:.4f} → TP {tp_pct:.1f}% SL {sl_pct:.1f}% (conf: {signal['confidence']})")

            except Exception as e:
                logger.error(f"[SCAN] {symbol} error: {e}")
                continue

        aconn.commit()
    return new_predictions


# ============================================
# MAIN LOOP
# ============================================

def sim_main(sim_end, sim_step=5):
    """Simulation Main Loop — Zeitmaschine, gleicher Code wie Live"""
    logger.info("=" * 60)
    logger.info(f"[SIM] Simulation Start: {clock.now().isoformat()}")
    logger.info(f"[SIM] Simulation Ende:  {sim_end.isoformat()}")
    logger.info(f"[SIM] Zeitschritt:      {sim_step} Minuten")
    logger.info(f"[SIM] Scanner Type:     {SCANNER_TYPE}")
    logger.info(f"[SIM] PID: {os.getpid()}")
    logger.info("=" * 60)

    model = get_cnn_model()
    if model is None:
        logger.error("[SIM] KEIN MODELL VERFÜGBAR — Simulation kann nicht starten!")
        return

    logger.info("[SIM] Modell bereit — Simulation startet")

    with app_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {CONFIG_TABLE} WHERE is_active = true")
        configs = cur.fetchall()

    if not configs:
        logger.error("[SIM] Keine aktive Scan-Config gefunden!")
        return

    config = configs[0]
    user_id = config['user_id']
    symbols = get_symbols_for_config(config)
    logger.info(f"[SIM] {len(symbols)} Symbole geladen, User {user_id}")

    learner_counter = 0
    cycle_count = 0
    total_predictions = 0
    total_resolved = 0
    last_day = None

    while clock.now() < sim_end:
        try:
            current_day = clock.now().strftime('%Y-%m-%d')
            if current_day != last_day:
                logger.info(f"[SIM] === {current_day} === (Zyklus {cycle_count}, Predictions bisher: {total_predictions}, Resolved: {total_resolved})")
                last_day = current_day

            # 1. Aktive Predictions prüfen
            resolved = check_active_predictions()
            total_resolved += resolved
            if resolved > 0:
                update_stats(user_id)

            # 2. Neue Symbole scannen
            new = scan_symbols(config, symbols)
            total_predictions += new

            # 3. Stats updaten
            update_stats(user_id)

            # 4. Learner alle 12h (720 Minuten)
            learner_counter += sim_step
            if learner_counter >= 720:
                logger.info(f"[SIM] Learner-Zyklus bei {clock.now().isoformat()}")
                try:
                    run_learning_cycle()
                except Exception as le:
                    logger.error(f"[SIM] Learner-Fehler: {le}")
                learner_counter = 0

            # 5. Zeit vorspulen
            clock.advance(sim_step)
            cycle_count += 1

        except Exception as e:
            logger.error(f"[SIM] Zyklus-Fehler bei {clock.now().isoformat()}: {e}")
            logger.error(traceback.format_exc())
            clock.advance(sim_step)
            cycle_count += 1

    logger.info("=" * 60)
    logger.info(f"[SIM] Simulation ABGESCHLOSSEN")
    logger.info(f"[SIM] Zyklen:      {cycle_count}")
    logger.info(f"[SIM] Predictions: {total_predictions}")
    logger.info(f"[SIM] Resolved:    {total_resolved}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("Momentum Scanner 2h (CNN) + Learning starting...")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Config: {CONFIG_TABLE}")
    logger.info(f"Scanner Type: {SCANNER_TYPE}")
    logger.info("=" * 60)

    model = get_cnn_model()
    if model is None:
        logger.error("[CNN] KEIN MODELL VERFÜGBAR — Scanner kann nicht starten!")
        return
    logger.info("[CNN] Modell bereit — Scanner 2h läuft")

    # Learning Thread DEAKTIVIERT — Simulation läuft, Learner werden dort getriggert
    # learner = LearnerThread(interval_hours=12)
    # learner.start()
    learner = None
    logger.info("[LEARNER] DEAKTIVIERT — wird über Simulation gesteuert")

    write_heartbeat('starting')

    while running:
        try:
            with app_db() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM {CONFIG_TABLE} WHERE is_active = true")
                configs = cur.fetchall()

            if not configs:
                time.sleep(10)
                continue

            for config in configs:
                if not running:
                    break

                user_id = config['user_id']
                idle = config['idle_seconds'] or 60

                # 1. Aktive Predictions prüfen
                logger.info(f"[LOOP] Checking active 2h predictions...")
                resolved = check_active_predictions()
                logger.info(f"[LOOP] Resolved: {resolved}")
                if resolved > 0:
                    update_stats(user_id)

                # 2. Neue Symbole scannen
                symbols = get_symbols_for_config(config)
                logger.info(f"[LOOP] Scanning {len(symbols) if symbols else 0} symbols...")
                if symbols:
                    new = scan_symbols(config, symbols)
                    if new > 0:
                        logger.info(f"[SCAN] Created {new} new 2h predictions for user {user_id}")

                # 3. Stats updaten
                update_stats(user_id)

                write_heartbeat('ok', {
                    'learner_alive': learner.is_alive() if learner else False,
                    'model_loaded': _cnn_model is not None,
                })

                logger.info(f"[LOOP] Cycle done, sleeping {idle}s...")
                for _ in range(idle):
                    if not running:
                        break
                    time.sleep(1)

        except Exception as e:
            logger.error(f"[MAIN] Error: {e}")
            logger.error(traceback.format_exc())
            write_heartbeat('error', {'error': str(e)})
            time.sleep(30)

    if learner:
        learner.stop()
        learner.join(timeout=5)
    write_heartbeat('stopped')
    logger.info("Momentum Scanner 2h stopped.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--sim':
        if len(sys.argv) < 4:
            print("Usage: scanner_2h.py --sim <start_iso> <end_iso> [step_minutes]")
            print("  Beispiel: scanner_2h.py --sim 2026-01-01T00:00:00+00:00 2026-03-01T00:00:00+00:00 5")
            sys.exit(1)
        sim_start = datetime.fromisoformat(sys.argv[2])
        sim_end = datetime.fromisoformat(sys.argv[3])
        sim_step = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        clock.set_time(sim_start)
        sim_main(sim_end, sim_step)
    else:
        main()
