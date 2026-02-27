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
from threshold_optimizer import run_threshold_optimization
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

import threading
import pickle
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
    """Average body/range ratio of last n candles"""
    if len(opens) < n: return None
    ratios = []
    for i in range(-n, 0):
        full = highs[i] - lows[i]
        if full <= 0: continue
        ratios.append(abs(closes[i] - opens[i]) / full)
    return sum(ratios) / len(ratios) if ratios else None

def calc_consecutive(closes, n=10):
    if len(closes) < n + 1: return 0, 0
    ups = sum(1 for i in range(-n, 0) if closes[i] > closes[i-1])
    return ups, n - ups

def calc_bb_position(closes, period=20):
    """Wo steht Preis in Bollinger Bändern? 0=unten, 1=oben"""
    if len(closes) < period: return None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    if std == 0: return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5

def calc_range_position(highs, lows, closes, n=7):
    """Wo steht Preis im High-Low Range der letzten n Perioden? 0=Low, 1=High"""
    if len(highs) < n: return None
    highest = max(highs[-n:])
    lowest = min(lows[-n:])
    if highest == lowest: return 0.5
    return (closes[-1] - lowest) / (highest - lowest)

def calc_wick_ratios(opens, highs, lows, closes, n=6):
    """Durchschnittliche upper/lower wick ratios der letzten n 1h-Candles"""
    if len(opens) < n: return None, None
    upper_wicks, lower_wicks = [], []
    for i in range(-n, 0):
        full = highs[i] - lows[i]
        if full <= 0: continue
        if closes[i] >= opens[i]:  # bullish
            upper_wicks.append((highs[i] - closes[i]) / full)
            lower_wicks.append((opens[i] - lows[i]) / full)
        else:  # bearish
            upper_wicks.append((highs[i] - opens[i]) / full)
            lower_wicks.append((closes[i] - lows[i]) / full)
    uw = sum(upper_wicks) / len(upper_wicks) if upper_wicks else None
    lw = sum(lower_wicks) / len(lower_wicks) if lower_wicks else None
    return uw, lw

# ============================================
# CNN MODEL DEFINITION (muss identisch zu ts_classifier.py sein)
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
    """Lädt das trainierte CNN-Modell (einmal, dann gecached)."""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    model_path = '/opt/coin/database/data/models/best_cnn_v2.pth'
    if not os.path.exists(model_path):
        logger.error(f"[CNN] Modell nicht gefunden: {model_path}")
        return None
    try:
        model = MultiTimeframeCNN()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        model.eval()
        _cnn_model = model
        logger.info(f"[CNN] Modell geladen: {model_path}")
        return model
    except Exception as e:
        logger.error(f"[CNN] Modell laden fehlgeschlagen: {e}")
        return None


def normalize_candles_for_cnn(candles, expected_len):
    """
    Normalisiert Candle-Daten zu 7 Kanälen für das CNN.
    Identisch zur Normalisierung in ts_classifier.py.
    """
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

    # 1. Returns
    price_ret = np.zeros(n)
    for j in range(1, n):
        if closes[j-1] > 0:
            price_ret[j] = (closes[j] / closes[j-1] - 1) * 100

    # 2. Volume relativ
    med_vol = np.median(volumes)
    volume_rel = volumes / med_vol if med_vol > 0 else np.ones(n)

    # 3. Trades relativ
    med_trades = np.median(trades)
    trades_rel = trades / med_trades if med_trades > 0 else np.ones(n)

    # 4. Taker Ratio
    taker_ratio = np.where(volumes > 0, taker / volumes, 0.5)

    # 5. Range %
    range_pct = np.where(closes > 0, (highs - lows) / closes * 100, 0)

    # 6. Body Direction
    full_range = highs - lows
    body_dir = np.where(full_range > 0, (closes - opens) / full_range, 0)

    # 7. HL Position
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


def analyze_symbol_cnn(ccur, symbol, current_price, market_context=None, scan_config=None):
    """
    CNN-basierte Signal-Erkennung.
    Lädt agg_5m (144), agg_1h (24), agg_4h (12), agg_1d (14),
    normalisiert, schickt durch CNN → Direction + Confidence.
    """
    model = get_cnn_model()
    if model is None:
        return None

    # Daten laden
    try:
        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_5m WHERE symbol = %s ORDER BY bucket DESC LIMIT 144""", (symbol,))
        candles_5m = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1h WHERE symbol = %s ORDER BY bucket DESC LIMIT 24""", (symbol,))
        candles_1h = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_4h WHERE symbol = %s ORDER BY bucket DESC LIMIT 12""", (symbol,))
        candles_4h = list(reversed(ccur.fetchall()))

        ccur.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1d WHERE symbol = %s ORDER BY bucket DESC LIMIT 14""", (symbol,))
        candles_1d = list(reversed(ccur.fetchall()))
    except Exception as e:
        logger.debug(f"[CNN] {symbol} Daten-Fehler: {e}")
        return None

    # Mindest-Daten
    if len(candles_5m) < 72 or len(candles_1h) < 12 or len(candles_4h) < 6 or len(candles_1d) < 7:
        return None

    # Normalisieren
    x5m = normalize_candles_for_cnn(candles_5m, 144)
    x1h = normalize_candles_for_cnn(candles_1h, 24)
    x4h = normalize_candles_for_cnn(candles_4h, 12)
    x1d = normalize_candles_for_cnn(candles_1d, 14)

    # Inference
    with torch.no_grad():
        t5 = torch.FloatTensor(x5m).unsqueeze(0)
        t1 = torch.FloatTensor(x1h).unsqueeze(0)
        t4 = torch.FloatTensor(x4h).unsqueeze(0)
        td = torch.FloatTensor(x1d).unsqueeze(0)
        logit = model(t5, t1, t4, td).item()
        prob = 1 / (1 + np.exp(-logit))  # sigmoid

    # Direction + Confidence
    if prob >= 0.5:
        direction = 'long'
        confidence = int(prob * 100)
    else:
        direction = 'short'
        confidence = int((1 - prob) * 100)

    # Marktkontext (Sicherheitsnetz, wie bisher)
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

    # ATR für TP/SL (aus 1h-Daten berechnen)
    closes_1h = [float(c['close']) for c in candles_1h]
    highs_1h = [float(c['high']) for c in candles_1h]
    lows_1h = [float(c['low']) for c in candles_1h]
    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    expected_move_pct = (atr * 2.5 / current_price * 100) if atr and current_price > 0 else 5.0

    reason = f"CNN v2: {direction.upper()} {prob*100:.1f}%" if direction == 'long' else f"CNN v2: {direction.upper()} {(1-prob)*100:.1f}%"

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


def analyze_symbol(candles_1h, candles_4h, candles_1d, current_price, market_context=None, scan_config=None):
    """
    v3 (LEGACY): Deep Filter basiert.
    Primary: 3 harte Filter pro Richtung (93%+ Precision aus Deep Filter).
    Secondary: Tier-Scoring als Confidence-Bonus.
    Marktkontext als Sicherheitsnetz.
    """
    if len(candles_1h) < 30 or len(candles_4h) < 15 or len(candles_1d) < 10:
        return None

    closes_1h  = [c['close'] for c in candles_1h]
    highs_1h   = [c['high'] for c in candles_1h]
    lows_1h    = [c['low'] for c in candles_1h]
    opens_1h   = [c['open'] for c in candles_1h]
    volumes_1h = [c['volume'] for c in candles_1h]
    closes_4h  = [c['close'] for c in candles_4h]
    highs_1d   = [c['high'] for c in candles_1d]
    lows_1d    = [c['low'] for c in candles_1d]
    closes_1d  = [c['close'] for c in candles_1d]

    # === CONFIG (DB, Optimizer-anpassbar) ===
    cfg = scan_config or {}

    # Deep Filter Thresholds
    df_long_range_min     = float(cfg.get('df_long_range_pos_7d_min') or 0.7288)
    df_long_ema50_max     = float(cfg.get('df_long_ema_price_50_max') or 6.5481)
    df_long_lwick_max     = float(cfg.get('df_long_lower_wick_max') or 0.3130)
    df_short_range_max    = float(cfg.get('df_short_range_pos_7d_max') or 0.0570)
    df_short_uwick_min    = float(cfg.get('df_short_upper_wick_min') or 0.2950)
    df_short_bb_max       = float(cfg.get('df_short_bb_position_max') or 0.4170)

    # === DEEP FILTER INDIKATOREN ===
    rsi_1h = calc_rsi(closes_1h, 14)
    if rsi_1h is None:
        return None

    ema_50 = calc_ema(closes_1h[-50:], 50) if len(closes_1h) >= 50 else None
    ema_price_50_pct = ((current_price - ema_50) / ema_50 * 100) if ema_50 and ema_50 > 0 else 0

    range_pos_7d = calc_range_position(highs_1d, lows_1d, closes_1d, 7)
    if range_pos_7d is None:
        return None

    bb_position_1h = calc_bb_position(closes_1h)
    upper_wick_1h, lower_wick_1h = calc_wick_ratios(opens_1h, highs_1h, lows_1h, closes_1h, 6)
    if upper_wick_1h is None or lower_wick_1h is None:
        return None

    # === PRIMARY: DEEP FILTER GATES (müssen ALLE passen) ===
    long_pass = (
        range_pos_7d >= df_long_range_min and
        ema_price_50_pct < df_long_ema50_max and
        lower_wick_1h < df_long_lwick_max
    )

    short_pass = (
        range_pos_7d < df_short_range_max and
        upper_wick_1h >= df_short_uwick_min and
        bb_position_1h < df_short_bb_max
    )

    # Kein Signal wenn keiner der Deep Filter passt
    if not long_pass and not short_pass:
        return None

    # === SECONDARY: CONFIDENCE SCORING (Bonus-Indikatoren) ===
    signals = []

    # Trend 4h
    trend_4h = 0
    if len(closes_4h) >= 5:
        r = closes_4h[-5:]
        trend_4h = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r) - 1)) * 2 - 1

    # Trend 1d
    trend_1d = 0
    if len(closes_1d) >= 5:
        r = closes_1d[-5:]
        trend_1d = (sum(1 for i in range(1, len(r)) if r[i] > r[i-1]) / (len(r) - 1)) * 2 - 1

    body_ratio = calc_body_ratio(opens_1h, highs_1h, lows_1h, closes_1h, 6) or 0.5
    consec_ups, consec_downs = calc_consecutive(closes_1h)

    hh_hl = 0
    if len(highs_1h) >= 6:
        rh, rl = highs_1h[-6:], lows_1h[-6:]
        hh_hl = (sum(1 for i in range(1, 6) if rh[i] > rh[i-1]) + sum(1 for i in range(1, 6) if rl[i] > rl[i-1])) / 10

    avg_vol = np.mean(volumes_1h[-20:]) if len(volumes_1h) >= 20 else np.mean(volumes_1h)
    vol_ratio = np.mean(volumes_1h[-3:]) / avg_vol if avg_vol > 0 else 1

    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    ema_9 = calc_ema(closes_1h, 9)
    ema_21 = calc_ema(closes_1h, 21)

    # Confidence: Base 70 (Deep Filter allein = 93%+), Bonus bis 100
    long_conf = 0
    short_conf = 0

    if long_pass:
        long_conf = 70
        signals.append({'name': f'DF: range7d={range_pos_7d:.2f} ema50={ema_price_50_pct:+.1f}% lwick={lower_wick_1h:.2f}', 'type': 'long', 'weight': 70})
        if trend_1d >= 0:    long_conf += 8;  signals.append({'name': f'1d trend {trend_1d:+.2f}', 'type': 'long', 'weight': 8})
        if trend_4h >= 0:    long_conf += 8;  signals.append({'name': f'4h trend {trend_4h:+.2f}', 'type': 'long', 'weight': 8})
        if rsi_1h >= 45:     long_conf += 5;  signals.append({'name': f'RSI {rsi_1h:.0f}', 'type': 'long', 'weight': 5})
        if hh_hl >= 0.5:     long_conf += 5;  signals.append({'name': f'HH/HL {hh_hl:.2f}', 'type': 'long', 'weight': 5})
        if vol_ratio > 1.3:  long_conf += 4;  signals.append({'name': f'Vol {vol_ratio:.1f}x', 'type': 'long', 'weight': 4})

    if short_pass:
        short_conf = 70
        signals.append({'name': f'DF: range7d={range_pos_7d:.2f} uwick={upper_wick_1h:.2f} bb={bb_position_1h:.2f}', 'type': 'short', 'weight': 70})
        if trend_1d <= 0:    short_conf += 8;  signals.append({'name': f'1d trend {trend_1d:+.2f}', 'type': 'short', 'weight': 8})
        if trend_4h <= 0:    short_conf += 8;  signals.append({'name': f'4h trend {trend_4h:+.2f}', 'type': 'short', 'weight': 8})
        if rsi_1h < 45:      short_conf += 5;  signals.append({'name': f'RSI {rsi_1h:.0f}', 'type': 'short', 'weight': 5})
        if hh_hl < 0.4:      short_conf += 5;  signals.append({'name': f'LL/LH {hh_hl:.2f}', 'type': 'short', 'weight': 5})
        if vol_ratio > 1.3:  short_conf += 4;  signals.append({'name': f'Vol {vol_ratio:.1f}x', 'type': 'short', 'weight': 4})

    # === MARKTKONTEXT (Sicherheitsnetz) ===
    if market_context:
        trend      = market_context['market_trend']
        avg_4h     = market_context['avg_4h']
        breadth_4h = market_context['breadth_4h']
        breadth_1h = market_context.get('breadth_1h', 0.5)

        if avg_4h < -1.5 and breadth_4h < 0.25:
            short_conf = 0
            signals.append({'name': 'Market oversold → short blocked', 'type': 'market_context', 'weight': -99})
        elif breadth_1h > 0.75:
            short_conf = 0
            signals.append({'name': f'Market bouncing (b1h={breadth_1h:.2f}) → short blocked', 'type': 'market_context', 'weight': -99})

        if breadth_1h < 0.25:
            long_conf = 0
            signals.append({'name': f'Market dump (b1h={breadth_1h:.2f}) → long blocked', 'type': 'market_context', 'weight': -99})

    # === ENTSCHEIDUNG ===
    if long_conf == 0 and short_conf == 0:
        return None

    if long_conf > short_conf:
        direction = 'long'
        confidence = min(long_conf, 100)
    elif short_conf > long_conf:
        direction = 'short'
        confidence = min(short_conf, 100)
    else:
        return None  # Gleichstand = kein Signal

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
            'ema_50': round(ema_50, 8) if ema_50 else None,
            'vol_ratio': round(vol_ratio, 2),
            'trend_4h': round(trend_4h, 2),
            'trend_1d': round(trend_1d, 2),
            'atr': round(atr, 8) if atr else None,
            'hh_hl': round(hh_hl, 2),
            'body_ratio': round(body_ratio, 3),
            'range_pos_7d': round(range_pos_7d, 3),
            'bb_position_1h': round(bb_position_1h, 3),
            'upper_wick_1h': round(upper_wick_1h, 3),
            'lower_wick_1h': round(lower_wick_1h, 3),
            'ema_price_50_pct': round(ema_price_50_pct, 2),
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

        acur.execute("SELECT * FROM momentum_predictions WHERE status = 'active' AND (scanner_type = 'default' OR scanner_type IS NULL)")
        active = acur.fetchall()

        if not active:
            return 0

        # Config laden für live TP/SL
        acur.execute("SELECT * FROM momentum_scan_config WHERE user_id = 1")
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
                pred['take_profit_price'] = new_tp_price
                pred['stop_loss_price'] = new_sl_price
                pred['take_profit_pct'] = new_tp_pct
                pred['stop_loss_pct'] = new_sl_pct
                logger.info(f"[TP/SL] #{pred['prediction_id']} {pred['symbol']} updated → TP {new_tp_pct}% SL {new_sl_pct}%")
        aconn.commit()

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

            # Live TP/SL aus Config (änderbar, gilt sofort für alle offenen Positionen)
            if pred['direction'] == 'long':
                live_tp = entry * (1 + long_tp / 100)
                live_sl = entry * (1 - long_sl / 100)
            else:
                live_tp = entry * (1 - short_tp / 100)
                live_sl = entry * (1 + short_sl / 100)

            # Status prüfen
            new_status = None
            duration = int((datetime.now(timezone.utc) - pred['detected_at']).total_seconds() / 60)

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

            # Expiry: max 72h
            if not new_status and pred['expires_at'] and datetime.now(timezone.utc) >= pred['expires_at']:
                new_status = 'expired'

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
                WHERE user_id = %s AND status != 'active' AND (scanner_type = 'default' OR scanner_type IS NULL) AND {time_where}{dir_where}
            """, (user_id,))
            
            row = cur.fetchone()
            total = row['total'] or 0
            correct = row['correct'] or 0
            hit_rate = (correct / total * 100) if total > 0 else 0

            cur.execute("""
                INSERT INTO momentum_stats (user_id, period, scanner_type, total_predictions, correct_predictions,
                    incorrect_predictions, expired_predictions, avg_confidence, avg_result_pct,
                    best_result_pct, worst_result_pct, hit_rate_pct, avg_duration_minutes, updated_at)
                VALUES (%s, %s, 'default', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
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
    """Prüft ob Optimierung laufen soll (Montag 08:00 Berlin, wöchentlich)"""
    global _last_optimization_run
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Europe/Berlin'))
    run_hours = [8]  # Wöchentlich (nur Montag)
    
    if now.weekday() == 0 and now.hour in run_hours:  # Nur Montag
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


# ============================================
# HEARTBEAT
# ============================================
SCANNER_HEARTBEAT_FILE = '/opt/coin/logs/.scanner_heartbeat'

def write_heartbeat(status='ok', extra=None):
    """Schreibt Scanner-Heartbeat für Watchdog-Monitoring."""
    try:
        data = {
            'status': status,
            'pid': os.getpid(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            data.update(extra)
        with open(SCANNER_HEARTBEAT_FILE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


# ============================================
# CNN LEARNING THREAD
# ============================================

class MultiTFDataset(Dataset):
    """Dataset für Multi-Timeframe CNN Training."""
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


def _load_recent_events(days=30, min_pct=5):
    """
    Lädt Events der letzten N Tage — identisch zu find_events_strict() aus differenz_scanner.py.
    Nutzt kline_metrics (pct_60m..pct_600m) für Event-Erkennung + Gegenextrem-Prüfung via agg_5m.
    """
    PCT_COLS = ['pct_60m','pct_90m','pct_120m','pct_180m','pct_240m',
                'pct_300m','pct_360m','pct_420m','pct_480m','pct_540m','pct_600m']
    DUR_MAP = {'pct_60m':60,'pct_90m':90,'pct_120m':120,'pct_180m':180,'pct_240m':240,
               'pct_300m':300,'pct_360m':360,'pct_420m':420,'pct_480m':480,'pct_540m':540,'pct_600m':600}

    with coins_db() as conn:
        cur = conn.cursor()

        # 1. Alle Kandidaten aus kline_metrics holen (alle pct-Spalten ≥min_pct%)
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
        logger.info(f"[LEARNER] kline_metrics Query: {len(rows)} Kandidaten in {time.time()-t0:.1f}s")

        if not rows:
            return []

        # 2. Bestes pct-Feld pro Row → Kandidaten-Liste
        candidates = []
        for row in rows:
            best_col = None
            best_pct = 0
            for col in PCT_COLS:
                val = row[col]
                if val is not None and abs(float(val)) >= min_pct:
                    best_col = col
                    best_pct = float(val)
                    break
            if best_col is None:
                continue
            duration = DUR_MAP[best_col]
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

        # 3. Dedup: max 1 Event pro Symbol pro 60 Min
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

        # 4. Gegenextrem-Prüfung via agg_5m
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
    """Lädt alle 4 Timeframe-Daten für Events."""
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
    """Normalisiert rohe DB-Candles zu 7 Kanälen (identisch zu normalize_candles_for_cnn)."""
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
    """Trainiert ein neues CNN-Modell und gibt (model, test_accuracy) zurück."""
    TF_EXPECTED = {'5m': 144, '1h': 24, '4h': 12, '1d': 14}

    # Tensoren vorbereiten
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

    # Zeitlicher Split 70/30
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

        # Train
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

        # Test
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
            # Temporär speichern
            torch.save(model.state_dict(), '/opt/coin/database/data/models/cnn_candidate.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[LEARNER] Early Stop: Epoch {epoch+1}, best {best_acc:.1f}% @ Epoch {best_epoch}")
                break

    # Bestes Modell laden
    model.load_state_dict(torch.load('/opt/coin/database/data/models/cnn_candidate.pth', weights_only=True))
    model.eval()
    return model, best_acc


def _get_current_model_accuracy():
    """Liest die Accuracy des aktuellen Modells aus der letzten Trainings-Info."""
    info_path = '/opt/coin/database/data/models/model_info.json'
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        return info.get('test_accuracy', 0)
    return 91.3  # Initiale Accuracy vom ersten Training


def run_learning_cycle():
    """Kompletter Lernzyklus: Events laden → Timeframes → Training → Hot-Swap."""
    logger.info("=" * 60)
    logger.info("[LEARNER] === LERNZYKLUS START ===")
    logger.info("=" * 60)
    t0 = time.time()

    try:
        # 1. Events laden (letzte 30 Tage)
        events = _load_recent_events(days=30)
        if len(events) < 500:
            logger.info(f"[LEARNER] Nur {len(events)} Events, brauche min. 500. Skip.")
            return

        # 2. Timeframe-Daten laden
        events_with_tf = _load_timeframes_for_events(events)
        if len(events_with_tf) < 400:
            logger.info(f"[LEARNER] Nur {len(events_with_tf)} Events mit TF-Daten. Skip.")
            return

        # 3. Neues Modell trainieren
        new_model, new_acc = _train_new_model(events_with_tf)
        if new_model is None:
            return

        # 4. Vergleich mit aktuellem Modell
        current_acc = _get_current_model_accuracy()
        logger.info(f"[LEARNER] Neues Modell: {new_acc:.1f}% vs Aktuelles: {current_acc:.1f}%")

        if new_acc > current_acc:
            # Hot-Swap: Neues Modell übernimmt
            global _cnn_model
            model_path = '/opt/coin/database/data/models/best_cnn_v2.pth'

            # Backup des alten Modells
            backup_path = f'/opt/coin/database/data/models/cnn_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            if os.path.exists(model_path):
                import shutil
                shutil.copy2(model_path, backup_path)
                logger.info(f"[LEARNER] Backup: {backup_path}")

            # Neues Modell speichern + aktivieren
            torch.save(new_model.state_dict(), model_path)
            _cnn_model = new_model

            # Info speichern
            info_path = '/opt/coin/database/data/models/model_info.json'
            with open(info_path, 'w') as f:
                json.dump({
                    'test_accuracy': round(new_acc, 2),
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'events_used': len(events_with_tf),
                    'previous_accuracy': round(current_acc, 2),
                }, f, indent=2)

            elapsed = (time.time() - t0) / 60
            logger.info(f"[LEARNER] HOT-SWAP: {current_acc:.1f}% → {new_acc:.1f}% (+{new_acc-current_acc:.1f}pp) in {elapsed:.1f}min")
        else:
            # Kandidat verworfen
            elapsed = (time.time() - t0) / 60
            logger.info(f"[LEARNER] KEIN SWAP: Neues {new_acc:.1f}% ≤ Aktuelles {current_acc:.1f}% ({elapsed:.1f}min)")
            # Kandidat-Datei aufräumen
            cand = '/opt/coin/database/data/models/cnn_candidate.pth'
            if os.path.exists(cand):
                os.remove(cand)

    except Exception as e:
        logger.error(f"[LEARNER] Fehler: {e}")
        logger.error(traceback.format_exc())


class LearnerThread(threading.Thread):
    """Background-Thread der alle 12h das CNN-Modell nachtrainiert."""

    def __init__(self, interval_hours=12):
        super().__init__(daemon=True, name='CNN-Learner')
        self.interval = interval_hours * 3600
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        logger.info(f"[LEARNER] Thread gestartet — Intervall: alle {self.interval // 3600}h")
        # Erster Lauf nach 1h (Scanner soll erst warm werden)
        self._stop_event.wait(3600)

        while not self._stop_event.is_set() and running:
            try:
                run_learning_cycle()
            except Exception as e:
                logger.error(f"[LEARNER] Thread-Fehler: {e}")
                logger.error(traceback.format_exc())

            # Warten bis zum nächsten Zyklus
            self._stop_event.wait(self.interval)

        logger.info("[LEARNER] Thread beendet")


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
            WHERE user_id = %s AND status = 'active' AND (scanner_type = 'default' OR scanner_type IS NULL)
        """, (user_id,))
        active_symbols = {r['symbol'] for r in acur.fetchall()}

        # Cooldown: Symbole die in den letzten 60 Min resolved wurden nicht nochmal scannen
        acur.execute("""
            SELECT DISTINCT symbol FROM momentum_predictions
            WHERE user_id = %s AND status != 'active' AND (scanner_type = 'default' OR scanner_type IS NULL)
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
                # Aktuellen Preis aus 1m-Klines holen
                ccur.execute("""
                    SELECT close FROM klines
                    WHERE symbol = %s AND interval = '1m'
                    ORDER BY open_time DESC LIMIT 1
                """, (symbol,))
                row_1m = ccur.fetchone()
                if not row_1m or not row_1m['close'] or row_1m['close'] == 0:
                    continue
                current_price = row_1m['close']

                # CNN-Analyse (lädt intern agg_5m/1h/4h/1d, normalisiert, Inference)
                signal = analyze_symbol_cnn(ccur, symbol, current_price, market_context=market_context, scan_config=config)

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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW() + INTERVAL '48 hours')
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
    logger.info("Momentum Scanner v4 (CNN) + Learning starting...")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)

    # CNN-Modell beim Start laden
    model = get_cnn_model()
    if model is None:
        logger.error("[CNN] KEIN MODELL VERFÜGBAR — Scanner kann nicht starten!")
        logger.error("[CNN] Bitte erst ts_classifier.py ausführen um das Modell zu trainieren.")
        return
    logger.info("[CNN] Modell bereit — Scanner läuft mit CNN-basierter Erkennung")

    # Learning Thread starten (trainiert alle 12h nach)
    learner = LearnerThread(interval_hours=12)
    learner.start()
    logger.info("[LEARNER] Background-Thread gestartet (12h Intervall)")

    # Initiales Heartbeat
    write_heartbeat('starting')

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

                # Heartbeat nach jedem Zyklus
                write_heartbeat('ok', {
                    'learner_alive': learner.is_alive(),
                    'model_loaded': _cnn_model is not None,
                })

                # Idle
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

    # Learner Thread stoppen
    learner.stop()
    learner.join(timeout=5)
    write_heartbeat('stopped')
    logger.info("Momentum Scanner stopped.")


if __name__ == '__main__':
    main()
