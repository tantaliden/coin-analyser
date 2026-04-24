#!/usr/bin/env python3
"""
CNN-Predict — Standalone CNN Signal-Erkennung.
Extrahiert aus scanner.py fuer Wiederverwendung (z.B. Ensemble-Service).

Keine Side-Effects, kein Logging-Setup, kein Signal-Handler.
Import-sicher: from momentum.cnn_predict import get_cnn_model, analyze_symbol_cnn
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================
# CNN MODEL DEFINITION (identisch zu scanner.py / ts_classifier.py)
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


# ============================================
# CNN Model Singleton
# ============================================

_cnn_model = None
CNN_MODEL_PATH = '/opt/coin/database/data/models/best_cnn_v2.pth'


def get_cnn_model(model_path=None):
    """Laedt das trainierte CNN-Modell (einmal, dann gecached)."""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    path = model_path or CNN_MODEL_PATH
    if not os.path.exists(path):
        logger.error(f"[CNN] Modell nicht gefunden: {path}")
        return None
    try:
        model = MultiTimeframeCNN()
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        model.eval()
        _cnn_model = model
        logger.info(f"[CNN] Modell geladen: {path}")
        return model
    except Exception as e:
        logger.error(f"[CNN] Modell laden fehlgeschlagen: {e}")
        return None


def reload_cnn_model():
    """Erzwingt Neuladen des CNN-Modells."""
    global _cnn_model
    _cnn_model = None
    return get_cnn_model()


# ============================================
# Normalisierung
# ============================================

def normalize_candles_for_cnn(candles, expected_len):
    """
    Normalisiert Candle-Daten zu 7 Kanaelen fuer das CNN.
    Identisch zur Normalisierung in ts_classifier.py.

    candles: Liste von dicts mit 'open','high','low','close','volume',
             'number_of_trades','taker_buy_base_asset_volume'
    """
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


# ============================================
# ATR Helper
# ============================================

def calc_atr(highs, lows, closes, period=14):
    """Average True Range."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])


# ============================================
# CNN Signal-Erkennung
# ============================================

def analyze_symbol_cnn(cursor, symbol, current_price, now_time=None):
    """
    CNN-basierte Signal-Erkennung.
    Laedt agg_5m (144), agg_1h (24), agg_4h (12), agg_1d (14),
    normalisiert, schickt durch CNN -> Direction + Confidence.

    Args:
        cursor: DB-Cursor (coins DB)
        symbol: z.B. 'BTCUSDC'
        current_price: aktueller Preis
        now_time: Zeitpunkt (default: jetzt)

    Returns:
        dict mit direction, confidence, entry_price, expected_move_pct, reason, indicators
        oder None bei kein Signal / zu wenig Daten
    """
    from datetime import datetime, timezone
    if now_time is None:
        now_time = datetime.now(timezone.utc)

    model = get_cnn_model()
    if model is None:
        return None

    # Daten laden
    try:
        cursor.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_5m WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 144""",
                       (symbol, now_time))
        candles_5m = list(reversed(cursor.fetchall()))

        cursor.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1h WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 24""",
                       (symbol, now_time))
        candles_1h = list(reversed(cursor.fetchall()))

        cursor.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_4h WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 12""",
                       (symbol, now_time))
        candles_4h = list(reversed(cursor.fetchall()))

        cursor.execute("""SELECT bucket, open, high, low, close, volume,
                               number_of_trades, taker_buy_base_asset_volume
                        FROM agg_1d WHERE symbol = %s AND bucket <= %s ORDER BY bucket DESC LIMIT 14""",
                       (symbol, now_time))
        candles_1d = list(reversed(cursor.fetchall()))
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

    # ATR fuer TP/SL (aus 1h-Daten berechnen)
    closes_1h = [float(c['close']) for c in candles_1h]
    highs_1h  = [float(c['high']) for c in candles_1h]
    lows_1h   = [float(c['low']) for c in candles_1h]
    atr = calc_atr(highs_1h, lows_1h, closes_1h, 14)
    expected_move_pct = (atr * 2.5 / current_price * 100) if atr and current_price > 0 else 5.0

    reason = f"CNN v2: {direction.upper()} {prob*100:.1f}%" if direction == 'long' else f"CNN v2: {direction.upper()} {(1-prob)*100:.1f}%"

    return {
        'direction': direction,
        'confidence': confidence,
        'entry_price': current_price,
        'expected_move_pct': round(expected_move_pct, 2),
        'reason': reason,
        'cnn_prob': round(prob, 4),
        'atr': round(atr, 8) if atr else None,
    }
