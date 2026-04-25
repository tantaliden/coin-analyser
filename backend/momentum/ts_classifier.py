#!/usr/bin/env python3
"""
TIME SERIES CLASSIFIER v2 — Multi-Timeframe CNN

ALLE verfügbaren Datenquellen:
- agg_5m: 144 Candles (12h) × 7 Rohdaten-Spalten
- agg_1h: 24 Candles (24h) × 7 Rohdaten-Spalten
- agg_4h: 12 Candles (48h) × 7 Rohdaten-Spalten
- agg_1d: 14 Candles (14 Tage) × 7 Rohdaten-Spalten

Jeder Timeframe wird separat normalisiert und hat eigene CNN-Branches.
Das Netz findet selbst welche Kombinationen aus Timeframe + Kanal + Zeitfenster
Long von Short unterscheiden.

Usage: nohup python3 ts_classifier.py > /opt/coin/logs/classifier_stdout.log 2>&1 &
"""

import os, sys, logging, time, pickle
from datetime import datetime, timedelta
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# === PATHS ===
LOG_FILE = '/opt/coin/logs/ts_classifier.log'
CHECKPOINT_DIR = '/opt/coin/database/data/checkpoints'
REPORT_DIR = '/opt/coin/database/data'
MODEL_DIR = '/opt/coin/database/data/models'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('classifier')

# === CONFIG ===
TRAIN_RATIO = 0.7
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.001
PATIENCE = 30

# Timeframe-Definitionen
TF_5M  = {'table': 'agg_5m',  'hours_back': 12, 'expected_candles': 144, 'min_candles': 72}
TF_1H  = {'table': 'agg_1h',  'hours_back': 24, 'expected_candles': 24,  'min_candles': 12}
TF_4H  = {'table': 'agg_4h',  'hours_back': 48, 'expected_candles': 12,  'min_candles': 6}
TF_1D  = {'table': 'agg_1d',  'hours_back': 336, 'expected_candles': 14, 'min_candles': 7}  # 14 Tage

# Rohdaten-Spalten pro Timeframe (7 Stück)
RAW_COLS = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'taker_buy_base_asset_volume']


def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


def save_checkpoint(name, data):
    path = os.path.join(CHECKPOINT_DIR, f'clf_{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    size = os.path.getsize(path) / 1024 / 1024
    logger.info(f"  Checkpoint: {path} ({size:.1f}MB)")


def load_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, f'clf_{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ========================================================================
# PHASE 1: ALLE TIMEFRAMES LADEN
# ========================================================================

def load_all_timeframes(conn, events):
    """
    Für jedes Event: Lade ALLE Timeframes (5m, 1h, 4h, 1d) mit ALLEN 7 Rohdaten-Spalten.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: ALLE TIMEFRAMES LADEN")
    logger.info(f"  {len(events)} Events")
    logger.info(f"  Timeframes: agg_5m (12h), agg_1h (24h), agg_4h (48h), agg_1d (14d)")
    logger.info(f"  Spalten pro TF: {len(RAW_COLS)} ({', '.join(RAW_COLS)})")
    logger.info("=" * 70)

    cur = conn.cursor()
    results = []
    skipped = 0
    t0 = time.time()

    timeframes = [
        ('5m', TF_5M),
        ('1h', TF_1H),
        ('4h', TF_4H),
        ('1d', TF_1D),
    ]

    for i, event in enumerate(events):
        if i > 0 and i % 2000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(events) - i) / rate / 60
            logger.info(f"  Lade: {i}/{len(events)} ({elapsed/60:.1f}min, ETA {eta:.1f}min, "
                        f"valid={len(results)}, skip={skipped})")

        entry_time = event['time']  # Start des Moves
        symbol = event['symbol']

        tf_data = {}
        valid = True

        for tf_name, tf_cfg in timeframes:
            window_start = entry_time - timedelta(hours=tf_cfg['hours_back'])
            try:
                cur.execute(f"""
                    SELECT bucket, open, high, low, close, volume,
                           number_of_trades, taker_buy_base_asset_volume
                    FROM {tf_cfg['table']}
                    WHERE symbol = %s
                      AND bucket >= %s
                      AND bucket < %s
                    ORDER BY bucket
                """, (symbol, window_start, entry_time))
                candles = cur.fetchall()
            except Exception as ex:
                conn.rollback()
                valid = False
                break

            if len(candles) < tf_cfg['min_candles']:
                valid = False
                break

            n = len(candles)
            data = {
                'n': n,
                'open':    np.array([float(c['open']) for c in candles]),
                'high':    np.array([float(c['high']) for c in candles]),
                'low':     np.array([float(c['low']) for c in candles]),
                'close':   np.array([float(c['close']) for c in candles]),
                'volume':  np.array([float(c['volume']) for c in candles]),
                'trades':  np.array([float(c['number_of_trades'] or 0) for c in candles]),
                'taker':   np.array([float(c['taker_buy_base_asset_volume'] or 0) for c in candles]),
            }
            tf_data[tf_name] = data

        if not valid:
            skipped += 1
            continue

        results.append({
            'event': event,
            'tf': tf_data,
        })

    elapsed = time.time() - t0
    logger.info(f"  Fertig: {len(results)} Events mit allen TFs, {skipped} übersprungen")
    logger.info(f"  Dauer: {elapsed/60:.1f} Minuten")
    return results


# ========================================================================
# PHASE 2: NORMALISIERUNG + TENSOR-AUFBEREITUNG
# ========================================================================

def normalize_timeframe(data, expected_len):
    """
    Normalisiert einen Timeframe zu 7 Kanälen:
    1. price_ret: Returns (close[i]/close[i-1] - 1), erster Wert = 0
    2. volume_rel: Volume / Median-Volume des Fensters
    3. trades_rel: Trades / Median-Trades des Fensters
    4. taker_ratio: Taker Buy / Volume (Käufer-Anteil, ~0.4-0.6)
    5. range_pct: (High-Low) / Close × 100 (Candle-Volatilität)
    6. body_dir: (Close-Open) / (High-Low) wenn Range>0, sonst 0 (Richtungsstärke, -1 bis +1)
    7. hl_position: (Close-Low) / (High-Low) wenn Range>0, sonst 0.5 (Position im Candle, 0-1)
    """
    closes = data['close']
    opens = data['open']
    highs = data['high']
    lows = data['low']
    volumes = data['volume']
    trades = data['trades']
    taker = data['taker']
    n = len(closes)

    # 1. Returns (prozentuale Veränderung Candle zu Candle)
    price_ret = np.zeros(n)
    for j in range(1, n):
        if closes[j-1] > 0:
            price_ret[j] = (closes[j] / closes[j-1] - 1) * 100

    # 2. Volume relativ zum Median
    med_vol = np.median(volumes)
    volume_rel = volumes / med_vol if med_vol > 0 else np.ones(n)

    # 3. Trades relativ zum Median
    med_trades = np.median(trades)
    trades_rel = trades / med_trades if med_trades > 0 else np.ones(n)

    # 4. Taker Ratio
    taker_ratio = np.where(volumes > 0, taker / volumes, 0.5)

    # 5. Range %
    range_pct = np.where(closes > 0, (highs - lows) / closes * 100, 0)

    # 6. Body Direction (-1 bis +1)
    full_range = highs - lows
    body_dir = np.where(full_range > 0, (closes - opens) / full_range, 0)

    # 7. HL Position (0-1)
    hl_pos = np.where(full_range > 0, (closes - lows) / full_range, 0.5)

    # Stack: (7, n)
    channels = np.stack([price_ret, volume_rel, trades_rel, taker_ratio,
                         range_pct, body_dir, hl_pos])

    # Auf expected_len bringen
    if n >= expected_len:
        channels = channels[:, -expected_len:]  # Letzte N (nächst am Event)
    else:
        pad = np.zeros((7, expected_len - n))
        channels = np.concatenate([pad, channels], axis=1)

    # Bereinigen
    channels = np.nan_to_num(channels, nan=0.0, posinf=20.0, neginf=-20.0)
    channels = np.clip(channels, -50, 50)

    return channels.astype(np.float32)


def prepare_tensors(events_with_tf):
    """Alle Events in Tensoren konvertieren."""
    logger.info("=" * 70)
    logger.info("PHASE 2: NORMALISIERUNG")
    logger.info(f"  {len(events_with_tf)} Events, 4 Timeframes × 7 Kanäle = 28 Datenströme")
    logger.info("=" * 70)

    X_5m_all = []
    X_1h_all = []
    X_4h_all = []
    X_1d_all = []
    y_all = []
    times_all = []

    for ed in events_with_tf:
        X_5m_all.append(normalize_timeframe(ed['tf']['5m'], TF_5M['expected_candles']))
        X_1h_all.append(normalize_timeframe(ed['tf']['1h'], TF_1H['expected_candles']))
        X_4h_all.append(normalize_timeframe(ed['tf']['4h'], TF_4H['expected_candles']))
        X_1d_all.append(normalize_timeframe(ed['tf']['1d'], TF_1D['expected_candles']))
        y_all.append(1.0 if ed['event']['direction'] == 'long' else 0.0)
        times_all.append(ed['event']['time'])

    X_5m = np.array(X_5m_all)  # (N, 7, 144)
    X_1h = np.array(X_1h_all)  # (N, 7, 24)
    X_4h = np.array(X_4h_all)  # (N, 7, 12)
    X_1d = np.array(X_1d_all)  # (N, 7, 14)
    y = np.array(y_all, dtype=np.float32)

    logger.info(f"  5m: {X_5m.shape}  (12h, {TF_5M['expected_candles']} Candles)")
    logger.info(f"  1h: {X_1h.shape}  (24h, {TF_1H['expected_candles']} Candles)")
    logger.info(f"  4h: {X_4h.shape}  (48h, {TF_4H['expected_candles']} Candles)")
    logger.info(f"  1d: {X_1d.shape}  (14d, {TF_1D['expected_candles']} Candles)")
    logger.info(f"  Labels: Long={int(y.sum())}, Short={int(len(y)-y.sum())}")
    logger.info(f"  Baseline: {max(y.mean(), 1-y.mean())*100:.1f}%")

    return X_5m, X_1h, X_4h, X_1d, y, times_all


def split_train_test(X_5m, X_1h, X_4h, X_1d, y, times):
    """Zeitlicher Split 70/30."""
    indices = np.argsort([t.timestamp() if hasattr(t, 'timestamp') else 0 for t in times])
    X_5m, X_1h, X_4h, X_1d, y = X_5m[indices], X_1h[indices], X_4h[indices], X_1d[indices], y[indices]

    s = int(len(y) * TRAIN_RATIO)
    train = (X_5m[:s], X_1h[:s], X_4h[:s], X_1d[:s], y[:s])
    test  = (X_5m[s:], X_1h[s:], X_4h[s:], X_1d[s:], y[s:])

    logger.info(f"  Train: {s} (Long={int(train[4].sum())}, Short={int(s-train[4].sum())})")
    logger.info(f"  Test:  {len(y)-s} (Long={int(test[4].sum())}, Short={int(len(y)-s-test[4].sum())})")
    return train, test


# ========================================================================
# CNN MODELL — Multi-Timeframe
# ========================================================================

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


class MultiTimeframeCNN(nn.Module):
    """
    Separater CNN-Branch pro Timeframe, dann Fusion.

    Jeder Branch hat Multi-Scale Convolutions (kurz + lang)
    damit er Muster auf verschiedenen Zeitskalen findet.
    """
    def __init__(self):
        super().__init__()
        n_ch = 7  # Kanäle pro Timeframe

        # === 5m Branch (144 Steps, 12h) — Feinste Auflösung ===
        self.branch_5m = nn.Sequential(
            # Kurze Muster (5-15 Candles = 25-75min)
            nn.Conv1d(n_ch, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48), nn.ReLU(),
            nn.MaxPool1d(2),  # → 72
            nn.Dropout(0.2),
            # Mittlere Muster
            nn.Conv1d(48, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),  # → 36
            nn.Dropout(0.2),
            # Lange Muster
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # → 8
        )

        # === 1h Branch (24 Steps, 24h) ===
        self.branch_1h = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),  # → 12
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # → 4
        )

        # === 4h Branch (12 Steps, 48h) ===
        self.branch_4h = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # → 4
        )

        # === 1d Branch (14 Steps, 14 Tage) ===
        self.branch_1d = nn.Sequential(
            nn.Conv1d(n_ch, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # → 4
        )

        # === Fusion ===
        # 5m: 64×8 = 512, 1h: 64×4 = 256, 4h: 64×4 = 256, 1d: 64×4 = 256
        # Total: 1280
        fusion_size = 64*8 + 64*4 + 64*4 + 64*4

        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x5m, x1h, x4h, x1d):
        f5m = self.branch_5m(x5m).flatten(1)  # (batch, 512)
        f1h = self.branch_1h(x1h).flatten(1)  # (batch, 256)
        f4h = self.branch_4h(x4h).flatten(1)  # (batch, 256)
        f1d = self.branch_1d(x1d).flatten(1)  # (batch, 256)

        fused = torch.cat([f5m, f1h, f4h, f1d], dim=1)  # (batch, 1280)
        out = self.fusion(fused)
        return out.squeeze(-1)


# ========================================================================
# TRAINING
# ========================================================================

def train_model(train_data, test_data):
    X5_tr, X1_tr, X4_tr, Xd_tr, y_tr = train_data
    X5_te, X1_te, X4_te, Xd_te, y_te = test_data

    logger.info("=" * 70)
    logger.info("TRAINING — Multi-Timeframe CNN")
    logger.info(f"  Epochs: max {EPOCHS}, Early Stopping Patience: {PATIENCE}")
    logger.info(f"  Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    logger.info("=" * 70)

    train_ds = MultiTFDataset(X5_tr, X1_tr, X4_tr, Xd_tr, y_tr)
    test_ds  = MultiTFDataset(X5_te, X1_te, X4_te, Xd_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTimeframeCNN()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameter: {n_params:,}")

    # Class weight
    n_long = y_tr.sum()
    n_short = len(y_tr) - n_long
    pos_weight = torch.FloatTensor([n_short / max(n_long, 1)])
    logger.info(f"  Class Weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6)

    best_test_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    t0 = time.time()
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x5, x1, x4, xd, yb in train_loader:
            optimizer.zero_grad()
            out = model(x5, x1, x4, xd)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(out) >= 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

        train_acc = train_correct / train_total * 100
        train_loss /= train_total

        # --- Test ---
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for x5, x1, x4, xd, yb in test_loader:
                out = model(x5, x1, x4, xd)
                loss = criterion(out, yb)
                test_loss += loss.item() * len(yb)
                probs = torch.sigmoid(out)
                preds = (probs >= 0.5).float()
                test_correct += (preds == yb).sum().item()
                test_total += len(yb)
                all_preds.extend(preds.numpy())
                all_labels.extend(yb.numpy())
                all_probs.extend(probs.numpy())

        test_acc = test_correct / test_total * 100
        test_loss /= test_total
        scheduler.step(test_acc)

        history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc, 2),
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Log alle 5 Epochen
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Epoch {epoch+1:4d}/{EPOCHS} | "
                        f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
                        f"Test: {test_loss:.4f} / {test_acc:.1f}% | "
                        f"LR={lr:.2e} | Best={best_test_acc:.1f}%@{best_epoch}")

        # Early Stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_cnn_v2.pth'))
            best_preds = all_preds
            best_labels = all_labels
            best_probs = all_probs
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early Stopping: Epoch {epoch+1}, keine Verbesserung seit {PATIENCE} Epochen")
                break

    elapsed = time.time() - t0
    logger.info(f"  Training fertig: {elapsed/60:.1f}min, Beste Test-Acc: {best_test_acc:.1f}% (Epoch {best_epoch})")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_cnn_v2.pth'), weights_only=True))
    return model, history, best_preds, best_labels, best_probs


# ========================================================================
# ANALYSE
# ========================================================================

def analyze_model(model, test_data):
    """Gradient-basierte Feature Importance pro Timeframe und Kanal."""
    logger.info("=" * 70)
    logger.info("ANALYSE: Feature Importance")
    logger.info("=" * 70)

    X5, X1, X4, Xd, y = test_data
    channel_names = ['price_ret', 'volume_rel', 'trades_rel', 'taker_ratio',
                     'range_pct', 'body_dir', 'hl_position']
    tf_names = ['5m', '1h', '4h', '1d']

    model.eval()

    # Tensoren mit Gradient-Tracking
    t5 = torch.FloatTensor(X5); t5.requires_grad = True
    t1 = torch.FloatTensor(X1); t1.requires_grad = True
    t4 = torch.FloatTensor(X4); t4.requires_grad = True
    td = torch.FloatTensor(Xd); td.requires_grad = True

    # Batch-weise (Speicher)
    batch = 2048
    grads = {tf: [] for tf in tf_names}
    tensors_map = {'5m': t5, '1h': t1, '4h': t4, '1d': td}

    for start in range(0, len(y), batch):
        end = min(start + batch, len(y))
        b5 = t5[start:end].detach().requires_grad_(True)
        b1 = t1[start:end].detach().requires_grad_(True)
        b4 = t4[start:end].detach().requires_grad_(True)
        bd = td[start:end].detach().requires_grad_(True)

        out = model(b5, b1, b4, bd)
        out.sum().backward()

        grads['5m'].append(b5.grad.abs().numpy())
        grads['1h'].append(b1.grad.abs().numpy())
        grads['4h'].append(b4.grad.abs().numpy())
        grads['1d'].append(bd.grad.abs().numpy())

    # Zusammenführen
    for tf in tf_names:
        grads[tf] = np.concatenate(grads[tf], axis=0)

    # === Timeframe-Wichtigkeit ===
    tf_importance = {}
    for tf in tf_names:
        tf_importance[tf] = grads[tf].mean()
    total_tf = sum(tf_importance.values())
    logger.info("  TIMEFRAME-WICHTIGKEIT:")
    for tf in sorted(tf_names, key=lambda t: -tf_importance[t]):
        pct = tf_importance[tf] / total_tf * 100
        bar = '█' * int(pct / 2)
        logger.info(f"    {tf:4s}: {pct:5.1f}% {bar}")

    # === Kanal-Wichtigkeit pro Timeframe ===
    logger.info("  KANAL-WICHTIGKEIT pro Timeframe:")
    for tf in tf_names:
        ch_imp = grads[tf].mean(axis=(0, 2))  # (7,)
        total_ch = ch_imp.sum()
        if total_ch == 0:
            continue
        logger.info(f"    --- {tf} ---")
        for ci in np.argsort(-ch_imp):
            pct = ch_imp[ci] / total_ch * 100
            bar = '█' * int(pct / 2)
            logger.info(f"      {channel_names[ci]:14s}: {pct:5.1f}% {bar}")

    # === Zeitliche Wichtigkeit (5m Branch detailliert) ===
    time_imp_5m = grads['5m'].mean(axis=(0, 1))  # (144,)
    blocks = [
        (0, 24,  '12-10h'), (24, 48,  '10-8h'), (48, 72,  '8-6h'),
        (72, 96,  '6-4h'),  (96, 120, '4-2h'),  (120, 132, '2-1h'),
        (132, 138, '1h-30m'), (138, 144, '30-0m'),
    ]
    total_t = time_imp_5m.sum()
    logger.info("  ZEITLICHE WICHTIGKEIT (5m):")
    for s, e, label in blocks:
        pct = time_imp_5m[s:e].sum() / total_t * 100
        bar = '█' * int(pct / 2)
        logger.info(f"    {label:10s}: {pct:5.1f}% {bar}")

    # === Top 20: TF × Kanal ===
    logger.info("  TOP 20: Timeframe × Kanal:")
    combos = []
    for tf in tf_names:
        ch_imp = grads[tf].mean(axis=(0, 2))
        for ci, cn in enumerate(channel_names):
            combos.append((tf, cn, ch_imp[ci]))
    combos.sort(key=lambda x: -x[2])
    total_combo = sum(c[2] for c in combos)
    for tf, cn, val in combos[:20]:
        pct = val / total_combo * 100
        logger.info(f"    {tf:4s} × {cn:14s}: {pct:5.1f}%")

    return tf_importance, grads


def analyze_confidence(probs_list, labels_list):
    logger.info("=" * 70)
    logger.info("CONFIDENCE ANALYSE")
    logger.info("=" * 70)

    probs = np.array(probs_list)
    labels = np.array(labels_list)

    logger.info(f"  {'Threshold':>10s} | {'Coverage':>10s} | {'Accuracy':>10s} | {'Long Acc':>10s} | {'Short Acc':>10s} | {'N':>6s}")
    logger.info(f"  {'-'*65}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        confident = (probs >= thresh) | (probs <= (1 - thresh))
        n_conf = confident.sum()
        if n_conf == 0:
            continue
        cp = probs[confident]
        cl = labels[confident]
        cpreds = (cp >= 0.5).astype(float)
        acc = (cpreds == cl).mean() * 100
        cov = n_conf / len(probs) * 100
        lm = cl == 1
        sm = cl == 0
        lacc = (cpreds[lm] == cl[lm]).mean() * 100 if lm.sum() > 0 else 0
        sacc = (cpreds[sm] == cl[sm]).mean() * 100 if sm.sum() > 0 else 0
        logger.info(f"  {thresh:10.2f} | {cov:9.1f}% | {acc:9.1f}% | {lacc:9.1f}% | {sacc:9.1f}% | {n_conf:6d}")


# ========================================================================
# REPORT
# ========================================================================

def write_report(history, best_preds, best_labels, best_probs, train_data, test_data, tf_importance):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORT_DIR, f'classifier_report_{ts}.txt')

    y_tr = train_data[4]
    y_te = test_data[4]
    preds = np.array(best_preds)
    labels = np.array(best_labels)
    probs = np.array(best_probs)
    baseline = max(y_te.mean(), 1 - y_te.mean()) * 100

    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-TIMEFRAME CNN CLASSIFIER v2 — ERGEBNISSE\n")
        f.write(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timeframes: agg_5m (12h), agg_1h (24h), agg_4h (48h), agg_1d (14d)\n")
        f.write(f"Kanäle pro TF: 7 (price_ret, volume_rel, trades_rel, taker_ratio, range_pct, body_dir, hl_pos)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Train: {len(y_tr)} (Long={int(y_tr.sum())}, Short={int(len(y_tr)-y_tr.sum())})\n")
        f.write(f"Test:  {len(y_te)} (Long={int(y_te.sum())}, Short={int(len(y_te)-y_te.sum())})\n")
        f.write(f"Baseline: {baseline:.1f}%\n\n")

        best_h = max(history, key=lambda h: h['test_acc'])
        f.write(f"BESTE TEST-ACCURACY: {best_h['test_acc']:.1f}% (Epoch {best_h['epoch']})\n")
        f.write(f"  vs Baseline: +{best_h['test_acc'] - baseline:.1f}pp\n\n")

        # History (jede 10. Epoche + beste)
        f.write("-" * 80 + "\n")
        f.write("TRAINING HISTORY (Auszug)\n")
        f.write("-" * 80 + "\n")
        for h in history:
            if h['epoch'] % 10 == 0 or h['epoch'] == best_h['epoch'] or h['epoch'] <= 5:
                marker = " <<<" if h['epoch'] == best_h['epoch'] else ""
                f.write(f"  Epoch {h['epoch']:4d} | Train: {h['train_acc']:.1f}% | "
                        f"Test: {h['test_acc']:.1f}% | LR={h['lr']:.2e}{marker}\n")
        f.write("\n")

        # Classification Report
        f.write("-" * 80 + "\n")
        f.write("CLASSIFICATION REPORT (Test-Set)\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(labels, preds, target_names=['SHORT', 'LONG']))
        f.write("\n")

        cm = confusion_matrix(labels, preds)
        f.write("Confusion Matrix:\n")
        f.write(f"              Pred SHORT  Pred LONG\n")
        f.write(f"  Act SHORT   {cm[0][0]:8d}   {cm[0][1]:8d}\n")
        f.write(f"  Act LONG    {cm[1][0]:8d}   {cm[1][1]:8d}\n\n")

        # Timeframe Importance
        f.write("-" * 80 + "\n")
        f.write("TIMEFRAME WICHTIGKEIT\n")
        f.write("-" * 80 + "\n")
        total = sum(tf_importance.values())
        for tf in sorted(tf_importance, key=lambda t: -tf_importance[t]):
            pct = tf_importance[tf] / total * 100
            f.write(f"  {tf:4s}: {pct:.1f}%\n")
        f.write("\n")

        # Confidence
        f.write("-" * 80 + "\n")
        f.write("CONFIDENCE ANALYSE\n")
        f.write("-" * 80 + "\n")
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            confident = (probs >= thresh) | (probs <= (1 - thresh))
            if confident.sum() == 0:
                continue
            cp = probs[confident]
            cl = labels[confident]
            acc = ((cp >= 0.5).astype(float) == cl).mean() * 100
            cov = confident.mean() * 100
            f.write(f"  ≥{thresh:.2f}: Coverage={cov:.1f}%, Accuracy={acc:.1f}%, N={confident.sum()}\n")

    logger.info(f"Report: {path}")
    return path


# ========================================================================
# MAIN
# ========================================================================

def main():
    logger.info("=" * 80)
    logger.info("MULTI-TIMEFRAME CNN CLASSIFIER v2 — START")
    logger.info("  Daten: agg_5m (12h) + agg_1h (24h) + agg_4h (48h) + agg_1d (14d)")
    logger.info("  Kanäle: 7 pro TF × 4 TFs = 28 Datenströme")
    logger.info("=" * 80)
    t_start = time.time()

    # --- Events laden ---
    events_path = os.path.join(CHECKPOINT_DIR, 'diff_events_strict.pkl')
    logger.info(f"Events: {events_path}")
    with open(events_path, 'rb') as f:
        events = pickle.load(f)
    logger.info(f"  {len(events)} Events geladen")

    # --- Alle Timeframes laden ---
    checkpoint = load_checkpoint('multi_tf_data')
    if checkpoint is not None:
        events_with_tf = checkpoint
        logger.info(f"  Multi-TF Daten aus Checkpoint: {len(events_with_tf)} Events")
    else:
        conn = coins_db()
        events_with_tf = load_all_timeframes(conn, events)
        conn.close()
        save_checkpoint('multi_tf_data', events_with_tf)

    # --- Normalisieren ---
    X_5m, X_1h, X_4h, X_1d, y, times = prepare_tensors(events_with_tf)
    del events_with_tf

    # --- Split ---
    train_data, test_data = split_train_test(X_5m, X_1h, X_4h, X_1d, y, times)
    del X_5m, X_1h, X_4h, X_1d, y

    # --- Training ---
    model, history, best_preds, best_labels, best_probs = train_model(train_data, test_data)

    # --- Analyse ---
    tf_importance, grads = analyze_model(model, test_data)
    analyze_confidence(best_probs, best_labels)

    # --- Report ---
    report_path = write_report(history, best_preds, best_labels, best_probs, train_data, test_data, tf_importance)

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"FERTIG — Gesamtdauer: {elapsed/60:.1f} Minuten")
    logger.info(f"Report: {report_path}")
    logger.info(f"Modell: {os.path.join(MODEL_DIR, 'best_cnn_v2.pth')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
