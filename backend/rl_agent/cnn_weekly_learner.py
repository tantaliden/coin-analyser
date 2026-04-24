#!/usr/bin/env python3
"""
CNN Wöchentlicher Learner — Fine-Tuning auf Events der letzten 7 Tage.

Läuft als Timer-Service, einmal pro Woche (Montag 03:00).
Lädt Events der letzten 7 Tage, fine-tuned das bestehende CNN-Modell,
übernimmt nur wenn besser als das alte.

Usage:
    /opt/coin/venv/bin/python3 /opt/coin/backend/rl_agent/cnn_weekly_learner.py
"""
import sys
import os
import json
import time
import copy
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'momentum'))

from momentum.scanner import (
    MultiTimeframeCNN, MultiTFDataset,
    _normalize_tf_candles, get_cnn_model, _cnn_model,
)

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = "/opt/coin/database/data/models/best_cnn_v2.pth"
BACKUP_DIR = "/opt/coin/database/data/models"
MIN_EVENTS = 200
LEARNING_RATE = 0.0001
PATIENCE = 15
BATCH_SIZE = 256
MIN_PCT = 5.0

PCT_COLS = [
    'pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
    'pct_300m', 'pct_330m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m',
]

TF_CONFIG = {
    '5m': {'table': 'agg_5m', 'candles': 144, 'hours_back': 12},
    '1h': {'table': 'agg_1h', 'candles': 24, 'hours_back': 24},
    '4h': {'table': 'agg_4h', 'candles': 12, 'hours_back': 48},
    '1d': {'table': 'agg_1d', 'candles': 14, 'hours_back': 336},
}


def get_conn(db_key='coins'):
    import psycopg2
    import psycopg2.extras
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def load_events(days=7):
    """Events der letzten X Tage aus kline_metrics."""
    conn = get_conn()
    cur = conn.cursor()
    where = ' OR '.join(f'ABS({col}) >= {MIN_PCT}' for col in PCT_COLS)
    start = datetime.now() - timedelta(days=days)

    cur.execute(f"""
        SELECT symbol, open_time, {', '.join(PCT_COLS)}
        FROM kline_metrics
        WHERE open_time >= %s AND ({where})
        ORDER BY symbol, open_time
    """, (start,))
    rows = cur.fetchall()
    print(f"  {len(rows)} Kandidaten aus kline_metrics")

    if not rows:
        conn.close()
        return []

    # Bestes pct-Feld pro Row
    events = []
    durations = {'pct_30m': 30, 'pct_60m': 60, 'pct_90m': 90, 'pct_120m': 120,
                 'pct_180m': 180, 'pct_240m': 240, 'pct_300m': 300, 'pct_330m': 330,
                 'pct_360m': 360, 'pct_420m': 420, 'pct_480m': 480, 'pct_540m': 540, 'pct_600m': 600}

    for row in rows:
        best_col = None
        best_pct = 0
        for col in PCT_COLS:
            val = float(row[col]) if row[col] is not None else 0
            if abs(val) >= MIN_PCT and (best_col is None or durations[col] < durations[best_col]):
                best_col = col
                best_pct = val

        if best_col is None:
            continue

        duration = durations[best_col]
        direction = 'long' if best_pct > 0 else 'short'
        entry_time = row['open_time'] - timedelta(minutes=duration)

        events.append({
            'symbol': row['symbol'],
            'time': entry_time,
            'direction': direction,
            'best_pct': round(best_pct, 2),
        })

    # Dedup: 1 Event pro Symbol pro 60 Min
    events.sort(key=lambda e: (e['symbol'], e['time']))
    deduped = []
    last_by_sym = {}
    for e in events:
        key = e['symbol']
        if key in last_by_sym:
            diff = (e['time'] - last_by_sym[key]).total_seconds() / 60
            if diff < 60:
                continue
        last_by_sym[key] = e['time']
        deduped.append(e)

    longs = sum(1 for e in deduped if e['direction'] == 'long')
    shorts = sum(1 for e in deduped if e['direction'] == 'short')
    print(f"  {len(deduped)} Events nach Dedup ({longs} Long, {shorts} Short)")

    conn.close()
    return deduped


def load_timeframes(events):
    """Multi-TF Candle-Daten laden."""
    conn = get_conn()
    cur = conn.cursor()
    results = []

    for i, event in enumerate(events):
        if i > 0 and i % 2000 == 0:
            print(f"  TF-Laden: {i}/{len(events)}, valid={len(results)}")

        tf_data = {}
        ok = True
        for tf_name, cfg in TF_CONFIG.items():
            start_time = event['time'] - timedelta(hours=cfg['hours_back'])
            try:
                cur.execute(f"""
                    SELECT open, high, low, close, volume, number_of_trades, taker_buy_base_asset_volume
                    FROM {cfg['table']}
                    WHERE symbol = %s AND bucket >= %s AND bucket < %s
                    ORDER BY bucket
                """, (event['symbol'], start_time, event['time']))
                candles = cur.fetchall()
            except Exception:
                conn.rollback()
                ok = False
                break

            if len(candles) < cfg['candles'] // 2:
                ok = False
                break

            arr = np.array([[float(c['open']), float(c['high']), float(c['low']),
                             float(c['close']), float(c['volume'] or 0),
                             float(c['number_of_trades'] or 0),
                             float(c['taker_buy_base_asset_volume'] or 0)]
                            for c in candles], dtype=np.float32)
            tf_data[tf_name] = arr

        if ok and len(tf_data) == 4:
            results.append({'event': event, 'tf': tf_data})

    cur.close()
    conn.close()
    print(f"  {len(results)} Events mit allen TFs geladen")
    return results


def prepare_tensors(events_with_tf):
    TF_EXPECTED = {'5m': 144, '1h': 24, '4h': 12, '1d': 14}
    X_5m, X_1h, X_4h, X_1d, y = [], [], [], [], []
    for ed in events_with_tf:
        X_5m.append(_normalize_tf_candles(ed['tf']['5m'], TF_EXPECTED['5m']))
        X_1h.append(_normalize_tf_candles(ed['tf']['1h'], TF_EXPECTED['1h']))
        X_4h.append(_normalize_tf_candles(ed['tf']['4h'], TF_EXPECTED['4h']))
        X_1d.append(_normalize_tf_candles(ed['tf']['1d'], TF_EXPECTED['1d']))
        y.append(1.0 if ed['event']['direction'] == 'long' else 0.0)
    return np.array(X_5m), np.array(X_1h), np.array(X_4h), np.array(X_1d), np.array(y, dtype=np.float32)


def run():
    print("=" * 60)
    print(f"CNN Weekly Learner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Events laden
    print("\n1. Events der letzten 7 Tage laden...")
    events = load_events(days=7)
    if len(events) < MIN_EVENTS:
        print(f"  Nur {len(events)} Events — brauche mindestens {MIN_EVENTS}. Abbruch.")
        return

    # 2. Timeframes laden
    print("\n2. Timeframe-Daten laden...")
    events_tf = load_timeframes(events)
    del events
    if len(events_tf) < MIN_EVENTS:
        print(f"  Nur {len(events_tf)} Events mit TF — zu wenig. Abbruch.")
        return

    # 3. Tensoren vorbereiten
    print("\n3. Tensoren vorbereiten...")
    X_5m, X_1h, X_4h, X_1d, y = prepare_tensors(events_tf)
    del events_tf

    split_idx = int(len(y) * 0.8)
    train_ds = MultiTFDataset(X_5m[:split_idx], X_1h[:split_idx], X_4h[:split_idx], X_1d[:split_idx], y[:split_idx])
    test_ds = MultiTFDataset(X_5m[split_idx:], X_1h[split_idx:], X_4h[split_idx:], X_1d[split_idx:], y[split_idx:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    del X_5m, X_1h, X_4h, X_1d, y

    n_train = len(train_ds)
    n_test = len(test_ds)
    print(f"  Train: {n_train}, Test: {n_test}")

    # 4. Altes Modell laden
    print("\n4. Bestehendes Modell laden...")
    model = MultiTimeframeCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        print(f"  Geladen: {MODEL_PATH}")
    else:
        print("  KEIN bestehendes Modell — from scratch!")

    # Altes Modell evaluieren
    model.eval()
    old_correct = 0
    old_total = 0
    with torch.no_grad():
        for x5, x1, x4, x1d, labels in test_loader:
            outputs = model(x5, x1, x4, x1d).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            old_correct += (preds == labels).sum().item()
            old_total += len(labels)
    old_acc = old_correct / max(old_total, 1) * 100
    print(f"  Alt: {old_acc:.1f}% auf Test-Set")

    # 5. Fine-Tuning
    print(f"\n5. Fine-Tuning (lr={LEARNING_RATE}, patience={PATIENCE})...")
    new_model = copy.deepcopy(model)

    n_long = sum(1 for ds_item in train_ds if ds_item[4].item() == 1.0)
    n_short = n_train - n_long
    pos_weight = torch.FloatTensor([n_short / max(n_long, 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(new_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(100):
        new_model.train()
        for x5, x1, x4, x1d, labels in train_loader:
            optimizer.zero_grad()
            outputs = new_model(x5, x1, x4, x1d).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        new_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x5, x1, x4, x1d, labels in test_loader:
                outputs = new_model(x5, x1, x4, x1d).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = correct / max(total, 1) * 100
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(new_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early Stop bei Epoch {epoch + 1}: {best_acc:.1f}%")
            break

    if best_state is None:
        print("  Training fehlgeschlagen!")
        return

    # 6. Vergleich
    print(f"\n6. Vergleich:")
    print(f"  Alt: {old_acc:.1f}%")
    print(f"  Neu: {best_acc:.1f}%")

    if best_acc > old_acc:
        # Backup + Swap
        backup = os.path.join(BACKUP_DIR, f"cnn_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        if os.path.exists(MODEL_PATH):
            import shutil
            shutil.copy2(MODEL_PATH, backup)
            print(f"  Backup: {backup}")

        new_model.load_state_dict(best_state)
        torch.save(best_state, MODEL_PATH)
        print(f"  HOT-SWAP: {old_acc:.1f}% -> {best_acc:.1f}% (+{best_acc - old_acc:.1f}pp)")
    else:
        print(f"  KEIN SWAP: Neu {best_acc:.1f}% <= Alt {old_acc:.1f}%")

    print(f"\nFertig: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    run()
