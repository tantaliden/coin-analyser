#!/usr/bin/env python3
"""
DIFFERENZ-SCANNER v1 — Volkers Konzept

Schritt 1: Strengere Event-Erkennung (Gegenextrem-Prüfung)
Schritt 2: 12h Rohdaten (agg_5m, 144 Candles) pro Event
Schritt 3: Anomalie-Erkennung pro Event (Zeitreihen-Analyse)
Schritt 4: Vergleich Long vs Short — Muster-Häufigkeiten
Schritt 5: Gegenprobe + Indikator-Sets
Schritt 6: Report

Usage: nohup python3 differenz_scanner.py [days] [min_pct] > /opt/coin/logs/differenz_stdout.log 2>&1 &
"""

import sys, os, logging, time, pickle, math
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# === PATHS ===
LOG_FILE = '/opt/coin/logs/differenz_scanner.log'
REPORT_DIR = '/opt/coin/database/data'
CHECKPOINT_DIR = '/opt/coin/database/data/checkpoints'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('differenz')

# === CONFIG ===
PCT_COLS = ['pct_60m','pct_90m','pct_120m','pct_180m','pct_240m',
            'pct_300m','pct_360m','pct_420m','pct_480m','pct_540m','pct_600m']
DUR_MAP = {'pct_60m':60,'pct_90m':90,'pct_120m':120,'pct_180m':180,'pct_240m':240,
           'pct_300m':300,'pct_360m':360,'pct_420m':420,'pct_480m':480,'pct_540m':540,'pct_600m':600}


def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)


def save_checkpoint(name, data):
    path = os.path.join(CHECKPOINT_DIR, f'diff_{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"  Checkpoint saved: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")


def load_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, f'diff_{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"  Checkpoint loaded: {path}")
        return data
    return None


# ========================================================================
# SCHRITT 1: STRENGERE EVENT-ERKENNUNG
# ========================================================================

def find_events_strict(conn, days, min_pct):
    """
    Findet Events (≥min_pct% Move) mit strenger Gegenextrem-Prüfung:
    - Long-Event: +5% erreicht, ABER -5% wurde NICHT vorher erreicht
    - Short-Event: -5% erreicht, ABER +5% wurde NICHT vorher erreicht

    Prüfung via agg_5m: Innerhalb des Move-Fensters prüfen ob das Gegenextrem
    vor dem Zielextrem erreicht wurde.
    """
    cur = conn.cursor()
    logger.info("=" * 70)
    logger.info("SCHRITT 1: STRENGERE EVENT-ERKENNUNG")
    logger.info(f"  kline_metrics ≥{min_pct}%, Gegenextrem-Prüfung via agg_5m")
    logger.info("=" * 70)

    # Erst alle Kandidaten aus kline_metrics holen (wie bisher)
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
    logger.info(f"  Query: {len(rows)} Kandidaten in {time.time()-t0:.1f}s")

    # Alle Kandidaten sammeln
    candidates = []
    for row in rows:
        best_col = None
        best_pct = 0
        for col in PCT_COLS:
            val = row[col]
            if val is not None and abs(val) >= min_pct:
                best_col = col
                best_pct = val
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

    logger.info(f"  Nach Dedup: {len(deduped)} Events")

    # Jetzt Gegenextrem-Prüfung via agg_5m
    # Für jedes Event: Lade die 5min-Candles im Move-Fenster (entry_time bis event_time)
    # Prüfe ob das Gegenextrem VOR dem Zielextrem erreicht wurde
    logger.info("  Gegenextrem-Prüfung läuft...")
    valid_events = []
    rejected = 0
    batch_size = 500

    for i, event in enumerate(deduped):
        if i > 0 and i % 5000 == 0:
            logger.info(f"    Gegenprüfung: {i}/{len(deduped)}, valid={len(valid_events)}, rejected={rejected}")

        try:
            cur.execute("""
                SELECT bucket, open, high, low, close
                FROM agg_5m
                WHERE symbol = %s
                  AND bucket >= %s
                  AND bucket < %s
                ORDER BY bucket
            """, (event['symbol'], event['time'], event['event_time']))
            candles = cur.fetchall()
        except Exception as ex:
            logger.warning(f"    DB error für {event['symbol']}: {ex}")
            conn.rollback()
            continue

        if len(candles) < 2:
            # Nicht genug Daten für Prüfung — Event behalten
            valid_events.append(event)
            continue

        # Referenzpreis = Open der ersten Candle (Start des Move-Fensters)
        ref_price = float(candles[0]['open'])
        if ref_price <= 0:
            valid_events.append(event)
            continue

        threshold = min_pct / 100.0  # z.B. 0.05 für 5%

        if event['direction'] == 'long':
            # Long-Event: +5% muss erreicht werden, -5% darf NICHT vorher erreicht werden
            for c in candles:
                low_pct = (float(c['low']) - ref_price) / ref_price
                high_pct = (float(c['high']) - ref_price) / ref_price
                # Prüfe ob Gegenextrem (-5%) zuerst erreicht
                if low_pct <= -threshold:
                    rejected += 1
                    break
                # Prüfe ob Ziel (+5%) erreicht — dann ist es valid
                if high_pct >= threshold:
                    valid_events.append(event)
                    break
            else:
                # Weder Ziel noch Gegenextrem klar erreicht in 5min-Daten
                # (kline_metrics hat es aber erkannt → Event behalten)
                valid_events.append(event)
        else:
            # Short-Event: -5% muss erreicht werden, +5% darf NICHT vorher erreicht werden
            for c in candles:
                high_pct = (float(c['high']) - ref_price) / ref_price
                low_pct = (float(c['low']) - ref_price) / ref_price
                # Prüfe ob Gegenextrem (+5%) zuerst erreicht
                if high_pct >= threshold:
                    rejected += 1
                    break
                # Prüfe ob Ziel (-5%) erreicht — dann ist es valid
                if low_pct <= -threshold:
                    valid_events.append(event)
                    break
            else:
                valid_events.append(event)

    longs = sum(1 for e in valid_events if e['direction'] == 'long')
    shorts = sum(1 for e in valid_events if e['direction'] == 'short')
    logger.info(f"  Gegenextrem-Prüfung abgeschlossen:")
    logger.info(f"    Vorher: {len(deduped)}")
    logger.info(f"    Rejected (Gegenextrem zuerst): {rejected}")
    logger.info(f"    Valid: {len(valid_events)} (LONG={longs}, SHORT={shorts})")

    return valid_events


# ========================================================================
# SCHRITT 2: 12h ROHDATEN PRO EVENT (agg_5m)
# ========================================================================

def load_12h_timeseries(conn, events):
    """
    Für jedes Event: Lade die letzten 12 Stunden (144 × 5min Candles) VOR dem Event.
    Speichert ALLE Datenströme: OHLCV, number_of_trades, taker_buy_base_asset_volume.
    """
    logger.info("=" * 70)
    logger.info("SCHRITT 2: 12h ROHDATEN LADEN (agg_5m)")
    logger.info(f"  {len(events)} Events × 144 Candles")
    logger.info("=" * 70)

    cur = conn.cursor()
    events_with_data = []
    skipped = 0
    t0 = time.time()

    for i, event in enumerate(events):
        if i > 0 and i % 2000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(events) - i) / rate / 60
            logger.info(f"  Lade Daten: {i}/{len(events)} ({elapsed/60:.1f}min, ETA {eta:.1f}min)")

        window_start = event['time'] - timedelta(hours=12)
        window_end = event['time']  # entry_time = Start des Moves

        try:
            cur.execute("""
                SELECT bucket, open, high, low, close, volume,
                       number_of_trades, taker_buy_base_asset_volume
                FROM agg_5m
                WHERE symbol = %s
                  AND bucket >= %s
                  AND bucket < %s
                ORDER BY bucket
            """, (event['symbol'], window_start, window_end))
            candles = cur.fetchall()
        except Exception as ex:
            logger.warning(f"  DB error {event['symbol']}: {ex}")
            conn.rollback()
            skipped += 1
            continue

        if len(candles) < 72:  # Mindestens 6h Daten (50% von 144)
            skipped += 1
            continue

        # Candles in numpy arrays konvertieren für schnelle Verarbeitung
        n = len(candles)
        ts = np.array([(c['bucket'] - window_start).total_seconds() / 60 for c in candles])  # Minuten seit window_start
        opens = np.array([float(c['open']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        closes = np.array([float(c['close']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])
        trades = np.array([float(c['number_of_trades'] or 0) for c in candles])
        taker_buy = np.array([float(c['taker_buy_base_asset_volume'] or 0) for c in candles])

        events_with_data.append({
            'event': event,
            'n_candles': n,
            'timestamps_min': ts,        # Minuten relativ zu window_start (0-720)
            'min_before_event': 720 - ts, # Minuten VOR dem Event
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'trades': trades,
            'taker_buy': taker_buy,
        })

    elapsed = time.time() - t0
    logger.info(f"  Fertig: {len(events_with_data)} Events mit Daten, {skipped} übersprungen")
    logger.info(f"  Dauer: {elapsed/60:.1f} Minuten")

    return events_with_data


# ========================================================================
# SCHRITT 3: ANOMALIE-ERKENNUNG PRO EVENT
# ========================================================================

def detect_anomalies(event_data):
    """
    Scannt die 12h-Zeitreihe EINES Events auf Anomalien.

    Gibt eine Liste von Anomalien zurück, jede mit:
    - typ: str (volume_spike, volume_drop, price_spike, trade_spike, trade_drop,
                taker_shift_buy, taker_shift_sell, volatility_spike, volatility_drop,
                momentum_up, momentum_down)
    - magnitude: float (Stärke der Anomalie, z.B. 2.5 = 250% über Normal)
    - minutes_before: float (Minuten vor dem Event)
    - duration_min: float (Dauer in Minuten)
    - direction: str ('up', 'down', 'neutral')
    """
    anomalies = []

    closes = event_data['close']
    volumes = event_data['volume']
    trades = event_data['trades']
    taker_buy = event_data['taker_buy']
    highs = event_data['high']
    lows = event_data['low']
    min_before = event_data['min_before_event']
    n = len(closes)

    if n < 12:  # Mindestens 1h Daten
        return anomalies

    # === ROLLING WINDOWS für "Normal" ===
    # Verschiedene Fenster: 6 Candles (30min), 12 (1h), 24 (2h), 36 (3h)
    windows = [6, 12, 24, 36]

    # ---------------------------------------------------------------
    # 1. VOLUMEN-ANOMALIEN (Spikes und Drops)
    # ---------------------------------------------------------------
    for w in windows:
        if n < w + 3:
            continue
        for i in range(w, n):
            window_vol = volumes[i-w:i]
            mean_vol = np.mean(window_vol)
            if mean_vol <= 0:
                continue
            current_vol = volumes[i]
            ratio = current_vol / mean_vol

            if ratio >= 2.0:  # ≥100% über Durchschnitt
                # Wie lange hält der Spike an?
                dur = 1
                for j in range(i+1, min(i+12, n)):
                    if volumes[j] / mean_vol >= 1.5:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'volume_spike',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })
            elif ratio <= 0.3 and mean_vol > 0:  # ≤70% unter Durchschnitt
                dur = 1
                for j in range(i+1, min(i+12, n)):
                    if volumes[j] / mean_vol <= 0.5:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'volume_drop',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })

    # ---------------------------------------------------------------
    # 2. PREIS-ANOMALIEN (schnelle Bewegungen)
    # ---------------------------------------------------------------
    # Preis-Change über verschiedene Lookbacks (1, 3, 6, 12 Candles = 5-60min)
    for lookback in [1, 3, 6, 12]:
        if n < lookback + 1:
            continue
        for i in range(lookback, n):
            if closes[i-lookback] <= 0:
                continue
            pct_change = (closes[i] - closes[i-lookback]) / closes[i-lookback] * 100

            # Normales Rauschen bestimmen (Std der Changes im gesamten Fenster)
            all_changes = []
            for k in range(lookback, n):
                if closes[k-lookback] > 0:
                    all_changes.append((closes[k] - closes[k-lookback]) / closes[k-lookback] * 100)
            if len(all_changes) < 10:
                continue
            std_change = np.std(all_changes)
            if std_change <= 0:
                continue

            z_score = abs(pct_change) / std_change
            if z_score >= 2.5:  # ≥2.5 Standardabweichungen
                anomalies.append({
                    'typ': 'price_spike',
                    'magnitude': round(z_score, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': lookback * 5,
                    'direction': 'up' if pct_change > 0 else 'down',
                    'window': lookback * 5,
                })

    # ---------------------------------------------------------------
    # 3. TRADE-ANOMALIEN (Anzahl Trades: Spikes und Drops)
    # ---------------------------------------------------------------
    for w in [6, 12, 24]:
        if n < w + 3:
            continue
        for i in range(w, n):
            window_trades = trades[i-w:i]
            mean_trades = np.mean(window_trades)
            if mean_trades <= 0:
                continue
            current_trades = trades[i]
            ratio = current_trades / mean_trades

            if ratio >= 2.5:
                dur = 1
                for j in range(i+1, min(i+12, n)):
                    if trades[j] / mean_trades >= 1.5:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'trade_spike',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })
            elif ratio <= 0.2:
                dur = 1
                for j in range(i+1, min(i+12, n)):
                    if trades[j] / mean_trades <= 0.4:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'trade_drop',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })

    # ---------------------------------------------------------------
    # 4. TAKER-RATIO-SHIFT (Käufer vs Verkäufer)
    # ---------------------------------------------------------------
    # taker_buy / volume = Anteil Käufer. Normal ~50%.
    for w in [6, 12, 24]:
        if n < w + 3:
            continue
        for i in range(w, n):
            if volumes[i] <= 0:
                continue
            current_ratio = taker_buy[i] / volumes[i] * 100  # in %

            # Vergleich mit Rolling-Fenster
            window_ratios = []
            for k in range(i-w, i):
                if volumes[k] > 0:
                    window_ratios.append(taker_buy[k] / volumes[k] * 100)
            if len(window_ratios) < 3:
                continue
            mean_ratio = np.mean(window_ratios)
            std_ratio = np.std(window_ratios)
            if std_ratio <= 0.5:
                std_ratio = 0.5

            shift = current_ratio - mean_ratio
            z = abs(shift) / std_ratio

            if z >= 2.0 and abs(shift) >= 5:  # ≥2 Std + mindestens 5pp Shift
                dur = 1
                for j in range(i+1, min(i+12, n)):
                    if volumes[j] > 0:
                        next_ratio = taker_buy[j] / volumes[j] * 100
                        if abs(next_ratio - mean_ratio) / std_ratio >= 1.5:
                            dur += 1
                        else:
                            break
                anomalies.append({
                    'typ': 'taker_shift_buy' if shift > 0 else 'taker_shift_sell',
                    'magnitude': round(abs(shift), 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'up' if shift > 0 else 'down',
                    'window': w * 5,
                })

    # ---------------------------------------------------------------
    # 5. VOLATILITÄTS-WECHSEL
    # ---------------------------------------------------------------
    # Candle-Range (high-low) relativ zum Preis
    candle_ranges = (highs - lows) / np.where(closes > 0, closes, 1) * 100  # in %

    for w in [12, 24, 36]:
        if n < w + 6:
            continue
        for i in range(w, n - 2):
            past_vol = np.mean(candle_ranges[i-w:i])
            if past_vol <= 0:
                continue
            # Aktuelle Volatilität (nächste 3 Candles = 15min)
            current_vol = np.mean(candle_ranges[i:i+3])
            ratio = current_vol / past_vol

            if ratio >= 2.5:  # Volatilität ≥150% höher
                dur = 3
                for j in range(i+3, min(i+12, n)):
                    if candle_ranges[j] / past_vol >= 1.5:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'volatility_spike',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })
            elif ratio <= 0.3:  # Volatilität ≥70% niedriger (Stille)
                dur = 3
                for j in range(i+3, min(i+12, n)):
                    if candle_ranges[j] / past_vol <= 0.5:
                        dur += 1
                    else:
                        break
                anomalies.append({
                    'typ': 'volatility_drop',
                    'magnitude': round(ratio, 2),
                    'minutes_before': float(min_before[i]),
                    'duration_min': dur * 5,
                    'direction': 'neutral',
                    'window': w * 5,
                })

    # ---------------------------------------------------------------
    # 6. MOMENTUM-WECHSEL
    # ---------------------------------------------------------------
    # Trend der letzten N Candles vs. aktuelle Richtung
    for w in [6, 12, 24]:
        if n < w + 6:
            continue
        for i in range(w + 3, n):
            # Trend vorher (lineare Regression)
            past_closes = closes[i-w:i]
            if past_closes[0] <= 0:
                continue
            past_trend = (past_closes[-1] - past_closes[0]) / past_closes[0] * 100

            # Aktuelle Richtung (3 Candles = 15min)
            recent_start = max(0, i-3)
            if closes[recent_start] <= 0:
                continue
            recent_trend = (closes[i] - closes[recent_start]) / closes[recent_start] * 100

            # Momentum-Wechsel = Richtungsumkehr mit Stärke
            if past_trend > 0.3 and recent_trend < -0.3:
                # Aufwärtstrend → plötzlich runter
                strength = abs(past_trend) + abs(recent_trend)
                if strength >= 1.0:
                    anomalies.append({
                        'typ': 'momentum_reversal',
                        'magnitude': round(strength, 2),
                        'minutes_before': float(min_before[i]),
                        'duration_min': (w + 3) * 5,
                        'direction': 'down',
                        'window': w * 5,
                    })
            elif past_trend < -0.3 and recent_trend > 0.3:
                # Abwärtstrend → plötzlich hoch
                strength = abs(past_trend) + abs(recent_trend)
                if strength >= 1.0:
                    anomalies.append({
                        'typ': 'momentum_reversal',
                        'magnitude': round(strength, 2),
                        'minutes_before': float(min_before[i]),
                        'duration_min': (w + 3) * 5,
                        'direction': 'up',
                        'window': w * 5,
                    })

    # ---------------------------------------------------------------
    # 7. WICK-ANOMALIEN (ungewöhnlich große Dochte)
    # ---------------------------------------------------------------
    body_sizes = np.abs(closes - event_data['open'])
    upper_wicks = highs - np.maximum(closes, event_data['open'])
    lower_wicks = np.minimum(closes, event_data['open']) - lows

    for i in range(12, n):
        body = body_sizes[i]
        if body <= 0:
            body = 0.0001
        candle_range = highs[i] - lows[i]
        if candle_range <= 0:
            continue

        upper_ratio = upper_wicks[i] / candle_range
        lower_ratio = lower_wicks[i] / candle_range

        # Vergleich mit lokaler Norm
        local_upper = np.mean(upper_wicks[max(0,i-12):i])
        local_lower = np.mean(lower_wicks[max(0,i-12):i])

        if local_upper > 0 and upper_wicks[i] / local_upper >= 3.0 and upper_ratio >= 0.6:
            anomalies.append({
                'typ': 'wick_anomaly',
                'magnitude': round(upper_wicks[i] / local_upper, 2),
                'minutes_before': float(min_before[i]),
                'duration_min': 5,
                'direction': 'up',
                'window': 60,
            })
        if local_lower > 0 and lower_wicks[i] / local_lower >= 3.0 and lower_ratio >= 0.6:
            anomalies.append({
                'typ': 'wick_anomaly',
                'magnitude': round(lower_wicks[i] / local_lower, 2),
                'minutes_before': float(min_before[i]),
                'duration_min': 5,
                'direction': 'down',
                'window': 60,
            })

    return anomalies


def process_all_anomalies(events_with_data):
    """
    Schritt 3: Anomalie-Erkennung für JEDES Event.
    Gibt events_with_anomalies zurück: Liste von {event, anomalies}.
    """
    logger.info("=" * 70)
    logger.info("SCHRITT 3: ANOMALIE-ERKENNUNG")
    logger.info(f"  {len(events_with_data)} Events werden einzeln gescannt")
    logger.info("=" * 70)

    results = []
    total_anomalies = 0
    t0 = time.time()
    anomaly_counts = defaultdict(int)

    for i, ed in enumerate(events_with_data):
        if i > 0 and i % 5000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(events_with_data) - i) / rate / 60
            avg = total_anomalies / i
            logger.info(f"  Anomalien: {i}/{len(events_with_data)} ({elapsed/60:.1f}min, "
                        f"ETA {eta:.1f}min, Ø{avg:.0f} Anomalien/Event)")

        anomalies = detect_anomalies(ed)
        total_anomalies += len(anomalies)

        for a in anomalies:
            anomaly_counts[a['typ']] += 1

        results.append({
            'event': ed['event'],
            'anomalies': anomalies,
        })

    elapsed = time.time() - t0
    avg = total_anomalies / max(len(results), 1)
    logger.info(f"  Fertig: {total_anomalies} Anomalien gefunden in {elapsed/60:.1f}min")
    logger.info(f"  Durchschnitt: {avg:.1f} Anomalien/Event")
    logger.info(f"  Verteilung:")
    for typ, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {typ}: {count} ({count/max(len(results),1):.1f}/Event)")

    return results


# ========================================================================
# SCHRITT 4: VERGLEICH LONG vs SHORT
# ========================================================================

def compare_anomaly_patterns(events_with_anomalies):
    """
    Schritt 4: Für jeden Anomalie-Typ + Zeitfenster + Stärkebereich:
    Wie oft taucht das bei Longs auf? Bei Shorts?

    Ergebnis: Muster-Liste mit Häufigkeiten pro Richtung.
    """
    logger.info("=" * 70)
    logger.info("SCHRITT 4: VERGLEICH LONG vs SHORT")
    logger.info("=" * 70)

    longs = [e for e in events_with_anomalies if e['event']['direction'] == 'long']
    shorts = [e for e in events_with_anomalies if e['event']['direction'] == 'short']
    n_long = len(longs)
    n_short = len(shorts)
    logger.info(f"  Long: {n_long}, Short: {n_short}")

    # Definiere Anomalie-Buckets: Typ × Zeitbereich × Stärkebereich
    # Zeitbereiche (Minuten vor Event)
    time_ranges = [
        (0, 30, '0-30min'),
        (30, 60, '30-60min'),
        (60, 120, '1-2h'),
        (120, 240, '2-4h'),
        (240, 360, '4-6h'),
        (360, 480, '6-8h'),
        (480, 720, '8-12h'),
    ]

    # Stärke-Bereiche (je nach Anomalie-Typ verschieden)
    magnitude_ranges = [
        (1.5, 2.5, 'mild'),
        (2.5, 4.0, 'moderate'),
        (4.0, 8.0, 'strong'),
        (8.0, float('inf'), 'extreme'),
    ]

    # Anomalie-Typen
    anomaly_types = [
        'volume_spike', 'volume_drop',
        'price_spike',
        'trade_spike', 'trade_drop',
        'taker_shift_buy', 'taker_shift_sell',
        'volatility_spike', 'volatility_drop',
        'momentum_reversal',
        'wick_anomaly',
    ]

    t0 = time.time()

    # Für jede Kombination: Zähle Events die MINDESTENS EINE solche Anomalie haben
    patterns = []

    for atyp in anomaly_types:
        for t_lo, t_hi, t_label in time_ranges:
            for m_lo, m_hi, m_label in magnitude_ranges:
                # Optional: Richtung der Anomalie
                for direction in ['any', 'up', 'down']:
                    # Zähle Long-Events die dieses Muster haben
                    long_count = 0
                    for e in longs:
                        has_match = False
                        for a in e['anomalies']:
                            if a['typ'] != atyp:
                                continue
                            if a['minutes_before'] < t_lo or a['minutes_before'] >= t_hi:
                                continue
                            if a['magnitude'] < m_lo or a['magnitude'] >= m_hi:
                                continue
                            if direction != 'any' and a.get('direction', 'neutral') != direction:
                                continue
                            has_match = True
                            break
                        if has_match:
                            long_count += 1

                    # Zähle Short-Events
                    short_count = 0
                    for e in shorts:
                        has_match = False
                        for a in e['anomalies']:
                            if a['typ'] != atyp:
                                continue
                            if a['minutes_before'] < t_lo or a['minutes_before'] >= t_hi:
                                continue
                            if a['magnitude'] < m_lo or a['magnitude'] >= m_hi:
                                continue
                            if direction != 'any' and a.get('direction', 'neutral') != direction:
                                continue
                            has_match = True
                            break
                        if has_match:
                            short_count += 1

                    long_pct = long_count / max(n_long, 1) * 100
                    short_pct = short_count / max(n_short, 1) * 100

                    # Nur Muster speichern wo mindestens eine Seite ≥5% Coverage hat
                    if long_pct < 5 and short_pct < 5:
                        continue

                    patterns.append({
                        'typ': atyp,
                        'time_range': t_label,
                        'time_lo': t_lo,
                        'time_hi': t_hi,
                        'mag_range': m_label,
                        'mag_lo': m_lo,
                        'mag_hi': m_hi,
                        'direction': direction,
                        'long_count': long_count,
                        'short_count': short_count,
                        'long_pct': round(long_pct, 2),
                        'short_pct': round(short_pct, 2),
                        'diff': round(long_pct - short_pct, 2),
                        'abs_diff': round(abs(long_pct - short_pct), 2),
                    })

    elapsed = time.time() - t0
    logger.info(f"  {len(patterns)} Muster gefunden in {elapsed/60:.1f}min")

    # Sortiere nach abs_diff (größter Unterschied zwischen Long und Short)
    patterns.sort(key=lambda p: -p['abs_diff'])

    # Top-Muster loggen
    logger.info(f"  Top 30 Muster (größter Long/Short-Unterschied):")
    for p in patterns[:30]:
        favors = 'LONG' if p['diff'] > 0 else 'SHORT'
        logger.info(f"    {p['typ']} | {p['time_range']} | {p['mag_range']} | dir={p['direction']} | "
                     f"Long={p['long_pct']:.1f}% Short={p['short_pct']:.1f}% | "
                     f"Diff={p['diff']:+.1f}pp → {favors}")

    return patterns


# ========================================================================
# SCHRITT 5: GEGENPROBE + INDIKATOR-SETS
# ========================================================================

def build_indicator_sets(patterns, events_with_anomalies, min_coverage=20, min_purity=80):
    """
    Schritt 5: Validierte Muster zu Sets kombinieren.

    Ein Muster ist "für Long" wenn:
    - long_pct deutlich > short_pct (long_pct ≥ 1.5× short_pct)
    - Contamination (short_pct) ≤ 10%... oder nach Verfeinerung niedrig genug

    Sets: Kombinationen von 2-3 Mustern die zusammen
    - Coverage ≥ min_coverage% der Ziel-Events
    - Purity ≥ min_purity%
    """
    logger.info("=" * 70)
    logger.info("SCHRITT 5: GEGENPROBE + INDIKATOR-SETS")
    logger.info(f"  min_coverage={min_coverage}%, min_purity={min_purity}%")
    logger.info("=" * 70)

    longs = [e for e in events_with_anomalies if e['event']['direction'] == 'long']
    shorts = [e for e in events_with_anomalies if e['event']['direction'] == 'short']
    n_long = len(longs)
    n_short = len(shorts)

    # --- Phase 5a: Einzelne Muster als Long- oder Short-Indikatoren ---
    long_indicators = []
    short_indicators = []

    for p in patterns:
        # Für Long: long_pct > short_pct, geringe Contamination bei Shorts
        if p['long_pct'] >= 15 and p['short_pct'] <= 10:
            long_indicators.append(p)
        # Für Short: short_pct > long_pct, geringe Contamination bei Longs
        if p['short_pct'] >= 15 and p['long_pct'] <= 10:
            short_indicators.append(p)

    # Auch Muster mit höherer Contamination aber starkem Unterschied
    for p in patterns:
        if p['diff'] >= 10 and p['long_pct'] >= 20:
            if p not in long_indicators:
                long_indicators.append(p)
        if p['diff'] <= -10 and p['short_pct'] >= 20:
            if p not in short_indicators:
                short_indicators.append(p)

    logger.info(f"  Long-Indikatoren (Einzel): {len(long_indicators)}")
    logger.info(f"  Short-Indikatoren (Einzel): {len(short_indicators)}")

    # --- Phase 5b: Einzel-Indikatoren per-Event validieren ---
    def event_matches_pattern(event_entry, pattern):
        """Prüft ob ein Event mindestens eine Anomalie hat die zum Muster passt."""
        for a in event_entry['anomalies']:
            if a['typ'] != pattern['typ']:
                continue
            if a['minutes_before'] < pattern['time_lo'] or a['minutes_before'] >= pattern['time_hi']:
                continue
            if a['magnitude'] < pattern['mag_lo'] or a['magnitude'] >= pattern['mag_hi']:
                continue
            if pattern['direction'] != 'any' and a.get('direction', 'neutral') != pattern['direction']:
                continue
            return True
        return False

    validated_long = []
    for p in long_indicators:
        # Per-Event Gegenprobe
        matching_longs = sum(1 for e in longs if event_matches_pattern(e, p))
        matching_shorts = sum(1 for e in shorts if event_matches_pattern(e, p))
        total_matching = matching_longs + matching_shorts
        if total_matching == 0:
            continue
        purity = matching_longs / total_matching * 100
        coverage = matching_longs / n_long * 100
        contamination = matching_shorts / max(n_short, 1) * 100

        validated_long.append({
            'pattern': p,
            'matching_longs': matching_longs,
            'matching_shorts': matching_shorts,
            'purity': round(purity, 2),
            'coverage': round(coverage, 2),
            'contamination': round(contamination, 2),
        })

    validated_short = []
    for p in short_indicators:
        matching_longs = sum(1 for e in longs if event_matches_pattern(e, p))
        matching_shorts = sum(1 for e in shorts if event_matches_pattern(e, p))
        total_matching = matching_longs + matching_shorts
        if total_matching == 0:
            continue
        purity = matching_shorts / total_matching * 100
        coverage = matching_shorts / n_short * 100
        contamination = matching_longs / max(n_long, 1) * 100

        validated_short.append({
            'pattern': p,
            'matching_longs': matching_longs,
            'matching_shorts': matching_shorts,
            'purity': round(purity, 2),
            'coverage': round(coverage, 2),
            'contamination': round(contamination, 2),
        })

    # Sortiere nach Purity (absteigend)
    validated_long.sort(key=lambda x: -x['purity'])
    validated_short.sort(key=lambda x: -x['purity'])

    logger.info(f"  Validierte Long-Indikatoren: {len(validated_long)}")
    for v in validated_long[:20]:
        p = v['pattern']
        logger.info(f"    {p['typ']} {p['time_range']} {p['mag_range']} dir={p['direction']} | "
                     f"Purity={v['purity']:.1f}% Coverage={v['coverage']:.1f}% "
                     f"Contamination={v['contamination']:.1f}%")

    logger.info(f"  Validierte Short-Indikatoren: {len(validated_short)}")
    for v in validated_short[:20]:
        p = v['pattern']
        logger.info(f"    {p['typ']} {p['time_range']} {p['mag_range']} dir={p['direction']} | "
                     f"Purity={v['purity']:.1f}% Coverage={v['coverage']:.1f}% "
                     f"Contamination={v['contamination']:.1f}%")

    # --- Phase 5c: Paare bilden ---
    logger.info("  Phase 5c: Paare bilden...")

    def build_pairs(indicators, target_events, counter_events, target_label):
        """Bildet Paare aus Einzel-Indikatoren, prüft per-Event."""
        n_target = len(target_events)
        n_counter = len(counter_events)
        top_singles = indicators[:50]  # Top 50 nach Purity

        pairs = []
        for i in range(len(top_singles)):
            for j in range(i+1, len(top_singles)):
                p1 = top_singles[i]['pattern']
                p2 = top_singles[j]['pattern']

                # Events die BEIDE Muster haben
                matching_target = 0
                matching_counter = 0

                for e in target_events:
                    if event_matches_pattern(e, p1) and event_matches_pattern(e, p2):
                        matching_target += 1
                for e in counter_events:
                    if event_matches_pattern(e, p1) and event_matches_pattern(e, p2):
                        matching_counter += 1

                total = matching_target + matching_counter
                if total == 0:
                    continue
                purity = matching_target / total * 100
                coverage = matching_target / n_target * 100

                if coverage >= 5 and purity >= min_purity:
                    pairs.append({
                        'p1': p1,
                        'p2': p2,
                        'matching_target': matching_target,
                        'matching_counter': matching_counter,
                        'purity': round(purity, 2),
                        'coverage': round(coverage, 2),
                        'label': target_label,
                    })

        pairs.sort(key=lambda x: (-x['purity'], -x['coverage']))
        return pairs

    long_pairs = build_pairs(validated_long, longs, shorts, 'LONG')
    short_pairs = build_pairs(validated_short, shorts, longs, 'SHORT')

    logger.info(f"  Long-Paare (Purity ≥ {min_purity}%): {len(long_pairs)}")
    for p in long_pairs[:15]:
        logger.info(f"    [{p['p1']['typ']} {p['p1']['time_range']} {p['p1']['mag_range']}] + "
                     f"[{p['p2']['typ']} {p['p2']['time_range']} {p['p2']['mag_range']}] | "
                     f"Purity={p['purity']:.1f}% Coverage={p['coverage']:.1f}% "
                     f"({p['matching_target']} target, {p['matching_counter']} counter)")

    logger.info(f"  Short-Paare (Purity ≥ {min_purity}%): {len(short_pairs)}")
    for p in short_pairs[:15]:
        logger.info(f"    [{p['p1']['typ']} {p['p1']['time_range']} {p['p1']['mag_range']}] + "
                     f"[{p['p2']['typ']} {p['p2']['time_range']} {p['p2']['mag_range']}] | "
                     f"Purity={p['purity']:.1f}% Coverage={p['coverage']:.1f}% "
                     f"({p['matching_target']} target, {p['matching_counter']} counter)")

    # --- Phase 5d: Triples bilden (Top-Paare erweitern) ---
    logger.info("  Phase 5d: Triples bilden...")

    def build_triples(pair_list, single_list, target_events, counter_events, target_label):
        top_pairs = pair_list[:20]
        top_singles = single_list[:50]
        n_target = len(target_events)

        triples = []
        for pair in top_pairs:
            for single in top_singles:
                p3 = single['pattern']
                # Nicht dasselbe Muster wie in dem Paar
                if (p3['typ'] == pair['p1']['typ'] and p3['time_range'] == pair['p1']['time_range'] and
                    p3['mag_range'] == pair['p1']['mag_range']):
                    continue
                if (p3['typ'] == pair['p2']['typ'] and p3['time_range'] == pair['p2']['time_range'] and
                    p3['mag_range'] == pair['p2']['mag_range']):
                    continue

                matching_target = 0
                matching_counter = 0
                for e in target_events:
                    if (event_matches_pattern(e, pair['p1']) and
                        event_matches_pattern(e, pair['p2']) and
                        event_matches_pattern(e, p3)):
                        matching_target += 1
                for e in counter_events:
                    if (event_matches_pattern(e, pair['p1']) and
                        event_matches_pattern(e, pair['p2']) and
                        event_matches_pattern(e, p3)):
                        matching_counter += 1

                total = matching_target + matching_counter
                if total == 0:
                    continue
                purity = matching_target / total * 100
                coverage = matching_target / n_target * 100

                if coverage >= 3 and purity >= min_purity + 5:  # Höhere Purity für Triples
                    triples.append({
                        'p1': pair['p1'],
                        'p2': pair['p2'],
                        'p3': p3,
                        'matching_target': matching_target,
                        'matching_counter': matching_counter,
                        'purity': round(purity, 2),
                        'coverage': round(coverage, 2),
                        'label': target_label,
                    })

        triples.sort(key=lambda x: (-x['purity'], -x['coverage']))
        return triples

    long_triples = build_triples(long_pairs, validated_long, longs, shorts, 'LONG')
    short_triples = build_triples(short_pairs, validated_short, shorts, longs, 'SHORT')

    logger.info(f"  Long-Triples (Purity ≥ {min_purity+5}%): {len(long_triples)}")
    for t in long_triples[:10]:
        logger.info(f"    [{t['p1']['typ']} {t['p1']['time_range']}] + [{t['p2']['typ']} {t['p2']['time_range']}] + "
                     f"[{t['p3']['typ']} {t['p3']['time_range']}] | "
                     f"Purity={t['purity']:.1f}% Coverage={t['coverage']:.1f}%")

    logger.info(f"  Short-Triples (Purity ≥ {min_purity+5}%): {len(short_triples)}")
    for t in short_triples[:10]:
        logger.info(f"    [{t['p1']['typ']} {t['p1']['time_range']}] + [{t['p2']['typ']} {t['p2']['time_range']}] + "
                     f"[{t['p3']['typ']} {t['p3']['time_range']}] | "
                     f"Purity={t['purity']:.1f}% Coverage={t['coverage']:.1f}%")

    return {
        'validated_long': validated_long,
        'validated_short': validated_short,
        'long_pairs': long_pairs,
        'short_pairs': short_pairs,
        'long_triples': long_triples,
        'short_triples': short_triples,
    }


# ========================================================================
# SCHRITT 6: TRAIN/TEST VALIDIERUNG
# ========================================================================

def validate_train_test(events_with_anomalies, indicator_sets):
    """
    Split Events in Train (70%) und Test (30%) nach Zeit.
    Prüfe ob Muster auf Test-Set auch halten.
    """
    logger.info("=" * 70)
    logger.info("SCHRITT 6: TRAIN/TEST VALIDIERUNG")
    logger.info("=" * 70)

    # Sortiere nach Zeit
    sorted_events = sorted(events_with_anomalies, key=lambda e: e['event']['time'])
    split_idx = int(len(sorted_events) * 0.7)
    train = sorted_events[:split_idx]
    test = sorted_events[split_idx:]

    train_longs = [e for e in train if e['event']['direction'] == 'long']
    train_shorts = [e for e in train if e['event']['direction'] == 'short']
    test_longs = [e for e in test if e['event']['direction'] == 'long']
    test_shorts = [e for e in test if e['event']['direction'] == 'short']

    logger.info(f"  Train: {len(train)} ({len(train_longs)} L, {len(train_shorts)} S)")
    logger.info(f"  Test:  {len(test)} ({len(test_longs)} L, {len(test_shorts)} S)")

    def event_matches_pattern(event_entry, pattern):
        for a in event_entry['anomalies']:
            if a['typ'] != pattern['typ']:
                continue
            if a['minutes_before'] < pattern['time_lo'] or a['minutes_before'] >= pattern['time_hi']:
                continue
            if a['magnitude'] < pattern['mag_lo'] or a['magnitude'] >= pattern['mag_hi']:
                continue
            if pattern['direction'] != 'any' and a.get('direction', 'neutral') != pattern['direction']:
                continue
            return True
        return False

    results = {}

    # Validiere Top Long-Singles
    logger.info("  --- Top Long-Indikatoren (Einzel) auf Test ---")
    for v in indicator_sets['validated_long'][:20]:
        p = v['pattern']
        test_long_match = sum(1 for e in test_longs if event_matches_pattern(e, p))
        test_short_match = sum(1 for e in test_shorts if event_matches_pattern(e, p))
        total = test_long_match + test_short_match
        test_purity = test_long_match / total * 100 if total > 0 else 0
        test_coverage = test_long_match / max(len(test_longs), 1) * 100
        logger.info(f"    {p['typ']} {p['time_range']} {p['mag_range']} | "
                     f"Train: Purity={v['purity']:.1f}% Cov={v['coverage']:.1f}% | "
                     f"Test: Purity={test_purity:.1f}% Cov={test_coverage:.1f}%")

    # Validiere Top Short-Singles
    logger.info("  --- Top Short-Indikatoren (Einzel) auf Test ---")
    for v in indicator_sets['validated_short'][:20]:
        p = v['pattern']
        test_long_match = sum(1 for e in test_longs if event_matches_pattern(e, p))
        test_short_match = sum(1 for e in test_shorts if event_matches_pattern(e, p))
        total = test_long_match + test_short_match
        test_purity = test_short_match / total * 100 if total > 0 else 0
        test_coverage = test_short_match / max(len(test_shorts), 1) * 100
        logger.info(f"    {p['typ']} {p['time_range']} {p['mag_range']} | "
                     f"Train: Purity={v['purity']:.1f}% Cov={v['coverage']:.1f}% | "
                     f"Test: Purity={test_purity:.1f}% Cov={test_coverage:.1f}%")

    # Validiere Long-Paare
    logger.info("  --- Top Long-Paare auf Test ---")
    for pair in indicator_sets['long_pairs'][:10]:
        test_long_match = sum(1 for e in test_longs
                             if event_matches_pattern(e, pair['p1']) and event_matches_pattern(e, pair['p2']))
        test_short_match = sum(1 for e in test_shorts
                              if event_matches_pattern(e, pair['p1']) and event_matches_pattern(e, pair['p2']))
        total = test_long_match + test_short_match
        test_purity = test_long_match / total * 100 if total > 0 else 0
        test_coverage = test_long_match / max(len(test_longs), 1) * 100
        logger.info(f"    [{pair['p1']['typ']} {pair['p1']['time_range']}] + "
                     f"[{pair['p2']['typ']} {pair['p2']['time_range']}] | "
                     f"Train: Purity={pair['purity']:.1f}% Cov={pair['coverage']:.1f}% | "
                     f"Test: Purity={test_purity:.1f}% Cov={test_coverage:.1f}%")

    # Validiere Short-Paare
    logger.info("  --- Top Short-Paare auf Test ---")
    for pair in indicator_sets['short_pairs'][:10]:
        test_long_match = sum(1 for e in test_longs
                             if event_matches_pattern(e, pair['p1']) and event_matches_pattern(e, pair['p2']))
        test_short_match = sum(1 for e in test_shorts
                              if event_matches_pattern(e, pair['p1']) and event_matches_pattern(e, pair['p2']))
        total = test_long_match + test_short_match
        test_purity = test_short_match / total * 100 if total > 0 else 0
        test_coverage = test_short_match / max(len(test_shorts), 1) * 100
        logger.info(f"    [{pair['p1']['typ']} {pair['p1']['time_range']}] + "
                     f"[{pair['p2']['typ']} {pair['p2']['time_range']}] | "
                     f"Train: Purity={pair['purity']:.1f}% Cov={pair['coverage']:.1f}% | "
                     f"Test: Purity={test_purity:.1f}% Cov={test_coverage:.1f}%")

    # Validiere Triples
    logger.info("  --- Top Triples auf Test ---")
    for t in (indicator_sets['long_triples'][:5] + indicator_sets['short_triples'][:5]):
        is_long = t['label'] == 'LONG'
        target_test = test_longs if is_long else test_shorts
        counter_test = test_shorts if is_long else test_longs

        match_target = sum(1 for e in target_test
                          if event_matches_pattern(e, t['p1']) and event_matches_pattern(e, t['p2']) and event_matches_pattern(e, t['p3']))
        match_counter = sum(1 for e in counter_test
                           if event_matches_pattern(e, t['p1']) and event_matches_pattern(e, t['p2']) and event_matches_pattern(e, t['p3']))
        total = match_target + match_counter
        test_purity = match_target / total * 100 if total > 0 else 0
        test_coverage = match_target / max(len(target_test), 1) * 100
        logger.info(f"    {t['label']}: [{t['p1']['typ']} {t['p1']['time_range']}] + "
                     f"[{t['p2']['typ']} {t['p2']['time_range']}] + [{t['p3']['typ']} {t['p3']['time_range']}] | "
                     f"Train: Purity={t['purity']:.1f}% Cov={t['coverage']:.1f}% | "
                     f"Test: Purity={test_purity:.1f}% Cov={test_coverage:.1f}%")


# ========================================================================
# REPORT
# ========================================================================

def write_report(events_with_anomalies, patterns, indicator_sets, config):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORT_DIR, f'differenz_report_{ts}.txt')

    longs = [e for e in events_with_anomalies if e['event']['direction'] == 'long']
    shorts = [e for e in events_with_anomalies if e['event']['direction'] == 'short']

    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIFFERENZ-SCANNER v1 — ERGEBNISSE\n")
        f.write(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {config['days']} Tage, ≥{config['min_pct']}% Moves\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Events total: {len(events_with_anomalies)}\n")
        f.write(f"  Long: {len(longs)}\n")
        f.write(f"  Short: {len(shorts)}\n\n")

        # Anomalie-Statistik
        all_anomalies = []
        for e in events_with_anomalies:
            all_anomalies.extend(e['anomalies'])
        f.write(f"Anomalien total: {len(all_anomalies)}\n")
        f.write(f"Durchschnitt pro Event: {len(all_anomalies)/max(len(events_with_anomalies),1):.1f}\n\n")

        anomaly_dist = defaultdict(int)
        for a in all_anomalies:
            anomaly_dist[a['typ']] += 1
        f.write("Anomalie-Verteilung:\n")
        for typ, count in sorted(anomaly_dist.items(), key=lambda x: -x[1]):
            f.write(f"  {typ}: {count}\n")
        f.write("\n")

        # Top Muster
        f.write("-" * 80 + "\n")
        f.write("TOP 50 MUSTER (größter Long/Short-Unterschied)\n")
        f.write("-" * 80 + "\n")
        for i, p in enumerate(patterns[:50]):
            favors = 'LONG' if p['diff'] > 0 else 'SHORT'
            f.write(f"  #{i+1}: {p['typ']} | {p['time_range']} | {p['mag_range']} | dir={p['direction']}\n")
            f.write(f"        Long={p['long_pct']:.1f}%  Short={p['short_pct']:.1f}%  "
                    f"Diff={p['diff']:+.1f}pp → {favors}\n")
        f.write("\n")

        # Validierte Einzel-Indikatoren
        f.write("-" * 80 + "\n")
        f.write("VALIDIERTE LONG-INDIKATOREN (Einzel)\n")
        f.write("-" * 80 + "\n")
        for v in indicator_sets['validated_long'][:30]:
            p = v['pattern']
            f.write(f"  {p['typ']} | {p['time_range']} | {p['mag_range']} | dir={p['direction']}\n")
            f.write(f"    Purity={v['purity']:.1f}%  Coverage={v['coverage']:.1f}%  "
                    f"Contamination={v['contamination']:.1f}%\n")
            f.write(f"    Long matches={v['matching_longs']}  Short matches={v['matching_shorts']}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("VALIDIERTE SHORT-INDIKATOREN (Einzel)\n")
        f.write("-" * 80 + "\n")
        for v in indicator_sets['validated_short'][:30]:
            p = v['pattern']
            f.write(f"  {p['typ']} | {p['time_range']} | {p['mag_range']} | dir={p['direction']}\n")
            f.write(f"    Purity={v['purity']:.1f}%  Coverage={v['coverage']:.1f}%  "
                    f"Contamination={v['contamination']:.1f}%\n")
            f.write(f"    Short matches={v['matching_shorts']}  Long matches={v['matching_longs']}\n")
        f.write("\n")

        # Paare
        f.write("-" * 80 + "\n")
        f.write("LONG-PAARE\n")
        f.write("-" * 80 + "\n")
        for p in indicator_sets['long_pairs'][:20]:
            f.write(f"  [{p['p1']['typ']} {p['p1']['time_range']} {p['p1']['mag_range']}]\n")
            f.write(f"  + [{p['p2']['typ']} {p['p2']['time_range']} {p['p2']['mag_range']}]\n")
            f.write(f"    Purity={p['purity']:.1f}%  Coverage={p['coverage']:.1f}%  "
                    f"({p['matching_target']} target, {p['matching_counter']} counter)\n\n")

        f.write("-" * 80 + "\n")
        f.write("SHORT-PAARE\n")
        f.write("-" * 80 + "\n")
        for p in indicator_sets['short_pairs'][:20]:
            f.write(f"  [{p['p1']['typ']} {p['p1']['time_range']} {p['p1']['mag_range']}]\n")
            f.write(f"  + [{p['p2']['typ']} {p['p2']['time_range']} {p['p2']['mag_range']}]\n")
            f.write(f"    Purity={p['purity']:.1f}%  Coverage={p['coverage']:.1f}%  "
                    f"({p['matching_target']} target, {p['matching_counter']} counter)\n\n")

        # Triples
        f.write("-" * 80 + "\n")
        f.write("TRIPLES\n")
        f.write("-" * 80 + "\n")
        for t in indicator_sets['long_triples'][:10]:
            f.write(f"  LONG: [{t['p1']['typ']} {t['p1']['time_range']}] + "
                    f"[{t['p2']['typ']} {t['p2']['time_range']}] + [{t['p3']['typ']} {t['p3']['time_range']}]\n")
            f.write(f"    Purity={t['purity']:.1f}%  Coverage={t['coverage']:.1f}%\n")
        for t in indicator_sets['short_triples'][:10]:
            f.write(f"  SHORT: [{t['p1']['typ']} {t['p1']['time_range']}] + "
                    f"[{t['p2']['typ']} {t['p2']['time_range']}] + [{t['p3']['typ']} {t['p3']['time_range']}]\n")
            f.write(f"    Purity={t['purity']:.1f}%  Coverage={t['coverage']:.1f}%\n")

    logger.info(f"Report: {path}")
    return path


# ========================================================================
# MAIN
# ========================================================================

def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    logger.info("=" * 80)
    logger.info("DIFFERENZ-SCANNER v1 — START")
    logger.info(f"  Zeitraum: {days} Tage, Min-Move: {min_pct}%")
    logger.info(f"  Konzept: Anomalie-Erkennung in 12h-Zeitreihen vor Events")
    logger.info("=" * 80)
    t_start = time.time()

    conn = coins_db()

    # --- SCHRITT 1: Event-Erkennung (streng) ---
    checkpoint = load_checkpoint('events_strict')
    if checkpoint is not None:
        events = checkpoint
        logger.info(f"  Events aus Checkpoint: {len(events)}")
    else:
        events = find_events_strict(conn, days, min_pct)
        save_checkpoint('events_strict', events)

    # --- SCHRITT 2: 12h Rohdaten ---
    checkpoint = load_checkpoint('timeseries_12h')
    if checkpoint is not None:
        events_with_data = checkpoint
        logger.info(f"  Timeseries aus Checkpoint: {len(events_with_data)}")
    else:
        events_with_data = load_12h_timeseries(conn, events)
        save_checkpoint('timeseries_12h', events_with_data)

    # --- SCHRITT 3: Anomalie-Erkennung ---
    checkpoint = load_checkpoint('anomalies')
    if checkpoint is not None:
        events_with_anomalies = checkpoint
        logger.info(f"  Anomalien aus Checkpoint: {len(events_with_anomalies)}")
    else:
        events_with_anomalies = process_all_anomalies(events_with_data)
        save_checkpoint('anomalies', events_with_anomalies)

    # --- SCHRITT 4: Vergleich Long vs Short ---
    patterns = compare_anomaly_patterns(events_with_anomalies)
    save_checkpoint('patterns', patterns)

    # --- SCHRITT 5: Gegenprobe + Sets ---
    indicator_sets = build_indicator_sets(patterns, events_with_anomalies)
    save_checkpoint('indicator_sets', indicator_sets)

    # --- SCHRITT 6: Train/Test ---
    validate_train_test(events_with_anomalies, indicator_sets)

    # --- Report ---
    config = {'days': days, 'min_pct': min_pct}
    report_path = write_report(events_with_anomalies, patterns, indicator_sets, config)

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"FERTIG — Gesamtdauer: {elapsed/3600:.1f} Stunden")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 80)

    conn.close()


if __name__ == '__main__':
    main()
