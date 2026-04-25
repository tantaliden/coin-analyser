#!/usr/bin/env python3
"""
AutoSearch Worker v2 - Slot-basierte Indikator-Erkennung
Findet häufige Wertebereiche in einzelnen Zeitslots
"""

import os
import sys
import time
import json
import logging
import signal
import traceback
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/coin/logs/autosearch_worker.log')
    ]
)
logger = logging.getLogger('autosearch_worker')

# DB Connection Settings
APP_DB = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'analyser_app',
    'user': 'volker_admin',
    'password': 'VoltiStrongPass2025'
}

COINS_DB = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'coins',
    'user': 'ingestor',
    'password': 'tX9RX2XOV8aQg5F01FVDwQeVFvs7cLTyqN4wB7Avv8Q='
}

# Felder die gescannt werden
SCAN_FIELDS = ['close', 'volume', 'trades', 'volatility', 'takerBuyRatio']

# Global flag für graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    logger.info(f"Signal {signum} received, shutting down gracefully...")
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@contextmanager
def get_app_db():
    conn = psycopg2.connect(**APP_DB, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_coins_db():
    conn = psycopg2.connect(**COINS_DB, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def update_job_progress(job_id, progress_percent, message=None):
    """Update job progress in DB"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            if message:
                cur.execute("""
                    UPDATE autosearch_jobs 
                    SET progress_percent = %s, progress_message = %s
                    WHERE id = %s
                """, (progress_percent, message, job_id))
            else:
                cur.execute("""
                    UPDATE autosearch_jobs 
                    SET progress_percent = %s
                    WHERE id = %s
                """, (progress_percent, job_id))
            conn.commit()


def check_job_cancelled(job_id):
    """Check if job was cancelled"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM autosearch_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            return row and row['status'] == 'cancelled'


def compute_slot_value(candles, slot_start_min, slot_end_min, field, base_values, zero_point):
    """Berechnet den normalisierten Wert für einen Slot"""
    slot_candles = [c for c in candles 
                   if slot_start_min <= (c['open_time'] - zero_point).total_seconds() / 60 < slot_end_min]
    
    if not slot_candles:
        return None
    
    if field == 'close':
        # % Change vom 0-Punkt
        avg_close = sum(c['close'] or 0 for c in slot_candles) / len(slot_candles)
        return ((avg_close - base_values['close']) / base_values['close']) * 100 if base_values['close'] else 0
    
    elif field == 'volume':
        # % Change vom 0-Punkt (erste 15 Minuten Durchschnitt)
        avg_volume = sum(c['volume'] or 0 for c in slot_candles) / len(slot_candles)
        return ((avg_volume - base_values['volume']) / base_values['volume']) * 100 if base_values['volume'] else 0
    
    elif field == 'trades':
        # % Change vom 0-Punkt
        avg_trades = sum(c['trades'] or 0 for c in slot_candles) / len(slot_candles)
        return ((avg_trades - base_values['trades']) / base_values['trades']) * 100 if base_values['trades'] else 0
    
    elif field == 'volatility':
        # Durchschnittliche Volatility in % (absolut, nicht relativ)
        volatilities = [(c['high'] - c['low']) / c['close'] * 100 if c['close'] else 0 for c in slot_candles]
        return sum(volatilities) / len(volatilities) if volatilities else 0
    
    elif field == 'takerBuyRatio':
        # Durchschnittliche Ratio (0-1, absolut)
        ratios = [c['taker_buy_base'] / c['volume'] if c['volume'] else 0.5 for c in slot_candles]
        return sum(ratios) / len(ratios) if ratios else 0.5
    
    return None


def find_value_clusters(values, min_support_percent=10, num_bins=20):
    """
    Findet Wertebereiche in denen viele Events liegen.
    Returns: Liste von (bin_min, bin_max, count, percent)
    """
    if not values:
        return []
    
    values = [v for v in values if v is not None]
    if len(values) < 10:
        return []
    
    arr = np.array(values)
    
    # Outlier entfernen (1. und 99. Perzentil)
    p1, p99 = np.percentile(arr, [1, 99])
    arr_clean = arr[(arr >= p1) & (arr <= p99)]
    
    if len(arr_clean) < 10:
        return []
    
    # Histogram erstellen
    hist, bin_edges = np.histogram(arr_clean, bins=num_bins)
    
    total = len(values)
    min_count = total * (min_support_percent / 100)
    
    clusters = []
    for i, count in enumerate(hist):
        if count >= min_count:
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            percent = (count / total) * 100
            clusters.append({
                'min': float(bin_min),
                'max': float(bin_max),
                'count': int(count),
                'percent': float(percent)
            })
    
    return clusters


def process_job(job):
    """Verarbeitet einen einzelnen Discovery Job - v2 Slot-basiert"""
    job_id = job['id']
    user_id = job['user_id']
    params = job['params']
    
    logger.info(f"[JOB {job_id}] Starting slot-based discovery for user {user_id}")
    logger.info(f"[JOB {job_id}] Params: {json.dumps(params)}")
    
    try:
        # Job als running markieren
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE autosearch_jobs 
                    SET status = 'running', started_at = NOW(), worker_pid = %s,
                        progress_percent = 0, progress_message = 'Starte Suche...'
                    WHERE id = %s
                """, (os.getpid(), job_id))
                conn.commit()
        
        # Parameter extrahieren
        date_from = params.get('date_from')
        date_to = params.get('date_to')
        percent_min = params.get('percent_min', 5)
        percent_max = params.get('percent_max', 100)
        duration_minutes = params.get('duration_minutes', 60)
        prehistory_minutes = params.get('prehistory_minutes', 1440)
        granularity_minutes = params.get('granularity_minutes', 15)
        min_cluster_size = params.get('min_cluster_size', 10)
        
        duration_col = f"pct_{duration_minutes}m"
        num_slots = prehistory_minutes // granularity_minutes
        
        # ========================================
        # PHASE 1: Events laden
        # ========================================
        update_job_progress(job_id, 5, "Lade Events aus kline_metrics...")
        logger.info(f"[JOB {job_id}] Phase 1: Loading events")
        
        events = []
        with get_coins_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT symbol, open_time, {duration_col} as change_pct
                    FROM kline_metrics
                    WHERE open_time >= %s AND open_time <= %s
                      AND {duration_col} >= %s AND {duration_col} <= %s
                      AND {duration_col} IS NOT NULL
                    ORDER BY open_time
                """, (date_from, date_to, percent_min, percent_max))
                events = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"[JOB {job_id}] Found {len(events)} events")
        
        if len(events) < min_cluster_size:
            raise ValueError(f"Nur {len(events)} Events gefunden, mindestens {min_cluster_size} benötigt")
        
        if check_job_cancelled(job_id):
            return
        
        # ========================================
        # PHASE 2: Slot-Werte für alle Events sammeln
        # ========================================
        update_job_progress(job_id, 10, f"Berechne Slot-Werte für {len(events)} Events...")
        logger.info(f"[JOB {job_id}] Phase 2: Computing slot values")
        
        # Struktur: slot_values[slot_idx][field] = [list of values]
        slot_values = {slot: {field: [] for field in SCAN_FIELDS} for slot in range(num_slots)}
        valid_event_count = 0
        
        with get_coins_db() as conn:
            with conn.cursor() as cur:
                total_events = len(events)
                for i, event in enumerate(events):
                    if i % 100 == 0:
                        progress = 10 + int((i / total_events) * 50)
                        update_job_progress(job_id, progress, f"Slot-Berechnung: {i}/{total_events} Events...")
                        logger.info(f"[JOB {job_id}] Processing event {i}/{total_events}")
                        
                        if check_job_cancelled(job_id):
                            return
                    
                    symbol = event['symbol']
                    event_time = event['open_time']
                    zero_point = event_time - timedelta(minutes=prehistory_minutes)
                    
                    # Vorlauf-Daten laden
                    cur.execute("""
                        SELECT open_time, close, volume, trades, high, low, taker_buy_base
                        FROM klines
                        WHERE symbol = %s AND interval = '1m'
                          AND open_time >= %s AND open_time < %s
                        ORDER BY open_time
                    """, (symbol, zero_point, event_time))
                    
                    candles = [dict(row) for row in cur.fetchall()]
                    
                    if len(candles) < prehistory_minutes * 0.8:
                        continue
                    
                    # Basis-Werte am 0-Punkt (erste 15 Minuten)
                    base_candles = [c for c in candles if (c['open_time'] - zero_point).total_seconds() / 60 < granularity_minutes]
                    if not base_candles:
                        continue
                    
                    base_values = {
                        'close': sum(c['close'] or 0 for c in base_candles) / len(base_candles),
                        'volume': sum(c['volume'] or 0 for c in base_candles) / len(base_candles),
                        'trades': sum(c['trades'] or 0 for c in base_candles) / len(base_candles),
                    }
                    
                    # Skip wenn Basis-Werte 0 sind
                    if base_values['close'] == 0 or base_values['volume'] == 0:
                        continue
                    
                    valid_event_count += 1
                    
                    # Für jeden Slot und jedes Feld den Wert berechnen
                    for slot in range(num_slots):
                        slot_start = slot * granularity_minutes
                        slot_end = (slot + 1) * granularity_minutes
                        
                        for field in SCAN_FIELDS:
                            val = compute_slot_value(candles, slot_start, slot_end, field, base_values, zero_point)
                            if val is not None:
                                slot_values[slot][field].append(val)
        
        logger.info(f"[JOB {job_id}] Valid events with full data: {valid_event_count}")
        
        if check_job_cancelled(job_id):
            return
        
        # ========================================
        # PHASE 3: Häufige Wertebereiche finden
        # ========================================
        update_job_progress(job_id, 65, "Suche häufige Wertebereiche...")
        logger.info(f"[JOB {job_id}] Phase 3: Finding frequent value ranges")
        
        # Mindestens 15% der Events müssen in einem Wertebereich liegen
        min_support_percent = 15
        
        significant_indicators = []
        
        for slot in range(num_slots):
            for field in SCAN_FIELDS:
                values = slot_values[slot][field]
                clusters = find_value_clusters(values, min_support_percent=min_support_percent)
                
                for cluster in clusters:
                    # Nur "interessante" Bereiche - nicht der Normalzustand
                    is_interesting = False
                    
                    if field == 'close':
                        # Close: Interessant wenn nicht um 0% herum
                        if cluster['max'] < -2 or cluster['min'] > 2:
                            is_interesting = True
                    elif field == 'volume':
                        # Volume: Interessant wenn deutlich unter oder über 0%
                        if cluster['max'] < -30 or cluster['min'] > 30:
                            is_interesting = True
                    elif field == 'trades':
                        # Trades: Interessant wenn deutlich abweichend
                        if cluster['max'] < -30 or cluster['min'] > 30:
                            is_interesting = True
                    elif field == 'volatility':
                        # Volatility: Interessant wenn hoch (> 0.5%)
                        if cluster['min'] > 0.5:
                            is_interesting = True
                    elif field == 'takerBuyRatio':
                        # TakerRatio: Interessant wenn nicht um 0.5 herum
                        if cluster['max'] < 0.45 or cluster['min'] > 0.55:
                            is_interesting = True
                    
                    if is_interesting:
                        slot_start_min = slot * granularity_minutes
                        slot_end_min = (slot + 1) * granularity_minutes
                        
                        significant_indicators.append({
                            'slot': slot,
                            'field': field,
                            'time_start': slot_start_min,
                            'time_end': slot_end_min,
                            'value_min': cluster['min'],
                            'value_max': cluster['max'],
                            'count': cluster['count'],
                            'percent': cluster['percent']
                        })
                        
                        logger.info(f"[JOB {job_id}] Found indicator: {field} @ {slot_start_min}-{slot_end_min}min = {cluster['min']:.2f} to {cluster['max']:.2f} ({cluster['percent']:.1f}%)")
        
        logger.info(f"[JOB {job_id}] Total significant indicators found: {len(significant_indicators)}")
        
        if check_job_cancelled(job_id):
            return
        
        # ========================================
        # PHASE 4: Patterns erstellen
        # ========================================
        update_job_progress(job_id, 80, f"Erstelle Patterns aus {len(significant_indicators)} Indikatoren...")
        logger.info(f"[JOB {job_id}] Phase 4: Creating patterns")
        
        pattern_ids = []
        
        if significant_indicators:
            # Gruppiere Indikatoren nach Zeit (früh, mittel, spät)
            early = [i for i in significant_indicators if i['time_end'] <= prehistory_minutes // 3]
            mid = [i for i in significant_indicators if prehistory_minutes // 3 < i['time_start'] < 2 * prehistory_minutes // 3]
            late = [i for i in significant_indicators if i['time_start'] >= 2 * prehistory_minutes // 3]
            
            logger.info(f"[JOB {job_id}] Indicators by time: early={len(early)}, mid={len(mid)}, late={len(late)}")
            
            # Sortiere nach Support (höchster zuerst)
            all_sorted = sorted(significant_indicators, key=lambda x: -x['percent'])
            
            # Erstelle ein Pattern mit den Top-Indikatoren (max 10)
            top_indicators = all_sorted[:10]
            
            if top_indicators:
                with get_app_db() as conn:
                    with conn.cursor() as cur:
                        # Pattern speichern
                        cur.execute("""
                            INSERT INTO discovered_patterns 
                            (user_id, name, status, search_date_from, search_date_to,
                             event_percent_min, event_percent_max, event_duration_minutes,
                             prehistory_minutes, source_event_count, cluster_event_count)
                            VALUES (%s, %s, 'discovered', %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (user_id, f"SlotPattern {job_id}", date_from, date_to,
                              percent_min, percent_max, duration_minutes,
                              prehistory_minutes, len(events), valid_event_count))
                        
                        pattern_id = cur.fetchone()['id']
                        pattern_ids.append(pattern_id)
                        
                        # Indikatoren nach Zeit sortieren
                        top_indicators.sort(key=lambda x: x['time_start'])
                        
                        for pos, ind in enumerate(top_indicators, 1):
                            cur.execute("""
                                INSERT INTO pattern_indicators
                                (pattern_id, position, field, time_start_minutes, time_end_minutes,
                                 value_min, value_max, match_count, match_percent)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (pattern_id, pos, ind['field'], ind['time_start'], ind['time_end'],
                                  ind['value_min'], ind['value_max'], ind['count'], ind['percent']))
                        
                        conn.commit()
                        logger.info(f"[JOB {job_id}] Created pattern {pattern_id} with {len(top_indicators)} indicators")
        
        # ========================================
        # PHASE 5: Job abschließen
        # ========================================
        result_summary = {
            'total_events': len(events),
            'valid_events': valid_event_count,
            'indicators_found': len(significant_indicators),
            'patterns_created': len(pattern_ids)
        }
        
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE autosearch_jobs 
                    SET status = 'completed', completed_at = NOW(),
                        progress_percent = 100, progress_message = 'Fertig!',
                        result_pattern_ids = %s, result_summary = %s
                    WHERE id = %s
                """, (pattern_ids, json.dumps(result_summary), job_id))
                conn.commit()
        
        logger.info(f"[JOB {job_id}] Completed: {result_summary}")
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"[JOB {job_id}] Failed: {error_msg}")
        logger.error(f"[JOB {job_id}] Traceback:\n{error_trace}")
        
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE autosearch_jobs 
                    SET status = 'failed', completed_at = NOW(),
                        progress_message = %s, error_message = %s
                    WHERE id = %s
                """, (f"Fehler: {error_msg}", error_trace, job_id))
                conn.commit()


def main():
    """Main Worker Loop"""
    logger.info("=" * 60)
    logger.info("AutoSearch Worker v2 (Slot-based) starting...")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)
    
    while running:
        try:
            with get_app_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM autosearch_jobs 
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        LIMIT 1
                    """)
                    job = cur.fetchone()
            
            if job:
                logger.info(f"Found queued job {job['id']}, processing...")
                process_job(job)
            else:
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(10)
    
    logger.info("AutoSearch Worker stopped")


if __name__ == '__main__':
    main()
