#!/usr/bin/env python3
"""
Backfill fehlende pct_* und abs_* Werte in kline_metrics
Zeitraum: 17.01.2026 00:00 bis 24.01.2026 00:00
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json
import pytz

BERLIN_TZ = pytz.timezone('Europe/Berlin')

with open('/opt/coin/database/settings.json') as f:
    SETTINGS = json.load(f)

DB_CONFIG = {
    'host': SETTINGS['databases']['coins']['host'],
    'port': SETTINGS['databases']['coins']['port'],
    'dbname': SETTINGS['databases']['coins']['name'],
    'user': 'volker_admin',
    'password': 'VoltiStrongPass2025'
}

DURATIONS = [30, 60, 90, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600]

def log(msg):
    timestamp = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_all_symbols(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval='1m' ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]

def backfill_symbol_range(conn, symbol, start_time, end_time):
    """Berechnet pct/abs Werte für einen Zeitraum mit 660min Lookback"""
    
    # Wir brauchen 660 Minuten VOR start_time für die Berechnungen
    lookback_start = start_time - timedelta(minutes=660)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, open, close 
            FROM klines 
            WHERE symbol = %s AND interval = '1m' 
              AND open_time >= %s AND open_time <= %s
            ORDER BY open_time
        """, (symbol, lookback_start, end_time))
        candles = cur.fetchall()
    
    if len(candles) < 100:
        return 0
    
    candle_map = {c['open_time']: c for c in candles}
    
    # Nur Candles im Zielbereich updaten
    target_candles = [c for c in candles if c['open_time'] >= start_time]
    
    if not target_candles:
        return 0
    
    update_sql = """
        UPDATE kline_metrics SET
            pct_30m = %s, pct_60m = %s, pct_90m = %s, pct_120m = %s,
            pct_180m = %s, pct_240m = %s, pct_300m = %s, pct_330m = %s,
            pct_360m = %s, pct_420m = %s, pct_480m = %s, pct_540m = %s, pct_600m = %s,
            abs_30m = %s, abs_60m = %s, abs_90m = %s, abs_120m = %s,
            abs_180m = %s, abs_240m = %s, abs_300m = %s, abs_330m = %s,
            abs_360m = %s, abs_420m = %s, abs_480m = %s, abs_540m = %s, abs_600m = %s
        WHERE symbol = %s AND open_time = %s
    """
    
    batch = []
    for candle in target_candles:
        if not candle['close']:
            continue
        
        pct_vals = []
        abs_vals = []
        
        for d in DURATIONS:
            start_t = candle['open_time'] - timedelta(minutes=d)
            start_c = candle_map.get(start_t)
            if start_c and start_c['open'] and float(start_c['open']) != 0:
                pct = ((float(candle['close']) - float(start_c['open'])) / float(start_c['open'])) * 100
                abs_val = float(candle['close']) - float(start_c['open'])
            else:
                pct = None
                abs_val = None
            pct_vals.append(pct)
            abs_vals.append(abs_val)
        
        batch.append(tuple(pct_vals + abs_vals + [symbol, candle['open_time']]))
    
    if batch:
        with conn.cursor() as cur:
            cur.executemany(update_sql, batch)
        conn.commit()
    
    return len(batch)

def main():
    # Korrigierter Zeitraum: ab 17.01 wo das Problem beginnt
    start_time = datetime(2026, 1, 17, 0, 0)   # 17.01.2026 00:00
    end_time = datetime(2026, 1, 24, 2, 0)     # 24.01.2026 02:00
    
    log(f"=== BACKFILL pct_* und abs_* Werte ===")
    log(f"Zeitraum: {start_time} bis {end_time}")
    log(f"Durations: {DURATIONS}")
    
    conn = get_connection()
    symbols = get_all_symbols(conn)
    log(f"Symbole: {len(symbols)}")
    
    total = 0
    for i, symbol in enumerate(symbols, 1):
        try:
            count = backfill_symbol_range(conn, symbol, start_time, end_time)
            total += count
            log(f"[{i}/{len(symbols)}] {symbol}: {count} rows")
        except Exception as e:
            log(f"[{i}/{len(symbols)}] {symbol} ERROR: {e}")
            conn.rollback()
    
    log(f"=== FERTIG: {total} Zeilen aktualisiert ===")
    conn.close()

if __name__ == '__main__':
    main()
