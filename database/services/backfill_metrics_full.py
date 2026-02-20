#!/usr/bin/env python3
"""
Backfill fehlende pct_* und abs_* Werte in kline_metrics
Füllt ALLE Lücken tagesweise ab einem Startdatum.
Usage: python3 backfill_metrics_full.py [START_DATE] [END_DATE]
Default: 2026-01-22 bis heute
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json, sys, pytz

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
    print(f"[{datetime.now(BERLIN_TZ):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

def backfill_symbol_day(conn, symbol, day_start, day_end):
    lookback_start = day_start - timedelta(minutes=660)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, open, close FROM klines
            WHERE symbol = %s AND interval = '1m' AND open_time >= %s AND open_time <= %s
            ORDER BY open_time
        """, (symbol, lookback_start, day_end))
        candles = cur.fetchall()
    
    if len(candles) < 100:
        return 0
    
    candle_map = {c['open_time']: c for c in candles}
    
    cols = ["symbol", "open_time"]
    cols += [f"pct_{d}m" for d in DURATIONS]
    cols += [f"abs_{d}m" for d in DURATIONS]
    
    insert_sql = f"""
        INSERT INTO kline_metrics ({', '.join(cols)}) VALUES ({', '.join(['%s']*len(cols))})
        ON CONFLICT (symbol, open_time) DO UPDATE SET
        {', '.join([f'pct_{d}m = EXCLUDED.pct_{d}m' for d in DURATIONS])},
        {', '.join([f'abs_{d}m = EXCLUDED.abs_{d}m' for d in DURATIONS])}
    """
    
    rows = []
    for c in candles:
        if c['open_time'] < day_start or c['open_time'] >= day_end:
            continue
        if not c['close']:
            continue
        row = [symbol, c['open_time']]
        for d in DURATIONS:
            start_t = c['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open'] and float(sc['open']) != 0:
                row.append(((float(c['close']) - float(sc['open'])) / float(sc['open'])) * 100)
            else:
                row.append(None)
        for d in DURATIONS:
            start_t = c['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open']:
                row.append(float(c['close']) - float(sc['open']))
            else:
                row.append(None)
        rows.append(row)
    
    if rows:
        with conn.cursor() as cur:
            # Batch in chunks of 500
            for i in range(0, len(rows), 500):
                cur.executemany(insert_sql, rows[i:i+500])
            conn.commit()
    return len(rows)

def main():
    start_str = sys.argv[1] if len(sys.argv) > 1 else '2026-01-22'
    end_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
    
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    
    log(f"Backfill metrics: {start_str} -> {end_str}")
    
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval='1m' ORDER BY symbol")
        symbols = [r[0] for r in cur.fetchall()]
    log(f"{len(symbols)} symbols")
    
    day = start
    while day <= end:
        day_start = day
        day_end = day + timedelta(days=1)
        total = 0
        for sym in symbols:
            total += backfill_symbol_day(conn, sym, day_start, day_end)
        log(f"{day:%Y-%m-%d}: {total} rows updated")
        day += timedelta(days=1)
    
    conn.close()
    log("DONE")

if __name__ == '__main__':
    main()
