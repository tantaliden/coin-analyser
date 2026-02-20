#!/usr/bin/env python3
"""
Binance USDC Ingestor - Vision API + Live API
BERLIN-ZEIT Version

Modi:
  python ingestor.py backfill  - Vision API für historische Daten
  python ingestor.py live      - Live API für Echtzeit-Updates
"""

import os
import sys
import time
import io
import zipfile
import csv
import json
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

BERLIN_TZ = pytz.timezone('Europe/Berlin')

# ===========================================
# CONFIGURATION - aus settings.json
# ===========================================

with open('/opt/coin/database/settings.json') as f:
    SETTINGS = json.load(f)

DB_HOST = SETTINGS['databases']['coins']['host']
DB_PORT = SETTINGS['databases']['coins']['port']
DB_NAME = SETTINGS['databases']['coins']['name']
DB_USER = SETTINGS['ingestor']['database']['user']
DB_PASSWORD = SETTINGS['ingestor']['database']['password']

BINANCE_API_KEY = SETTINGS['binance']['apiKey']
BINANCE_API = SETTINGS['binance']['liveApiUrl']
VISION_BASE = SETTINGS['binance']['visionApi'] + '/monthly/klines'
VISION_DAILY = SETTINGS['binance']['visionApi'] + '/daily/klines'

START_YEAR = SETTINGS['ingestor']['startYear']
BATCH_SIZE = SETTINGS['ingestor']['batchSize']
HTTP_TIMEOUT = SETTINGS['ingestor']['httpTimeout']
HTTP_RETRIES = SETTINGS['ingestor']['httpRetries']
LIVE_OFFSET_MIN = SETTINGS['ingestor']['liveOffsetMinutes']
LIVE_LOOKBACK_HOURS = SETTINGS['ingestor']['liveLookbackHours']
LIVE_INTERVAL_SEC = SETTINGS['ingestor']['liveIntervalSeconds']
RATE_LIMIT_SLEEP = SETTINGS["ingestor"]["rateLimitSleep"]
SYMBOL_REFRESH_HOURS = SETTINGS["ingestor"]["symbolRefreshHours"]
QUOTE_ASSET = SETTINGS["ingestor"]["quoteAsset"]

# Initial Lookback für Lückenfüllung: 2000 Minuten = ~34 Stunden
INITIAL_LOOKBACK_HOURS = 34

print(f"[CONFIG] Start Year: {START_YEAR}")
print(f"[CONFIG] Batch Size: {BATCH_SIZE}")
print(f"[CONFIG] Live Lookback: {LIVE_LOOKBACK_HOURS} hours")
print(f"[CONFIG] Initial Lookback: {INITIAL_LOOKBACK_HOURS} hours (2000 min)")
print(f"[CONFIG] Live Interval: {LIVE_INTERVAL_SEC} seconds")
print(f"[CONFIG] Timezone: Europe/Berlin")

# ===========================================
# HTTP SESSION
# ===========================================

SESSION = requests.Session()
if BINANCE_API_KEY:
    SESSION.headers.update({'X-MBX-APIKEY': BINANCE_API_KEY})

# ===========================================
# TIMEZONE CONVERSION
# ===========================================

def utc_to_berlin(ts_ms):
    """Konvertiere UTC Millisekunden/Mikrosekunden zu Berlin datetime (naive)"""
    ts = int(ts_ms)
    # Binance hat ab 2025 Mikrosekunden (16 Ziffern)
    if ts > 100_000_000_000_000:
        ts = ts // 1_000_000
    elif ts > 100_000_000_000:
        ts = ts // 1_000
    
    utc_dt = datetime.utcfromtimestamp(ts)
    utc_dt = pytz.UTC.localize(utc_dt)
    berlin_dt = utc_dt.astimezone(BERLIN_TZ)
    return berlin_dt.replace(tzinfo=None)  # naive datetime für DB

def now_berlin():
    """Aktuelle Zeit in Berlin"""
    return datetime.now(BERLIN_TZ).replace(tzinfo=None)

# ===========================================
# DATABASE
# ===========================================

def db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

UPSERT_SQL = """
INSERT INTO klines (symbol, interval, open_time, open, high, low, close, volume, 
                    close_time, trades, quote_asset_volume, taker_buy_base, taker_buy_quote) 
VALUES %s 
ON CONFLICT (symbol, interval, open_time) DO UPDATE SET 
    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, 
    volume=EXCLUDED.volume, close_time=EXCLUDED.close_time, trades=EXCLUDED.trades, 
    quote_asset_volume=EXCLUDED.quote_asset_volume, taker_buy_base=EXCLUDED.taker_buy_base, 
    taker_buy_quote=EXCLUDED.taker_buy_quote
"""

# ===========================================
# HTTP HELPERS
# ===========================================

def http_get(url, timeout=HTTP_TIMEOUT, retries=HTTP_RETRIES):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            if i < retries - 1:
                time.sleep(1 + i * 2)
    return 'ERROR'

# ===========================================
# BINANCE SYMBOLS
# ===========================================

def fetch_symbols():
    print(f'[SYMBOLS] Fetching {QUOTE_ASSET} trading pairs...')
    r = SESSION.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    
    syms = []
    for s in data.get('symbols', []):
        if s.get('status') != 'TRADING':
            continue
        if s.get('quoteAsset') != QUOTE_ASSET:
            continue
        if s.get('isSpotTradingAllowed') is False:
            continue
        syms.append(s['symbol'])
    
    syms.sort()
    print(f'[SYMBOLS] Found {len(syms)} {QUOTE_ASSET} pairs')
    return syms

# ===========================================
# VISION API URLs (für Backfill)
# ===========================================

def vision_url(symbol, year, month):
    return f"{VISION_BASE}/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"

def daily_url(symbol, y, m, d):
    return f"{VISION_DAILY}/{symbol}/1m/{symbol}-1m-{y}-{m:02d}-{d:02d}.zip"

# ===========================================
# DATA PARSING - BERLIN ZEIT
# ===========================================

def parse_and_upsert(conn, symbol, rows_iter):
    """Parse Vision API CSV rows und speichere in Berlin-Zeit"""
    cur = conn.cursor()
    batch = []
    total = 0
    
    for row in rows_iter:
        open_time_berlin = utc_to_berlin(row[0])
        close_time_berlin = utc_to_berlin(row[6])
        
        vals = (
            symbol, '1m', open_time_berlin,
            float(row[1]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5]),
            close_time_berlin,
            int(row[8]),
            float(row[7]),
            float(row[9]), float(row[10])
        )
        batch.append(vals)
        
        if len(batch) >= BATCH_SIZE:
            execute_values(cur, UPSERT_SQL, batch)
            conn.commit()
            total += len(batch)
            batch = []
    
    if batch:
        execute_values(cur, UPSERT_SQL, batch)
        conn.commit()
        total += len(batch)
    
    return total

def parse_live_klines(conn, symbol, klines_data):
    """Parse Binance Live API klines response und speichere in Berlin-Zeit"""
    cur = conn.cursor()
    batch = []
    
    for row in klines_data:
        open_time_berlin = utc_to_berlin(row[0])
        close_time_berlin = utc_to_berlin(row[6])
        
        vals = (
            symbol, '1m', open_time_berlin,
            float(row[1]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5]),
            close_time_berlin,
            int(row[8]),
            float(row[7]),
            float(row[9]), float(row[10])
        )
        batch.append(vals)
    
    if batch:
        execute_values(cur, UPSERT_SQL, batch)
        conn.commit()
    
    return len(batch)

# ===========================================
# COMPLETENESS CHECKS - BERLIN ZEIT
# ===========================================

def count_minutes_range(conn, symbol, start_dt, end_dt):
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol=%s AND interval='1m' AND open_time BETWEEN %s AND %s",
        (symbol, start_dt, end_dt)
    )
    return cur.fetchone()[0] or 0

def is_month_complete(conn, symbol, y, m):
    start = datetime(y, m, 1, 0, 0)
    nextm = start + relativedelta(months=1)
    end = nextm - timedelta(minutes=1)
    expected = int((end - start).total_seconds() // 60) + 1
    cnt = count_minutes_range(conn, symbol, start, end)
    return cnt > 0 and cnt >= int(expected * 0.98)

def is_day_complete(conn, symbol, dday):
    start = datetime(dday.year, dday.month, dday.day, 0, 0)
    end = datetime(dday.year, dday.month, dday.day, 23, 59)
    cnt = count_minutes_range(conn, symbol, start, end)
    return cnt > 0 and cnt >= 1400

def get_last_candle_time(conn, symbol):
    """Letzte Candle-Zeit für ein Symbol (Berlin-Zeit)"""
    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(open_time) FROM klines WHERE symbol=%s AND interval='1m'",
        (symbol,)
    )
    result = cur.fetchone()[0]
    return result

# ===========================================
# BACKFILL LOGIC (Vision API)
# ===========================================

def get_earliest_month(symbol):
    """Find earliest month with data on Vision API"""
    now = now_berlin()
    for year in range(START_YEAR, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            url = vision_url(symbol, year, month)
            resp = http_get(url, timeout=5, retries=1)
            if resp is not None and resp != 'ERROR':
                return datetime(year, month, 1)
    return datetime(now.year, now.month, 1)

def backfill_symbol(conn, symbol):
    print(f"\n[BF] === {symbol} ===")
    
    start_dt = get_earliest_month(symbol)
    
    if start_dt.year < START_YEAR:
        start_dt = datetime(START_YEAR, 1, 1)
    
    print(f"[BF] {symbol} starting from: {start_dt:%Y-%m}")
    
    now = now_berlin()
    until_day = (now - timedelta(days=1)).date()
    last_full_month = (datetime(until_day.year, until_day.month, 1) - timedelta(days=1)).replace(day=1)
    
    total_inserted = 0
    
    # Monthly ZIPs
    current = start_dt
    while current <= last_full_month:
        y, m = current.year, current.month
        
        if is_month_complete(conn, symbol, y, m):
            print(f"[BF] {symbol} {y}-{m:02d} SKIP (complete)")
            current += relativedelta(months=1)
            continue
        
        url = vision_url(symbol, y, m)
        resp = http_get(url)
        
        if resp is None:
            print(f"[BF] {symbol} {y}-{m:02d} NOT FOUND")
            current += relativedelta(months=1)
            continue
        
        if resp == 'ERROR':
            current += relativedelta(months=1)
            continue
        
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'))
                    inserted = parse_and_upsert(conn, symbol, reader)
                    total_inserted += inserted
                    print(f"[BF] {symbol} {y}-{m:02d} +{inserted:,}")
        except Exception as e:
            print(f"[BF] {symbol} {y}-{m:02d} ERROR: {e}")
        
        current += relativedelta(months=1)
        time.sleep(0.1)
    
    # Daily ZIPs for current month
    cur_month_start = datetime(until_day.year, until_day.month, 1)
    if cur_month_start >= start_dt:
        start_day = max(cur_month_start.date(), start_dt.date())
        day = start_day
        
        while day <= until_day:
            if is_day_complete(conn, symbol, day):
                day += timedelta(days=1)
                continue
            
            url = daily_url(symbol, day.year, day.month, day.day)
            resp = http_get(url, timeout=10, retries=2)
            
            if resp and resp != 'ERROR':
                try:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        name = zf.namelist()[0]
                        with zf.open(name) as f:
                            reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'))
                            inserted = parse_and_upsert(conn, symbol, reader)
                            total_inserted += inserted
                            if inserted > 0:
                                print(f"[BF] {symbol} {day} +{inserted}")
                except:
                    pass
            
            day += timedelta(days=1)
            time.sleep(0.05)
    
    print(f"[BF] {symbol} DONE: +{total_inserted:,} candles")
    return total_inserted

def backfill_all(symbols):
    conn = db()
    total = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print(f"{'='*50}")
        
        try:
            added = backfill_symbol(conn, symbol)
            total += added
        except Exception as e:
            print(f"[BF] {symbol} FATAL: {e}")
            conn.rollback()
        
        time.sleep(0.5)
    
    conn.close()
    print(f"\n[DONE] Total: {total:,} candles")

# ===========================================
# LIVE SYNC LOGIC (Live API) - BERLIN ZEIT
# ===========================================

def fetch_live_klines(symbol, start_time_ms, end_time_ms, limit=1000):
    """Holt Klines von Binance Live API"""
    url = f"{BINANCE_API}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1m',
        'startTime': start_time_ms,
        'endTime': end_time_ms,
        'limit': limit
    }
    
    try:
        r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            time.sleep(60)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def berlin_to_utc_ms(dt_berlin):
    """Konvertiere Berlin naive datetime zu UTC Millisekunden für Binance API"""
    berlin_aware = BERLIN_TZ.localize(dt_berlin)
    utc_dt = berlin_aware.astimezone(pytz.UTC)
    return int(utc_dt.timestamp() * 1000)

def live_sync_symbol(conn, symbol, lookback_hours):
    """Synchronisiert ein Symbol mit Live API (Berlin-Zeit)"""
    now = now_berlin()
    
    # Bestimme Startzeit: entweder letzte Candle oder lookback_hours zurück
    last_candle = get_last_candle_time(conn, symbol)
    
    if last_candle:
        start_time = last_candle + timedelta(minutes=1)
    else:
        start_time = now - timedelta(hours=lookback_hours)
    
    # End-Zeit: jetzt minus offset
    end_time = now
    
    if start_time >= end_time:
        return 0
    
    # Konvertiere zu UTC für Binance API
    start_ms = berlin_to_utc_ms(start_time)
    end_ms = berlin_to_utc_ms(end_time)
    
    total_inserted = 0
    current_start = start_ms
    
    while current_start < end_ms:
        klines = fetch_live_klines(symbol, current_start, end_ms)
        
        if not klines:
            break
        
        inserted = parse_live_klines(conn, symbol, klines)
        total_inserted += inserted
        
        if len(klines) < 1000:
            break
        
        last_time = klines[-1][0]
        current_start = last_time + 60000
        
        time.sleep(0.005)
    
    return total_inserted

def live_sync_all(symbols, lookback_hours):
    """Einmaliger Sync aller Symbole"""
    conn = db()
    total = 0
    
    print(f"[LIVE] Starting sync for {len(symbols)} symbols (lookback: {lookback_hours}h)")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            added = live_sync_symbol(conn, symbol, lookback_hours)
            total += added
            if added > 0:
                print(f"[LIVE] [{i}/{len(symbols)}] {symbol} +{added}")
        except Exception as e:
            print(f"[LIVE] {symbol} ERROR: {e}")
            conn.rollback()
        
        time.sleep(0.1)
    
    conn.close()
    return total

def live_loop(symbols):
    """Endlos-Loop für Live-Updates (Berlin-Zeit)"""
    print(f"[LIVE] Starting live sync loop (Berlin-Zeit)")
    print(f"[LIVE] Initial Lookback: {INITIAL_LOOKBACK_HOURS} hours (2000 min)")
    print(f"[LIVE] Normal Lookback: {LIVE_LOOKBACK_HOURS} hours")
    print(f"[LIVE] Interval: {LIVE_INTERVAL_SEC} seconds")
    print(f"[LIVE] Symbol refresh: every {SYMBOL_REFRESH_HOURS} hours")
    
    first_run = True
    last_symbol_refresh = now_berlin()
    current_symbols = symbols
    
    while True:
        try:
            hours_since_refresh = (now_berlin() - last_symbol_refresh).total_seconds() / 3600
            if hours_since_refresh >= SYMBOL_REFRESH_HOURS:
                print(f"[LIVE] Refreshing symbol list...")
                old_count = len(current_symbols)
                current_symbols = fetch_symbols()
                new_count = len(current_symbols)
                if new_count != old_count:
                    print(f"[LIVE] Symbols changed: {old_count} -> {new_count}")
                last_symbol_refresh = now_berlin()
            
            if first_run:
                lookback = INITIAL_LOOKBACK_HOURS  # 34h = 2000 min
                first_run = False
            else:
                lookback = 1
            
            start_time = now_berlin()
            total = live_sync_all(current_symbols, lookback)
            duration = (now_berlin() - start_time).total_seconds()
            
            print(f"[LIVE] Cycle complete: +{total} candles in {duration:.1f}s")
            
            
        except KeyboardInterrupt:
            print("\n[LIVE] Stopped by user")
            break
        except Exception as e:
            print(f"[LIVE] Error in loop: {e}")
            time.sleep(30)


# ===========================================
# MAIN
# ===========================================

if __name__ == '__main__':
    print('[BOOT] Binance USDC Ingestor starting (BERLIN-ZEIT)...')
    print(f'[BOOT] Data from {START_YEAR} onwards')
    
    symbols = fetch_symbols()
    
    if len(sys.argv) < 2:
        print("\n[USAGE]")
        print("  python ingestor.py backfill  - Vision API für historische Daten")
        print("  python ingestor.py live      - Live API für Echtzeit-Updates")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "backfill":
        backfill_all(symbols)
        print("[BOOT] Backfill complete!")
    
    elif mode == "live":
        live_loop(symbols)
    
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("  Use 'backfill' or 'live'")
SCRIPT_EOF # Neuer ingestor.py mit Berlin-Zeit