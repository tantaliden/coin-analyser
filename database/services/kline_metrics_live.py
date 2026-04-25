#!/usr/bin/env python3
"""
kline_metrics Live Service - BERLIN-ZEIT Version
- Alle 2 Minuten: letzte 660 Candles pro Coin laden und pct/abs berechnen
- Alle 12h: Lückenprüfung für die letzten 24h, automatischer Backfill
- Einmal täglich: Check auf neue Coins, ggf. Backfill
- Beim Start: 2000 Minuten Rückblick für Lückenfüllung
- Telegram-Alerts bei Lücken und Backfill-Status
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from datetime import datetime, timedelta
import time
import json
import pytz
import urllib.request
import urllib.parse

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

# Telegram Config
BOT_TOKEN = '8430890812:AAFAbxXkc9-hw19FMuaRQwN4d6s0O8rULrM'
CHAT_ID = '2112844328'

DURATIONS = [30, 60, 90, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600]
LOOKBACK_MINUTES = 660
INITIAL_LOOKBACK_MINUTES = 2000
UPDATE_INTERVAL_SEC = 120
DAILY_CHECK_HOUR = 3
GAP_CHECK_INTERVAL_SEC = 43200  # 12 Stunden
GAP_BACKFILL_HOURS = 24         # Lückenprüfung: letzte 24h
LOG_FILE = '/opt/coin/database/logs/kline_metrics_live.log'
GAP_STATE_FILE = '/opt/coin/database/logs/.metrics_gap_state'

def log(msg):
    timestamp = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass

def now_berlin():
    return datetime.now(BERLIN_TZ).replace(tzinfo=None)

def send_telegram(text):
    """Telegram-Nachricht senden"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            'chat_id': CHAT_ID,
            'text': text,
            'parse_mode': 'HTML'
        }).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as response:
            return True
    except Exception as e:
        log(f"[TELEGRAM] Send error: {e}")
        return False

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_all_symbols(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval='1m' ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]

def update_symbol(conn, symbol, lookback_minutes=LOOKBACK_MINUTES):
    """Lädt Klines mit erweitertem Lookback für Referenz-Klines,
    berechnet aber nur für die letzten lookback_minutes."""
    now = now_berlin()
    max_duration = max(DURATIONS)  # 600
    # Lade Klines weit genug zurück damit ALLE Referenz-Klines vorhanden sind
    data_start = now - timedelta(minutes=lookback_minutes + max_duration + 10)
    # Berechne nur für Klines ab calc_start
    calc_start = now - timedelta(minutes=lookback_minutes)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, open, close
            FROM klines
            WHERE symbol = %s AND interval = '1m' AND open_time >= %s
            ORDER BY open_time
        """, (symbol, data_start))
        candles = cur.fetchall()

    if len(candles) < max_duration:
        return 0

    candle_map = {c['open_time']: c for c in candles}
    # Nur Klines ab calc_start berechnen (nicht das ganze erweiterte Fenster)
    calc_candles = [c for c in candles if c['open_time'] >= calc_start]

    cols = ["symbol", "open_time"]
    cols += [f"pct_{d}m" for d in DURATIONS]
    cols += [f"abs_{d}m" for d in DURATIONS]

    upsert_sql = f"""
        INSERT INTO kline_metrics ({', '.join(cols)}) VALUES %s
        ON CONFLICT (symbol, open_time) DO UPDATE SET
        {', '.join([f'pct_{d}m = COALESCE(EXCLUDED.pct_{d}m, kline_metrics.pct_{d}m)' for d in DURATIONS])},
        {', '.join([f'abs_{d}m = COALESCE(EXCLUDED.abs_{d}m, kline_metrics.abs_{d}m)' for d in DURATIONS])}
    """

    rows = []
    for candle in calc_candles:
        if not candle['close']:
            continue
        row = [symbol, candle['open_time']]
        for d in DURATIONS:
            start_t = candle['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open'] and float(sc['open']) != 0:
                row.append(((float(candle['close']) - float(sc['open'])) / float(sc['open'])) * 100)
            else:
                row.append(None)
        for d in DURATIONS:
            start_t = candle['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open']:
                row.append(float(candle['close']) - float(sc['open']))
            else:
                row.append(None)
        rows.append(tuple(row))

    if rows:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows, page_size=1000)
        conn.commit()

    return len(rows)

# ============ LÜCKENPRÜFUNG + AUTOBACKFILL ============

def check_gaps(conn):
    """Prüft ob in den letzten 24h Lücken bei IRGENDEINER Duration existieren.
    Gibt dict zurück: {symbol: gap_count}"""
    now = now_berlin()
    start = now - timedelta(hours=GAP_BACKFILL_HOURS)

    log(f"[GAP-CHECK] Prüfe Lücken {start:%Y-%m-%d %H:%M} bis {now:%Y-%m-%d %H:%M}")

    with conn.cursor() as cur:
        # Zähle pro Symbol: Zeilen wo IRGENDEINE Duration NULL ist
        null_conditions = " OR ".join([f"pct_{d}m IS NULL" for d in DURATIONS])
        cur.execute(f"""
            SELECT symbol, COUNT(*) as gap_count
            FROM kline_metrics
            WHERE open_time >= %s AND open_time < %s
              AND ({null_conditions})
            GROUP BY symbol
            HAVING COUNT(*) > 10
            ORDER BY COUNT(*) DESC
        """, (start, now))
        gaps = {row[0]: row[1] for row in cur.fetchall()}

    return gaps

def backfill_gaps(conn, gaps):
    """Füllt Lücken für alle betroffenen Symbole.
    Überspringt Zeilen die bereits befüllt sind (pct_120m IS NOT NULL, ausgenommen 0)."""
    now = now_berlin()
    max_duration = max(DURATIONS)  # 600
    gap_start = now - timedelta(hours=GAP_BACKFILL_HOURS)
    # Lookback muss max_duration VOR dem Gap-Start beginnen damit alle Referenz-Klines vorhanden
    data_start = gap_start - timedelta(minutes=max_duration + 10)

    total_filled = 0
    symbols_filled = 0

    cols = ["symbol", "open_time"]
    cols += [f"pct_{d}m" for d in DURATIONS]
    cols += [f"abs_{d}m" for d in DURATIONS]

    upsert_sql = f"""
        INSERT INTO kline_metrics ({', '.join(cols)}) VALUES %s
        ON CONFLICT (symbol, open_time) DO UPDATE SET
        {', '.join([f'pct_{d}m = COALESCE(EXCLUDED.pct_{d}m, kline_metrics.pct_{d}m)' for d in DURATIONS])},
        {', '.join([f'abs_{d}m = COALESCE(EXCLUDED.abs_{d}m, kline_metrics.abs_{d}m)' for d in DURATIONS])}
    """

    for symbol, gap_count in gaps.items():
        try:
            # Klines laden (mit Lookback für Duration-Berechnung)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT open_time, open, close
                    FROM klines
                    WHERE symbol = %s AND interval = '1m'
                      AND open_time >= %s AND open_time <= %s
                    ORDER BY open_time
                """, (symbol, data_start, now))
                candles = cur.fetchall()

            if len(candles) < 100:
                continue

            candle_map = {c['open_time']: c for c in candles}

            # Finde Lücken-Zeiten für dieses Symbol (irgendeine Duration IS NULL)
            null_conditions = " OR ".join([f"pct_{d}m IS NULL" for d in DURATIONS])
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT open_time FROM kline_metrics
                    WHERE symbol = %s AND open_time >= %s AND open_time < %s
                      AND ({null_conditions})
                    ORDER BY open_time
                """, (symbol, gap_start, now))
                gap_times = set(row[0] for row in cur.fetchall())

            if not gap_times:
                continue

            rows = []
            for gt in gap_times:
                c = candle_map.get(gt)
                if not c or not c['close']:
                    continue

                row = [symbol, gt]
                for d in DURATIONS:
                    start_t = gt - timedelta(minutes=d)
                    sc = candle_map.get(start_t)
                    if sc and sc['open'] and float(sc['open']) != 0:
                        row.append(((float(c['close']) - float(sc['open'])) / float(sc['open'])) * 100)
                    else:
                        row.append(None)
                for d in DURATIONS:
                    start_t = gt - timedelta(minutes=d)
                    sc = candle_map.get(start_t)
                    if sc and sc['open']:
                        row.append(float(c['close']) - float(sc['open']))
                    else:
                        row.append(None)
                rows.append(tuple(row))

            if rows:
                with conn.cursor() as cur:
                    execute_values(cur, upsert_sql, rows, page_size=1000)
                conn.commit()
                total_filled += len(rows)
                symbols_filled += 1

        except Exception as e:
            log(f"[GAP-FILL] {symbol} ERROR: {e}")
            try:
                conn.rollback()
            except:
                pass

    return symbols_filled, total_filled

def run_gap_check(conn):
    """12h-Lückenprüfung mit Autobackfill und Telegram-Alerts"""
    log("[GAP-CHECK] === 12h Lückenprüfung gestartet ===")

    gaps = check_gaps(conn)

    if not gaps:
        log("[GAP-CHECK] Keine Lücken gefunden ✓")
        return

    total_gaps = sum(gaps.values())
    log(f"[GAP-CHECK] LÜCKEN: {len(gaps)} Symbole, {total_gaps} fehlende Zeilen")

    # Telegram-Alert: Lücken gefunden
    top5 = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:5]
    top_text = "\n".join([f"  {s}: {c} Lücken" for s, c in top5])
    send_telegram(
        f"⚠️ <b>METRICS LÜCKEN</b>\n\n"
        f"{len(gaps)} Symbole mit fehlenden pct-Werten\n"
        f"Total: {total_gaps} leere Zeilen (24h)\n\n"
        f"Top 5:\n<pre>{top_text}</pre>\n\n"
        f"🔄 Starte Auto-Backfill..."
    )

    # Backfill ausführen
    start_time = time.time()
    symbols_filled, rows_filled = backfill_gaps(conn, gaps)
    duration = time.time() - start_time

    log(f"[GAP-CHECK] Backfill fertig: {symbols_filled} Symbole, {rows_filled} Zeilen, {duration:.0f}s")

    # Telegram-Alert: Backfill fertig
    send_telegram(
        f"✅ <b>METRICS BACKFILL FERTIG</b>\n\n"
        f"{symbols_filled} Symbole aufgefüllt\n"
        f"{rows_filled} Zeilen berechnet\n"
        f"Dauer: {duration:.0f}s\n\n"
        f"🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}"
    )

    # Nochmal prüfen ob alles gefüllt ist
    remaining = check_gaps(conn)
    if remaining:
        still = sum(remaining.values())
        log(f"[GAP-CHECK] WARNUNG: Noch {still} Lücken übrig (fehlende Klines)")
        send_telegram(
            f"⚠️ <b>NOCH LÜCKEN</b>\n\n"
            f"{len(remaining)} Symbole haben noch {still} Lücken\n"
            f"(vermutlich fehlende 1m Klines)\n\n"
            f"Nächster Check in 12h"
        )

def check_new_coins(conn):
    """Prüft auf Coins deren erster klines-Eintrag < 25h alt ist"""
    log("[DAILY] Checking for new coins...")
    cutoff = now_berlin() - timedelta(hours=25)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, MIN(open_time) as first_candle
            FROM klines
            WHERE interval = '1m'
            GROUP BY symbol
            HAVING MIN(open_time) > %s
        """, (cutoff,))
        new_coins = cur.fetchall()

    if not new_coins:
        log("[DAILY] No new coins found")
        return

    log(f"[DAILY] Found {len(new_coins)} new coin(s)")
    send_telegram(f"🆕 <b>{len(new_coins)} neue Coins</b>\n\n" +
                  "\n".join([f"• {s}" for s, _ in new_coins]))

    for symbol, first_candle in new_coins:
        log(f"[DAILY] Backfilling {symbol} (first candle: {first_candle})")
        backfill_symbol(conn, symbol)

def backfill_symbol(conn, symbol):
    """Backfill für einen einzelnen Coin"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, open, close
            FROM klines
            WHERE symbol = %s AND interval = '1m'
            ORDER BY open_time
        """, (symbol,))
        candles = cur.fetchall()

    if len(candles) < 600:
        log(f"  {symbol}: Not enough candles ({len(candles)})")
        return 0

    candle_map = {c['open_time']: c for c in candles}
    start_from = candles[0]['open_time'] + timedelta(minutes=600)

    cols = ["symbol", "open_time"]
    cols += [f"pct_{d}m" for d in DURATIONS]
    cols += [f"abs_{d}m" for d in DURATIONS]

    upsert_sql = f"""
        INSERT INTO kline_metrics ({', '.join(cols)}) VALUES %s
        ON CONFLICT (symbol, open_time) DO NOTHING
    """

    batch = []
    total = 0

    for candle in candles:
        if candle['open_time'] < start_from:
            continue
        if not candle['close']:
            continue

        row = [symbol, candle['open_time']]
        for d in DURATIONS:
            start_t = candle['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open'] and float(sc['open']) != 0:
                row.append(((float(candle['close']) - float(sc['open'])) / float(sc['open'])) * 100)
            else:
                row.append(None)
        for d in DURATIONS:
            start_t = candle['open_time'] - timedelta(minutes=d)
            sc = candle_map.get(start_t)
            if sc and sc['open']:
                row.append(float(candle['close']) - float(sc['open']))
            else:
                row.append(None)
        batch.append(tuple(row))

        if len(batch) >= 5000:
            with conn.cursor() as cur:
                execute_values(cur, upsert_sql, batch, page_size=1000)
            conn.commit()
            total += len(batch)
            batch = []

    if batch:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, batch, page_size=1000)
        conn.commit()
        total += len(batch)

    log(f"  {symbol}: Backfilled {total} rows")
    return total

# ============ MAIN LOOP ============

def main():
    log("=== kline_metrics Live Service (BERLIN-ZEIT) ===")
    log(f"Update interval: {UPDATE_INTERVAL_SEC}s")
    log(f"Normal lookback: {LOOKBACK_MINUTES} minutes")
    log(f"Initial lookback: {INITIAL_LOOKBACK_MINUTES} minutes")
    log(f"Daily check: {DAILY_CHECK_HOUR}:00 Berlin-Zeit")
    log(f"Gap check interval: {GAP_CHECK_INTERVAL_SEC}s ({GAP_CHECK_INTERVAL_SEC//3600}h)")

    conn = None
    last_daily_check = None
    last_gap_check = 0
    first_run = True

    while True:
        try:
            # Connection prüfen und ggf. neu erstellen
            if conn is None or conn.closed:
                log("[RECONNECT] Establishing database connection...")
                try:
                    conn = get_connection()
                    log("[RECONNECT] Connection established successfully")
                except Exception as e:
                    log(f"[RECONNECT] Failed to connect: {e}")
                    time.sleep(30)
                    continue

            cycle_start = now_berlin()
            now_ts = time.time()

            # Daily check um 03:00 Berlin-Zeit
            current_hour = cycle_start.hour
            if current_hour == DAILY_CHECK_HOUR:
                if last_daily_check is None or last_daily_check.date() < cycle_start.date():
                    check_new_coins(conn)
                    last_daily_check = cycle_start

            # 12h Lückenprüfung
            if now_ts - last_gap_check >= GAP_CHECK_INTERVAL_SEC:
                run_gap_check(conn)
                last_gap_check = now_ts

            # Lookback: beim ersten Lauf 2000 min, danach 660 min
            if first_run:
                lookback = INITIAL_LOOKBACK_MINUTES
                log(f"[INIT] First run - using {lookback} min lookback")
                first_run = False
            else:
                lookback = LOOKBACK_MINUTES

            # Alle Symbole updaten
            symbols = get_all_symbols(conn)
            total = 0

            for i, symbol in enumerate(symbols, 1):
                try:
                    count = update_symbol(conn, symbol, lookback)
                    total += count
                    if lookback > LOOKBACK_MINUTES:
                        log(f"[{i}/{len(symbols)}] {symbol}: {count}")
                except Exception as e:
                    log(f"  {symbol} ERROR: {e}")
                    try:
                        conn.rollback()
                    except:
                        pass

            duration = (now_berlin() - cycle_start).total_seconds()
            log(f"[CYCLE] {len(symbols)} symbols, {total} rows, {duration:.1f}s")

            sleep_time = max(UPDATE_INTERVAL_SEC - duration, 1)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log("Stopped by user")
            break
        except Exception as e:
            log(f"ERROR: {e} - will reconnect next cycle")
            conn = None
            time.sleep(30)

    if conn:
        conn.close()

if __name__ == '__main__':
    main()
