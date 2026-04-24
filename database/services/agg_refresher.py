#!/usr/bin/env python3
"""
Continuous Aggregate Refresher + kline_metrics — Exakte Refresh-Zeiten.

Refresht jedes Agg + kline_metrics genau 45 Sekunden nach Bucket-Ende.
Ersetzt die TimescaleDB Auto-Policies fuer praezisere Kontrolle.

Usage:
    nohup /opt/coin/venv/bin/python3 /opt/coin/database/services/agg_refresher.py &
"""
import time
import sys
import json
import psycopg2
from datetime import datetime
from pathlib import Path

# kline_metrics Funktionen importieren
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kline_metrics_live import update_symbol as update_metrics_symbol, get_all_symbols as get_metrics_symbols, get_connection as get_metrics_conn

SETTINGS_PATH = "/opt/coin/settings.json"

# Refresh-Schedule: Sekunde 38 jeder relevanten Minute
# Key = agg_name, Value = set von Minuten in der Stunde wo refresht wird
# Agent-Aggs ZUERST (5m, 1h, 4h, 1d), dann Rest
SCHEDULE = {
    'agg_5m':  set(range(0, 60, 5)),           # Agent: Entry-Daten
    'agg_1h':  {0},                            # Agent: Multi-TF
    'agg_4h':  {0},                            # Agent: Multi-TF
    'agg_1d':  {0},                            # Agent: Daily
    'agg_2m':  set(range(0, 60, 2)),           # 0,2,4,...,58
    'agg_10m': set(range(0, 60, 10)),          # 0,10,20,30,40,50
    'agg_15m': {0, 15, 30, 45},
    'agg_30m': {0, 30},
    'agg_45m': {0, 15, 30, 45},
    'agg_2h':  {0},                            # nur gerade Stunden
}

# Lookback pro Agg (wie weit zurueck refreshen)
LOOKBACK = {
    'agg_2m':  '1 hour',
    'agg_5m':  '2 hours',
    'agg_10m': '2 hours',
    'agg_15m': '2 hours',
    'agg_30m': '3 hours',
    'agg_45m': '3 hours',
    'agg_1h':  '4 hours',
    'agg_2h':  '6 hours',
    'agg_4h':  '12 hours',
    'agg_1d':  '3 days',
}

# agg_2h nur bei geraden Stunden, agg_4h nur bei 0,4,8,12,16,20
HOUR_FILTER = {
    'agg_2h': set(range(0, 24, 2)),
    'agg_4h': {0, 4, 8, 12, 16, 20},
    'agg_1d': {0},
}


def get_conn():
    """Verbindet als postgres (Owner der Continuous Aggregates)."""
    conn = psycopg2.connect(dbname='coins', user='postgres', host='/var/run/postgresql')
    conn.autocommit = True
    return conn


def refresh_agg(conn, agg_name, lookback):
    """Einzelnes Agg refreshen."""
    try:
        cur = conn.cursor()
        cur.execute(f"""
            CALL refresh_continuous_aggregate('{agg_name}',
                NOW()::timestamp - INTERVAL '{lookback}',
                NOW()::timestamp)
        """)
        cur.close()
        return True
    except Exception as e:
        print(f"[AGG-REFRESH] FEHLER {agg_name}: {e}")
        return False


def run_metrics_update(metrics_conn):
    """kline_metrics fuer alle Symbole aktualisieren."""
    try:
        symbols = get_metrics_symbols(metrics_conn)
        total = 0
        for symbol in symbols:
            try:
                count = update_metrics_symbol(metrics_conn, symbol, 60)
                total += count
            except Exception as e:
                try:
                    metrics_conn.rollback()
                except:
                    pass
        return total
    except Exception as e:
        print(f"[METRICS] FEHLER: {e}")
        return -1


def main():
    print("=" * 60)
    print("  Continuous Aggregate + kline_metrics Refresher")
    print("  Exakte Zeiten: Bucket-Ende + 45 Sekunden")
    print("=" * 60)

    conn = get_conn()
    metrics_conn = get_metrics_conn()
    last_refreshed = {}  # agg_name -> (hour, minute) des letzten Refresh
    last_metrics = None  # (hour, minute) des letzten Metrics-Updates

    while True:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second

        # Nur bei Sekunde 38-55 pruefen (Fenster fuer den Refresh)
        if second >= 35:
            refreshed_this_tick = []

            for agg_name, minutes in SCHEDULE.items():
                if minute not in minutes:
                    continue

                # Stunden-Filter (fuer 2h, 4h, 1d)
                if agg_name in HOUR_FILTER:
                    if hour not in HOUR_FILTER[agg_name]:
                        continue

                # Schon in dieser Minute refresht?
                key = (hour, minute)
                if last_refreshed.get(agg_name) == key:
                    continue

                # Refresh ausfuehren
                lookback = LOOKBACK[agg_name]
                ok = refresh_agg(conn, agg_name, lookback)
                if ok:
                    last_refreshed[agg_name] = key
                    refreshed_this_tick.append(agg_name)

            # kline_metrics bei jeder geraden Minute aktualisieren
            metrics_key = (hour, minute)
            if minute % 2 == 0 and last_metrics != metrics_key:
                m_count = run_metrics_update(metrics_conn)
                if m_count >= 0:
                    last_metrics = metrics_key
                    refreshed_this_tick.append(f"metrics({m_count})")
                else:
                    # Reconnect bei Fehler
                    try:
                        metrics_conn.close()
                    except:
                        pass
                    metrics_conn = get_metrics_conn()

            if refreshed_this_tick:
                print(f"[{now.strftime('%H:%M:%S')}] Refreshed: {', '.join(refreshed_this_tick)}")

        # DB-Connection pruefen
        if now.minute == 0 and now.second < 5:
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            except Exception:
                print("[AGG-REFRESH] DB reconnect...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_conn()

        # Schlafen bis naechste Sekunde
        elapsed = time.time() % 1
        time.sleep(max(0.1, 1.0 - elapsed))


if __name__ == '__main__':
    main()
