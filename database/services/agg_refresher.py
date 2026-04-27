#!/usr/bin/env python3
"""
Continuous Aggregate Refresher — Exakte Refresh-Zeiten.

Refresht jedes Agg ab Sekunde 5 nach Bucket-Ende, damit klines_10s schon
finalisiert sind (bucket_grace 1s + flush 0.5s + DB-Insert ~2s).
Ergaenzt die TimescaleDB Auto-Policies fuer schnellere Sichtbarkeit
(insbesondere fuer den AKTUELL laufenden 1h/4h/1d-Bucket).

kline_metrics laeuft als eigener Service (kline-metrics-updater.service).

Usage:
    systemctl start agg-refresher
"""
import time
import sys
import json
import psycopg2
from datetime import datetime
from pathlib import Path

SETTINGS_PATH = "/opt/coin/settings.json"

# Refresh-Schedule: Sekunde 5 jeder relevanten Minute (klein zuerst, gross folgt sequentiell).
# Key = agg_name, Value = set von Minuten in der Stunde wo refresht wird.
# Reihenfolge wichtig: agg_1m zuerst, danach 5m, 15m, 30m, 1h, 4h, 1d.
# Stunden-Aggs (1h, 4h, 1d) werden auch jede Stunde refresht damit der laufende Bucket sichtbar ist.
SCHEDULE = {
    'agg_1m':  set(range(0, 60)),              # jede Minute
    'agg_5m':  set(range(0, 60, 5)),           # alle 5 Min
    'agg_15m': set(range(0, 60, 5)),           # alle 5 Min (laufender 15m-Bucket)
    'agg_30m': set(range(0, 60, 5)),           # alle 5 Min (laufender 30m-Bucket)
    'agg_1h':  set(range(0, 60, 5)),           # alle 5 Min (laufender 1h-Bucket)
    'agg_4h':  set(range(0, 60, 15)),          # alle 15 Min (laufender 4h-Bucket)
    'agg_1d':  {0, 30},                        # alle 30 Min (laufender 1d-Bucket)
}

# Lookback pro Agg (wie weit zurueck refreshen)
LOOKBACK = {
    'agg_1m':  '5 minutes',
    'agg_5m':  '2 hours',
    'agg_15m': '2 hours',
    'agg_30m': '3 hours',
    'agg_1h':  '4 hours',
    'agg_4h':  '12 hours',
    'agg_1d':  '3 days',
}

# Keine HOUR_FILTER mehr - 1h/4h/1d sollen jede Stunde refresht werden, damit
# der aktuell laufende Bucket bereits sichtbar ist (statt erst nach Bucket-Ende).
HOUR_FILTER = {}


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


def main():
    print("=" * 60)
    print("  Continuous Aggregate Refresher")
    print("  Trigger: Sekunde 5 nach Minutenwechsel")
    print("=" * 60)

    conn = get_conn()
    last_refreshed = {}  # agg_name -> (hour, minute) des letzten Refresh

    # Initial-Refresh ALLER Aggs beim Start, damit laufende Buckets sofort sichtbar sind.
    # Wichtig nach Service-Restart oder fuer Aggs die lange nicht refresht wurden (z.B. agg_1d).
    print("[INIT] Initial refresh aller Aggs...")
    for agg_name, lookback in LOOKBACK.items():
        if refresh_agg(conn, agg_name, lookback):
            print(f"[INIT] {agg_name} ok (lookback {lookback})")
        else:
            print(f"[INIT] {agg_name} FAILED")
    print("[INIT] done.")

    while True:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second

        # Trigger-Fenster: Sekunde 5-30 (genug Reserve falls ein Refresh laenger dauert)
        if second >= 5:
            refreshed_this_tick = []

            for agg_name, minutes in SCHEDULE.items():
                if minute not in minutes:
                    continue

                if agg_name in HOUR_FILTER:
                    if hour not in HOUR_FILTER[agg_name]:
                        continue

                key = (hour, minute)
                if last_refreshed.get(agg_name) == key:
                    continue

                lookback = LOOKBACK[agg_name]
                ok = refresh_agg(conn, agg_name, lookback)
                if ok:
                    last_refreshed[agg_name] = key
                    refreshed_this_tick.append(agg_name)

            if refreshed_this_tick:
                print(f"[{now.strftime('%H:%M:%S')}] Refreshed: {', '.join(refreshed_this_tick)}")

        # DB-Connection pruefen (zur vollen Stunde, vor dem Trigger-Fenster)
        if now.minute == 0 and now.second < 3:
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
