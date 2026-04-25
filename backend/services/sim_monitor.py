#!/usr/bin/env python3
"""
Sim-Monitor: Prüft alle 30 Min ob ein neuer Sim-Tag abgeschlossen wurde.
Pushed Tages- und Gesamtstatistik via Telegram.
"""

import json
import time
import signal
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

SETTINGS_PATH = "/opt/coin/settings.json"
BOT_TOKEN = "8430890812:AAFAbxXkc9-hw19FMuaRQwN4d6s0O8rULrM"
CHAT_ID = "2112844328"
CHECK_INTERVAL = 1800  # 30 Minuten
SIM_START = datetime(2026, 1, 1)

running = True
last_reported_day = None


def signal_handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_conn():
    s = json.load(open(SETTINGS_PATH))
    db = s["databases"]["app"]
    return psycopg2.connect(
        dbname=db["name"], user=db["user"], password=db["password"],
        host=db["host"], port=db["port"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")


def check_and_report():
    global last_reported_day

    conn = get_conn()
    cur = conn.cursor()

    # Aktueller Sim-Tag
    cur.execute("""
        SELECT MAX(detected_at) FROM momentum_predictions
        WHERE scanner_type = 'default' OR scanner_type IS NULL
    """)
    last = cur.fetchone()["max"]
    if not last:
        conn.close()
        return

    current_day = (last.replace(tzinfo=None) - SIM_START).days + 1
    current_date = last.replace(tzinfo=None).date()

    # Prüfe ob ein neuer Tag abgeschlossen wurde
    # Ein Tag gilt als abgeschlossen wenn der Scanner schon im nächsten Tag ist
    completed_date = current_date - timedelta(days=1)
    completed_day = (completed_date - SIM_START.date()).days + 1

    if last_reported_day is not None and completed_day <= last_reported_day:
        conn.close()
        return

    # Tages-Stats für den abgeschlossenen Tag
    cur.execute("""
        SELECT direction, status, COUNT(*) as cnt FROM momentum_predictions
        WHERE (scanner_type = 'default' OR scanner_type IS NULL)
        AND status IN ('hit_tp', 'hit_sl')
        AND detected_at >= %s AND detected_at < %s
        GROUP BY direction, status
    """, (str(completed_date), str(completed_date + timedelta(days=1))))

    day_data = {"long": {"hit_tp": 0, "hit_sl": 0}, "short": {"hit_tp": 0, "hit_sl": 0}}
    for row in cur.fetchall():
        day_data[row["direction"]][row["status"]] = row["cnt"]

    day_tp = day_data["long"]["hit_tp"] + day_data["short"]["hit_tp"]
    day_sl = day_data["long"]["hit_sl"] + day_data["short"]["hit_sl"]
    day_total = day_tp + day_sl

    if day_total == 0:
        conn.close()
        return

    day_hr = day_tp / day_total * 100

    # Long/Short Tag
    lt, ls = day_data["long"]["hit_tp"], day_data["long"]["hit_sl"]
    st, ss = day_data["short"]["hit_tp"], day_data["short"]["hit_sl"]
    day_long_hr = (lt / (lt + ls) * 100) if (lt + ls) > 0 else 0
    day_short_hr = (st / (st + ss) * 100) if (st + ss) > 0 else 0

    # Gesamt-Stats
    cur.execute("""
        SELECT status, COUNT(*) as cnt FROM momentum_predictions
        WHERE (scanner_type = 'default' OR scanner_type IS NULL)
        AND status IN ('hit_tp', 'hit_sl')
        GROUP BY status
    """)
    total_tp = total_sl = 0
    for row in cur.fetchall():
        if row["status"] == "hit_tp":
            total_tp = row["cnt"]
        else:
            total_sl = row["cnt"]

    total_hr = (total_tp / (total_tp + total_sl) * 100) if (total_tp + total_sl) > 0 else 0

    # Gesamt Long/Short
    cur.execute("""
        SELECT direction, status, COUNT(*) as cnt FROM momentum_predictions
        WHERE (scanner_type = 'default' OR scanner_type IS NULL)
        AND status IN ('hit_tp', 'hit_sl')
        GROUP BY direction, status
    """)
    g = {"long": {"hit_tp": 0, "hit_sl": 0}, "short": {"hit_tp": 0, "hit_sl": 0}}
    for row in cur.fetchall():
        g[row["direction"]][row["status"]] = row["cnt"]

    glt, gls = g["long"]["hit_tp"], g["long"]["hit_sl"]
    gst, gss = g["short"]["hit_tp"], g["short"]["hit_sl"]
    g_long_hr = (glt / (glt + gls) * 100) if (glt + gls) > 0 else 0
    g_short_hr = (gst / (gst + gss) * 100) if (gst + gss) > 0 else 0

    conn.close()

    # Emoji basierend auf Tages-Performance
    if day_hr >= 80:
        emoji = "🟢"
    elif day_hr >= 70:
        emoji = "🟡"
    elif day_hr >= 60:
        emoji = "🟠"
    else:
        emoji = "🔴"

    msg = (
        f"{emoji} <b>Sim Tag {completed_day} — {completed_date.strftime('%d.%m.%Y')}</b>\n"
        f"\n"
        f"<b>Tag:</b> {day_hr:.1f}% ({day_tp} TP / {day_sl} SL)\n"
        f"  Long: {day_long_hr:.1f}% | Short: {day_short_hr:.1f}%\n"
        f"\n"
        f"<b>Gesamt:</b> {total_hr:.1f}% ({total_tp} TP / {total_sl} SL)\n"
        f"  Long: {g_long_hr:.1f}% | Short: {g_short_hr:.1f}%\n"
        f"  Total: {total_tp + total_sl} Predictions"
    )

    send_telegram(msg)
    last_reported_day = completed_day
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tag {completed_day} ({completed_date}) reported: {day_hr:.1f}%")


def main():
    global last_reported_day

    print(f"Sim-Monitor gestartet. Check alle {CHECK_INTERVAL}s.")

    # Initialen Stand ermitteln um nicht alte Tage nachzupushen
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(detected_at) FROM momentum_predictions
            WHERE scanner_type = 'default' OR scanner_type IS NULL
        """)
        last = cur.fetchone()["max"]
        if last:
            current_date = last.replace(tzinfo=None).date()
            completed_date = current_date - timedelta(days=1)
            last_reported_day = (completed_date - SIM_START.date()).days + 1
            print(f"Start bei Tag {last_reported_day} ({completed_date})")
        conn.close()
    except Exception as e:
        print(f"Init error: {e}")

    while running:
        try:
            check_and_report()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(CHECK_INTERVAL)

    print("Sim-Monitor gestoppt.")


if __name__ == "__main__":
    main()
