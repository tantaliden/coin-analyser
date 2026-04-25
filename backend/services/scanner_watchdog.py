#!/usr/bin/env python3
"""
Scanner Watchdog — Überwacht beide Momentum-Scanner (default + 2h).
Bei Ausfall: Restart versuchen. Bei anhaltendem Ausfall: Telegram-Alert.
"""
import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime
import pytz

BERLIN_TZ = pytz.timezone('Europe/Berlin')

BOT_TOKEN = '8430890812:AAFAbxXkc9-hw19FMuaRQwN4d6s0O8rULrM'
CHAT_ID = '2112844328'

LOG_FILE = '/opt/coin/logs/scanner_watchdog.log'

# Überwachte Scanner
SCANNERS = {
    'default': {
        'heartbeat': '/opt/coin/logs/.scanner_heartbeat',
        'service': 'momentum-scanner.service',
        'label': 'Scanner (Default)',
    },
    'cnn_2h': {
        'heartbeat': '/opt/coin/logs/.scanner_2h_heartbeat',
        'service': 'momentum-scanner-2h.service',
        'label': 'Scanner (2h)',
    },
}

# Timing
CHECK_INTERVAL_SECONDS = 60
HEARTBEAT_MAX_AGE_MINUTES = 5
FAILURES_BEFORE_RESTART = 3
FAILURES_BEFORE_ALERT = 5
ALERT_COOLDOWN_MINUTES = 30


def log(msg):
    timestamp = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass


def send_telegram(text):
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
        log(f"Telegram send error: {e}")
        return False


def check_heartbeat(hb_path):
    try:
        if not os.path.exists(hb_path):
            return False, "Keine Heartbeat-Datei"
        age_minutes = (time.time() - os.path.getmtime(hb_path)) / 60
        if age_minutes > HEARTBEAT_MAX_AGE_MINUTES:
            return False, f"{age_minutes:.1f} Min alt"
        return True, f"{age_minutes:.1f} Min"
    except Exception as e:
        return False, f"Fehler: {e}"


def restart_service(service_name):
    try:
        log(f"Starte {service_name} neu...")
        result = subprocess.run(['systemctl', 'restart', service_name], timeout=30, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"{service_name} neugestartet")
            return True
        else:
            log(f"Restart fehlgeschlagen: {result.stderr}")
            return False
    except Exception as e:
        log(f"Restart error: {e}")
        return False


def main():
    log("=== Scanner Watchdog gestartet ===")
    log(f"Überwache: {', '.join(s['label'] for s in SCANNERS.values())}")
    log(f"Toleranz: {FAILURES_BEFORE_RESTART} Fehler → Restart, {FAILURES_BEFORE_ALERT} → Telegram")

    # State pro Scanner
    state = {}
    for key in SCANNERS:
        state[key] = {
            'consecutive_failures': 0,
            'restart_attempts': 0,
            'last_alert_time': 0,
            'was_alerting': False,
        }

    while True:
        try:
            for scanner_key, scanner_cfg in SCANNERS.items():
                s = state[scanner_key]
                ok, msg = check_heartbeat(scanner_cfg['heartbeat'])

                if not ok:
                    s['consecutive_failures'] += 1
                    label = scanner_cfg['label']

                    if s['consecutive_failures'] == FAILURES_BEFORE_RESTART:
                        log(f"[{label}] {s['consecutive_failures']} Fehler — versuche Restart")

                    # Nach FAILURES_BEFORE_RESTART: Restart versuchen
                    if s['consecutive_failures'] >= FAILURES_BEFORE_RESTART:
                        restart_service(scanner_cfg['service'])
                        s['restart_attempts'] += 1

                        # Nach Restart: Counter leicht zurücksetzen
                        s['consecutive_failures'] = FAILURES_BEFORE_RESTART - 1

                    # Nach FAILURES_BEFORE_ALERT: Telegram senden
                    if s['consecutive_failures'] >= FAILURES_BEFORE_ALERT:
                        now = time.time()
                        if (now - s['last_alert_time']) / 60 >= ALERT_COOLDOWN_MINUTES:
                            send_telegram(
                                f"🔴 <b>SCANNER WATCHDOG</b>\n\n"
                                f"<b>{label}</b> antwortet nicht!\n"
                                f"Heartbeat: {msg}\n"
                                f"Restart-Versuche: {s['restart_attempts']}\n\n"
                                f"⚠️ Manueller Eingriff nötig!\n\n"
                                f"🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}"
                            )
                            s['last_alert_time'] = now
                            s['was_alerting'] = True
                            log(f"[{label}] TELEGRAM ALERT gesendet — manueller Eingriff nötig")

                    if s['consecutive_failures'] <= FAILURES_BEFORE_RESTART:
                        log(f"[{label}] Heartbeat fehlt ({s['consecutive_failures']}/{FAILURES_BEFORE_RESTART}): {msg}")

                else:
                    # OK
                    if s['consecutive_failures'] > 0:
                        log(f"[{scanner_cfg['label']}] Heartbeat OK ({msg}) — Zähler zurückgesetzt")
                    if s['was_alerting']:
                        send_telegram(
                            f"🟢 <b>SCANNER OK</b>\n\n"
                            f"<b>{scanner_cfg['label']}</b> wieder aktiv!\n"
                            f"Heartbeat: {msg}\n\n"
                            f"🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}"
                        )
                        log(f"[{scanner_cfg['label']}] Recovery — Telegram OK gesendet")
                    s['consecutive_failures'] = 0
                    s['restart_attempts'] = 0
                    s['was_alerting'] = False

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(30)


if __name__ == '__main__':
    main()
