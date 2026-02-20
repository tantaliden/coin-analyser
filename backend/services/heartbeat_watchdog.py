#!/usr/bin/env python3
"""
Heartbeat Watchdog - Der aufpassende Kumpel
Tolerant bei einzelnen Aussetzern, erst nach mehreren Fehlern aktiv
"""
import urllib.request
import urllib.parse
import json
import subprocess
import os
import time
from datetime import datetime
import pytz

BERLIN_TZ = pytz.timezone('Europe/Berlin')

BOT_TOKEN = '8430890812:AAFAbxXkc9-hw19FMuaRQwN4d6s0O8rULrM'
CHAT_ID = '2112844328'

HEARTBEAT_FILE = '/opt/coin/logs/.health_heartbeat'
LOG_FILE = '/opt/coin/logs/heartbeat_watchdog.log'
ALERT_STATE_FILE = '/opt/coin/logs/.watchdog_alert_sent'

# Timing-Konfiguration
CHECK_INTERVAL_SECONDS = 60      # Watchdog prüft alle 60 Sek
HEARTBEAT_MAX_AGE_MINUTES = 3    # Heartbeat gilt als "alt" wenn > 3 Min
FAILURES_BEFORE_ACTION = 5       # Erst nach 5 aufeinanderfolgenden Fehlern aktiv
                                 # = ca. 5 Minuten ohne Lebenszeichen

def log(msg):
    timestamp = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass

def send_alert(text):
    """NUR senden, kein getUpdates!"""
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
        log(f"Send error: {e}")
        return False

def check_heartbeat():
    """Prüft ob Health-Bot noch lebt"""
    try:
        if not os.path.exists(HEARTBEAT_FILE):
            return False, "Keine Heartbeat-Datei!"
        
        mtime = os.path.getmtime(HEARTBEAT_FILE)
        age_minutes = (time.time() - mtime) / 60
        
        if age_minutes > HEARTBEAT_MAX_AGE_MINUTES:
            return False, f"{age_minutes:.1f} Min alt"
        
        return True, f"{age_minutes:.1f} Min"
    except Exception as e:
        return False, f"Fehler: {e}"

def restart_health_bot():
    """Startet Health-Bot neu"""
    try:
        log("Starte Health-Bot neu...")
        subprocess.run(['systemctl', 'restart', 'analyser-telegram-bot.service'], timeout=30)
        return True
    except Exception as e:
        log(f"Restart failed: {e}")
        return False

def restart_services():
    """Startet NUR den Telegram-Bot neu - der Bot ueberwacht alles andere"""
    try:
        log("Starte analyser-telegram-bot neu...")
        subprocess.run(['systemctl', 'restart', 'analyser-telegram-bot.service'], timeout=30)
        return True
    except Exception as e:
        log(f"Service restart failed: {e}")
        return False

def should_send_alert():
    try:
        if os.path.exists(ALERT_STATE_FILE):
            mtime = os.path.getmtime(ALERT_STATE_FILE)
            if (time.time() - mtime) / 60 < 30:
                return False
    except:
        pass
    return True

def mark_alert_sent():
    with open(ALERT_STATE_FILE, 'w') as f:
        f.write(datetime.now().isoformat())

def clear_alert():
    try:
        if os.path.exists(ALERT_STATE_FILE):
            os.remove(ALERT_STATE_FILE)
    except:
        pass

def main():
    log("=== Watchdog gestartet ===")
    log(f"Toleranz: {FAILURES_BEFORE_ACTION} Fehlprüfungen bevor Aktion")
    
    consecutive_failures = 0
    was_alerting = False
    restart_attempts = 0
    
    while True:
        try:
            ok, msg = check_heartbeat()
            
            if not ok:
                consecutive_failures += 1
                log(f"Heartbeat fehlt ({consecutive_failures}/{FAILURES_BEFORE_ACTION}): {msg}")
                
                # Erst nach X aufeinanderfolgenden Fehlern aktiv werden
                if consecutive_failures >= FAILURES_BEFORE_ACTION:
                    log(f"ALARM: {consecutive_failures} Fehlprüfungen - Health-Bot scheint tot!")
                    
                    # Health-Bot neustarten
                    restart_health_bot()
                    restart_attempts += 1
                    
                    # Nach 3 Restart-Versuchen auch Services neustarten
                    if restart_attempts >= 3:
                        log("3x Restart fehlgeschlagen, starte auch Services neu...")
                        restart_services()
                        restart_attempts = 0
                    
                    # Alert senden (max alle 30 Min)
                    if should_send_alert():
                        send_alert(
                            f"🛡️🚨 <b>WATCHDOG ALARM</b>\n\n"
                            f"Health-Bot antwortet nicht!\n"
                            f"Heartbeat: {msg}\n"
                            f"Fehlversuche: {consecutive_failures}\n\n"
                            f"➜ Health-Bot neugestartet\n\n"
                            f"🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}"
                        )
                        mark_alert_sent()
                    
                    was_alerting = True
                    # Nach Restart Zähler zurücksetzen, aber nicht auf 0
                    consecutive_failures = FAILURES_BEFORE_ACTION - 2
                
            else:
                # Heartbeat OK - Zähler zurücksetzen
                if consecutive_failures > 0:
                    log(f"Heartbeat OK ({msg}) - Zähler zurückgesetzt")
                consecutive_failures = 0
                restart_attempts = 0
                
                if was_alerting:
                    log("Health-Bot wieder aktiv!")
                    send_alert(
                        f"🛡️✅ <b>WATCHDOG OK</b>\n\n"
                        f"Health-Bot wieder aktiv!\n"
                        f"Heartbeat: {msg}\n\n"
                        f"🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}"
                    )
                    clear_alert()
                    was_alerting = False
            
            time.sleep(CHECK_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(30)

if __name__ == '__main__':
    main()
