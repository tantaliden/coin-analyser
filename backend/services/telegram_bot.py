#!/usr/bin/env python3
"""
Analyser Health-Bot - Der fleißige Kumpel
- Telegram-Befehle lesen & verarbeiten
- Health-Checks durchführen
- Heartbeat schreiben (beweist: ich lebe!)
- Watchdog-Service überwachen
"""
import urllib.request
import urllib.parse
import json
import subprocess
import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
import pytz
import psycopg2

BERLIN_TZ = pytz.timezone('Europe/Berlin')

BOT_TOKEN = '8430890812:AAFAbxXkc9-hw19FMuaRQwN4d6s0O8rULrM'
CHAT_ID = '2112844328'

PASSWORD_FILE = '/opt/coin/backend/services/.bot_password'
SESSION_FILE = '/opt/coin/backend/services/.bot_session'
LOCKOUT_FILE = '/opt/coin/backend/services/.bot_lockout'
HEARTBEAT_FILE = '/opt/coin/logs/.health_heartbeat'
LOG_FILE = '/opt/coin/logs/telegram_bot.log'
ALERT_STATE_FILE = '/opt/coin/logs/.alert_sent'

SESSION_TIMEOUT = 3600
MAX_ATTEMPTS = 3
LOCKOUT_MINUTES = 5
HEALTH_CHECK_INTERVAL = 300  # 5 Minuten
HEARTBEAT_WRITE_INTERVAL = 60  # Heartbeat alle 60 Sek wenn Telegram OK

# Health Thresholds
METRICS_WARNING_MIN = 10
METRICS_CRITICAL_MIN = 30
CPU_WARNING = 80
RAM_WARNING = 85
DISK_WARNING = 85

# DB Config laden
with open('/opt/coin/database/settings.json') as f:
    SETTINGS = json.load(f)

DB_CONFIG = {
    'host': SETTINGS['databases']['coins']['host'],
    'port': SETTINGS['databases']['coins']['port'],
    'dbname': SETTINGS['databases']['coins']['name'],
    'user': SETTINGS['ingestor']['database']['user'],
    'password': SETTINGS['ingestor']['database']['password']
}

def log(msg):
    timestamp = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass

def write_heartbeat():
    """Heartbeat schreiben - beweist Watchdog dass wir leben"""
    try:
        with open(HEARTBEAT_FILE, 'w') as f:
            f.write(datetime.now(BERLIN_TZ).isoformat())
    except:
        pass

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_password():
    try:
        with open(PASSWORD_FILE, 'r') as f:
            return f.read().strip()
    except:
        save_password('1234')
        return hash_password('1234')

def save_password(new_pw):
    with open(PASSWORD_FILE, 'w') as f:
        f.write(hash_password(new_pw))
    os.chmod(PASSWORD_FILE, 0o600)

def is_locked_out():
    try:
        if os.path.exists(LOCKOUT_FILE):
            with open(LOCKOUT_FILE, 'r') as f:
                data = json.load(f)
            if data.get('attempts', 0) >= MAX_ATTEMPTS:
                elapsed = time.time() - data.get('time', 0)
                if elapsed < LOCKOUT_MINUTES * 60:
                    return True, int((LOCKOUT_MINUTES * 60 - elapsed) / 60) + 1
                else:
                    clear_lockout()
    except:
        pass
    return False, 0

def add_failed_attempt():
    try:
        data = {'attempts': 0, 'time': time.time()}
        if os.path.exists(LOCKOUT_FILE):
            with open(LOCKOUT_FILE, 'r') as f:
                data = json.load(f)
        data['attempts'] = data.get('attempts', 0) + 1
        data['time'] = time.time()
        with open(LOCKOUT_FILE, 'w') as f:
            json.dump(data, f)
        return data['attempts']
    except:
        return 0

def clear_lockout():
    try:
        if os.path.exists(LOCKOUT_FILE):
            os.remove(LOCKOUT_FILE)
    except:
        pass

def is_authenticated(chat_id):
    try:
        with open(SESSION_FILE, 'r') as f:
            data = json.load(f)
            if data.get('chat_id') == chat_id:
                if time.time() - data.get('time', 0) < SESSION_TIMEOUT:
                    return True
    except:
        pass
    return False

def set_authenticated(chat_id):
    with open(SESSION_FILE, 'w') as f:
        json.dump({'chat_id': chat_id, 'time': time.time()}, f)
    os.chmod(SESSION_FILE, 0o600)

def send_message(text):
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

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=30"
        if offset:
            url += f"&offset={offset}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=35) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('result', [])
    except Exception as e:
        log(f"GetUpdates error: {e}")
        time.sleep(5)
        return None  # None = Fehler, [] = keine Nachrichten

def run_command(cmd, timeout=30):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "⏱ Timeout!"
    except Exception as e:
        return f"Fehler: {e}"

# ============ HEALTH CHECKS ============

def check_service(name):
    try:
        result = subprocess.run(['systemctl', 'is-active', name], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() == 'active'
    except:
        return False

def check_kline_metrics():
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute("SELECT MAX(open_time) FROM kline_metrics")
            result = cur.fetchone()
        conn.close()
        
        if result[0] is None:
            return 'CRITICAL', 'Keine Daten!'
        
        last_data = result[0]
        if last_data.tzinfo is None:
            last_data = BERLIN_TZ.localize(last_data)
        
        age_minutes = (datetime.now(BERLIN_TZ) - last_data).total_seconds() / 60
        
        if age_minutes > METRICS_CRITICAL_MIN:
            return 'CRITICAL', f'{age_minutes:.0f} min alt!'
        elif age_minutes > METRICS_WARNING_MIN:
            return 'WARNING', f'{age_minutes:.0f} min alt'
        return 'OK', f'{age_minutes:.0f} min'
    except Exception as e:
        return 'CRITICAL', f'DB: {e}'

def check_agg_freshness():
    """Prueft ob agg_5m, agg_1h, agg_4h aktuell sind"""
    results = []
    thresholds = {
        'agg_5m':  (12, 20),
        'agg_1h':  (125, 180),
        'agg_4h':  (485, 600),
    }
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            for table, (warn_min, crit_min) in thresholds.items():
                cur.execute(f"SELECT MAX(bucket) FROM {table}")
                result = cur.fetchone()
                if result[0] is None:
                    results.append(('CRITICAL', f'{table}: Keine Daten!'))
                    continue
                last = result[0]
                if last.tzinfo is None:
                    last = BERLIN_TZ.localize(last)
                age = (datetime.now(BERLIN_TZ) - last).total_seconds() / 60
                if age > crit_min:
                    results.append(('CRITICAL', f'{table}: {age:.0f} min alt!'))
                elif age > warn_min:
                    results.append(('WARNING', f'{table}: {age:.0f} min alt'))
        conn.close()
    except Exception as e:
        results.append(('CRITICAL', f'Agg-Check DB: {e}'))
    return results

def check_cpu():
    try:
        with open('/proc/loadavg', 'r') as f:
            load = float(f.read().split()[0])
        cpu_count = os.cpu_count() or 1
        pct = (load / cpu_count) * 100
        if pct > CPU_WARNING:
            return 'WARNING', f'CPU {pct:.0f}%'
        return 'OK', f'CPU {pct:.0f}%'
    except:
        return 'OK', 'CPU ?'

def check_ram():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                mem[parts[0].rstrip(':')] = int(parts[1])
        total = mem.get('MemTotal', 1)
        available = mem.get('MemAvailable', 0)
        pct = ((total - available) / total) * 100
        if pct > RAM_WARNING:
            return 'WARNING', f'RAM {pct:.0f}%'
        return 'OK', f'RAM {pct:.0f}%'
    except:
        return 'OK', 'RAM ?'

def check_disk():
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            pct = int(parts[4].rstrip('%'))
            if pct > DISK_WARNING:
                return 'WARNING', f'SSD {pct}%'
            return 'OK', f'SSD {pct}%'
    except:
        pass
    return 'OK', 'SSD ?'

def check_watchdog_service():
    """Prüft ob Watchdog-Service läuft, startet ihn ggf. neu"""
    if not check_service('heartbeat-watchdog.service'):
        log("Watchdog-Service tot, starte neu...")
        subprocess.run(['systemctl', 'restart', 'heartbeat-watchdog.service'], timeout=30)
        return False
    return True

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

def clear_alert_state():
    try:
        if os.path.exists(ALERT_STATE_FILE):
            os.remove(ALERT_STATE_FILE)
    except:
        pass

def run_health_check():
    """Führt kompletten Health-Check durch"""
    log("=== Health Check ===")
    
    problems = []
    
    # Service Check - critical services
    critical_services = [
        'analyser-ingestor', 'agg-refresher',
        'coin-analyser-api', 'coin-info-updater',
        'rl-agent',
    ]
    for svc in critical_services:
        if not check_service(f'{svc}.service'):
            problems.append(f"Service: {svc} DOWN!")
    
    # Metrics Check
    status, msg = check_kline_metrics()
    if status != 'OK':
        problems.append(f"Metrics: {msg}")

    # Agg Freshness Check
    for status, msg in check_agg_freshness():
        if status != 'OK':
            problems.append(msg)

    # Learner Check
    status, msg = check_learner_health()
    if status == 'CRITICAL':
        problems.append(f"Learner: {msg}")

    # Server Checks
    status, msg = check_cpu()
    if status != 'OK':
        problems.append(msg)
    
    status, msg = check_ram()
    if status != 'OK':
        problems.append(msg)
    
    status, msg = check_disk()
    if status != 'OK':
        problems.append(msg)
    
    # Watchdog prüfen
    check_watchdog_service()
    
    if problems:
        log(f"Problems: {problems}")
        if should_send_alert():
            emoji = "🚨" if any('CRITICAL' in p or 'DOWN' in p for p in problems) else "⚠️"
            send_message(f"{emoji} <b>HEALTH ALERT</b>\n\n" + "\n".join(problems) + f"\n\n🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}")
            mark_alert_sent()
    else:
        if os.path.exists(ALERT_STATE_FILE):
            send_message(f"✅ <b>HEALTH OK</b>\n\nAlles wieder normal.\n\n🕐 {datetime.now(BERLIN_TZ).strftime('%H:%M')}")
        clear_alert_state()
        log("Health OK")

# ============ TELEGRAM COMMANDS ============

def cmd_help():
    return """🤖 <b>Analyser Bot</b>

<b>Ohne Login:</b>
help - Hilfe
login [pw]

<b>Nach Login:</b>
status - System-Status
health - Health-Check jetzt
balance - Kontostand (BIN + HL)
metrics - Metrics Coverage (24h)
backfill - Backfill auslösen
logs - Letzte Logs
restart - Services neustarten
agent - RL-Agent Status
agent on/off - Agent starten/stoppen
agent max N - Max Positionen
agent trade N - Trade-Size ($, ab 15)
logout
changepw [alt] [neu]

⏱ Session: 1h | 🔒 3 Versuche"""

def cmd_status():
    output = run_command('/opt/coin/database/status.sh', timeout=15)
    if len(output) > 3500:
        output = output[:3500] + "\n..."
    return f"<pre>{output}</pre>"

def check_learner_health() -> tuple:
    """Prüft ob der Learner in den letzten 26h mindestens 2x gelaufen ist."""
    health_log = '/opt/coin/logs/learner_health.json'
    try:
        if not os.path.exists(health_log):
            return 'WARNING', 'Kein Health-Log'
        with open(health_log) as f:
            entries = json.loads(f.read())
        if not entries:
            return 'WARNING', 'Log leer'

        cutoff = datetime.now(timezone.utc) - timedelta(hours=26)
        recent = [e for e in entries if datetime.fromisoformat(e['timestamp']) > cutoff]
        ok_runs = [e for e in recent if e.get('status') == 'ok']

        last = entries[-1]
        last_time = datetime.fromisoformat(last['timestamp'])
        age_h = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600

        if len(ok_runs) >= 2:
            return 'OK', f'{len(ok_runs)}x in 26h (letzter vor {age_h:.0f}h)'
        elif len(ok_runs) == 1:
            return 'WARNING', f'Nur 1x in 26h (vor {age_h:.0f}h)'
        else:
            return 'CRITICAL', f'0x in 26h (letzter vor {age_h:.0f}h)'
    except Exception as e:
        return 'WARNING', f'Fehler: {e}'


def cmd_health():
    """Manueller Health-Check"""
    lines = []

    # Services
    services = [
        'analyser-ingestor', 'agg-refresher',
        'analyser-telegram-bot', 'heartbeat-watchdog',
        'coin-info-updater',
        'coin-analyser-api', 'coin-analyser-frontend',
        'rl-agent',
    ]
    for svc in services:
        ok = check_service(f'{svc}.service')
        emoji = "✅" if ok else "❌"
        lines.append(f"{emoji} {svc}")

    # Learner Health
    status, msg = check_learner_health()
    emoji = "✅" if status == 'OK' else "⚠️" if status == 'WARNING' else "❌"
    lines.append(f"{emoji} Learner: {msg}")

    # Metrics
    status, msg = check_kline_metrics()
    emoji = "✅" if status == 'OK' else "⚠️" if status == 'WARNING' else "❌"
    lines.append(f"{emoji} Metrics: {msg}")

    # Server
    _, msg = check_cpu()
    lines.append(f"💻 {msg}")
    _, msg = check_ram()
    lines.append(f"💻 {msg}")
    _, msg = check_disk()
    lines.append(f"💻 {msg}")

    return "🏥 <b>Health Check</b>\n\n" + "\n".join(lines)

def cmd_metrics():
    """Zeigt Metrics Coverage für die letzten 24h"""
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15s'")
            # Gesamt Zeilen vs befüllte in letzten 24h
            cur.execute("""
                SELECT COUNT(*) as total,
                       COUNT(pct_120m) as filled,
                       COUNT(*) - COUNT(pct_120m) as gaps
                FROM kline_metrics
                WHERE open_time >= NOW() - INTERVAL '24 hours'
            """)
            total, filled, gaps = cur.fetchone()
            
            pct = (filled / total * 100) if total > 0 else 0
            
            # Top 5 Symbole mit meisten Lücken
            cur.execute("""
                SELECT symbol, COUNT(*) as gap_count
                FROM kline_metrics
                WHERE open_time >= NOW() - INTERVAL '24 hours'
                  AND pct_120m IS NULL
                GROUP BY symbol
                ORDER BY COUNT(*) DESC LIMIT 5
            """)
            top_gaps = cur.fetchall()
        conn.close()
        
        emoji = "✅" if pct >= 99 else "⚠️" if pct >= 90 else "❌"
        lines = [f"{emoji} <b>Metrics Coverage (24h)</b>", ""]
        lines.append(f"Total: {total:,} Zeilen")
        lines.append(f"Befüllt: {filled:,} ({pct:.1f}%)")
        lines.append(f"Lücken: {gaps:,}")
        
        if top_gaps:
            lines.append("")
            lines.append("<b>Top Lücken:</b>")
            for sym, cnt in top_gaps:
                lines.append(f"  {sym}: {cnt}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Fehler: {e}"

def cmd_backfill():
    """Löst manuell einen Gap-Check + Backfill aus"""
    ps = run_command("pgrep -f 'kline_metrics_live' > /dev/null && echo 'RUNNING' || echo 'NOT'")
    if 'NOT' in ps:
        return "❌ kline-metrics-live Service läuft nicht!"
    
    # Signal an den Service senden: Gap-State-File löschen um sofortigen Check auszulösen
    run_command("rm -f /opt/coin/database/logs/.metrics_gap_state")
    run_command("systemctl restart kline-metrics-live.service")
    return "🔄 <b>Backfill ausgelöst</b>\n\nService wird neu gestartet.\nBeim Start: 2000min Lookback + sofortiger Gap-Check.\n\nCheck mit: metrics"

def cmd_logs():
    logs = run_command("tail -15 /opt/coin/database/logs/kline_metrics_live.log 2>/dev/null")
    return f"📋 <b>Logs</b>\n\n<pre>{logs[-3000:]}</pre>" if logs.strip() else "ℹ️ Keine Logs"

def cmd_restart():
    send_message("🔄 <b>Restart...</b>")
    run_command("systemctl restart kline-metrics-live.service", timeout=30)
    time.sleep(3)
    run_command("systemctl restart analyser-ingestor.service", timeout=30)
    time.sleep(3)
    s1 = run_command("systemctl is-active kline-metrics-live.service").strip()
    s2 = run_command("systemctl is-active analyser-ingestor.service").strip()
    return f"✅ <b>Restart fertig</b>\n\nmetrics: {s1}\ningestor: {s2}"

def cmd_balance():
    """Kontostand Binance + Hyperliquid + Gesamt"""
    import sys
    sys.path.insert(0, '/opt/coin/backend')
    lines = []
    binance_total = 0
    hl_total = 0

    # Binance
    try:
        from auth.auth import decrypt_value
        with open('/opt/coin/settings.json') as f:
            app_settings = json.load(f)
        app_db = app_settings['databases']['app']
        conn = psycopg2.connect(
            dbname=app_db['name'], user=app_db['user'], password=app_db['password'],
            host=app_db['host'], port=app_db['port']
        )
        cur = conn.cursor()
        cur.execute("SELECT binance_api_key_encrypted, binance_api_secret_encrypted, binance_api_valid FROM users WHERE user_id = 1")
        user = cur.fetchone()
        if user and user[0] and user[2]:
            from binance.client import Client as BinanceClient
            client = BinanceClient(decrypt_value(user[0]), decrypt_value(user[1]))
            account = client.get_account()
            usdc_bal = 0
            pos_val = 0
            for asset in account.get('balances', []):
                total = float(asset['free']) + float(asset['locked'])
                if total > 0:
                    if asset['asset'] == 'USDC':
                        usdc_bal = total
                    else:
                        try:
                            ticker = client.get_symbol_ticker(symbol=f"{asset['asset']}USDC")
                            pos_val += total * float(ticker['price'])
                        except:
                            pass
            binance_total = usdc_bal + pos_val
            lines.append(f"🟡 <b>Binance:</b> ${binance_total:,.2f}")
            if pos_val > 0:
                lines.append(f"   USDC: ${usdc_bal:,.2f} | Pos: ${pos_val:,.2f}")
        else:
            lines.append("🟡 <b>Binance:</b> nicht konfiguriert")

        # Hyperliquid
        cur.execute("SELECT hyperliquid_wallet_address, hyperliquid_api_valid FROM users WHERE user_id = 1")
        hl_user = cur.fetchone()
        conn.close()

        if hl_user and hl_user[0] and hl_user[1]:
            from hyperliquid.info import Info as HLInfo
            from hyperliquid.utils import constants as hl_constants
            info = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True)
            state = info.user_state(hl_user[0])
            margin = state.get("marginSummary", {})
            hl_total = float(margin.get("accountValue", 0))
            withdrawable = float(state.get("withdrawable", 0))
            positions = [a for a in state.get("assetPositions", []) if float(a.get("position", {}).get("szi", 0)) != 0]
            lines.append(f"🟢 <b>Hyperliquid:</b> ${hl_total:,.2f}")
            if positions:
                upnl = sum(float(a["position"].get("unrealizedPnl", 0)) for a in positions)
                lines.append(f"   {len(positions)} Pos | uPnL: ${upnl:+,.2f} | Free: ${withdrawable:,.2f}")
        else:
            lines.append("🟢 <b>Hyperliquid:</b> nicht konfiguriert")

    except Exception as e:
        lines.append(f"❌ Fehler: {e}")

    grand_total = binance_total + hl_total
    lines.append(f"\n💰 <b>Gesamt: ${grand_total:,.2f}</b>")

    return "📊 <b>Kontostand</b>\n\n" + "\n".join(lines)


def cmd_agent(args):
    """RL-Agent Status + Konfiguration.

    agent        → Status anzeigen
    agent max N  → Max gleichzeitige Positionen setzen (1-50)
    """
    try:
        with open('/opt/coin/settings.json') as f:
            app_settings = json.load(f)
        app_db = app_settings['databases']['app']
        conn = psycopg2.connect(
            dbname=app_db['name'], user=app_db['user'], password=app_db['password'],
            host=app_db['host'], port=app_db['port']
        )
        cur = conn.cursor()

        # Subcommand: on / off
        if args and args[0] in ('on', 'off'):
            action = args[0]
            if action == 'on':
                cur.execute("UPDATE rl_agent_config SET is_active = true, updated_at = NOW() WHERE user_id = 1")
                conn.commit()
                conn.close()
                os.system('/usr/bin/systemctl start rl-agent')
                time.sleep(1)
                svc_ok = check_service('rl-agent.service')
                if svc_ok:
                    return "✅ RL-Agent <b>gestartet</b>"
                return "⚠️ DB aktiviert, aber Service startet nicht. Logs prüfen!"
            else:
                cur.execute("UPDATE rl_agent_config SET is_active = false, updated_at = NOW() WHERE user_id = 1")
                conn.commit()
                conn.close()
                os.system('/usr/bin/systemctl stop rl-agent')
                return "🔴 RL-Agent <b>gestoppt</b>"

        # Subcommand: max N
        if args and len(args) >= 2 and args[0] == 'max':
            try:
                new_max = int(args[1])
            except ValueError:
                conn.close()
                return "❌ Ungültige Zahl. Bsp: agent max 30"
            if new_max < 1:
                conn.close()
                return "❌ Mindestens 1."
            cur.execute(
                "UPDATE rl_agent_config SET max_concurrent_positions = %s, updated_at = NOW() WHERE user_id = 1",
                (new_max,)
            )
            conn.commit()
            conn.close()
            return f"✅ Max Positionen auf <b>{new_max}</b> gesetzt."

        # Subcommand: trade N
        if args and len(args) >= 2 and args[0] == 'trade':
            try:
                new_size = float(args[1])
            except ValueError:
                conn.close()
                return "❌ Ungültige Zahl. Bsp: agent trade 20"
            if new_size < 15:
                conn.close()
                return "❌ Mindestens $15."
            cur.execute(
                "UPDATE rl_agent_config SET base_trade_size = %s, updated_at = NOW() WHERE user_id = 1",
                (new_size,)
            )
            conn.commit()
            conn.close()
            return f"✅ Trade-Size auf <b>${new_size:.0f}</b> gesetzt.\n(Gilt bis 1/100 greift ab $1500 Portfolio)"

        # Status anzeigen
        cur.execute("SELECT is_active, max_concurrent_positions, max_leverage, base_trade_size FROM rl_agent_config WHERE user_id = 1")
        config = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM rl_positions WHERE status = 'open'")
        open_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE pnl_usd > 0) as winners,
                   COALESCE(SUM(pnl_usd), 0) as total_pnl
            FROM rl_positions WHERE status = 'closed'
        """)
        perf = cur.fetchone()
        conn.close()

        # State-Datei für Punkte + Portfolio
        state = {}
        state_path = '/opt/coin/database/data/models/rl_agent_state.json'
        try:
            with open(state_path) as f:
                state = json.load(f)
        except:
            pass

        total_points = state.get('total_points', 0)
        portfolio = state.get('portfolio', 0)

        if total_points >= 5000:
            bonus = '2.0x'
        elif total_points >= 2000:
            bonus = '1.5x'
        elif total_points >= 500:
            bonus = '1.2x'
        else:
            bonus = '1.0x'

        is_active = config[0] if config else False
        max_pos = config[1] if config else 50
        max_lev = config[2] if config else 10
        base_size = float(config[3]) if config and config[3] else 15.0

        status_emoji = "🟢" if is_active else "🔴"
        svc_ok = check_service('rl-agent.service')
        svc_emoji = "✅" if svc_ok else "❌"

        total_trades = perf[0] if perf else 0
        winners = perf[1] if perf else 0
        total_pnl = float(perf[2]) if perf else 0
        wr = f"{winners/total_trades*100:.0f}%" if total_trades > 0 else "-"

        lines = [
            f"🤖 <b>RL-Agent V3</b> (PPO)",
            f"",
            f"{status_emoji} Agent: {'AKTIV' if is_active else 'INAKTIV'}",
            f"{svc_emoji} Service: {'läuft' if svc_ok else 'gestoppt'}",
            f"",
            f"📊 <b>Positionen:</b> {open_count}/{max_pos} offen",
            f"⚙️ Max Hebel: {max_lev}x | Trade: ${base_size:.0f}",
            f"",
            f"🏆 <b>Punkte:</b> {total_points:,.0f} (Bonus {bonus})",
            f"💰 Portfolio: ${portfolio:,.2f}",
            f"",
            f"📈 <b>Performance:</b>",
            f"   {total_trades} Trades | WR: {wr}",
            f"   PnL: ${total_pnl:+,.2f}",
            f"",
            f"💡 <code>agent max N</code> — Max Positionen",
            f"💡 <code>agent trade N</code> — Trade-Size ($)",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"❌ Fehler: {e}"


def handle_message(text, chat_id):
    text = text.strip()
    parts = text.split()
    cmd = parts[0].lower().lstrip('/') if parts else ""

    if str(chat_id) != CHAT_ID:
        return None

    if cmd in ('help', 'start', 'hilfe'):
        return cmd_help()

    if cmd == 'login':
        locked, remaining = is_locked_out()
        if locked:
            return f"🔒 Gesperrt! Noch {remaining} Min."

        if len(parts) < 2:
            return "❌ login [passwort]"

        if hash_password(parts[1]) == get_password():
            set_authenticated(chat_id)
            clear_lockout()
            return "✅ Eingeloggt!"
        else:
            attempts = add_failed_attempt()
            remaining = MAX_ATTEMPTS - attempts
            if remaining <= 0:
                return f"❌ Falsch!\n🔒 {LOCKOUT_MINUTES} Min Sperre!"
            return f"❌ Falsch! Noch {remaining} Versuche."

    if not is_authenticated(chat_id):
        return "🔒 Erst: login [passwort]"

    if cmd == 'status':
        return cmd_status()
    elif cmd == 'health':
        return cmd_health()
    elif cmd == 'balance':
        return cmd_balance()
    elif cmd == 'metrics':
        return cmd_metrics()
    elif cmd == 'backfill':
        return cmd_backfill()
    elif cmd == 'logs':
        return cmd_logs()
    elif cmd == 'restart':
        return cmd_restart()
    elif cmd == 'agent':
        return cmd_agent(parts[1:])
    elif cmd == 'logout':
        try:
            os.remove(SESSION_FILE)
        except:
            pass
        return "👋 Ausgeloggt!"
    elif cmd == 'changepw':
        if len(parts) < 3:
            return "❌ changepw [alt] [neu]"
        if hash_password(parts[1]) == get_password():
            if len(parts[2]) < 4:
                return "❌ Mind. 4 Zeichen!"
            save_password(parts[2])
            return "✅ Passwort geändert!"
        return "❌ Altes Passwort falsch!"
    
    return "❓ Unbekannt. help"

# ============ MAIN LOOP ============

def main():
    log("=== Health-Bot gestartet ===")
    send_message("🤖 <b>Health-Bot gestartet</b>\n\nPasswort: 1234")
    
    offset = None
    last_health_check = 0
    last_heartbeat = 0
    telegram_ok = True
    
    while True:
        try:
            now = time.time()
            
            # Telegram Updates holen
            updates = get_updates(offset)
            
            if updates is None:
                # Telegram-Fehler
                telegram_ok = False
            else:
                # Telegram funktioniert
                telegram_ok = True
                
                for update in updates:
                    offset = update['update_id'] + 1
                    if 'message' in update:
                        msg = update['message']
                        chat_id = msg['chat']['id']
                        text = msg.get('text', '')
                        if text:
                            log(f"Msg: {text[:50]}")
                            response = handle_message(text, chat_id)
                            if response:
                                send_message(response)
            
            # Heartbeat schreiben wenn Telegram OK (alle 60 Sek)
            if telegram_ok and (now - last_heartbeat >= HEARTBEAT_WRITE_INTERVAL):
                write_heartbeat()
                last_heartbeat = now
            
            # Health-Check alle 5 Minuten
            if now - last_health_check >= HEALTH_CHECK_INTERVAL:
                run_health_check()
                last_health_check = now
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
