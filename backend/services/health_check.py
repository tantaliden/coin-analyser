"""HL Daten-Health-Check mit Telegram-Alerts.
Keine Fallbacks — Fehler werden gemeldet statt ignoriert.
Alle Konfigurationen aus settings.json."""

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("health_check")


def load_settings():
    with open('/opt/coin/settings.json') as f:
        return json.load(f)


def db_conn(s):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def send_telegram(s, text):
    tg = s["telegram"]
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": tg["chat_id"], "text": text, "parse_mode": "HTML"
    }).encode()
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        log.warning("telegram send failed: %s", e)


def run_checks(s) -> list:
    """Gibt Liste von (severity, message) zurueck. severity = 'ok'|'warn'|'crit'."""
    cfg = s["health_check"]
    issues = []

    # 1) Service-Status
    for svc in cfg["required_services"]:
        r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
        if r.stdout.strip() != "active":
            issues.append(("crit", f"Service {svc}: {r.stdout.strip()}"))

    # 2) DB-Queries
    try:
        with db_conn(s) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT EXTRACT(EPOCH FROM (now() - max(open_time)))::int FROM klines")
                age_klines = cur.fetchone()[0]
                if cfg.get("ctx_max_age_seconds") is not None:
                    cur.execute("SELECT EXTRACT(EPOCH FROM (now() - max(ts)))::int FROM hl_asset_ctx")
                    age_ctx = cur.fetchone()[0]
                else:
                    age_ctx = None
                cur.execute("SELECT EXTRACT(EPOCH FROM (now() - max(bucket)))::int FROM agg_1m")
                age_agg1m = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(DISTINCT symbol) FROM klines WHERE open_time >= now() - (%s || ' seconds')::interval",
                    (cfg["window_seconds"],)
                )
                syms_recent = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM hl_meta")
                total_syms = cur.fetchone()[0]
    except Exception as e:
        issues.append(("crit", f"DB-Query fehlgeschlagen: {e}"))
        return issues

    if age_klines is None:
        issues.append(("crit", "klines leer"))
    elif age_klines > cfg["klines_max_age_seconds"]:
        issues.append(("crit", f"klines-Lag {age_klines}s > {cfg['klines_max_age_seconds']}s"))

    if cfg.get("ctx_max_age_seconds") is not None:
        if age_ctx is None:
            issues.append(("crit", "hl_asset_ctx leer"))
        elif age_ctx > cfg["ctx_max_age_seconds"]:
            issues.append(("crit", f"ctx-Lag {age_ctx}s > {cfg['ctx_max_age_seconds']}s"))

    if age_agg1m is not None and age_agg1m > cfg["agg_1m_max_age_seconds"]:
        issues.append(("warn", f"agg_1m-Lag {age_agg1m}s > {cfg['agg_1m_max_age_seconds']}s"))

    if syms_recent < cfg["min_symbols_last_window"]:
        issues.append(("crit",
            f"nur {syms_recent}/{total_syms} Symbole aktiv (letzte {cfg['window_seconds']}s)"))

    return issues


def main():
    s = load_settings()
    cfg = s["health_check"]
    hb_file = Path(cfg["heartbeat_file"])
    hb_file.parent.mkdir(parents=True, exist_ok=True)

    last_alert_key = None
    last_alert_ts = 0

    while True:
        try:
            issues = run_checks(s)
            crit = [m for sev, m in issues if sev == "crit"]
            warn = [m for sev, m in issues if sev == "warn"]

            # Heartbeat
            hb_file.write_text(str(int(time.time())))

            if not crit:
                if last_alert_key is not None:
                    label = cfg["server_label"]
                    send_telegram(s, f"[HL-Health {label}] recovery — alles OK wieder")
                    last_alert_key = None
            else:
                alert_key = "|".join(sorted(crit))
                now = time.time()
                if alert_key != last_alert_key or (now - last_alert_ts) >= cfg["alert_cooldown_seconds"]:
                    label = cfg["server_label"]
                    text = f"[HL-Health {label}] CRITICAL\n" + "\n".join(f"- {m}" for m in crit)
                    if warn:
                        text += "\n(warn: " + "; ".join(warn) + ")"
                    send_telegram(s, text)
                    last_alert_key = alert_key
                    last_alert_ts = now
                    log.warning("ALERT: %s", crit)
            log.info("check: crit=%d warn=%d", len(crit), len(warn))
        except Exception as e:
            log.exception("check failed: %s", e)
        time.sleep(cfg["interval_seconds"])


if __name__ == "__main__":
    main()
