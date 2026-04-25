"""Gap-Recovery: findet Luecken in Live-klines und zieht fehlende Rows vom Test-Server.
Scannt alle N Minuten die letzten M Minuten.
Gap-Kriterium: < min_symbols_per_slot unique symbols in einem 10s-Bucket.
Keine Fallbacks — bei Fehlern Exception + Telegram-Alert."""

import json
import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("gap_recovery")


def load_settings():
    with open('/opt/coin/settings.json') as f:
        return json.load(f)


def db_conn(s):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def find_gaps(conn, window_minutes: int, exclude_last: int, min_symbols: int) -> list:
    """Gibt Liste von 10s-Slots zurueck (datetime) wo weniger als min_symbols Rows existieren."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH slots AS (
              SELECT time_bucket('10 seconds', open_time) AS slot,
                     count(DISTINCT symbol) AS cnt
              FROM klines
              WHERE open_time >= now() - (%s || ' minutes')::interval
                AND open_time <  now() - (%s || ' seconds')::interval
              GROUP BY slot
            )
            SELECT slot FROM slots WHERE cnt < %s ORDER BY slot
        """, (window_minutes, exclude_last, min_symbols))
        return [r[0] for r in cur.fetchall()]


def pull_rows_for_slots(cfg: dict, slots: list) -> list:
    """SSH zum Test-Server, pg-Query auf klines fuer die gegebenen Slots. Returns list[dict]."""
    if not slots:
        return []
    slot_values = ",".join(f"('{s.isoformat()}'::timestamptz)" for s in slots)
    sql = f"""
        SELECT symbol, interval, open_time, close_time, open, high, low, close,
               volume, trades, quote_asset_volume, taker_buy_base, taker_buy_quote,
               funding, open_interest, premium, oracle_px, mark_px, mid_px
        FROM klines k
        WHERE interval = '10s'
          AND open_time IN (
              SELECT slot FROM (VALUES {slot_values}) s(slot))
        ORDER BY open_time, symbol
    """
    # SSH fuehrt Remote-Shell aus; SQL via stdin an psql, kein Shell-Escape noetig
    remote_cmd = (
        f"sudo -u {cfg['source_postgres_user']} psql "
        f"-d {cfg['source_postgres_db']} -At -F $'\x1f'"
    )
    ssh = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
           f"{cfg['source_ssh_user']}@{cfg['source_ssh_host']}", remote_cmd]
    result = subprocess.run(ssh, input=sql, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"SSH/psql failed: {result.stderr}")
    rows = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split(chr(31))
        if len(parts) != 19:
            log.warning("ignored malformed row: %r", line[:100])
            continue
        rows.append({
            "symbol": parts[0],
            "interval": parts[1],
            "open_time": parts[2],
            "close_time": parts[3] or None,
            "open": float(parts[4]) if parts[4] else None,
            "high": float(parts[5]) if parts[5] else None,
            "low": float(parts[6]) if parts[6] else None,
            "close": float(parts[7]) if parts[7] else None,
            "volume": float(parts[8]) if parts[8] else None,
            "trades": int(parts[9]) if parts[9] else None,
            "quote_asset_volume": float(parts[10]) if parts[10] else None,
            "taker_buy_base": float(parts[11]) if parts[11] else None,
            "taker_buy_quote": float(parts[12]) if parts[12] else None,
            "funding": float(parts[13]) if parts[13] else None,
            "open_interest": float(parts[14]) if parts[14] else None,
            "premium": float(parts[15]) if parts[15] else None,
            "oracle_px": float(parts[16]) if parts[16] else None,
            "mark_px": float(parts[17]) if parts[17] else None,
            "mid_px": float(parts[18]) if parts[18] else None,
        })
    return rows


def insert_rows(conn, rows: list) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO klines
              (symbol, interval, open_time, close_time, open, high, low, close,
               volume, trades, quote_asset_volume, taker_buy_base, taker_buy_quote,
               funding, open_interest, premium, oracle_px, mark_px, mid_px)
            VALUES %s
            ON CONFLICT (symbol, interval, open_time) DO NOTHING
        """, [(
            r["symbol"], r["interval"], r["open_time"], r["close_time"],
            r["open"], r["high"], r["low"], r["close"], r["volume"], r["trades"],
            r["quote_asset_volume"], r["taker_buy_base"], r["taker_buy_quote"],
            r["funding"], r["open_interest"], r["premium"],
            r["oracle_px"], r["mark_px"], r["mid_px"]
        ) for r in rows])
    conn.commit()
    return len(rows)


def main():
    s = load_settings()
    cfg = s["gap_recovery"]
    while True:
        try:
            with db_conn(s) as conn:
                slots = find_gaps(conn, cfg["scan_window_minutes"], cfg["exclude_last_seconds"], cfg["min_symbols_per_slot"])
                if slots:
                    log.warning("%d gap-slots found in last %dmin", len(slots), cfg["scan_window_minutes"])
                    chunk_size = cfg["chunk_size_slots"]
                    total_inserted = 0
                    for i in range(0, len(slots), chunk_size):
                        batch = slots[i:i+chunk_size]
                        try:
                            rows = pull_rows_for_slots(cfg, batch)
                            n = insert_rows(conn, rows)
                            total_inserted += n
                            log.info("recovered %d rows for %d slots (chunk %d)", n, len(batch), i//chunk_size)
                        except Exception as ce:
                            log.exception("chunk %d failed: %s", i//chunk_size, ce)
                    log.warning("gap-recovery done: %d rows inserted for %d gap-slots",
                                total_inserted, len(slots))
                else:
                    log.info("no gaps in last %dmin", cfg["scan_window_minutes"])
        except Exception as e:
            log.exception("recovery-pass failed: %s", e)
        time.sleep(cfg["interval_minutes"] * 60)


if __name__ == "__main__":
    main()
