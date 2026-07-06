"""Gap-Recovery (bidirektional, 18.06.2026): jeder Server ergaenzt fehlende Live-Daten
vom jeweils anderen (source_ssh_host). Zwei Paesse:
  1) Fehlende ROWS  : 10s-Slots mit < min_symbols Coins  -> Rows vom anderen Server INSERTen.
  2) NULL-FELDER    : vorhandene Rows mit fehlenden HL-Feldern (funding/premium/book_*/…)
                      -> beim anderen Server holen und NUR die NULL-Loecher fuellen
                      (COALESCE: lokaler Wert wird NIE durch Remote ueberschrieben).
Betrifft ausschliesslich `klines` (Live, 10s). hist_* (Backfill) bleibt unberuehrt.
Keine Fallbacks — bei Fehlern Exception + Log."""

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

# Spalten in klines (Reihenfolge fix fuer SELECT/Parse/INSERT)
COLS = ["symbol", "interval", "open_time", "close_time", "open", "high", "low", "close",
        "volume", "trades", "quote_asset_volume", "taker_buy_base", "taker_buy_quote",
        "funding", "open_interest", "premium", "oracle_px", "mark_px", "mid_px",
        "bbo_bid_px", "bbo_ask_px", "bbo_bid_sz", "bbo_ask_sz",
        "spread_bps", "book_imbalance_5", "book_depth_5",
        "book_imbalance_10", "book_depth_10", "book_imbalance_20", "book_depth_20"]
COL_SQL = ", ".join(COLS)
# Felder, die ein Row haben MUSS, sonst gilt er als unvollstaendig (HL-Mikrostruktur)
NULL_DETECT = ["funding", "premium", "open_interest", "mid_px", "mark_px",
               "bbo_bid_px", "spread_bps", "book_imbalance_5"]
# Nicht-Key-Felder, die beim NULL-Fill per COALESCE aufgefuellt werden (alles ausser symbol/interval/open_time)
KEYS = {"symbol", "interval", "open_time"}
FILL_FIELDS = [c for c in COLS if c not in KEYS]
# Typen fuer sauberes Parsen
_INT = {"trades"}


def load_settings():
    with open('/opt/coin/settings.json') as f:
        return json.load(f)


def db_conn(s):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def _parse_rows(stdout: str) -> list:
    """psql -At -F <0x1f> Output -> list[dict] mit allen COLS."""
    rows = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split(chr(31))
        if len(parts) != len(COLS):
            log.warning("ignored malformed row: %r", line[:100])
            continue
        r = {}
        for col, val in zip(COLS, parts):
            if col in ("symbol", "interval"):
                r[col] = val
            elif col in ("open_time", "close_time"):
                r[col] = val or None
            elif col in _INT:
                r[col] = int(val) if val else None
            else:
                r[col] = float(val) if val else None
        rows.append(r)
    return rows


def _run_remote_sql(cfg: dict, sql: str) -> list:
    """SSH zum Quell-Server, SQL via stdin an psql, Output geparst."""
    remote_cmd = (
        f"sudo -u {cfg['source_postgres_user']} psql "
        f"-d {cfg['source_postgres_db']} -At -F $'\x1f'"
    )
    ssh = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
           f"{cfg['source_ssh_user']}@{cfg['source_ssh_host']}", remote_cmd]
    result = subprocess.run(ssh, input=sql, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"SSH/psql failed: {result.stderr}")
    return _parse_rows(result.stdout)


# ---------- Pass 1: fehlende Rows ----------

def find_gaps(conn, window_minutes: int, exclude_last: int, min_symbols: int) -> list:
    """10s-Slots mit weniger als min_symbols Rows (komplett fehlende Daten)."""
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
    if not slots:
        return []
    slot_values = ",".join(f"('{s.isoformat()}'::timestamptz)" for s in slots)
    sql = f"""
        SELECT {COL_SQL}
        FROM klines k
        WHERE interval = '10s'
          AND open_time IN (SELECT slot FROM (VALUES {slot_values}) s(slot))
        ORDER BY open_time, symbol
    """
    return _run_remote_sql(cfg, sql)


def insert_rows(conn, rows: list) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, f"""
            INSERT INTO klines ({COL_SQL})
            VALUES %s
            ON CONFLICT (symbol, interval, open_time) DO NOTHING
        """, [tuple(r[c] for c in COLS) for r in rows])
    conn.commit()
    return len(rows)


# ---------- Pass 2: NULL-Felder in vorhandenen Rows ----------

def find_null_rows(conn, window_minutes: int, exclude_last: int) -> list:
    """(symbol, open_time) von Rows, bei denen ein HL-Feld NULL ist."""
    cond = " OR ".join(f"{f} IS NULL" for f in NULL_DETECT)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT symbol, open_time FROM klines
            WHERE interval = '10s'
              AND open_time >= now() - (%s || ' minutes')::interval
              AND open_time <  now() - (%s || ' seconds')::interval
              AND ({cond})
            ORDER BY open_time, symbol
        """, (window_minutes, exclude_last))
        return [(r[0], r[1]) for r in cur.fetchall()]


def pull_rows_for_keys(cfg: dict, keys: list) -> list:
    """Volle Rows vom Quell-Server fuer gegebene (symbol, open_time)-Paare."""
    if not keys:
        return []
    vals = ",".join(f"('{sym}','{ot.isoformat()}'::timestamptz)" for sym, ot in keys)
    sql = f"""
        SELECT {COL_SQL}
        FROM klines k
        WHERE interval = '10s'
          AND (symbol, open_time) IN (
              SELECT symbol, open_time FROM (VALUES {vals}) s(symbol, open_time))
        ORDER BY open_time, symbol
    """
    return _run_remote_sql(cfg, sql)


def update_null_fields(conn, rows: list) -> int:
    """Fuellt NUR NULL-Loecher: COALESCE(lokal, remote). Lokaler Wert wird nie ueberschrieben."""
    if not rows:
        return 0
    sets = ", ".join(f"{f} = COALESCE({f}, %s)" for f in FILL_FIELDS)
    sql = f"UPDATE klines SET {sets} WHERE symbol=%s AND interval=%s AND open_time=%s"
    n = 0
    with conn.cursor() as cur:
        for r in rows:
            params = [r[f] for f in FILL_FIELDS] + [r["symbol"], r["interval"], r["open_time"]]
            cur.execute(sql, params)
            n += cur.rowcount
    conn.commit()
    return n


def main():
    s = load_settings()
    cfg = s["gap_recovery"]
    chunk = cfg["chunk_size_slots"]
    while True:
        try:
            with db_conn(s) as conn:
                # Pass 1: fehlende Rows
                slots = find_gaps(conn, cfg["scan_window_minutes"], cfg["exclude_last_seconds"], cfg["min_symbols_per_slot"])
                if slots:
                    log.warning("%d gap-slots found in last %dmin", len(slots), cfg["scan_window_minutes"])
                    total = 0
                    for i in range(0, len(slots), chunk):
                        batch = slots[i:i+chunk]
                        try:
                            total += insert_rows(conn, pull_rows_for_slots(cfg, batch))
                        except Exception as ce:
                            log.exception("row-chunk %d failed: %s", i//chunk, ce)
                    log.warning("row-recovery done: %d rows inserted for %d gap-slots", total, len(slots))
                else:
                    log.info("no row-gaps in last %dmin", cfg["scan_window_minutes"])

                # Pass 2: NULL-Felder in vorhandenen Rows
                nkeys = find_null_rows(conn, cfg["scan_window_minutes"], cfg["exclude_last_seconds"])
                if nkeys:
                    log.warning("%d rows with NULL HL-fields in last %dmin", len(nkeys), cfg["scan_window_minutes"])
                    filled = 0
                    for i in range(0, len(nkeys), chunk):
                        batch = nkeys[i:i+chunk]
                        try:
                            filled += update_null_fields(conn, pull_rows_for_keys(cfg, batch))
                        except Exception as ce:
                            log.exception("null-chunk %d failed: %s", i//chunk, ce)
                    log.warning("null-field fill done: %d field-updates for %d rows", filled, len(nkeys))
                else:
                    log.info("no NULL-field rows in last %dmin", cfg["scan_window_minutes"])
        except Exception as e:
            log.exception("recovery-pass failed: %s", e)
        time.sleep(cfg["interval_minutes"] * 60)


if __name__ == "__main__":
    main()
