"""kline_metrics Updater — aus agg_1m.
Pro open_time t: pct_Xm = (close[t] - open[t-Xm]) / open[t-Xm] * 100
                abs_Xm = close[t] - open[t-Xm]
Keine Fallbacks. Alle Konfigurationen in settings.json."""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("kline_metrics_updater")


def load_settings():
    root = Path(__file__).resolve().parents[2]
    with open(root / 'settings.json') as f:
        return json.load(f)


def get_conn(s: dict):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def update_symbol(conn, symbol: str, lookback_minutes: int, durations: list) -> int:
    max_d = max(durations)
    start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes + max_d)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT bucket AS open_time, open, close
            FROM agg_1m WHERE symbol=%s AND bucket >= %s
            ORDER BY bucket
        """, (symbol, start))
        candles = cur.fetchall()
    if not candles:
        return 0
    by_time = {c["open_time"]: c for c in candles}
    calc_start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    rows = []
    for c in candles:
        if c["open_time"] < calc_start or c["close"] is None:
            continue
        row = {"symbol": symbol, "open_time": c["open_time"]}
        close_val = float(c["close"])
        for d in durations:
            ref = by_time.get(c["open_time"] - timedelta(minutes=d))
            if ref is None or ref["open"] is None or float(ref["open"]) == 0:
                row[f"pct_{d}m"] = None
                row[f"abs_{d}m"] = None
            else:
                open_ref = float(ref["open"])
                row[f"pct_{d}m"] = (close_val - open_ref) / open_ref * 100.0
                row[f"abs_{d}m"] = close_val - open_ref
        rows.append(row)
    if not rows:
        return 0
    cols = ["symbol", "open_time"] + [f"pct_{d}m" for d in durations] + [f"abs_{d}m" for d in durations]
    values = [tuple(r[c] for c in cols) for r in rows]
    sql = f"""
        INSERT INTO kline_metrics ({', '.join(cols)}) VALUES %s
        ON CONFLICT (symbol, open_time) DO UPDATE SET
        {', '.join(f'{c}=EXCLUDED.{c}' for c in cols[2:])}
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=1000)
    conn.commit()
    return len(rows)


def main():
    s = load_settings()
    cfg = s["kline_metrics_updater"]
    interval = cfg["interval_seconds"]
    lookback = cfg["lookback_minutes"]
    durations = cfg["durations_minutes"]
    while True:
        try:
            with get_conn(s) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT symbol FROM hl_meta ORDER BY symbol")
                    symbols = [r[0] for r in cur.fetchall()]
                total = 0
                for sym in symbols:
                    try:
                        n = update_symbol(conn, sym, lookback, durations)
                        total += n
                    except Exception as e:
                        log.warning("symbol %s failed: %s", sym, e)
            log.info("pass done: %d symbols, %d rows upserted", len(symbols), total)
        except Exception as e:
            log.exception("pass failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
