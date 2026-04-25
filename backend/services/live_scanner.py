"""Live-Scanner: prueft alle aktiven Indikator-Sets gegen aktuelle klines-Daten.
Alle interval Minuten fuer jeden Coin: fuer jedes aktive Set testen ob Kriterien matchen.
Match -> INSERT live_predictions. Keine Fallbacks.
Konfiguration in settings.json.live_scanner."""
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("live_scanner")



ALLOWED_COLUMNS = frozenset([
    'close', 'open', 'high', 'low', 'volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'quote_asset_volume',
    'funding', 'open_interest', 'premium',
    'oracle_px', 'mark_px', 'mid_px',
    'bbo_bid_px', 'bbo_ask_px', 'bbo_bid_sz', 'bbo_ask_sz',
    'spread_bps', 'book_imbalance_5', 'book_depth_5',
])

def load_settings():
    with open('/opt/coin/settings.json') as f:
        return json.load(f)


def db_coins(s):
    db = s["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def db_app(s):
    db = s["databases"]["app"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def fetch_active_sets(app_conn):
    with app_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT s.set_id, s.name, s.direction, s.target_percent, s.duration_minutes,
                   s.prehistory_minutes, s.candle_timeframe, s.is_active
            FROM indicator_sets s
            WHERE s.is_active = TRUE
            ORDER BY s.set_id
        """)
        sets = cur.fetchall()
        for row in sets:
            cur.execute("""
                SELECT * FROM indicator_items WHERE set_id = %s ORDER BY sort_order
            """, (row['set_id'],))
            row['items'] = [dict(r) for r in cur.fetchall()]
        return sets


def evaluate_set(coins_conn, s_set, now_utc) -> list:
    """Scannt alle Coins auf match mit diesem Set. Returns list of prediction dicts.
    Anchor ist now — time_offset_from/to werden relativ zu now ausgewertet."""
    candle_timeframe = s_set['candle_timeframe'] or 1
    table = f"klines_{_tf_alias(candle_timeframe)}"
    prehistory = s_set['prehistory_minutes'] or 60
    scan_window_start = now_utc - timedelta(minutes=prehistory)

    # Alle Coins
    with coins_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM hl_meta ORDER BY symbol")
        symbols = [r[0] for r in cur.fetchall()]

    matches = []
    for symbol in symbols:
        result = _check_all_items(coins_conn, table, symbol, s_set, now_utc)
        if result is None:
            continue
        if result['matched']:
            matches.append({
                "set_id": s_set['set_id'],
                "symbol": symbol,
                "match_score": result['score'],
                "details": result['details'],
            })
    return matches


def _tf_alias(tf_minutes: int) -> str:
    # Muss konsistent mit den klines_Xm-Views sein
    mapping = {1: '1m', 5: '5m', 15: '15m', 30: '30m', 60: '1h', 240: '4h', 1440: '1d'}
    if tf_minutes not in mapping:
        raise ValueError(f"candle_timeframe {tf_minutes} nicht in klines_* Views verfuegbar")
    return mapping[tf_minutes]


def _check_all_items(coins_conn, table, symbol, s_set, now_utc) -> dict:
    items = s_set['items']
    if not items:
        return None
    score = 0
    total = len(items)
    details = []
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        for item in items:
            t_start = item['time_start_minutes']
            t_end = item['time_end_minutes']
            start_dt = now_utc + timedelta(minutes=t_start)
            end_dt = now_utc + timedelta(minutes=t_end)
            itype = item['indicator_type']
            op = item['condition_operator'] or '>'
            v = item['condition_value']
            v2 = item['condition_value2']

            if itype == 'candle_pattern':
                details.append({"item_id": item['item_id'], "hit": False, "note": "pattern skipped"})
                continue
            if itype not in ALLOWED_COLUMNS:
                details.append({"item_id": item['item_id'], "hit": False,
                                "note": f"unknown field: {itype}"})
                continue

            cur.execute(
                f"SELECT MIN({itype}) AS mn, MAX({itype}) AS mx, last({itype}, open_time) AS lt "
                f"FROM {table} WHERE symbol=%s AND open_time >= %s AND open_time <= %s",
                (symbol, start_dt, end_dt)
            )
            row = cur.fetchone()
            if row is None or row['lt'] is None:
                details.append({"item_id": item['item_id'], "hit": False, "note": "no data"})
                continue

            val = float(row['lt'])
            hit = _check_op(val, op, v, v2, float(row['mn']) if row['mn'] is not None else None,
                                          float(row['mx']) if row['mx'] is not None else None)
            if hit:
                score += 1
            details.append({
                "item_id": item['item_id'], "field": itype, "op": op,
                "val": val, "threshold": float(v) if v is not None else None,
                "hit": hit,
            })

    return {
        "matched": score == total,
        "score": round(score / total, 3) if total > 0 else 0,
        "details": details,
    }


def _check_op(val, op, v, v2, mn, mx):
    if v is None:
        return False
    v = float(v)
    if op == '>':
        return val > v
    if op == '<':
        return val < v
    if op == '>=':
        return val >= v
    if op == '<=':
        return val <= v
    if op == '=':
        return abs(val - v) < 1e-9
    if op == '!=':
        return abs(val - v) >= 1e-9
    if op == 'between' and v2 is not None:
        v2 = float(v2)
        lo, hi = min(v, v2), max(v, v2)
        # Maximum im Fenster innerhalb [lo,hi]
        if mn is not None and mx is not None:
            return not (mx < lo or mn > hi)
        return lo <= val <= hi
    return False


def insert_predictions(app_conn, preds: list):
    if not preds:
        return
    with app_conn.cursor() as cur:
        for p in preds:
            cur.execute("""
                INSERT INTO live_predictions (set_id, symbol, match_score, details)
                VALUES (%s, %s, %s, %s::jsonb)
            """, (p["set_id"], p["symbol"], p["match_score"], json.dumps(p["details"])))
    app_conn.commit()


def send_telegram(s, text):
    tg = s.get("telegram") or {}
    if not tg.get("bot_token") or not tg.get("chat_id"):
        return
    import urllib.parse, urllib.request
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": tg["chat_id"], "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=10).read()
    except Exception as e:
        log.warning("telegram failed: %s", e)


def main():
    s = load_settings()
    cfg = s["live_scanner"]
    interval = cfg["interval_seconds"]
    alert = cfg["alert_on_match"]
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            with db_app(s) as app_conn:
                sets = fetch_active_sets(app_conn)
            log.info("scanning %d active sets", len(sets))
            if sets:
                with db_coins(s) as coins_conn:
                    for ss in sets:
                        matches = evaluate_set(coins_conn, ss, now_utc)
                        if not matches:
                            continue
                        with db_app(s) as app_conn:
                            insert_predictions(app_conn, matches)
                        log.info("set %d '%s': %d matches", ss['set_id'], ss['name'], len(matches))
                        if alert:
                            top = ", ".join(m["symbol"] for m in matches[:10])
                            send_telegram(s, f"[LiveScanner] Set '{ss['name']}': "
                                             f"{len(matches)} Matches — {top}")
        except Exception as e:
            log.exception("scan pass failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
