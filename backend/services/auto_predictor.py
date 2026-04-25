"""Regelbasierte Live-Predictions aus technischen Indikatoren.
Keine ML, kein Training. Regeln aus settings.json.auto_predictor.rules.
Ergebnisse -> live_predictions (analyser_app). Optional Telegram-Alert."""

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("auto_predictor")


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


# Operatoren ---------------------------------------------------------------

def op_gt(val, threshold, *_):
    return val is not None and val > threshold

def op_lt(val, threshold, *_):
    return val is not None and val < threshold

def op_rising_pct(series, threshold_pct, *_):
    """Series = list[float], relative Aenderung end vs start > threshold_pct %."""
    if not series or len(series) < 2:
        return False
    first = next((x for x in series if x is not None), None)
    last = series[-1]
    if first is None or last is None or first == 0:
        return False
    return (last / first - 1) * 100 >= threshold_pct

def op_abs_change_pct_max(series, threshold_pct, *_):
    """Max |pct change| im Fenster MUSS kleiner sein (Preis-Stabilitaet)."""
    if not series or len(series) < 2:
        return False
    first = next((x for x in series if x is not None), None)
    if first is None or first == 0:
        return False
    mx = max(series)
    mn = min(series)
    return max(abs(mx/first - 1), abs(mn/first - 1)) * 100 <= threshold_pct

def op_z_score(series, threshold_z, *_):
    """last z-score > threshold_z (oberhalb)."""
    if not series or len(series) < 10:
        return False
    baseline = [x for x in series[:-1] if x is not None]
    if len(baseline) < 3:
        return False
    mean = sum(baseline) / len(baseline)
    var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    std = math.sqrt(var)
    if std == 0:
        return False
    last = series[-1]
    if last is None:
        return False
    return (last - mean) / std >= threshold_z

def op_bollinger_pos(series, threshold, period, *_):
    """Letzter Wert relativ zu Bollinger-Position: (close - SMA) / (2*std)."""
    if not series or len(series) < period:
        return False
    win = series[-period:]
    clean = [x for x in win if x is not None]
    if len(clean) < period:
        return False
    mean = sum(clean) / period
    var = sum((x - mean) ** 2 for x in clean) / period
    std = math.sqrt(var)
    if std == 0:
        return False
    pos = (clean[-1] - mean) / (2 * std)
    # threshold < 0 = unter Unterband; > 0 = ueber Oberband
    if threshold < 0:
        return pos <= threshold
    return pos >= threshold

OP_FUNCS = {
    ">": op_gt, "<": op_lt,
    "rising_pct": op_rising_pct,
    "abs_change_pct_max": op_abs_change_pct_max,
    "z_score": op_z_score,
    "bollinger_pos": op_bollinger_pos,
}


# Feld-Loader --------------------------------------------------------------

def load_series(cur, symbol: str, field: str, minutes: int):
    """Holt Zeitreihe aus agg_1m fuer die letzten N Minuten.
    Sonderfall: taker_buy_base_ratio = taker_buy_base / volume."""
    lookback = max(minutes, 30)
    cur.execute("""
        SELECT bucket, open, close, volume, trades,
               taker_buy_base, taker_buy_quote,
               funding, open_interest, premium,
               spread_bps, book_imbalance_5, book_depth_5,
               mark_px, mid_px
        FROM agg_1m WHERE symbol=%s AND bucket >= now() - (%s || ' minutes')::interval
        ORDER BY bucket
    """, (symbol, lookback))
    rows = cur.fetchall()
    if not rows:
        return []
    if field == "taker_buy_base_ratio":
        out = []
        for r in rows:
            v = r['volume']; t = r['taker_buy_base']
            if v and v > 0 and t is not None:
                out.append(t / v)
            else:
                out.append(None)
        return out
    return [r.get(field) for r in rows]


# Regel-Auswertung ---------------------------------------------------------

def evaluate_rule(cur, symbol: str, rule: dict, base_lookback: int) -> dict:
    details = []
    for cond in rule["conditions"]:
        field = cond["field"]
        op = cond["op"]
        val = cond["value"]
        window = cond.get("window_min", base_lookback)
        series = load_series(cur, symbol, field, window)
        series_window = series[-max(1, int(window)):] if series else []

        fn = OP_FUNCS.get(op)
        if fn is None:
            details.append({"cond": cond, "ok": False, "reason": "unknown op"})
            return {"ok": False, "details": details}

        if op in (">", "<"):
            last = next((s for s in reversed(series_window) if s is not None), None)
            ok = fn(last, val)
            details.append({"cond": cond, "ok": ok, "value": last})
        elif op == "bollinger_pos":
            period = cond.get("window_min", 20)
            ok = op_bollinger_pos(series, val, period)
            details.append({"cond": cond, "ok": ok})
        else:
            ok = fn(series_window, val)
            details.append({"cond": cond, "ok": ok, "n": len(series_window)})
        if not ok:
            return {"ok": False, "details": details}
    return {"ok": True, "details": details}


# Telegram -----------------------------------------------------------------

def send_telegram(s, text):
    tg = s.get("telegram") or {}
    if not tg.get("bot_token") or not tg.get("chat_id"):
        return
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": tg["chat_id"], "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=10).read()
    except Exception as e:
        log.warning("telegram failed: %s", e)


# Main ---------------------------------------------------------------------

def main():
    s = load_settings()
    cfg = s["auto_predictor"]
    interval = cfg["interval_seconds"]
    cooldown = cfg["cooldown_seconds_per_symbol"]
    lookback = cfg["lookback_minutes"]
    rules = cfg["rules"]
    last_alert = {}  # (symbol, rule_name) -> ts

    while True:
        try:
            with db_coins(s) as coins, db_app(s) as app:
                with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
                    cur_c.execute("SELECT symbol FROM hl_meta ORDER BY symbol")
                    symbols = [r["symbol"] for r in cur_c.fetchall()]
                matches_total = 0
                with coins.cursor(cursor_factory=RealDictCursor) as cur_c:
                    for sym in symbols:
                        for rule in rules:
                            result = evaluate_rule(cur_c, sym, rule, lookback)
                            if not result["ok"]:
                                continue
                            now_s = time.time()
                            key = (sym, rule["name"])
                            if key in last_alert and now_s - last_alert[key] < cooldown:
                                continue
                            last_alert[key] = now_s
                            matches_total += 1
                            # Aktuellen Preis holen fuer entry/TP/SL
                            with coins.cursor() as cur_p:
                                cur_p.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
                                prow = cur_p.fetchone()
                            if not prow or prow[0] is None:
                                continue
                            coins.commit()
                            entry = float(prow[0])
                            tp_pct = 3.0
                            sl_pct = 2.0
                            direction = rule["direction"]
                            if direction == "long":
                                tp_price = entry * (1 + tp_pct/100.0)
                                sl_price = entry * (1 - sl_pct/100.0)
                            else:
                                tp_price = entry * (1 - tp_pct/100.0)
                                sl_price = entry * (1 + sl_pct/100.0)

                            with app.cursor() as cur_a:
                                cur_a.execute("""
                                    INSERT INTO live_predictions (set_id, symbol, match_score, details)
                                    VALUES (NULL, %s, 1.0, %s::jsonb)
                                    ON CONFLICT DO NOTHING
                                """, (sym, json.dumps({
                                    "rule": rule["name"],
                                    "direction": direction,
                                    "description": rule["description"],
                                    "details": result["details"],
                                })))
                                # Zusaetzlich in momentum_predictions (UI-kompatibel)
                                cur_a.execute("""
                                    INSERT INTO momentum_predictions
                                      (user_id, symbol, direction, entry_price, take_profit_price, stop_loss_price,
                                       take_profit_pct, stop_loss_pct, confidence, reason, signals,
                                       scanner_type, expires_at)
                                    VALUES (1, %s, %s, %s, %s, %s, %s, %s, 100, %s, %s::jsonb,
                                            'auto_predictor', now() + interval '4 hours')
                                """, (sym, direction, entry, tp_price, sl_price, tp_pct, sl_pct,
                                       rule["name"] + ": " + rule["description"],
                                       json.dumps({"rule": rule["name"], "details": result["details"]})))
                            app.commit()
                            log.info("MATCH %s / %s", sym, rule["name"])
                            if cfg["alert_on_match"]:
                                send_telegram(s, f"[AutoPredictor] {sym}: {rule['name']} "
                                                f"({rule['direction']})")
                log.info("pass done: %d matches across %d rules x %d symbols",
                         matches_total, len(rules), len(symbols))
        except Exception as e:
            log.exception("pass failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
