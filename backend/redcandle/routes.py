"""Kapitulations-Bounce Paper-Shadow API (read-only).
Eigene Paper-Welt (redcandle_paper_positions / redcandle_paper_wallet_state). Live-Werte der
offenen Positionen + Historie + Equity-Kurve. Config live aus settings.redcandle_shadow.
Keine Fallbacks, keine Defaults, keine Hardcodes."""
import json
from fastapi import APIRouter
from psycopg2.extras import RealDictCursor
import psycopg2

router = APIRouter(prefix="/api/v1/redcandle", tags=["redcandle"])
SETTINGS_PATH = "/opt/coin/settings.json"


def _s():
    return json.load(open(SETTINGS_PATH))


def _db_app():
    d = _s()["databases"]["app"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])


def _db_coins():
    d = _s()["databases"]["coins"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])


def _current_prices(symbols):
    if not symbols:
        return {}
    with _db_coins() as coins:
        with coins.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("""SELECT DISTINCT ON (symbol) symbol, close FROM agg_1m
                         WHERE symbol = ANY(%s) ORDER BY symbol, bucket DESC""", (list(symbols),))
            return {r["symbol"]: float(r["close"]) for r in c.fetchall()}


@router.get("/paper")
def redcandle_paper():
    x = _s()["redcandle_shadow"]
    with _db_app() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT balance, start_balance, peak_balance, n_trades FROM redcandle_paper_wallet_state WHERE id=1")
            ws = dict(cur.fetchone() or {})
            start_bal = float(ws.get("start_balance") or 0)
            balance = float(ws.get("balance") if ws.get("balance") is not None else start_bal)
            cur.execute("""SELECT count(*) n,
                count(*) FILTER (WHERE close_reason='TP') tp,
                count(*) FILTER (WHERE close_reason='SL') sl,
                count(*) FILTER (WHERE close_reason='TIMEOUT') timeout,
                count(*) FILTER (WHERE pnl_usd>0) wins,
                COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct
                FROM redcandle_paper_positions WHERE status='closed'""")
            cl = dict(cur.fetchone())
            cur.execute("""SELECT id,symbol,entry_px,tp_px,sl_px,qty,margin_usd,drop_pct,pre_green,opened_at
                           FROM redcandle_paper_positions WHERE status='open' ORDER BY opened_at DESC""")
            opens = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT symbol,entry_px,exit_px,drop_pct,pre_green,pnl_pct,pnl_usd,close_reason,opened_at,closed_at
                           FROM redcandle_paper_positions WHERE status='closed'
                           ORDER BY closed_at DESC LIMIT 100""")
            history = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT closed_at, pnl_usd FROM redcandle_paper_positions
                           WHERE status='closed' ORDER BY closed_at""")
            curve = []; eq = start_bal
            for r in cur.fetchall():
                eq += float(r["pnl_usd"] or 0)
                curve.append({"t": r["closed_at"].isoformat() if r["closed_at"] else None, "equity": round(eq, 2)})

    prices = _current_prices({o["symbol"] for o in opens})
    unrealized = 0.0
    for o in opens:
        entry = float(o["entry_px"]); px = prices.get(o["symbol"]); qty = float(o["qty"]); margin = float(o["margin_usd"])
        o["drop_pct"] = round(float(o["drop_pct"]) * 100, 2) if o["drop_pct"] is not None else None
        if px and entry:
            pnl_usd = qty * (px - entry)
            o["current_px"] = px
            o["live_pnl_usd"] = round(pnl_usd, 3)
            o["live_pnl_pct"] = round(pnl_usd / margin * 100.0, 2) if margin else None
            unrealized += pnl_usd
        else:
            o["current_px"] = None; o["live_pnl_usd"] = None; o["live_pnl_pct"] = None

    realized = float(cl["sum_usd"]); n_closed = cl["n"]
    return {
        "strategy": "redcandle_capitulation_bounce",
        "candle_interval": x["candle_interval"], "n_red": x["n_red"], "pre_n": x["pre_n"],
        "pre_green_min": x["pre_green_min"], "min_drop_pct": x["min_drop_pct"],
        "tp_pct": x["tp_pct"], "sl_pct": x["sl_pct"], "timeout_hours": x["timeout_hours"],
        "trade_size_usd": float(x["trade_size_usd"]), "leverage": x["leverage"],
        "fee_roundtrip_pct": x["fee_roundtrip_pct"], "max_open": x["max_open"],
        "start_balance": round(start_bal, 2), "balance": round(balance, 2),
        "equity": round(balance + unrealized, 2),
        "unrealized_usd": round(unrealized, 2), "realized_usd": round(realized, 2),
        "total_return_pct": round((balance + unrealized - start_bal) / start_bal * 100, 2) if start_bal else 0,
        "closed": n_closed, "open": len(opens),
        "tp": cl["tp"], "sl": cl["sl"], "timeout": cl["timeout"], "wins": cl["wins"],
        "win_rate_pct": round(100.0 * cl["wins"] / n_closed, 1) if n_closed else None,
        "avg_pnl_pct": round(float(cl["avg_pct"]), 3),
        "open_positions": opens, "history": history, "equity_curve": curve,
        "note": "Long-only Kapitulations-Bounce: genau 5 rote 5m-Candles (gruen umschlossen) + Vorlauf >=6 gruen + Rutsch>=min_drop. Einstieg=1. gruene. Edge sitzt in der Rutsch-Tiefe. Kein Echtgeld.",
    }
