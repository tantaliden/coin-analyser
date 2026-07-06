"""nnf-Predictor Paper-Shadow API (read-only). Zwei Varianten (A=OHLCV, B=+Funding),
je eigenes Paper-Buch. Live-Werte (akt. Preis + unrealisierter PnL) wie reach/seq.
settings.json live gelesen. Keine Fallbacks."""
import json
from fastapi import APIRouter
from psycopg2.extras import RealDictCursor
import psycopg2

router = APIRouter(prefix="/api/v1/nnf", tags=["nnf"])
SETTINGS_PATH = "/opt/coin/settings.json"

def _s(): return json.load(open(SETTINGS_PATH))
def _db_app():
    d = _s()["databases"]["app"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])
def _db_coins():
    d = _s()["databases"]["coins"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])

def _current_prices(symbols):
    if not symbols: return {}
    try:
        with _db_coins() as coins, coins.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("""SELECT DISTINCT ON (symbol) symbol, mid_px, close FROM klines
                         WHERE symbol = ANY(%s) AND interval='10s' ORDER BY symbol, open_time DESC""", (list(symbols),))
            return {r['symbol']: (r['mid_px'] or r['close']) for r in c.fetchall()}
    except Exception:
        return {}

def _variant(conn, table, label, start_bal, size):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""SELECT count(*) n, count(*) FILTER (WHERE exit_reason='TP') wins,
            count(*) FILTER (WHERE exit_reason='SL') losses, count(*) FILTER (WHERE exit_reason='TIMEOUT') timeouts,
            COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct,
            count(*) FILTER (WHERE learned) learned
            FROM {table} WHERE status='closed'""")
        cl = dict(cur.fetchone())
        cur.execute(f"""SELECT id,symbol,side,entry_px,tp_px,sl_px,tp_pct,sl_pct,conf,leverage,opened_at
                        FROM {table} WHERE status='open' ORDER BY opened_at DESC""")
        opens = [dict(r) for r in cur.fetchall()]
        cur.execute(f"""SELECT symbol,side,entry_px,exit_px,pnl_pct,pnl_usd,conf,exit_reason,opened_at,closed_at
                        FROM {table} WHERE status='closed' ORDER BY closed_at DESC LIMIT 100""")
        history = [dict(r) for r in cur.fetchall()]
        cur.execute(f"SELECT closed_at, pnl_usd FROM {table} WHERE status='closed' ORDER BY closed_at")
        curve = []; eq = start_bal
        for r in cur.fetchall():
            eq += float(r['pnl_usd'] or 0)
            curve.append({"t": r['closed_at'].isoformat() if r['closed_at'] else None, "equity": round(eq, 2)})
    prices = _current_prices({o['symbol'] for o in opens}); unreal = 0.0
    for o in opens:
        entry = float(o['entry_px']); px = prices.get(o['symbol']); lev = o.get('leverage') or 1
        if px and entry:
            px = float(px); move = (px-entry)/entry if o['side'] == 'long' else (entry-px)/entry
            o['current_px'] = px; o['live_pnl_pct'] = round(move*100*lev, 3); o['live_pnl_usd'] = round(move*size*lev, 3)
            unreal += move*size*lev
        else:
            o['current_px'] = None; o['live_pnl_pct'] = None; o['live_pnl_usd'] = None
    realized = float(cl['sum_usd']); n = cl['n']; wl = cl['wins'] + cl['losses']
    return {
        "label": label, "trade_size_usd": size, "start_balance": start_bal,
        "balance": round(start_bal + realized, 2), "equity": round(start_bal + realized + unreal, 2),
        "unrealized_usd": round(unreal, 2), "realized_usd": round(realized, 2),
        "total_return_pct": round((realized + unreal) / start_bal * 100, 2) if start_bal else 0,
        "closed": n, "open": len(opens), "learned": cl['learned'],
        "wins": cl['wins'], "losses": cl['losses'], "timeouts": cl['timeouts'],
        "win_rate_pct": round(100.0 * cl['wins'] / wl, 1) if wl else None,
        "avg_pnl_pct": round(float(cl['avg_pct']), 4),
        "open_positions": opens, "history": history, "equity_curve": curve,
    }

@router.get("/paper")
def nnf_paper():
    c = _s()["nnf_predictor"]; lv = c["live"]; vs = c["variants"]
    start_bal = float(lv["paper_start_balance"]); size = float(lv["trade_size_usd"])
    out = {"min_conf": float(lv["min_confidence"]), "timeout_hours": int(lv["timeout_hours"]),
           "trade_size_usd": size, "leverage_cap": int(lv["leverage_cap"]), "max_open": int(lv["max_open"]),
           "start_balance": start_bal}
    with _db_app() as conn:
        for k in vs.keys():
            out[k] = _variant(conn, vs[k]["table"], vs[k]["label"], start_bal, size)
    return out
