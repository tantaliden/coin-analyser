"""Reach-Predictor Paper-Shadow API (read-only).
Eigenes Paper-Buch (reach_paper_positions) mit Live-Werten (akt. Preis + unrealisierter
PnL offener Positionen, analog seq/Paper-Wallet) + Historie + Equity-Kurve.
settings.json live gelesen. Keine Fallbacks."""
import json
from fastapi import APIRouter
from psycopg2.extras import RealDictCursor
import psycopg2

router = APIRouter(prefix="/api/v1/reach", tags=["reach"])
SETTINGS_PATH = "/opt/coin/settings.json"

def _s(): return json.load(open(SETTINGS_PATH))
def _db_app():
    d = _s()["databases"]["app"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])
def _db_coins():
    d = _s()["databases"]["coins"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])

def _current_prices(symbols):
    """Aktueller Preis je Symbol aus den neuesten 10s-klines (mid_px, sonst close) —
    dieselbe Quelle wie der Shadow-Service. KEIN Fallback: fehlt der Preis -> None."""
    if not symbols: return {}
    try:
        with _db_coins() as coins:
            with coins.cursor(cursor_factory=RealDictCursor) as c:
                c.execute("""SELECT DISTINCT ON (symbol) symbol, mid_px, close
                             FROM klines WHERE symbol = ANY(%s) AND interval='10s'
                             ORDER BY symbol, open_time DESC""", (list(symbols),))
                return {r['symbol']: (r['mid_px'] or r['close']) for r in c.fetchall()}
    except Exception:
        return {}

@router.get("/paper")
def reach_paper():
    c = _s()["reach_model"]; lv = c["live"]; tg = c["target"]
    start_bal = float(lv["paper_start_balance"]); size = float(lv["trade_size_usd"])
    with _db_app() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT count(*) n,
                count(*) FILTER (WHERE exit_reason='TP') wins,
                count(*) FILTER (WHERE exit_reason='SL') losses,
                count(*) FILTER (WHERE exit_reason='DRIFT') drifts,
                COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct
                FROM reach_paper_positions WHERE status='closed'""")
            cl = dict(cur.fetchone())
            cur.execute("""SELECT id,symbol,side,entry_px,tp_px,sl_px,tp_pct,sl_pct,conf,n_match,leverage,opened_at
                           FROM reach_paper_positions WHERE status='open' ORDER BY opened_at DESC""")
            opens = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT symbol,side,entry_px,exit_px,pnl_pct,pnl_usd,exit_reason,opened_at,closed_at
                           FROM reach_paper_positions WHERE status='closed'
                           ORDER BY closed_at DESC LIMIT 100""")
            history = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT closed_at, pnl_usd FROM reach_paper_positions
                           WHERE status='closed' ORDER BY closed_at""")
            curve = []; eq = start_bal
            for r in cur.fetchall():
                eq += float(r['pnl_usd'] or 0)
                curve.append({"t": r['closed_at'].isoformat() if r['closed_at'] else None, "equity": round(eq, 2)})

    prices = _current_prices({o['symbol'] for o in opens})
    unrealized = 0.0
    for o in opens:
        entry = float(o['entry_px']); px = prices.get(o['symbol']); lev = o.get('leverage') or 1
        if px and entry:
            px = float(px)
            move = (px - entry) / entry if o['side'] == 'long' else (entry - px) / entry
            o['current_px'] = px
            o['live_pnl_pct'] = round(move * 100.0 * lev, 3)   # gehebelte Rendite auf Margin
            o['live_pnl_usd'] = round(move * size * lev, 3)
            unrealized += move * size * lev
        else:
            o['current_px'] = None; o['live_pnl_pct'] = None; o['live_pnl_usd'] = None

    realized = float(cl['sum_usd']); n_closed = cl['n']; wl = cl['wins'] + cl['losses']
    return {
        "gate": {"min_net_expectancy_pct": tg["min_net_expectancy_pct"], "min_reach_rate": tg["min_reach_rate"]},
        "trade_size_usd": size, "start_balance": start_bal,
        "balance": round(start_bal + realized, 2),
        "equity": round(start_bal + realized + unrealized, 2),
        "unrealized_usd": round(unrealized, 2),
        "realized_usd": round(realized, 2),
        "total_return_pct": round((realized + unrealized) / start_bal * 100, 2) if start_bal else 0,
        "closed": n_closed, "open": len(opens),
        "wins": cl['wins'], "losses": cl['losses'], "drifts": cl['drifts'],
        "win_rate_pct": round(100.0 * cl['wins'] / wl, 1) if wl else None,
        "tp_rate_pct": round(100.0 * cl['wins'] / n_closed, 1) if n_closed else None,
        "avg_pnl_pct": round(float(cl['avg_pct']), 4),
        "open_positions": opens,
        "history": history,
        "equity_curve": curve,
    }
