"""GBM-Predictor Paper-Shadow API (read-only).
Cross-Sectional Composite (marktneutral): eigene Paper-Welt (gbm_paper_positions/
gbm_paper_wallet_state). Live-Werte der offenen Positionen + Historie + Equity-Kurve.
Config aus settings.gbm_predictor.xsection (live gelesen). Keine Fallbacks/Hardcodes."""
import json
from fastapi import APIRouter
from psycopg2.extras import RealDictCursor
import psycopg2

router = APIRouter(prefix="/api/v1/gbm", tags=["gbm"])
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
    """Aktueller Preis je Symbol aus den neuesten agg_1m (close). Fehlt er -> None."""
    if not symbols:
        return {}
    try:
        with _db_coins() as coins:
            with coins.cursor(cursor_factory=RealDictCursor) as c:
                c.execute("""SELECT DISTINCT ON (symbol) symbol, close
                             FROM agg_1m WHERE symbol = ANY(%s)
                             ORDER BY symbol, bucket DESC""", (list(symbols),))
                return {r['symbol']: r['close'] for r in c.fetchall()}
    except Exception:
        return {}


@router.get("/paper")
def gbm_paper():
    x = _s()["gbm_predictor"]["xsection"]
    with _db_app() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT balance, start_balance, peak_balance, n_trades FROM gbm_paper_wallet_state WHERE id=1")
            ws = dict(cur.fetchone() or {})
            start_bal = float(ws.get("start_balance") or 0)
            balance = float(ws.get("balance") if ws.get("balance") is not None else start_bal)
            cur.execute("""SELECT count(*) n,
                count(*) FILTER (WHERE status='win') wins,
                count(*) FILTER (WHERE status='loss') losses,
                COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct
                FROM gbm_paper_positions WHERE status<>'open'""")
            cl = dict(cur.fetchone())
            cur.execute("""SELECT id,symbol,side,entry_px,qty,margin_usd,dir_conf,opened_at
                           FROM gbm_paper_positions WHERE status='open'
                           ORDER BY side, dir_conf DESC""")
            opens = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT symbol,side,entry_px,exit_px,pnl_pct,pnl_usd,status,opened_at,closed_at
                           FROM gbm_paper_positions WHERE status<>'open'
                           ORDER BY closed_at DESC LIMIT 100""")
            history = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT closed_at, pnl_usd FROM gbm_paper_positions
                           WHERE status<>'open' ORDER BY closed_at""")
            curve = []; eq = start_bal
            for r in cur.fetchall():
                eq += float(r['pnl_usd'] or 0)
                curve.append({"t": r['closed_at'].isoformat() if r['closed_at'] else None, "equity": round(eq, 2)})

    # Live-Werte der offenen Positionen (qty-exakt, inkl. Hebel)
    prices = _current_prices({o['symbol'] for o in opens})
    unrealized = 0.0
    for o in opens:
        entry = float(o['entry_px']); px = prices.get(o['symbol']); qty = float(o['qty'])
        margin = float(o['margin_usd'])
        o['score'] = round(float(o['dir_conf']), 3) if o['dir_conf'] is not None else None
        if px and entry:
            px = float(px)
            pnl_usd = qty * (px - entry) if o['side'] == 'long' else qty * (entry - px)
            o['current_px'] = px
            o['live_pnl_usd'] = round(pnl_usd, 3)
            o['live_pnl_pct'] = round(pnl_usd / margin * 100.0, 3) if margin else None
            unrealized += pnl_usd
        else:
            o['current_px'] = None; o['live_pnl_pct'] = None; o['live_pnl_usd'] = None

    realized = float(cl['sum_usd']); n_closed = cl['n']; wl = cl['wins'] + cl['losses']
    return {
        "strategy": "cross_sectional_composite_marktneutral",
        "market_coin": x["market_coin"], "universe_top_n": x["universe_top_n"],
        "lookback_hours_list": x["lookback_hours_list"], "beta_window_hours": x["beta_window_hours"],
        "weight_roh": x["weight_roh"], "weight_residual": x["weight_residual"],
        "k": x["k"], "rebalance_hours": x["rebalance_hours"],
        "trade_size_usd": float(x["trade_size_usd"]), "leverage": x["leverage"],
        "fee_roundtrip_pct": x["fee_roundtrip_pct"],
        "start_balance": round(start_bal, 2),
        "balance": round(balance, 2),
        "equity": round(balance + unrealized, 2),
        "unrealized_usd": round(unrealized, 2),
        "realized_usd": round(realized, 2),
        "total_return_pct": round((balance + unrealized - start_bal) / start_bal * 100, 2) if start_bal else 0,
        "closed": n_closed, "open": len(opens),
        "wins": cl['wins'], "losses": cl['losses'],
        "leg_win_rate_pct": round(100.0 * cl['wins'] / wl, 1) if wl else None,
        "avg_pnl_pct": round(float(cl['avg_pct']), 4),
        "open_positions": opens,
        "history": history,
        "equity_curve": curve,
        "note": "Marktneutraler Cross-Sectional-Spread (Top-k long / Bottom-k short, alle Horizonte roh+residual gleichgewichtet). Einziger Ansatz der alle ehrlichen Tests ueberlebt hat (t>2, beide Haelften+).",
    }
