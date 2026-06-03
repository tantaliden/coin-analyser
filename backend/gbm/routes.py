"""GBM-Predictor Paper-Shadow API (read-only).
Eigene Paper-Welt (gbm_paper_positions/wallet/regime_state) — Live-Werte der
offenen Positionen + Historie + Equity-Kurve + Regime-Zustand (das Besondere
dieses Predictors: er dreht das Vorzeichen bei Trendwenden).
settings.json live gelesen. Keine Fallbacks."""
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
    g = _s()["gbm_predictor"]
    sv = g["service"]
    start_bal = float(sv["paper_start_balance"])
    size = float(g["dollar"]["trade_size_usd"])
    with _db_app() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT count(*) n,
                count(*) FILTER (WHERE status='win') wins,
                count(*) FILTER (WHERE status='loss') losses,
                count(*) FILTER (WHERE status='timeout') drifts,
                COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct
                FROM gbm_paper_positions WHERE status<>'open'""")
            cl = dict(cur.fetchone())
            cur.execute("""SELECT id,symbol,side,entry_px,tp_px,sl_px,qty,margin_usd,
                                  dir_conf,p_resolve,regime_sign,opened_at
                           FROM gbm_paper_positions WHERE status='open' ORDER BY opened_at DESC""")
            opens = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT symbol,side,entry_px,exit_px,pnl_pct,pnl_usd,status,
                                  regime_sign,opened_at,closed_at
                           FROM gbm_paper_positions WHERE status<>'open'
                           ORDER BY closed_at DESC LIMIT 100""")
            history = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT closed_at, pnl_usd FROM gbm_paper_positions
                           WHERE status<>'open' ORDER BY closed_at""")
            curve = []; eq = start_bal
            for r in cur.fetchall():
                eq += float(r['pnl_usd'] or 0)
                curve.append({"t": r['closed_at'].isoformat() if r['closed_at'] else None, "equity": round(eq, 2)})
            cur.execute("SELECT current_sign, n_closed, hit_rate FROM gbm_regime_state WHERE id=1")
            reg = dict(cur.fetchone() or {})

    # Live-Werte der offenen Positionen (qty-exakt, inkl. Hebel)
    prices = _current_prices({o['symbol'] for o in opens})
    unrealized = 0.0
    for o in opens:
        entry = float(o['entry_px']); px = prices.get(o['symbol']); qty = float(o['qty'])
        margin = float(o['margin_usd'])
        if px and entry:
            px = float(px)
            pnl_usd = qty * (px - entry) if o['side'] == 'long' else qty * (entry - px)
            o['current_px'] = px
            o['live_pnl_usd'] = round(pnl_usd, 3)
            o['live_pnl_pct'] = round(pnl_usd / margin * 100.0, 3) if margin else None
            o['conf'] = o['dir_conf']
            unrealized += pnl_usd
        else:
            o['current_px'] = None; o['live_pnl_pct'] = None; o['live_pnl_usd'] = None; o['conf'] = o['dir_conf']

    realized = float(cl['sum_usd']); n_closed = cl['n']; wl = cl['wins'] + cl['losses']
    return {
        "gate_threshold": sv["gate_threshold"], "dir_threshold": sv["dir_threshold"],
        "trade_size_usd": size, "leverage": sv["leverage"], "max_open": sv["max_open"],
        "tp_pct": g["label"]["tp_pct"], "sl_pct": g["label"]["sl_pct"],
        "start_balance": start_bal,
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
        "regime_sign": reg.get("current_sign"), "regime_hit": reg.get("hit_rate"),
        "regime_n": reg.get("n_closed"),
        "open_positions": opens,
        "history": history,
        "equity_curve": curve,
        "backtest_expectation": {"auc": 0.63, "dir_hit_pct": 58.8, "note": "adaptives Vorzeichen, OOS"},
    }
