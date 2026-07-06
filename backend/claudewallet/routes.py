"""Claude Paper Wallet API (read-only). Mein Experimentier-Sandkasten. Kein Echtgeld."""
import json
from fastapi import APIRouter
from psycopg2.extras import RealDictCursor
import psycopg2

router = APIRouter(prefix="/api/v1/claudewallet", tags=["claudewallet"])
SETTINGS_PATH="/opt/coin/settings.json"
def _s(): return json.load(open(SETTINGS_PATH))
def _db(w):
    d=_s()["databases"][w]; return psycopg2.connect(host=d["host"],port=d["port"],dbname=d["name"],user=d["user"],password=d["password"])

def _prices(syms):
    if not syms: return {}
    try:
        with _db("coins") as c, c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT DISTINCT ON (symbol) symbol, mid_px, close FROM klines WHERE symbol=ANY(%s) AND interval='10s' ORDER BY symbol,open_time DESC",(list(syms),))
            return {r['symbol']:(r['mid_px'] or r['close']) for r in cur.fetchall()}
    except Exception: return {}

# --- Edge-Map: pro Muster (Strategie/Seite/Trigger-Staerke) Profitabilitaet -> behalten/drehen/wegwerfen ---
def _stats(pnls):
    n=len(pnls)
    if n==0: return {"n":0,"exp":None,"wr":None,"sum":0.0}
    wins=sum(1 for p in pnls if p>0)
    return {"n":n,"exp":round(sum(pnls)/n,3),"wr":round(100.0*wins/n),"sum":round(sum(pnls),2)}

def _verdict(s, cfg):
    if s["n"] < int(cfg["min_n"]): return "zu früh"
    e=s["exp"]
    if e > float(cfg["keep_exp"]): return "behalten"
    if e < float(cfg["flip_exp"]): return "drehen?"   # extremes Minus -> Gegenrichtung pruefen
    return "wegwerfen"                                 # nahe 0 = kein Edge

def compute_edgemap(rows, cfg):
    from collections import defaultdict
    byst=defaultdict(list)
    for r in rows: byst[r["strategy"]].append(r)
    out=[]
    for st in sorted(byst):
        trs=byst[st]
        node={"strategy":st}
        node["overall"]={**_stats([float(r["pnl_usd"]) for r in trs])}
        node["overall"]["verdict"]=_verdict(node["overall"], cfg)
        sides=[]
        for side in ("long","short"):
            sp=[float(r["pnl_usd"]) for r in trs if r["side"]==side]
            if not sp: continue
            srow={"side":side, **_stats(sp)}; srow["verdict"]=_verdict(srow, cfg)
            wt=[(abs(float(r["trigger_ret"])), float(r["pnl_usd"])) for r in trs if r["side"]==side and r["trigger_ret"] is not None]
            tiers=[]
            if len(wt) >= 2*int(cfg["min_n"]):
                med=sorted(a for a,_ in wt)[len(wt)//2]
                for tname,sel in (("schwach",[p for a,p in wt if a<med]),("stark",[p for a,p in wt if a>=med])):
                    tr={"tier":tname, **_stats(sel)}; tr["verdict"]=_verdict(tr,cfg); tiers.append(tr)
            srow["tiers"]=tiers; sides.append(srow)
        node["sides"]=sides; out.append(node)
    return out

@router.get("/paper")
def paper():
    C=_s()["claude_wallet"]; start=float(C["paper_start_balance"]); size=float(C["trade_size_usd"])
    with _db("app") as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT count(*) n, count(*) FILTER (WHERE exit_reason='TP') wins,
            count(*) FILTER (WHERE exit_reason='SL') losses, count(*) FILTER (WHERE exit_reason='TIMEOUT') timeouts,
            COALESCE(sum(pnl_usd),0) sum_usd, COALESCE(avg(pnl_pct),0) avg_pct FROM claude_paper_positions WHERE status='closed'""")
        cl=dict(cur.fetchone())
        cur.execute("SELECT id,symbol,side,strategy,entry_px,tp_px,sl_px,tp_pct,sl_pct,leverage,trigger_ret,opened_at FROM claude_paper_positions WHERE status='open' ORDER BY opened_at DESC")
        opens=[dict(r) for r in cur.fetchall()]
        cur.execute("SELECT symbol,side,strategy,entry_px,exit_px,pnl_pct,pnl_usd,exit_reason,trigger_ret,opened_at,closed_at FROM claude_paper_positions WHERE status='closed' ORDER BY closed_at DESC LIMIT 100")
        hist=[dict(r) for r in cur.fetchall()]
        cur.execute("SELECT closed_at,pnl_usd FROM claude_paper_positions WHERE status='closed' ORDER BY closed_at")
        curve=[]; eq=start
        for r in cur.fetchall():
            eq+=float(r['pnl_usd'] or 0); curve.append({"t":r['closed_at'].isoformat() if r['closed_at'] else None,"equity":round(eq,2)})
        # Pro-Strategie-Aufschluesselung
        cur.execute("""SELECT strategy,
            count(*) FILTER (WHERE status='closed') closed,
            count(*) FILTER (WHERE status='open') open,
            COALESCE(sum(pnl_usd) FILTER (WHERE status='closed'),0) realized,
            count(*) FILTER (WHERE status='closed' AND pnl_usd>0) wins,
            count(*) FILTER (WHERE status='closed' AND pnl_usd<=0) losses,
            count(*) FILTER (WHERE exit_reason='TP') tp, count(*) FILTER (WHERE exit_reason='SL') sl,
            count(*) FILTER (WHERE exit_reason='TIMEOUT') tmo
            FROM claude_paper_positions GROUP BY strategy ORDER BY strategy""")
        strat_rows=[dict(r) for r in cur.fetchall()]
        cur.execute("SELECT strategy,side,trigger_ret,pnl_usd FROM claude_paper_positions WHERE status='closed' AND pnl_usd IS NOT NULL")
        emrows=[dict(r) for r in cur.fetchall()]
    edgemap=compute_edgemap(emrows, C.get("edgemap",{"min_n":10,"keep_exp":0.0,"flip_exp":-0.05}))
    # aktive Strategien ohne Trades trotzdem als (leere) Box zeigen
    present={r['strategy'] for r in strat_rows}
    for st in C.get("active_strategies",{}):
        if st not in present:
            strat_rows.append({"strategy":st,"closed":0,"open":0,"realized":0,"wins":0,"losses":0,"tp":0,"sl":0,"tmo":0})
    strat_rows.sort(key=lambda r:r["strategy"])
    prices=_prices({o['symbol'] for o in opens}); unreal=0.0
    for o in opens:
        e=float(o['entry_px']); px=prices.get(o['symbol']); lev=o.get('leverage') or 1
        if px and e:
            px=float(px); move=(px-e)/e if o['side']=='long' else (e-px)/e
            o['current_px']=px; o['live_pnl_pct']=round(move*100*lev,3); o['live_pnl_usd']=round(size*move*lev,3); unreal+=size*move*lev
        else: o['current_px']=None; o['live_pnl_pct']=None; o['live_pnl_usd']=None
    realized=float(cl['sum_usd']); n=cl['n']; wl=cl['wins']+cl['losses']
    return {
        "strategy":",".join(C.get("active_strategies",{}).keys()), "note":"Claude Experimentier-Wallet — Strategie iterierbar, kein Echtgeld",
        "trade_size_usd":size,"leverage_cap":int(C["leverage_cap"]),"max_open":int(C["max_open"]),
        "timeout_hours":C["timeout_hours"],"start_balance":start,
        "balance":round(start+realized,2),"equity":round(start+realized+unreal,2),
        "unrealized_usd":round(unreal,2),"realized_usd":round(realized,2),
        "total_return_pct":round((realized+unreal)/start*100,2) if start else 0,
        "closed":n,"open":len(opens),"wins":cl['wins'],"losses":cl['losses'],"timeouts":cl['timeouts'],
        "win_rate_pct":round(100.0*cl['wins']/wl,1) if wl else None,
        "open_positions":opens,"history":hist,"equity_curve":curve,"edgemap":edgemap,
        "strategies":[{**r,
            "realized":round(float(r["realized"]),2),
            "wr":round(100.0*r["wins"]/(r["wins"]+r["losses"]),0) if (r["wins"]+r["losses"]) else None,
            "exp":round(float(r["realized"])/r["closed"],3) if r["closed"] else None,
            "verdict":("zu früh" if r["closed"]<20 else ("Edge+" if float(r["realized"])>0 else "Edge−"))
          } for r in strat_rows]}
