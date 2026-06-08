#!/usr/bin/env python3
"""Cross-Sectional Momentum Paper-Shadow (marktneutral) — Forward-Validierung des
einzigen Kandidaten, der alle ehrlichen Tests überlebt (liquid-only, t=2.77, beide
Hälften+). KEIN Echtgeld. Eigene Paper-Welt (xs_paper_*).

Alle rebalance_hours: alte Positionen schließen (PnL realisieren), Top-N liquideste
Coins nach Trailing-lookback_hours-Return ranken, Top-K LONG + Bottom-K SHORT öffnen
(gleiche Größe je Leg -> marktneutral). Alles aus settings.cross_sectional. KEINE
Defaults/Fallbacks."""
import json, time
import numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
SETTINGS="/opt/coin/settings.json"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def req(d,k,ctx):
    if k not in d: raise RuntimeError(f"settings.{ctx}.{k} fehlt — KEINE Defaults")
    return d[k]
def load_cfg():
    s=json.load(open(SETTINGS)); f=req(s,"cross_sectional","root")
    return s, dict(
        top_n=int(req(f,"universe_top_n","cross_sectional")),
        lookback_h=int(req(f,"lookback_hours","cross_sectional")),
        rebal_h=float(req(f,"rebalance_hours","cross_sectional")),
        k=int(req(f,"k","cross_sectional")),
        size=float(req(f,"trade_size_usd","cross_sectional")),
        lev=int(req(f,"leverage","cross_sectional")),
        fee=float(req(f,"fee_roundtrip_pct","cross_sectional")),
        min_vol=float(req(f,"min_dollar_volume_24h","cross_sectional")),
        scan=int(req(f,"scan_interval_seconds","cross_sectional")),
    )
def dbc(s,which):
    d=s["databases"][which]
    _c = psycopg2.connect(host=d["host"],port=d["port"],dbname=d["name"],user=d["user"],password=d["password"]); _c.autocommit = True; return _c
def price_now(coins,sym):
    with coins.cursor() as cur:
        cur.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1",(sym,))
        r=cur.fetchone()
    return float(r[0]) if r and r[0] else None
def trailing_ret(coins,sym,lb_h):
    with coins.cursor() as cur:
        cur.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1",(sym,)); now=cur.fetchone()
        cur.execute("SELECT close FROM agg_1m WHERE symbol=%s AND bucket<now()-make_interval(hours=>%s) ORDER BY bucket DESC LIMIT 1",(sym,lb_h)); old=cur.fetchone()
    if now and old and now[0] and old[0] and float(old[0])>0: return float(now[0])/float(old[0])-1.0
    return None
def top_universe(coins,cfg):
    with coins.cursor() as cur:
        cur.execute("""SELECT symbol,sum(quote_asset_volume) v FROM agg_1m WHERE bucket>now()-interval '24 hours'
                       GROUP BY symbol HAVING sum(quote_asset_volume)>=%s ORDER BY v DESC LIMIT %s""",(cfg["min_vol"],cfg["top_n"]))
        return [r[0] for r in cur.fetchall()]
def close_all(coins,app,cfg):
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM xs_paper_positions WHERE status='open'"); opens=cur.fetchall()
    if not opens: return 0
    total=0.0
    for p in opens:
        px=price_now(coins,p["symbol"])
        if px is None: continue
        notional=float(p["qty"])*p["entry_px"]
        gross=(px-p["entry_px"])/p["entry_px"] if p["side"]=="long" else (p["entry_px"]-px)/p["entry_px"]
        pnl=notional*(gross-cfg["fee"]/100.0); total+=pnl
        with app.cursor() as c2:
            c2.execute("UPDATE xs_paper_positions SET status='closed',exit_px=%s,pnl_usd=%s,closed_at=now() WHERE id=%s",(px,pnl,p["id"]))
    with app.cursor() as c2:
        c2.execute("""UPDATE xs_paper_wallet_state SET balance=balance+%s, n_rebalances=n_rebalances+1,
                      peak_balance=GREATEST(peak_balance,balance+%s), updated_at=now() WHERE id=1""",(total,total))
    app.commit(); log(f"REBALANCE close: {len(opens)} Positionen, realisiert ${total:.3f}")
    return len(opens)
def open_basket(coins,app,cfg):
    uni=top_universe(coins,cfg)
    rets=[(sym,trailing_ret(coins,sym,cfg["lookback_h"])) for sym in uni]
    rets=[(s,r) for s,r in rets if r is not None]
    if len(rets)<2*cfg["k"]: log(f"zu wenige Coins ({len(rets)}) — kein Rebalance"); return 0
    rets.sort(key=lambda x:x[1])
    longs=[s for s,_ in rets[-cfg["k"]:]]; shorts=[s for s,_ in rets[:cfg["k"]]]
    n=0
    for side,syms in [("long",longs),("short",shorts)]:
        for sym in syms:
            px=price_now(coins,sym)
            if px is None: continue
            qty=cfg["size"]*cfg["lev"]/px
            with app.cursor() as c2:
                c2.execute("""INSERT INTO xs_paper_positions(symbol,side,entry_px,qty,margin_usd)
                              VALUES(%s,%s,%s,%s,%s)""",(sym,side,px,qty,cfg["size"]))
            n+=1
    app.commit(); log(f"REBALANCE open: LONG {longs} | SHORT {shorts}")
    return n
def last_rebalance(app):
    with app.cursor() as cur:
        cur.execute("SELECT max(opened_at) FROM xs_paper_positions"); r=cur.fetchone()
    return r[0] if r else None
def main():
    s,cfg=load_cfg()
    log(f"Cross-Sectional-Shadow start: Top-{cfg['top_n']} K={cfg['k']} lookback={cfg['lookback_h']}h rebalance={cfg['rebal_h']}h lev={cfg['lev']}")
    coins=dbc(s,"coins"); app=dbc(s,"app")
    import datetime as dt
    while True:
        t0=time.time()
        try:
            s,cfg=load_cfg()
            lr=last_rebalance(app); now=dt.datetime.now(dt.timezone.utc)
            due = lr is None or (now-lr).total_seconds()>=cfg["rebal_h"]*3600
            if due:
                close_all(coins,app,cfg); open_basket(coins,app,cfg)
        except Exception as e:
            log(f"loop error: {e}")
            try: coins.rollback(); app.rollback()
            except Exception: coins=dbc(s,"coins"); app=dbc(s,"app")
        time.sleep(max(5,cfg["scan"]-(time.time()-t0)))
if __name__=="__main__": main()
