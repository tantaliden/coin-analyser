#!/usr/bin/env python3
"""CLAUDE PAPER WALLET (Volker 24.06.2026): mein Experimentier-Sandkasten. Mehrere Strategien
LAUFEN PARALLEL (jede eigener Slot-Budget + Tag), ich vergleiche live, kuendige schlechte ab,
baue neue. Ziel: ins PLUS. Kein Echtgeld. Welt claude_paper_*. Alles aus settings.claude_wallet.

STRATEGIEN (frei iterierbar — strategy-tag pro Position gespeichert):
  v1_meanrev : Fade kurzfristiger Extrem-Moves (+-X% in 30min -> Gegenrichtung). Mean-Reversion.
  v2_momentum: GLEICHE Extrem-Moves + Volumenanstieg -> MITgehen (Continuation). Gegenprobe zu v1.
  v3_rsi     : RSI(14,5m) Extrem -> <low long / >high short. Klassisches Oversold/Overbought-Fade.
"""
import json, time, datetime as dt
import psycopg2
from psycopg2.extras import RealDictCursor

SETTINGS="/opt/coin/settings.json"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def dbc(s):
    d=s["databases"]["coins"]; c=psycopg2.connect(host=d["host"],port=d["port"],dbname=d["name"],user=d["user"],password=d["password"]); c.autocommit=True; return c
def dba(s):
    d=s["databases"]["app"]; c=psycopg2.connect(host=d["host"],port=d["port"],dbname=d["name"],user=d["user"],password=d["password"]); c.autocommit=True; return c

DDL="""
CREATE TABLE IF NOT EXISTS claude_paper_positions(
  id BIGSERIAL PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL, strategy TEXT,
  entry_px DOUBLE PRECISION NOT NULL, tp_px DOUBLE PRECISION, sl_px DOUBLE PRECISION,
  tp_pct DOUBLE PRECISION, sl_pct DOUBLE PRECISION, leverage INT,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT now(), status TEXT NOT NULL DEFAULT 'open',
  exit_px DOUBLE PRECISION, pnl_pct DOUBLE PRECISION, pnl_usd DOUBLE PRECISION,
  closed_at TIMESTAMPTZ, exit_reason TEXT, trigger_ret DOUBLE PRECISION);
CREATE INDEX IF NOT EXISTS claude_pp_status ON claude_paper_positions(status);
CREATE TABLE IF NOT EXISTS claude_paper_meta(id INT PRIMARY KEY DEFAULT 1, start_balance DOUBLE PRECISION);
"""

# --- Marktdaten-Helfer ---
def top_coins(coins, n):
    cur=coins.cursor()
    cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
                   GROUP BY symbol ORDER BY SUM(quote_asset_volume) DESC NULLS LAST LIMIT %s""",(n,))
    return [r[0] for r in cur.fetchall()]
def coin_lev(coins, sym, cap):
    cur=coins.cursor(); cur.execute("SELECT max_leverage FROM hl_meta WHERE symbol=%s",(sym,)); r=cur.fetchone()
    return min(cap,int(r[0])) if r and r[0] is not None else None
def price_now(coins, sym):
    cur=coins.cursor(); cur.execute("SELECT close FROM klines WHERE symbol=%s AND interval='10s' ORDER BY open_time DESC LIMIT 1",(sym,))
    r=cur.fetchone()
    if r and r[0] is not None: return float(r[0])
    cur.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1",(sym,)); r=cur.fetchone()
    return float(r[0]) if r and r[0] is not None else None

def market_snapshot(coins, uni):
    """Pro Coin: ret30 (%), vol_ratio (letzte 30min vs davor), rsi14 (5m). Ein Pass."""
    out={}
    cur=coins.cursor()
    for sym in uni:
        cur.execute("""SELECT close, volume FROM agg_5m WHERE symbol=%s ORDER BY bucket DESC LIMIT 18""",(sym,))
        rows=cur.fetchall()
        if len(rows)<14: continue
        cl=[float(r[0]) for r in rows][::-1]; vo=[float(r[1] or 0) for r in rows][::-1]
        ret30=(cl[-1]/cl[-7]-1)*100 if len(cl)>=7 and cl[-7]>0 else None
        v2=sum(vo[-6:]); v1=sum(vo[-12:-6]); vr=(v2/v1) if v1>0 else None
        d=[cl[i+1]-cl[i] for i in range(len(cl)-1)][-14:]
        up=sum(x for x in d if x>0); dn=-sum(x for x in d if x<0)
        rsi=up/(up+dn)*100 if (up+dn)>0 else 50.0
        out[sym]=dict(ret30=ret30, vr=vr, rsi=rsi)
    return out

# ============ STRATEGIEN ============ (jede: returns list (sym, side, tp_pct, sl_pct, trig))
def strat_v1_meanrev(snap, held, cfg):
    t=float(cfg["trigger_pct"]); tp=float(cfg["tp_pct"]); sl=float(cfg["sl_pct"]); out=[]
    for sym,m in snap.items():
        if sym in held or m["ret30"] is None: continue
        if m["ret30"]>=t:    out.append((sym,'short',tp,sl,m["ret30"]))
        elif m["ret30"]<=-t: out.append((sym,'long', tp,sl,m["ret30"]))
    out.sort(key=lambda x:-abs(x[4])); return out
def strat_v2_momentum(snap, held, cfg):
    t=float(cfg["trigger_pct"]); vmin=float(cfg["vol_mult"]); tp=float(cfg["tp_pct"]); sl=float(cfg["sl_pct"]); out=[]
    for sym,m in snap.items():
        if sym in held or m["ret30"] is None or m["vr"] is None or m["vr"]<vmin: continue
        if m["ret30"]>=t:    out.append((sym,'long', tp,sl,m["ret30"]))   # mitgehen
        elif m["ret30"]<=-t: out.append((sym,'short',tp,sl,m["ret30"]))
    out.sort(key=lambda x:-abs(x[4])); return out
def strat_v3_rsi(snap, held, cfg):
    lo=float(cfg["rsi_low"]); hi=float(cfg["rsi_high"]); tp=float(cfg["tp_pct"]); sl=float(cfg["sl_pct"]); out=[]
    for sym,m in snap.items():
        if sym in held: continue
        if m["rsi"]<=lo:   out.append((sym,'long', tp,sl,m["rsi"]))
        elif m["rsi"]>=hi: out.append((sym,'short',tp,sl,m["rsi"]))
    out.sort(key=lambda x:abs(50-x[4]) if False else -abs(50-x[4])); return out
STRATS={"v1_meanrev":strat_v1_meanrev,"v2_momentum":strat_v2_momentum,"v3_rsi":strat_v3_rsi}
# =====================================

# ===== v5_conj: Konjunktion taker_30 x breadth_30 -> SHORT (1:1 aus conjunction_mining2.py) =====
# Belegter Edge (H2-validiert, 62-73%): taker_30<=p10 UND breadth_30<=p10 -> short, TP1/SL1/4h.
# taker_30 = sum(taker_buy_base[30m])/sum(volume[30m])*100 ; breadth_30 = % Coins mit +30m-Return.
def conj_pctl(coins, uni, days, q_tk, q_br):
    """p-Quantile (p10) von taker_30 (alle Coins x Zeit) und breadth_30 (global) ueber `days`.
    Positional rolling(30) via SQL — gleiche Konvention wie conj_now (dichte agg_1m). RULE_6: None bei NULL."""
    cur=coins.cursor()
    cur.execute("""
      WITH base AS (
        SELECT sum(taker_buy_base) OVER w AS tb30, sum(volume) OVER w AS v30
        FROM agg_1m WHERE symbol=ANY(%s) AND bucket > now()-make_interval(days=>%s)
        WINDOW w AS (PARTITION BY symbol ORDER BY bucket ROWS BETWEEN 29 PRECEDING AND CURRENT ROW))
      SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY (tb30/NULLIF(v30,0))*100) FROM base WHERE v30>0
    """,(uni,days,q_tk))
    tk=cur.fetchone()[0]
    cur.execute("""
      WITH r AS (
        SELECT bucket, close/NULLIF(lag(close,30) OVER (PARTITION BY symbol ORDER BY bucket),0)-1 AS ret30
        FROM agg_1m WHERE symbol=ANY(%s) AND bucket > now()-make_interval(days=>%s)),
      b AS (SELECT count(*) FILTER (WHERE ret30>0)::float/NULLIF(count(ret30),0)*100 AS breadth
            FROM r GROUP BY bucket HAVING count(ret30)>=10)
      SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY breadth) FROM b
    """,(uni,days,q_br))
    br=cur.fetchone()[0]
    if tk is None or br is None:
        log("FALLBACK_TRIGGERED conj_pctl: p-thresholds NULL -> v5_conj pausiert"); return None
    return (float(tk), float(br))

def conj_now(coins, uni):
    """JETZT: taker_30 je Coin + globaler breadth_30. agg_1m, 30-min-Fenster (positional)."""
    cur=coins.cursor(); taker={}; rets={}
    for sym in uni:
        cur.execute("SELECT close,volume,taker_buy_base FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 31",(sym,))
        rows=cur.fetchall()[::-1]
        if len(rows)<31: continue
        cl=[r[0] for r in rows]; vol=[float(r[1] or 0) for r in rows]; tbb=[float(r[2] or 0) for r in rows]
        v=sum(vol[-30:])
        if v>0: taker[sym]=sum(tbb[-30:])/v*100
        if cl[-1] is not None and cl[-31] is not None and cl[-31]>0: rets[sym]=cl[-1]/cl[-31]-1
    nn=len(rets)
    breadth=(sum(1 for x in rets.values() if x>0)/nn*100) if nn>=10 else None
    return taker, breadth

def conj_candidates(taker, breadth, thr, held, cfg):
    """Konjunktion taker_30<=thr UND breadth_30<=thr. Seite aus cfg['side'] (Extrem-p10=short bewaehrt;
    lockere Schwelle p20 drehte 8/8 SL -> Dips bouncen -> 'long' faden). Liefert (sym,side,tp,sl,trig)."""
    if breadth is None or thr is None: return []
    tk_thr, br_thr = thr
    if breadth > br_thr: return []          # Markt-Breadth nicht im Extrem -> Konjunktion feuert nicht
    side=str(cfg["side"]); tp=float(cfg["tp_pct"]); sl=float(cfg["sl_pct"]); out=[]
    for sym,tk in taker.items():
        if sym in held: continue
        if tk <= tk_thr: out.append((sym,side,tp,sl,tk))
    out.sort(key=lambda x:x[4])             # extremster (niedrigster taker) zuerst
    return out
# =====================================

def timeout_for(C, strategy, global_to):
    """Per-Strategie-Timeout (z.B. v5_conj 4h) statt globalem; fehlt der Key -> globaler Wert."""
    sc=C.get(strategy)
    if sc and "timeout_hours" in sc: return float(sc["timeout_hours"])
    return global_to

def market_vola(coins, hours):
    """Markt-Vola = avg BTC 1h-Range (%) ueber `hours`. RULE_6: None bei fehlenden Daten (Gate faellt offen)."""
    cur=coins.cursor()
    cur.execute("SELECT avg((high-low)/NULLIF(low,0)*100) FROM agg_1h WHERE symbol='BTC' AND bucket>now()-make_interval(hours=>%s)",(hours,))
    r=cur.fetchone()
    if not r or r[0] is None:
        log("FALLBACK_TRIGGERED market_vola: keine BTC-agg_1h -> Gate offen"); return None
    return float(r[0])

# ===== v6_squeeze: Funding-Extrem + OI-Anstieg -> overcrowded-Reversal (Derivate-Positionierung) =====
# Hypothese (NICHT vorab validiert, Vergleichsmodell): funding_z>=+sigma & OI steigt -> Longs ueberfuellt
# -> SHORT (Long-Squeeze); funding_z<=-sigma & OI steigt -> Shorts ueberfuellt -> LONG (Short-Squeeze).
def sqz_baseline(coins, uni, days):
    """Pro Coin funding-Mittel+Std ueber `days` (z-Score-Baseline). Cache, periodisch refresht."""
    cur=coins.cursor()
    cur.execute("""SELECT symbol, avg(funding), stddev_samp(funding) FROM agg_1m
                   WHERE symbol=ANY(%s) AND bucket>now()-make_interval(days=>%s) AND funding IS NOT NULL
                   GROUP BY symbol""",(uni,days))
    out={r[0]:(float(r[1]), float(r[2])) for r in cur.fetchall() if r[1] is not None and r[2] and r[2]>0}
    if not out: log("FALLBACK_TRIGGERED sqz_baseline: keine funding-Baseline -> v6_squeeze pausiert")
    return out

def sqz_now(coins, uni):
    """JETZT pro Coin: aktuelles funding + oi_d_60 (% OI-Aenderung ueber 60min)."""
    cur=coins.cursor(); out={}
    cur.execute("""WITH base AS (
        SELECT symbol, funding, open_interest,
               lag(open_interest,60) OVER (PARTITION BY symbol ORDER BY bucket) AS oi60,
               row_number() OVER (PARTITION BY symbol ORDER BY bucket DESC) rn
        FROM agg_1m WHERE symbol=ANY(%s) AND bucket>now()-interval '90 minutes')
        SELECT symbol, funding, CASE WHEN oi60>0 THEN (open_interest/oi60-1)*100 END AS oi_d
        FROM base WHERE rn=1""",(uni,))
    for r in cur.fetchall():
        if r[1] is not None: out[r[0]]=(float(r[1]), (float(r[2]) if r[2] is not None else None))
    return out

def sqz_candidates(now_map, baseline, cfg, held):
    """funding_z>=+zsig & OI steigt -> SHORT ; <=-zsig & OI steigt -> LONG. (sym,side,tp,sl,trig)."""
    if not baseline: return []
    zsig=float(cfg["funding_z_sigma"]); oi_min=float(cfg["oi_rise_min_pct"])
    sides=set(cfg["sides"])                              # Volker-Daten 26.06: short verliert -> ["long"]
    tp=float(cfg["tp_pct"]); sl=float(cfg["sl_pct"]); out=[]
    for sym,(fund,oi_d) in now_map.items():
        if sym in held or oi_d is None: continue
        b=baseline.get(sym)
        if b is None: continue
        mu,sd=b; fz=(fund-mu)/sd if sd>0 else 0.0
        if oi_d < oi_min: continue                       # nur bei aktivem OI-Aufbau
        if fz >= zsig and 'short' in sides:    out.append((sym,'short',tp,sl,fz))
        elif fz <= -zsig and 'long' in sides:  out.append((sym,'long', tp,sl,fz))
    out.sort(key=lambda x:-abs(x[4]))                    # extremstes funding zuerst
    return out

# ===== v4_whale: Wal-Eskalation als Entry (Smart-Money folgen), Exit = eigenes TP/SL/Timeout =====
# Eskalationen liefert whale_esc_sync.service (Cross-Server-Feed). Entry in Wal-Richtung, NICHT die
# Wal-Position spiegeln -> Exit vom Wal entkoppelt (TP +4 / SL -2 / 12h). Belegter Edge 70%/12h.
def open_v4_whale(coins, app, cfg, cap, allheld, free):
    """Unverarbeitete Eskalationen (ts_event < max_age) -> Position in Wal-Richtung. Returns opened."""
    tp_pct=float(cfg["tp_pct"]); sl_pct=float(cfg["sl_pct"]); age_min=float(cfg["max_event_age_minutes"])
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT id,symbol,side,pct FROM whale_escalations
                       WHERE processed=false AND ts_event > now()-(%s * interval '1 minute')
                       ORDER BY ts_event""",(age_min,))
        rows=cur.fetchall()
    opened=0
    for r in rows:
        sym=r["symbol"]; side=r["side"]
        if free<=0: break
        if sym in allheld:
            with app.cursor() as cu: cu.execute("UPDATE whale_escalations SET processed=true WHERE id=%s",(r["id"],))
            continue
        lev=coin_lev(coins,sym,cap)
        if lev is None: log(f"v4_whale {sym}: kein hl_meta-Hebel -> skip"); continue
        px=price_now(coins,sym)
        if px is None: continue
        if side=='long': tp=px*(1+tp_pct/100); slx=px*(1-sl_pct/100)
        else:            tp=px*(1-tp_pct/100); slx=px*(1+sl_pct/100)
        with app.cursor() as cu:
            cu.execute("""INSERT INTO claude_paper_positions(symbol,side,strategy,entry_px,tp_px,sl_px,tp_pct,sl_pct,leverage,trigger_ret)
                          VALUES(%s,%s,'v4_whale',%s,%s,%s,%s,%s,%s,%s)""",(sym,side,px,tp,slx,tp_pct,sl_pct,lev,float(r["pct"] or 0)))
            cu.execute("UPDATE whale_escalations SET processed=true WHERE id=%s",(r["id"],))
        allheld.add(sym); free-=1; opened+=1
        log(f"OPEN [v4_whale] {side} {sym} @ {px:.6f} {lev}x tp+{tp_pct}/sl-{sl_pct} (wal {float(r['pct'] or 0)*100:.0f}%)")
    return opened

def check_close(coins, p, timeout_h, be_pct):
    """TP/SL/Timeout auf 10s-klines, bar-fuer-bar KAUSAL. Breakeven: sobald der Move >= be_pct erreicht
    hat, gilt ab dem FOLGENDEN Bar der SL auf Einstand (+/-0.05%) -> Exit-Grund 'BE'. TP wird in jedem
    Bar VOR dem Breakeven geprueft. Kein nachtraegliches Verbiegen des DB-SL (das triggerte alte Bars
    rueckwirkend und frass jeden TP - Bug, Volker 25.06.)."""
    cur=coins.cursor()
    cur.execute("SELECT high,low,close,open_time FROM klines WHERE symbol=%s AND interval='10s' AND open_time>%s ORDER BY open_time",(p['symbol'],p['opened_at']))
    rows=cur.fetchall()
    now=dt.datetime.now(dt.timezone.utc)
    wall_age_h=(now-p['opened_at']).total_seconds()/3600
    if not rows:
        # Coin-Daten eingefroren (delisted/tot): keine Kerzen nach Entry -> Timeout nach Wanduhr,
        # Exit zum letzten bekannten Preis (sonst haengt die Position ewig, Volker 01.07.).
        if wall_age_h>=timeout_h:
            lp=price_now(coins,p['symbol'])
            return (lp if lp is not None else float(p['entry_px'])),'TIMEOUT',now
        return None
    e=float(p['entry_px']); tp=float(p['tp_px']); sl0=float(p['sl_px']); long=(p['side']=='long')
    be_sl = e*1.0005 if long else e*0.9995
    be_active=False
    for h,l,c,ot in rows:
        if h is None or l is None: continue
        h=float(h); l=float(l)
        cur_sl = be_sl if be_active else sl0
        if long:
            if l<=cur_sl: return cur_sl,('BE' if be_active else 'SL'),ot
            if h>=tp:     return tp,'TP',ot
            if be_pct>0 and (h/e-1)*100 >= be_pct: be_active=True
        else:
            if h>=cur_sl: return cur_sl,('BE' if be_active else 'SL'),ot
            if l<=tp:     return tp,'TP',ot
            if be_pct>0 and (1-l/e)*100 >= be_pct: be_active=True
    if wall_age_h>=timeout_h: return float(rows[-1][2]),'TIMEOUT',rows[-1][3]
    return None

def main():
    s=json.load(open(SETTINGS)); C=s["claude_wallet"]
    size=float(C["trade_size_usd"]); cap=int(C["leverage_cap"]); fee=float(C["fee_roundtrip_pct"])
    scan=int(C["scan_interval_seconds"]); top_n=int(C["top_n_coins"]); timeout_h=float(C["timeout_hours"])
    active=C["active_strategies"]   # dict strat -> max_open
    log(f"CLAUDE-WALLET start: {list(active.keys())} | size${size} lev<={cap} scan{scan}s timeout{timeout_h}h")
    with dba(s) as app:
        with app.cursor() as cur:
            cur.execute(DDL)
            cur.execute("INSERT INTO claude_paper_meta(id,start_balance) VALUES(1,%s) ON CONFLICT(id) DO NOTHING",(float(C["paper_start_balance"]),))
    conj_thr=None; conj_thr_at=0.0   # v5_conj p10-Cache (taker_thr, breadth_thr)
    sqz_base=None; sqz_base_at=0.0    # v6_squeeze funding-Baseline-Cache {sym:(mu,sd)}
    regime_note=None                  # zuletzt geloggter Regime-Gate-Zustand
    while True:
        t0=time.time()
        try:
            with dbc(s) as coins, dba(s) as app:
                # 1) schliessen (alle Strategien, generisch)
                with app.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM claude_paper_positions WHERE status='open'")
                    opens=cur.fetchall()
                closed=0
                be_pct=float(C["breakeven_pct"])
                for p in opens:
                    r=check_close(coins,p,timeout_for(C,p['strategy'],timeout_h),be_pct)
                    if r is None: continue
                    px,reason,ct=r; lev=p['leverage'] or 1
                    move=(px/p['entry_px']-1)*100 if p['side']=='long' else (1-px/p['entry_px'])*100
                    pnl=(move-fee)*lev; usd=size*pnl/100
                    with app.cursor() as cu:
                        cu.execute("UPDATE claude_paper_positions SET status='closed',exit_px=%s,pnl_pct=%s,pnl_usd=%s,closed_at=%s,exit_reason=%s WHERE id=%s",(px,pnl,usd,ct,reason,p['id']))
                    closed+=1
                # 2) Markt-Snapshot einmal, dann je Strategie eigenes Budget
                uni=top_coins(coins,top_n)
                snap=market_snapshot(coins,uni)
                # v5_conj: p10-Schwellen periodisch refreshen + aktuelle taker/breadth
                conj_taker={}; conj_breadth=None
                if "v5_conj" in active:
                    cc=C["v5_conj"]
                    if conj_thr is None or (time.time()-conj_thr_at) > float(cc["pctl_refresh_hours"])*3600:
                        nt=conj_pctl(coins,uni,int(cc["pctl_lookback_days"]),float(cc["taker_pctl"]),float(cc["breadth_pctl"]))
                        if nt: conj_thr=nt; conj_thr_at=time.time(); log(f"conj p10: taker<={nt[0]:.2f} breadth<={nt[1]:.2f}")
                    conj_taker,conj_breadth=conj_now(coins,uni)
                # v6_squeeze: funding-Baseline periodisch + aktuelle funding/oi
                sqz_map={}
                if "v6_squeeze" in active:
                    sc=C["v6_squeeze"]
                    if sqz_base is None or (time.time()-sqz_base_at) > float(sc["baseline_refresh_hours"])*3600:
                        nb=sqz_baseline(coins,uni,int(sc["baseline_days"]))
                        if nb: sqz_base=nb; sqz_base_at=time.time(); log(f"sqz baseline: {len(nb)} Coins")
                    sqz_map=sqz_now(coins,uni)
                # offene je Strategie
                heldby={}
                with app.cursor() as cur:
                    cur.execute("SELECT strategy,symbol FROM claude_paper_positions WHERE status='open'")
                    for st,sym in cur.fetchall(): heldby.setdefault(st,set()).add(sym)
                allheld={sym for sset in heldby.values() for sym in sset}
                # per-Coin-Cooldown nach Close: verhindert dass eine Strategie denselben Coin haemmert
                cooldown=int(C["cooldown_seconds_per_symbol"]); recent={}
                if cooldown>0:
                    with app.cursor() as cur:
                        cur.execute("""SELECT strategy,symbol FROM claude_paper_positions
                                       WHERE closed_at > now()-(%s * interval '1 second') GROUP BY strategy,symbol""",(cooldown,))
                        for st,sym in cur.fetchall(): recent.setdefault(st,set()).add(sym)
                # Regime-Gate: pausiert gelistete Strats wenn Markt-Vola zu niedrig fuer ihren TP (Erreichbarkeit)
                rg=C.get("regime",{}); gated=set(rg.get("gated_strategies",[])); blocked=set()
                if gated:
                    mvola=market_vola(coins,int(rg["vola_window_hours"])); saf=float(rg["reach_safety_factor"])
                    if mvola is not None:
                        for gst in gated:
                            req=(float(C[gst]["tp_pct"])/(timeout_for(C,gst,timeout_h)**0.5))*saf
                            if mvola<req: blocked.add(gst)
                    if blocked!=regime_note:
                        log(f"REGIME-GATE: BTC-Vola {mvola}% -> pausiert {sorted(blocked) or 'nichts'}"); regime_note=blocked
                opened=0
                for st,maxo in active.items():
                    held_st=heldby.get(st,set())
                    free=int(maxo)-len(held_st)
                    if free<=0: continue
                    if st in blocked: continue   # Regime-Gate: Markt zu ruhig, TP nicht erreichbar
                    if st=="v4_whale":
                        opened+=open_v4_whale(coins, app, C["v4_whale"], cap, allheld, free); continue
                    if st=="v5_conj":
                        cand=conj_candidates(conj_taker, conj_breadth, conj_thr, allheld, C["v5_conj"])
                    elif st=="v6_squeeze":
                        cand=sqz_candidates(sqz_map, sqz_base, C["v6_squeeze"], allheld)
                    else:
                        cand=STRATS[st](snap, allheld, C[st])   # nicht 2x denselben Coin global
                    for sym,side,tp_pct,sl_pct,trig in cand:
                        if free<=0: break
                        if sym in recent.get(st,set()): continue   # per-Coin-Cooldown
                        lev=coin_lev(coins,sym,cap)
                        if lev is None: continue
                        px=price_now(coins,sym)
                        if px is None: continue
                        if side=='long': tp=px*(1+tp_pct/100); slx=px*(1-sl_pct/100)
                        else:            tp=px*(1-tp_pct/100); slx=px*(1+sl_pct/100)
                        with app.cursor() as cu:
                            cu.execute("""INSERT INTO claude_paper_positions(symbol,side,strategy,entry_px,tp_px,sl_px,tp_pct,sl_pct,leverage,trigger_ret)
                                          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",(sym,side,st,px,tp,slx,tp_pct,sl_pct,lev,trig))
                        allheld.add(sym); free-=1; opened+=1
                        log(f"OPEN [{st}] {side} {sym} @ {px:.6f} {lev}x tp+{tp_pct}/sl-{sl_pct} (trig {trig:+.1f})")
                if closed or opened: log(f"scan: +{opened} open, {closed} closed, offen={len(opens)-closed+opened}")
        except Exception as e:
            log(f"loop error: {e}")
        time.sleep(max(1,scan-(time.time()-t0)))

if __name__=="__main__": main()
