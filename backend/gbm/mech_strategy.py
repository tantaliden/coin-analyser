#!/usr/bin/env python3
"""Funding-Squeeze als $-Strategie: Event |z|>3 -> Entry in Squeeze-Richtung,
TP/SL-Sweep. Pro Geometrie: WR (resolved), Netto-EV/Event (Timeout=-fee), beide
Hälften (Stabilität). Da Richtung stabil ~57%, soll asymm. R:R klar +EV bringen.
Read-only agg_1m, 40d. fee aus gbm_predictor.label.fee_roundtrip_pct."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
S = json.load(open("/opt/coin/settings.json"))
s = S["databases"]["coins"]; fee = float(S["gbm_predictor"]["label"]["fee_roundtrip_pct"])
c = psycopg2.connect(host=s["host"], port=s["port"], dbname=s["name"], user=s["user"], password=s["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r["symbol"] for r in cur.fetchall()]
H=6*60; STEP=10; ZTHR=3.0; LV=[1.0,1.5,2.0,3.0]
EV=[]  # (t, sq, {up_idx[L]}, {dn_idx[L]})
for sym in coins:
    cur.execute("""SELECT bucket,close,high,low,funding FROM agg_1m WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>40) ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)<2000: continue
    fund=np.array([float(x["funding"]) if x["funding"] is not None else np.nan for x in r])
    if np.nanstd(fund)<1e-9: continue
    close=np.array([float(x["close"]) for x in r]);high=np.array([float(x["high"]) for x in r]);low=np.array([float(x["low"]) for x in r])
    mu,sd=np.nanmean(fund),np.nanstd(fund); z=(fund-mu)/sd; n=len(r); bk=[x["bucket"] for x in r]
    INF=10**9
    for i in range(0,n-1,STEP):
        if not np.isfinite(z[i]) or abs(z[i])<ZTHR: continue
        e=close[i]
        if e<=0: continue
        end=min(i+H,n); fh=high[i+1:end]; fl=low[i+1:end]
        up={}; dn={}
        for L in LV:
            cu=fh>=e*(1+L/100); cd=fl<=e*(1-L/100)
            up[L]=int(np.argmax(cu)) if cu.any() else INF
            dn[L]=int(np.argmax(cd)) if cd.any() else INF
        EV.append((bk[i], 1 if z[i]<0 else -1, up, dn))
c.close()
EV.sort(key=lambda x:x[0]); m=len(EV); INF=10**9
print(f"Funding-Events |z|>{ZTHR}: {m:,} (~{m/40:.0f}/Tag)  fee={fee}%")
print(f"{'TP':>4} {'SL':>4} {'R:R':>4} {'resolve%':>8} {'WR%':>5} {'EV/Ev%':>7} {'H1':>6} {'H2':>6}")
def evalg(tp,sl):
    out=[]
    for _,sq,up,dn in EV:
        if sq==1: tphit,slhit=up[tp],dn[sl]
        else:     tphit,slhit=dn[tp],up[sl]
        if tphit==INF and slhit==INF: out.append(-fee); continue   # Timeout
        out.append((tp-fee) if tphit<slhit else (-sl-fee))
    return np.array(out)
for tp in [1.5,2.0,3.0]:
    for sl in [1.0,1.5,2.0]:
        a=evalg(tp,sl); h=len(a)//2
        resolved=np.array([1 for _,sq,up,dn in EV if (up[tp] if sq==1 else dn[tp])!=INF or (dn[sl] if sq==1 else up[sl])!=INF])
        rr=resolved.sum()/m*100
        wr=(a>0).sum()/max(1,(a>0).sum()+(a<-fee).sum())*100  # win unter aufgelösten
        print(f"{tp:>4.1f} {sl:>4.1f} {tp/sl:>4.1f} {rr:>7.0f}% {wr:>4.0f}% {a.mean():>+6.3f} {a[:h].mean():>+5.3f} {a[h:].mean():>+5.3f}")
