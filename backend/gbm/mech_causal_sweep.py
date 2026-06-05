#!/usr/bin/env python3
"""Kausaler z×TP/SL-Sweep: gibt es IRGENDEINE ehrliche (kausal-z) Konfiguration des
Funding-Squeeze mit robust +EV (beide Hälften)? Wenn nein -> Edge tot. Read-only."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
S=json.load(open("/opt/coin/settings.json")); s=S["databases"]["coins"]; f=S["funding_squeeze"]
fee=float(f["fee_roundtrip_pct"]); W=int(f["z_lookback_hours"])*60; H=int(f["timeout_hours"])*60
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"])
cur=c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=%s""",(f["min_dollar_volume_24h"],))
coins=[r["symbol"] for r in cur.fetchall()]; LV=[1.0,1.5,2.0,3.0]; INF=10**9
def causal_z(fund,W):
    n=len(fund); out=np.full(n,np.nan)
    cs=np.concatenate([[0],np.cumsum(fund)]); cs2=np.concatenate([[0],np.cumsum(fund*fund)])
    for i in range(n):
        lo=max(0,i-W); k=i-lo
        if k<60: continue
        m=(cs[i]-cs[lo])/k; v=(cs2[i]-cs2[lo])/k-m*m; sd=np.sqrt(v) if v>1e-18 else 0
        if sd>0: out[i]=(fund[i]-m)/sd
    return out
EV=[]  # (t, z, sq, up{L}, dn{L})
for sym in coins:
    cur.execute("""SELECT bucket,close,high,low,funding FROM agg_1m WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>40) ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)<2000: continue
    fund=np.array([float(x["funding"]) if x["funding"] is not None else np.nan for x in r])
    if np.nanstd(fund)<1e-9: continue
    fund=np.nan_to_num(fund,nan=np.nanmean(fund))
    close=np.array([float(x["close"]) for x in r]);high=np.array([float(x["high"]) for x in r]);low=np.array([float(x["low"]) for x in r])
    z=causal_z(fund,W); n=len(r); bk=[x["bucket"] for x in r]; i=0
    while i<n-1:
        if not np.isfinite(z[i]) or abs(z[i])<3.0: i+=1; continue
        e=close[i]; end=min(i+H,n); fh=high[i+1:end]; fl=low[i+1:end]
        up={};dn={}
        for L in LV:
            cu=fh>=e*(1+L/100); cd=fl<=e*(1-L/100)
            up[L]=int(np.argmax(cu)) if cu.any() else INF; dn[L]=int(np.argmax(cd)) if cd.any() else INF
        EV.append((bk[i], abs(z[i]), 1 if z[i]<0 else -1, up, dn)); i+=60
c.close(); EV.sort(key=lambda x:x[0]); m=len(EV)
print(f"Kausale Events |z|>3: {m}")
print(f"{'zmin':>4} {'tp':>4} {'sl':>4} {'n':>5} {'WR%':>5} {'EV/Ev%':>7} {'H1':>6} {'H2':>6}")
def ev_for(zmin,tp,sl):
    out=[]
    for _,az,sq,up,dn in EV:
        if az<zmin: continue
        tph,slh=(up[tp],dn[sl]) if sq==1 else (dn[tp],up[sl])
        if tph==INF and slh==INF: out.append(-fee)
        else: out.append((tp-fee) if tph<slh else (-sl-fee))
    return np.array(out)
for zmin in [3.0,4.0,5.0]:
    for tp,sl in [(1.5,2.0),(1.5,1.5),(2.0,2.0),(2.0,1.5),(1.5,1.0),(3.0,2.0)]:
        a=ev_for(zmin,tp,sl)
        if len(a)<50: print(f"{zmin:>4.0f} {tp:>4.1f} {sl:>4.1f} {len(a):>5}  zu wenig"); continue
        h=len(a)//2; wr=(a>0).sum()/max(1,(a>0).sum()+(a<-fee).sum())*100
        print(f"{zmin:>4.0f} {tp:>4.1f} {sl:>4.1f} {len(a):>5} {wr:>4.0f}% {a.mean():>+6.3f} {a[:h].mean():>+5.3f} {a[h:].mean():>+5.3f}")
