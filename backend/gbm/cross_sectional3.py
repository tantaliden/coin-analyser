#!/usr/bin/env python3
"""Robustheit des Cross-Sectional-Momentum (mehr Historie gibt's nicht, nur 42d).
Bricht der Edge bei anderem K (3/5/8/10) oder höheren Kosten (Slippage auf illiquide
Shorts) zusammen? Nicht-überlappend, t-Stat, beide Hälften. Wenn robust -> Forward-
Shadow gerechtfertigt. Read-only agg_1h."""
import json, numpy as np, psycopg2
S=json.load(open("/opt/coin/settings.json")); s=S["databases"]["coins"]
fee=float(S["funding_squeeze"]["fee_roundtrip_pct"])
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"]);cur=c.cursor()
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r[0] for r in cur.fetchall()]
series={}
for sym in coins:
    cur.execute("""SELECT bucket,close,quote_asset_volume FROM agg_1h WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>42) ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)>200: series[sym]={b:(float(cl),float(qv or 0)) for b,cl,qv in r if cl is not None}
c.close()
allt=sorted(set().union(*[set(v) for v in series.values()])); syms=list(series.keys())
P=np.full((len(allt),len(syms)),np.nan); V=np.full((len(allt),len(syms)),0.0)
for j,sym in enumerate(syms):
    for i,t in enumerate(allt):
        if t in series[sym]: P[i,j],V[i,j]=series[sym][t]
T=len(allt)
avgvol=np.nanmean(np.where(V>0,V,np.nan),axis=0)  # mittl. Stundenvolumen je Coin
print(f"Coins={len(syms)} Stunden={T}")
def run(L,H,K,cost):
    sp=[]
    for i in range(L,T-H,H):
        trail=P[i]/P[i-L]-1.0; fwd=P[i+H]/P[i]-1.0
        valid=np.isfinite(trail)&np.isfinite(fwd)
        if valid.sum()<2*K+5: continue
        idx=np.where(valid)[0]; order=idx[np.argsort(trail[idx])]
        lo=order[-K:]; sh=order[:K]
        sp.append((np.mean(fwd[lo])-np.mean(fwd[sh]))/2 - cost)
    sp=np.array(sp)*100; n=len(sp); h=n//2
    if n<5: return n,0,0,0,0
    t=sp.mean()/(sp.std(ddof=1)/np.sqrt(n)) if sp.std()>0 else 0
    return n,sp.mean(),t,sp[:h].mean(),sp[h:].mean()
print(f"\n{'L':>4}{'H':>4}{'K':>3} {'Kosten':>7} {'n':>4} {'Spread%':>8} {'t':>6} {'H1':>7} {'H2':>7}")
for L,H in [(72,24),(48,24),(24,24)]:
    for K in [3,5,8,10]:
        for cst in [2*fee/100, 5*fee/100]:   # normal vs hohe Slippage
            n,m,t,h1,h2=run(L,H,K,cst)
            flag=" <<<" if t>2 and h1>0 and h2>0 else ""
            print(f"{L:>4}{H:>4}{K:>3} {cst*100:>6.2f}% {n:>4} {m:>+7.3f} {t:>+5.2f} {h1:>+6.3f} {h2:>+6.3f}{flag}")
print("\n(<<< = t>2 & beide Haelften+ trotz Variation. Bricht es bei K/Kosten -> fragil.)")
