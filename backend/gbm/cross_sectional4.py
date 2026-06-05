#!/usr/bin/env python3
"""Liquid-only-Test: hält der Cross-Sectional-Edge, wenn nur die TOP-N liquidesten
Coins gehandelt werden (kleine reale Slippage)? Universum = Top-N nach 24h-Volumen.
K klein (liquide = wenige Coins). Kosten realistisch-niedrig für liquide (0.10-0.18%).
Nicht-überlappend, t-Stat, beide Hälften. Read-only agg_1h 42d."""
import json, numpy as np, psycopg2
S=json.load(open("/opt/coin/settings.json")); s=S["databases"]["coins"]
fee=float(S["funding_squeeze"]["fee_roundtrip_pct"])
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"]);cur=c.cursor()
# Top-Coins nach 24h-Volumen (absteigend)
cur.execute("""SELECT symbol, sum(quote_asset_volume) v FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000 ORDER BY v DESC""")
ranked=[r[0] for r in cur.fetchall()]
print(f"Liquideste Coins (Top-10): {ranked[:10]}")
def load(univ):
    series={}
    for sym in univ:
        cur.execute("""SELECT bucket,close FROM agg_1h WHERE symbol=%s AND bucket>now()-make_interval(days=>42)
                       ORDER BY bucket ASC""",(sym,))
        r=cur.fetchall()
        if len(r)>200: series[sym]={b:float(cl) for b,cl in r if cl is not None}
    allt=sorted(set().union(*[set(v) for v in series.values()])); syms=list(series.keys())
    P=np.full((len(allt),len(syms)),np.nan)
    for j,sym in enumerate(syms):
        for i,t in enumerate(allt):
            if t in series[sym]: P[i,j]=series[sym][t]
    return P,len(allt)
def run(P,T,L,H,K,cost):
    sp=[]
    for i in range(L,T-H,H):
        trail=P[i]/P[i-L]-1.0; fwd=P[i+H]/P[i]-1.0
        valid=np.isfinite(trail)&np.isfinite(fwd)
        if valid.sum()<2*K+2: continue
        idx=np.where(valid)[0]; order=idx[np.argsort(trail[idx])]
        sp.append((np.mean(fwd[order[-K:]])-np.mean(fwd[order[:K]]))/2 - cost)
    sp=np.array(sp)*100; n=len(sp); h=n//2
    if n<5: return n,0,0,0,0
    t=sp.mean()/(sp.std(ddof=1)/np.sqrt(n)) if sp.std()>0 else 0
    return n,sp.mean(),t,sp[:h].mean(),sp[h:].mean()
print(f"\n{'Top-N':>6}{'L':>4}{'H':>4}{'K':>3} {'Kosten':>7} {'n':>4} {'Spread%':>8} {'t':>6} {'H1':>7} {'H2':>7}")
for N in [10,15,20]:
    P,T=load(ranked[:N])
    for L in [24,72]:
        for K in [3,5]:
            if 2*K+2> N: continue
            for cst in [1*fee/100, 2*fee/100]:   # liquide: 0.09% bzw 0.18%
                n,m,t,h1,h2=run(P,T,L,24,K,cst)
                flag=" <<<" if t>2 and h1>0 and h2>0 else ""
                print(f"{N:>6}{L:>4}{24:>4}{K:>3} {cst*100:>6.2f}% {n:>4} {m:>+7.3f} {t:>+5.2f} {h1:>+6.3f} {h2:>+6.3f}{flag}")
c.close()
print("\n(Hält t>2 & beide Haelften+ auf liquiden Coins mit kleinen Kosten -> tradebar. Sonst: Edge lebt nur von illiquiden.)")
