#!/usr/bin/env python3
"""Ehrlicher Signifikanztest des Cross-Sectional-Momentum: NICHT-überlappende
Halteperioden (rebalance alle H Stunden) -> unabhängige Stichproben. Pro Periode
Long/Short-Spread minus Kosten (2 Legs round-trip). Ausgabe: mittlerer Spread,
t-Stat, %positive, beide Hälften, Anzahl unabhängiger Perioden. Nur so weiß ich,
ob die +2.7% echt sind oder Überlappungs-Illusion. Read-only agg_1h, 40d."""
import json, numpy as np, psycopg2
S=json.load(open("/opt/coin/settings.json")); s=S["databases"]["coins"]
fee=float(S["funding_squeeze"]["fee_roundtrip_pct"])
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"]);cur=c.cursor()
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r[0] for r in cur.fetchall()]
series={}
for sym in coins:
    cur.execute("""SELECT bucket,close FROM agg_1h WHERE symbol=%s AND bucket>now()-make_interval(days=>40)
                   ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)>200: series[sym]={b:float(cl) for b,cl in r if cl is not None}
c.close()
allt=sorted(set().union(*[set(v) for v in series.values()])); syms=list(series.keys())
P=np.full((len(allt),len(syms)),np.nan)
for j,sym in enumerate(syms):
    for i,t in enumerate(allt):
        if t in series[sym]: P[i,j]=series[sym][t]
T=len(allt); K=5; cost=2*fee/100.0   # 2 Legs round-trip pro Rebalance
print(f"Coins={len(syms)} Stunden={T}  K={K}  Kosten/Periode={cost*100:.2f}%")
print(f"{'L':>4} {'H':>4} {'Perioden':>8} {'Spread%':>8} {'t-Stat':>7} {'%+':>5} {'H1':>7} {'H2':>7} {'Sharpe/Per':>10}")
def run(L,H):
    sp=[]
    for i in range(L, T-H, H):            # SCHRITT = H -> nicht überlappend
        trail=P[i]/P[i-L]-1.0; fwd=P[i+H]/P[i]-1.0
        valid=np.isfinite(trail)&np.isfinite(fwd)
        if valid.sum()<2*K+5: continue
        idx=np.where(valid)[0]; order=idx[np.argsort(trail[idx])]
        lo=order[-K:]; sh=order[:K]
        sp.append((np.mean(fwd[lo])-np.mean(fwd[sh]))/2 - cost)
    sp=np.array(sp)*100; n=len(sp); h=n//2
    if n<5: return n,0,0,0,0,0,0
    t=sp.mean()/(sp.std(ddof=1)/np.sqrt(n)) if sp.std()>0 else 0
    return n, sp.mean(), t, (sp>0).mean()*100, sp[:h].mean(), sp[h:].mean(), sp.mean()/sp.std() if sp.std()>0 else 0
for L in [24,48,72,168]:
    for H in [24,72]:
        n,m,t,pp,h1,h2,sh=run(L,H)
        flag=" <<<" if t>2 and h1>0 and h2>0 else ""
        print(f"{L:>4} {H:>4} {n:>8} {m:>+7.3f} {t:>+6.2f} {pp:>4.0f}% {h1:>+6.3f} {h2:>+6.3f} {sh:>+9.3f}{flag}")
print("\n(t>2 + beide Hälften>0 = belastbarer Kandidat. Perioden=unabhängige Stichproben. Sharpe/Per = pro Periode.)")
