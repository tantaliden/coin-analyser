#!/usr/bin/env python3
"""Scope-c, NEUE Richtung (markt-neutral, umgeht die Regime-Wand): Cross-Sectional
Momentum/Reversion. Jede Stunde: Coins nach Trailing-Return (lookback L) ranken,
Top-k LONG + Bottom-k SHORT gleichzeitig -> gegen Marktrichtung gehedged. Forward-
Return des Long/Short-Spreads über Halte-H. Kausal (nur Vergangenheit fürs Ranking).
Aggregiert: mittlerer Spread, beide Hälften (Stabilität), %positive Perioden, vs Markt.
Read-only agg_1h, 40d."""
import json, numpy as np, psycopg2
S=json.load(open("/opt/coin/settings.json")); s=S["databases"]["coins"]
fee=float(S["funding_squeeze"]["fee_roundtrip_pct"])
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"])
cur=c.cursor()
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r[0] for r in cur.fetchall()]
# agg_1h Close-Matrix
series={}
for sym in coins:
    cur.execute("""SELECT bucket,close FROM agg_1h WHERE symbol=%s AND bucket>now()-make_interval(days=>40)
                   ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)>200: series[sym]={b:float(cl) for b,cl in r if cl is not None}
c.close()
# gemeinsame Zeitachse
allt=sorted(set().union(*[set(v) for v in series.values()]))
syms=list(series.keys())
P=np.full((len(allt),len(syms)),np.nan)
for j,sym in enumerate(syms):
    for i,t in enumerate(allt):
        if t in series[sym]: P[i,j]=series[sym][t]
T=len(allt); print(f"Coins={len(syms)} Stunden={T} (~{T/24:.0f} Tage)")
K=5  # Top/Bottom-k
print(f"{'L(h)':>5} {'H(h)':>5} {'Perioden':>8} {'MOM-Spread%':>11} {'H1':>7} {'H2':>7} {'%+':>5} {'Markt%':>7}")
def run(L,H):
    sp=[]; mk=[]
    for i in range(L, T-H):
        trail=P[i]/P[i-L]-1.0
        fwd=P[i+H]/P[i]-1.0
        valid=np.isfinite(trail)&np.isfinite(fwd)
        if valid.sum()<2*K+5: continue
        idx=np.where(valid)[0]
        order=idx[np.argsort(trail[idx])]
        lo=order[-K:]   # höchste Trailing = Winner -> LONG (momentum)
        sh=order[:K]    # niedrigste -> SHORT
        mom=(np.mean(fwd[lo])-np.mean(fwd[sh]))/2 - fee/100  # /2: Kapital je Seite; fee beide Seiten
        sp.append(mom); mk.append(np.mean(fwd[valid]))
    sp=np.array(sp)*100; mk=np.array(mk)*100; h=len(sp)//2
    return len(sp), sp.mean(), sp[:h].mean(), sp[h:].mean(), (sp>0).mean()*100, mk.mean()
for L in [6,24,72,168]:
    for H in [6,24,72]:
        n,m,h1,h2,pp,mkt=run(L,H)
        print(f"{L:>5} {H:>5} {n:>8} {m:>+10.3f} {h1:>+6.3f} {h2:>+6.3f} {pp:>4.0f}% {mkt:>+6.3f}")
print("\n(MOM-Spread>0 = Momentum: Winner schlagen Loser. <0 = Reversion. Markt% = mittlerer Forward aller, zur Kontrolle Neutralität.)")
