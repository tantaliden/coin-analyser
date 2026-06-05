#!/usr/bin/env python3
"""Volkers Selbstkorrektur-These testen: bei jedem geklärten Sturm in dessen Richtung
drehen, bis zum nächsten Sturm halten. Dreht die Richtung -> zweiter Sturm -> Flip.
Trägt das NETTO (mit Gebühren) oder zerhackt Whipsaw es? BTC 30d, 1-min.
Sweep Vola-Schwelle (höher = weniger/größere Stürme = weniger Whipsaw). Read-only."""
import json, numpy as np, psycopg2
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("SELECT close,quote_asset_volume,open_interest FROM agg_1m WHERE symbol='BTC' AND bucket>now()-interval '30 days' ORDER BY bucket")
r=cur.fetchall(); c.close()
close=np.array([float(x[0]) for x in r]); vol=np.array([float(x[1] or 0) for x in r]); oi=np.array([float(x[2] or 0) for x in r]); n=len(close)
ret1=np.zeros(n); ret1[1:]=close[1:]/close[:-1]-1
VW=30; WAIT=10; VOLR=1.5; OIT=0.3; FEE=0.0009  # Flip-Gebühr (2 Fills)
# rollende Vola (std der 1-min returns ueber VW)
vola=np.full(n,np.nan)
for i in range(VW,n):
    seg=ret1[i-VW+1:i+1]; vola[i]=np.std(seg)*100
volr=np.full(n,0.0); oichg=np.full(n,0.0)
for i in range(60,n):
    a=vol[i-30:i].sum(); b=vol[i-60:i-30].sum(); volr[i]=a/b if b>0 else 0
    if oi[i-60]>0: oichg[i]=(oi[i]/oi[i-60]-1)*100
bh=(close[-1]/close[VW]-1)*100
print(f"BTC 30d: Buy&Hold {bh:+.1f}%  (n={n} min)")
print(f"{'VolaThr':>8}{'Stürme%':>8}{'Flips':>6}{'NettoRet%':>10}{'WR-Flip%':>9}{'vs B&H':>8}")
def runthr(vt):
    pos=0; entry_i=None; pnl=0.0; flips=0; flip_res=[]; storm_min=0
    run=0
    for i in range(VW+60,n):
        storm=(vola[i]>=vt) and ((volr[i]>=VOLR) or (oichg[i]>=OIT))
        if storm: storm_min+=1
        # onset-run zaehlen
        if vola[i]>=vt: run+=1
        else: run=0
        clarified = storm and run>=WAIT
        if clarified:
            d=1 if close[i]/close[i-run]-1>0 else -1
            if d!=pos:
                if pos!=0 and entry_i is not None:
                    seg=(close[i]/close[entry_i]-1)*100*pos - FEE*100
                    pnl+=seg; flip_res.append(seg)
                else:
                    pnl-=FEE*100
                pos=d; entry_i=i; flips+=1
    if pos!=0 and entry_i is not None:
        pnl+=(close[-1]/close[entry_i]-1)*100*pos
    wr=np.mean([x>0 for x in flip_res])*100 if flip_res else 0
    return storm_min/n*100, flips, pnl, wr
for vt in [0.07,0.10,0.14,0.18]:
    sp,fl,pn,wr=runthr(vt)
    print(f"{vt:>8.2f}{sp:>8.1f}{fl:>6}{pn:>+9.1f}{wr:>8.0f}%{pn-bh:>+8.1f}")
print("\n(NettoRet = Strategie-Return mit Flip-Gebühren. >0 und >B&H = Volkers Logik trägt. Negativ = Whipsaw frisst es.)")
