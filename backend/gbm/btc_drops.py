#!/usr/bin/env python3
"""Volkers Kern-Hypothese direkt an BTC: gibt es 1h VOR einem Abrutsch ein
Vorzeichen — zieht Volumen an oder bricht ein, dreht Taker-Flow, OI, Vola?
Drop = BTC Forward-30min <= -Schwelle. Pro Drop: Features in [t-60,t] (nur
Vergangenheit). Vs Baseline (alle Minuten) + AUC + Fehlalarm. Read-only BTC agg_1m."""
import json, numpy as np, psycopg2
from sklearn.metrics import roc_auc_score
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("""SELECT bucket,close,quote_asset_volume,trades,taker_buy_quote,open_interest,funding
               FROM agg_1m WHERE symbol='BTC' AND bucket>now()-interval '30 days' ORDER BY bucket""")
r=cur.fetchall(); c.close()
n=len(r)
close=np.array([float(x[1]) for x in r]); vol=np.array([float(x[2] or 0) for x in r])
trd=np.array([float(x[3] or 0) for x in r]); tbq=np.array([float(x[4] or 0) for x in r])
oi=np.array([float(x[5] or 0) for x in r]); fund=np.array([float(x[6] or 0) for x in r])
taker=np.divide(tbq,vol,out=np.full(n,0.5),where=vol>0)
print(f"BTC Minuten: {n} (~{n/1440:.0f} Tage)")
def feats(i):
    if i<120 or i+30>=n: return None
    fwd=(close[i+30]/close[i]-1)*100
    volr=vol[i-30:i].sum()/vol[i-60:i-30].sum() if vol[i-60:i-30].sum()>0 else None  # >1 zieht an
    trdr=trd[i-30:i].sum()/trd[i-60:i-30].sum() if trd[i-60:i-30].sum()>0 else None
    rr=np.diff(close[i-30:i+1])/close[i-30:i]; vola=np.std(rr)*100
    ret60=(close[i]/close[i-60]-1)*100
    tak=np.mean(taker[i-30:i])
    oichg=(oi[i]/oi[i-60]-1)*100 if oi[i-60]>0 else None
    return dict(fwd=fwd,volr=volr,trdr=trdr,vola=vola,ret60=ret60,tak=tak,oichg=oichg)
F=[feats(i) for i in range(n)]; F=[x for x in F if x]
for THR in [0.5,0.8,1.2]:
    drop=np.array([1 if x["fwd"]<=-THR else 0 for x in F])
    base=drop.mean()
    print(f"\n=== Drop = Forward-30min <= -{THR}%  ({drop.sum()} Drops, Basisrate {base*100:.1f}%) ===")
    print(f"{'Feature':10}{'vor Drop':>10}{'Baseline':>10}{'AUC':>7}  Deutung")
    for key,desc in [("volr","Volumen-Ratio (>1 zieht an)"),("trdr","Trades-Ratio"),
                     ("vola","Vola 30m"),("ret60","Return 60m vor"),("tak","Taker-Buy"),("oichg","OI-Change 60m")]:
        x=np.array([v[key] if v[key] is not None else np.nan for v in F]); m=np.isfinite(x)
        md=np.nanmean(x[(drop==1)&m]); mb=np.nanmean(x[m])
        auc=roc_auc_score(drop[m],x[m]); auc=max(auc,1-auc)
        flag="<- Signal" if auc>0.58 else ""
        print(f"{key:10}{md:>10.3f}{mb:>10.3f}{auc:>7.3f}  {desc} {flag}")
print("\n(AUC>0.58 = echtes Vorzeichen. ~0.5 = BTC-Drops kommen ohne Vorwarnung in diesen Daten.)")
