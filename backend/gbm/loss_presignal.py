#!/usr/bin/env python3
"""Volkers Frage: Gab es VOR den Verlust-Trades des Predictors ein Vorzeichen?
(1) BTC-Kopplung: verlieren Trades, wenn BTC gegen die Position läuft?
(2) Pre-Entry (30/60min vor Open): unterscheidet sich BTC-Volumen/Vola/Return
    zwischen Loss und Win? Fokus Volumen (zieht an ODER bricht ein).
(3) Trennkraft (AUC) je Feature für Loss-Vorhersage + Fehlalarm.
Kausal: nur Daten VOR dem Entry. Read-only."""
import json, numpy as np, psycopg2
S=json.load(open("/opt/coin/settings.json")); ap=S["databases"]["app"]; co=S["databases"]["coins"]
a=psycopg2.connect(host=ap["host"],port=ap["port"],dbname=ap["name"],user=ap["user"],password=ap["password"]);ca=a.cursor()
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cc=c.cursor()
# Trades der letzten 5 Tage (win/loss), nur multi_head
ca.execute("""SELECT created_at,symbol,side,status FROM open_predictions
              WHERE created_at>now()-interval '5 days' AND status IN ('win','loss')
              ORDER BY created_at""")
trades=ca.fetchall()
# BTC-Minutenreihe laden (close + quote_volume), 5d + Puffer
cc.execute("""SELECT bucket,close,quote_asset_volume FROM agg_1m WHERE symbol='BTC'
              AND bucket>now()-interval '5 days 3 hours' ORDER BY bucket""")
br=cc.fetchall()
bt=[x[0] for x in br]; bc=np.array([float(x[1]) for x in br]); bv=np.array([float(x[2] or 0) for x in br])
import bisect
def idx_at(t):
    i=bisect.bisect_right(bt,t)-1
    return i if i>=0 else None
def ret(i,mins):
    j=i-mins
    return (bc[i]/bc[j]-1)*100 if j>=0 and bc[j]>0 else None
def volratio(i,mins):  # Volumen letzte X min vs davor X min (>1 zieht an, <1 bricht ein)
    if i-2*mins<0: return None
    rec=bv[i-mins:i].sum(); prev=bv[i-2*mins:i-mins].sum()
    return rec/prev if prev>0 else None
def vola(i,mins):
    if i-mins<0: return None
    r=np.diff(bc[i-mins:i+1])/bc[i-mins:i]; return np.std(r)*100
rows=[]
for t,sym,side,status in trades:
    i=idx_at(t)
    if i is None or i<120: continue
    rows.append(dict(status=status,side=side,
        btc_ret30=ret(i,30), btc_ret60=ret(i,60),
        btc_volr30=volratio(i,30), btc_vola30=vola(i,30),
        btc_fwd30=ret(min(i+30,len(bc)-1),30)))  # BTC waehrend Trade-Start (Kopplung)
import statistics as st
def stats(key,filt):
    v=[r[key] for r in rows if r[key] is not None and filt(r)]
    return (np.mean(v),np.median(v),len(v)) if v else (None,None,0)
nL=sum(1 for r in rows if r["status"]=="loss"); nW=sum(1 for r in rows if r["status"]=="win")
print(f"Trades 5d: Loss={nL} Win={nW}")
print("\n=== (1) BTC-Kopplung: BTC-Move in den 30min NACH Entry, vorzeichen-justiert (favorable>0) ===")
for stt in ["win","loss"]:
    vals=[]
    for r in rows:
        if r["status"]!=stt or r["btc_fwd30"] is None: continue
        fav = r["btc_fwd30"] if r["side"]=="long" else -r["btc_fwd30"]
        vals.append(fav)
    if vals: print(f"  {stt}: BTC lief im Schnitt {np.mean(vals):+.3f}% FÜR die Position (n={len(vals)})")
print("\n=== (2) Pre-Entry BTC-Features: Loss vs Win (Mittel) ===")
print(f"{'Feature':14}{'Loss':>10}{'Win':>10}  Deutung")
for key,desc in [("btc_ret30","BTC 30m-Return vor Entry"),("btc_ret60","BTC 60m-Return vor Entry"),
                 ("btc_volr30","BTC Vol-Ratio (>1 zieht an)"),("btc_vola30","BTC Vola 30m vor Entry")]:
    ml=stats(key,lambda r:r["status"]=="loss")[0]; mw=stats(key,lambda r:r["status"]=="win")[0]
    print(f"{key:14}{ml:>10.3f}{mw:>10.3f}  {desc}")
print("\n=== (3) Trennkraft (AUC) je Pre-Entry-Feature für Loss-Vorhersage ===")
from sklearn.metrics import roc_auc_score
y=np.array([1 if r["status"]=="loss" else 0 for r in rows])
for key in ["btc_ret30","btc_ret60","btc_volr30","btc_vola30"]:
    x=np.array([r[key] if r[key] is not None else np.nan for r in rows])
    m=np.isfinite(x)
    if m.sum()>50 and len(set(y[m]))>1:
        auc=roc_auc_score(y[m],x[m]); auc=max(auc,1-auc)
        print(f"  {key:14} AUC={auc:.3f} {'<- Trennkraft' if auc>0.58 else '(~Zufall)'}")
a.close();c.close()
print("\n(AUC~0.5 = wertlos. Vol-Ratio Loss vs Win zeigt, ob Volumen vor Verlusten anders war.)")
