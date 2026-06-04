#!/usr/bin/env python3
"""Volkers Präzisions-These: nicht alle trennen, sondern eine KLEINE Top-Scheibe
mit ~75% WR finden (30 von 150). Test: Win-Klassifikator OOS, dann WR der Top-k
nach Score (egal welche absolute P) + beste Coin/Stunden-Tasche. Zeigt, ob eine
hochpräzise Minderheit existiert. Read-only, train früh / test spät."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
a = json.load(open("/opt/coin/settings.json"))["databases"]["app"]
c = psycopg2.connect(host=a["host"], port=a["port"], dbname=a["name"], user=a["user"], password=a["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT created_at, symbol, status, features FROM open_predictions
               WHERE status IN ('win','loss','timeout') AND features IS NOT NULL ORDER BY created_at""")
rows = cur.fetchall(); c.close()
n = len(rows); y = np.array([1 if r["status"]=="win" else 0 for r in rows])
def F(r): f=r["features"]; return f if isinstance(f,dict) else json.loads(f)
keys = [k for k in F(rows[0]) if not k.startswith("_") and isinstance(F(rows[0]).get(k),(int,float))]
X = np.full((n,len(keys)),np.nan,np.float32)
for i,r in enumerate(rows):
    f=F(r)
    for j,k in enumerate(keys):
        v=f.get(k)
        if isinstance(v,(int,float)): X[i,j]=v
sym = np.array([r["symbol"] for r in rows]); hr = np.array([r["created_at"].hour for r in rows])
split=int(n*0.65)
import xgboost as xgb
clf=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,min_child_weight=20,
                      subsample=1.0,colsample_bytree=1.0,random_state=42,n_jobs=-1,eval_metric="logloss",tree_method="hist")
clf.fit(X[:split],y[:split])
p=clf.predict_proba(X[split:])[:,1]; yte=y[split:]
order=np.argsort(p)[::-1]   # beste zuerst
print(f"OOS-Test: {len(yte):,} Predictions, Basis-WR {yte.mean()*100:.1f}%")
print("\n=== WR der TOP-k nach Win-Score (Volkers '30 von 150') ===")
for k in [10,30,50,100,200,500,1000,2000]:
    if k>len(yte): break
    sel=order[:k]; print(f"  Top-{k:>5}: WR={yte[sel].mean()*100:5.1f}%")
# Beste Coin-Tasche OOS (min n=40)
print("\n=== Beste Coin-WR (OOS, min n=40) — gibt es 'saubere' Coins? ===")
res=[]
for s in np.unique(sym[split:]):
    m=sym[split:]==s
    if m.sum()>=40: res.append((yte[m].mean(),int(m.sum()),s))
res.sort(reverse=True)
for wr,nn,s in res[:6]: print(f"  {s:10} n={nn:>4} WR={wr*100:4.1f}%")
# Beste Stunde OOS
print("\n=== Beste Stunden-WR (OOS) ===")
hh=hr[split:]; best=[]
for h in range(24):
    m=hh==h
    if m.sum()>=80: best.append((yte[m].mean(),int(m.sum()),h))
best.sort(reverse=True)
for wr,nn,h in best[:5]: print(f"  {h:02d}h n={nn:>4} WR={wr*100:4.1f}%")
