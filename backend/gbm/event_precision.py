#!/usr/bin/env python3
"""Scope-c, Volkers Präzisions-These auf RICHTUNG: gibt es eine kleine, hoch-
präzise Klasse von Events, wo die Richtung OOS zu ~75% stimmt? Nutzt den gbm-
Cache (Vorboten-Features X + Label y=1long/2short, =sauberer 3%-vor-1.5%-Move).
Binär long/short auf aufgelösten Events, train früh / test spät, XGB, dann
Top-k-Präzision der Richtungs-Calls. Kein Look-ahead."""
import json, numpy as np
cache = json.load(open("/opt/coin/settings.json"))["gbm_predictor"]["paths"]["dataset_cache"]
import os
if not os.path.exists(cache):
    print("kein Cache — abbruch"); raise SystemExit
d = np.load(cache, allow_pickle=True)
X, y, fc = d["X"], d["y"], list(d["feat_cols"]); t = d["t"].astype("datetime64[ns]")
o = np.argsort(t); X, y, t = X[o], y[o], t[o]
res = y != 0
Xr, yb = X[res], (y[res] == 1).astype(int)   # 1=long, 0=short
n = len(yb); split = int(n*0.65)
print(f"aufgelöste Events: {n:,}  long-Anteil {yb.mean()*100:.1f}%  train={split:,} test={n-split:,}")
import xgboost as xgb
from sklearn.metrics import roc_auc_score
clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, min_child_weight=20,
                        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1,
                        eval_metric="logloss", tree_method="hist")
clf.fit(Xr[:split], yb[:split])
p = clf.predict_proba(Xr[split:])[:,1]; yt = yb[split:]
print(f"Richtungs-AUC OOS = {roc_auc_score(yt,p):.4f}")
# Konfidenz = Abstand von 0.5; Top-k nach Konfidenz, WR = picked-richtig
conf = np.abs(p-0.5); pick = (p>=0.5).astype(int); correct = (pick==yt).astype(int)
order = np.argsort(conf)[::-1]
print("\n=== Top-k konfidenteste Richtungs-Calls: WR (>=75% = Muster!) ===")
for k in [30,50,100,200,500,1000]:
    if k>len(yt): break
    sel=order[:k]; print(f"  Top-{k:>5}: WR={correct[sel].mean()*100:5.1f}%")
