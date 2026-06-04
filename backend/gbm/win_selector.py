#!/usr/bin/env python3
"""Selektions-Ansatz (Volker 04.06.): WR ist eine Auswahl-Frage. Lerne aus den
aufgelösten Predictions des alten Predictors einen WIN-Klassifikator (P(win) aus
den 179 gespeicherten Features inkl. HL-Orderflow), und finde OUT-OF-SAMPLE die
Filter-Schwelle, ab der die behaltene Teilmenge >=75% WR erreicht — und wie viele
Trades dann noch übrig sind. Wenn das OOS hält -> Muster gefunden.

Train = frühe 65% (zeitlich), Test = späte 35%. Kein Look-ahead. Read-only.
"""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor

a = json.load(open("/opt/coin/settings.json"))["databases"]["app"]
c = psycopg2.connect(host=a["host"], port=a["port"], dbname=a["name"], user=a["user"], password=a["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT created_at, side, status, pnl_pct, features FROM open_predictions
               WHERE status IN ('win','loss','timeout') AND features IS NOT NULL ORDER BY created_at""")
rows = cur.fetchall(); c.close()
n = len(rows)
y = np.array([1 if r["status"] == "win" else 0 for r in rows], dtype=int)
pnl = np.array([float(r["pnl_pct"]) for r in rows])
def F(r): f = r["features"]; return f if isinstance(f, dict) else json.loads(f)
keys = [k for k in F(rows[0]).keys() if not k.startswith("_") and isinstance(F(rows[0]).get(k), (int, float))]
X = np.full((n, len(keys)), np.nan, np.float32)
for i, r in enumerate(rows):
    f = F(r)
    for j, k in enumerate(keys):
        v = f.get(k)
        if isinstance(v, (int, float)): X[i, j] = v
print(f"n={n:,}  Features={len(keys)}  Gesamt-WR={y.mean()*100:.1f}%")

split = int(n * 0.65)
Xtr, ytr, Xte, yte, pte = X[:split], y[:split], X[split:], y[split:], pnl[split:]
print(f"train={split:,} (WR {ytr.mean()*100:.1f}%)  test={n-split:,} (WR {yte.mean()*100:.1f}%, OOS spätere Zeit)")

import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        min_child_weight=20, subsample=1.0, colsample_bytree=1.0,
                        random_state=42, n_jobs=-1, eval_metric="logloss", tree_method="hist")
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
from sklearn.metrics import roc_auc_score
print(f"\nWin-Klassifikator OOS-AUC = {roc_auc_score(yte, p):.4f}  (0.5=wertlos)")

print("\n=== Behaltene Teilmenge je P(win)-Schwelle (OOS) ===")
print(f"{'thr':>5} {'n_keep':>7} {'%test':>6} {'WR%':>6} {'Ø pnl%':>8} {'sum pnl%':>9} {'/Tag~':>6}")
span_days = max(1.0, (rows[-1]["created_at"] - rows[split]["created_at"]).total_seconds()/86400)
for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    m = p >= thr; nk = int(m.sum())
    if nk < 20: print(f"{thr:>5.2f} {nk:>7}"); continue
    print(f"{thr:>5.2f} {nk:>7} {nk/len(yte)*100:>5.1f}% {yte[m].mean()*100:>5.1f}% {pte[m].mean():>+7.3f} {pte[m].sum():>+8.1f} {nk/span_days:>5.0f}")

imp = clf.feature_importances_; top = np.argsort(imp)[::-1][:15]
print("\n=== Top-15 Trennfeatures (was unterscheidet Win/Loss) ===")
for j in top: print(f"  {keys[j]:24} {imp[j]:.4f}")
