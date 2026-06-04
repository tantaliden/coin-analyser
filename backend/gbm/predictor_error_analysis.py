#!/usr/bin/env python3
"""Fehler-Analyse des ALTEN Predictors (open_predictions): WO macht er es
systematisch falsch? Sucht Muster in den 29.900 aufgelösten Predictions
(win/loss) anhand der 179 gespeicherten Features + Meta.

Ausgabe:
 1) Kalibrierung: WR je score-Dezil (invers?)
 2) WR je Seite + je _predicted_side
 3) WR je Stunde (UTC)
 4) Coins mit schlechtester/bester WR (min n)
 5) Diskriminanteste Features (standardisierte Mittelwert-Differenz win vs loss)
 6) Für Top-Discriminatoren: WR je Quantil (zeigt Richtung des Musters)
Read-only. Keine Defaults nötig."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor

a = json.load(open("/opt/coin/settings.json"))["databases"]["app"]
c = psycopg2.connect(host=a["host"], port=a["port"], dbname=a["name"], user=a["user"], password=a["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol, side, score, status, created_at, features,
                      predicted_up_pct, predicted_down_pct
               FROM open_predictions WHERE status IN ('win','loss') AND features IS NOT NULL""")
rows = cur.fetchall()
c.close()
n = len(rows)
print(f"Aufgelöst mit Features: {n:,}")

y = np.array([1 if r["status"] == "win" else 0 for r in rows])  # 1=win
print(f"Gesamt-WR: {y.mean()*100:.1f}%")

def feats(r):
    f = r["features"]
    return f if isinstance(f, dict) else json.loads(f)

# 1) Kalibrierung nach score
score = np.array([float(r["score"]) if r["score"] is not None else np.nan for r in rows])
print("\n=== 1) Kalibrierung: WR je score-Dezil (steigend = gut, fallend = INVERS) ===")
valid = ~np.isnan(score)
qs = np.quantile(score[valid], np.linspace(0, 1, 11))
for i in range(10):
    lo, hi = qs[i], qs[i+1]
    m = valid & (score >= lo) & (score <= hi if i == 9 else score < hi)
    if m.sum() > 30:
        print(f"  score[{lo:.3f},{hi:.3f}) n={m.sum():>6} WR={y[m].mean()*100:5.1f}%")

# 2) Seite + predicted_side
print("\n=== 2) WR je Seite ===")
side = np.array([r["side"] for r in rows])
for s in ("long", "short"):
    m = side == s
    if m.sum(): print(f"  {s:5} n={m.sum():>6} WR={y[m].mean()*100:5.1f}%")
ps = np.array([feats(r).get("_predicted_side", "?") for r in rows])
print("  -- nach _predicted_side --")
for s in np.unique(ps):
    m = ps == s
    if m.sum() > 30: print(f"  pred={s:6} n={m.sum():>6} WR={y[m].mean()*100:5.1f}%")

# 3) Stunde
print("\n=== 3) WR je Stunde (lokal) ===")
hr = np.array([r["created_at"].hour for r in rows])
for h in range(0, 24, 2):
    m = (hr == h) | (hr == h+1)
    if m.sum() > 50: print(f"  {h:02d}-{h+1:02d}h n={m.sum():>5} WR={y[m].mean()*100:5.1f}%")

# 4) Coins
print("\n=== 4) Coins: schlechteste/beste WR (min n=80) ===")
sym = np.array([r["symbol"] for r in rows])
res = []
for s in np.unique(sym):
    m = sym == s
    if m.sum() >= 80: res.append((y[m].mean(), int(m.sum()), s))
res.sort()
print("  schlechteste:"); [print(f"    {s:10} n={nn:>5} WR={wr*100:4.1f}%") for wr, nn, s in res[:8]]
print("  beste:"); [print(f"    {s:10} n={nn:>5} WR={wr*100:4.1f}%") for wr, nn, s in res[-6:]]

# 5) Diskriminanteste Features
print("\n=== 5) Diskriminanteste Features (|standardisierte Differenz win-loss|) ===")
keys = [k for k in feats(rows[0]).keys() if not k.startswith("_")]
M = np.full((n, len(keys)), np.nan)
for i, r in enumerate(rows):
    f = feats(r)
    for j, k in enumerate(keys):
        v = f.get(k)
        if isinstance(v, (int, float)): M[i, j] = v
diffs = []
for j, k in enumerate(keys):
    col = M[:, j]; ok = ~np.isnan(col)
    if ok.sum() < n*0.5: continue
    w = col[ok & (y == 1)]; l = col[ok & (y == 0)]
    if len(w) < 50 or len(l) < 50: continue
    sd = np.std(col[ok]) or 1.0
    diffs.append((abs(w.mean()-l.mean())/sd, (w.mean()-l.mean())/sd, k, w.mean(), l.mean()))
diffs.sort(reverse=True)
for ad_, d, k, wm, lm in diffs[:18]:
    print(f"  {k:22} std-diff={d:+.3f}  win_avg={wm:+.4g} loss_avg={lm:+.4g}")

# 6) WR je Quantil für Top-3 Discriminatoren
print("\n=== 6) WR je Quantil der Top-Discriminatoren ===")
for ad_, d, k, wm, lm in diffs[:3]:
    j = keys.index(k); col = M[:, j]; ok = ~np.isnan(col)
    qq = np.quantile(col[ok], [0, .2, .4, .6, .8, 1.0])
    print(f"  {k}:")
    for i in range(5):
        lo, hi = qq[i], qq[i+1]
        m = ok & (col >= lo) & (col <= hi if i == 4 else col < hi)
        if m.sum() > 30: print(f"    [{lo:+.4g},{hi:+.4g}) n={m.sum():>5} WR={y[m].mean()*100:5.1f}%")
