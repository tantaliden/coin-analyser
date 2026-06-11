#!/usr/bin/env python3
"""Meta-Modell-Test (Volker 10.06.: 'beliebig viele KIs, Hauptsache das Ergebnis stimmt').

Ehrliche Frage: holt ein trainiertes Modell (Gradient Boosting, walk-forward) aus
DEMSELBEN kausalen Feature-Panel (conj_panel.npz, 135k Anchors × 12 Features,
Outcome = Rennen +1% vor −1%/240min) OOS mehr Präzision als die handgeminte
Konjunktions-Regel (62-73%)?

Setup gegen Selbstbetrug:
  - Zeitlich sortiert, 5 expanding-window Folds über die hintere Hälfte:
    train = alles davor, test = nächster 10%-Block. NIE in die Zukunft.
  - Metrik: AUC je Fold + Präzision der Top-Konfidenz-Buckets (1%/5%/10%) je
    Richtung + Monotonie der Kalibrierung. In-sample interessiert nicht.
"""
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = "/opt/coin/backend/gbm/conj_panel.npz"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    z = np.load(CACHE, allow_pickle=True)
    F = z["F"].astype(float); t = z["t"]; RES = z["RES"]; YL = z["YL"]
    feats = list(z["feats"])
    m = RES
    F, t, y = F[m], t[m], YL[m].astype(int)
    o = np.argsort(t)
    F, t, y = F[o], t[o], y[o]
    n = len(y)
    log(f"{n} aufgelöste Anchors, {len(feats)} Features, Basisrate long {y.mean():.3f}")

    folds = []
    for k in range(5):
        te0 = int(n * (0.5 + 0.1 * k)); te1 = int(n * (0.5 + 0.1 * (k + 1)))
        folds.append((np.arange(0, te0), np.arange(te0, te1)))

    all_p, all_y = [], []
    for k, (tr, te) in enumerate(folds):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=42)
        clf.fit(F[tr], y[tr])
        p = clf.predict_proba(F[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        log(f"Fold {k+1}: train={len(tr)} test={len(te)} AUC={auc:.3f}")
        all_p.append(p); all_y.append(y[te])
    P = np.concatenate(all_p); Y = np.concatenate(all_y)
    log(f"OOS gesamt: n={len(P)} AUC={roc_auc_score(Y, P):.3f}")

    print("\n=== Top-Konfidenz-Präzision OOS (das Einzige was zählt) ===")
    for frac in (0.01, 0.05, 0.10):
        kq = int(len(P) * frac)
        iy = np.argsort(-P)[:kq]   # höchste P(long)
        isd = np.argsort(P)[:kq]   # niedrigste P(long) = short-Calls
        print(f"top {frac:4.0%}: LONG-Calls hit={Y[iy].mean():.3f} (n={kq}) | "
              f"SHORT-Calls hit={(1 - Y[isd]).mean():.3f} (n={kq})")

    print("\n=== Kalibrierung OOS (Dezile von P(long)) ===")
    qs = np.quantile(P, np.linspace(0, 1, 11))
    for i in range(10):
        msk = (P >= qs[i]) & (P < qs[i + 1] if i < 9 else P <= qs[i + 1])
        if msk.sum():
            print(f"  dezil {i+1:2d}: P̄={P[msk].mean():.3f} real={Y[msk].mean():.3f} n={msk.sum()}")

    imp = sorted(zip(feats, np.abs(np.corrcoef(F.T, y)[:-1, -1])), key=lambda x: -x[1])
    print("\n(grobe Einzel-Korrelationen:", ", ".join(f"{k}={v:.3f}" for k, v in imp[:5]), ")")
    print("\nfertig")

if __name__ == "__main__":
    main()
