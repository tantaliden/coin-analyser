#!/usr/bin/env python3
"""Schritt 2: Tuning des Regime-Detektors (Vorzeichen).

Isoliert die Qualität des adaptiven Vorzeichens: sweept base_train_frac x
regime_window x resolution_lag und misst auf den AUFGELÖSTEN Test-Fällen die
adaptive AUC + Richtungstreffer + Netto-%/Trade (win=+tp-fee, lose=-sl-fee).
Kein Gate hier (das ist separat) — es geht rein um die beste Vorzeichen-Logik.

Liest Cache. Schreibt KEINE Settings (Vorschlag wird geloggt; Übernahme bewusst).
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, make_xgb


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_dataset(cfg):
    cache = json.load(open("/opt/coin/settings.json"))["gbm_predictor"]["paths"]["dataset_cache"]
    d = np.load(cache, allow_pickle=True)
    return d["X"], d["y"], d["t"].astype("datetime64[ns]")


def auc(yb, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else float("nan")


def main():
    cfg = load_cfg()
    tp, sl, fee = cfg["tp"], cfg["sl"], cfg["fee"]
    X, y, t = get_dataset(cfg)
    o = np.argsort(t); X, y, t = X[o], y[o], t[o]
    res = y != 0
    Xr, yb, tr = X[res], (y[res] == 1).astype(int), t[res]
    n = len(yb)
    minw = 300
    log(f"Aufgelöst {n:,}. Sweep base_frac x window x lag  (Metrik: adaptive AUC / hit / netto%/Trade)")
    log(f"{'bfrac':>5} {'W':>5} {'lag':>4} {'n_te':>6} {'AUC':>6} {'hit%':>6} {'net%/Tr':>8} {'inv%':>5}")

    best = None
    for bf in [0.30, 0.40, 0.50]:
        cut = int(n * bf)
        base = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
        base.fit(Xr[:cut], yb[:cut])
        pL = base.predict_proba(Xr)[:, 1]
        correct = ((pL >= 0.5).astype(int) == yb).astype(np.int8)
        for lag_m in [30, 90, 180]:
            known = tr + np.timedelta64(lag_m, "m")
            ko = np.argsort(known); kn = known[ko]; cs = correct[ko]
            cum = np.concatenate([[0], np.cumsum(cs)])
            for W in [500, 1000, 1500, 3000]:
                sign = np.ones(n, dtype=np.int8); have = np.zeros(n, dtype=bool)
                for i in range(cut, n):
                    pos = np.searchsorted(kn, tr[i], side="right")
                    lo = max(0, pos - W); cnt = pos - lo
                    if cnt < minw:
                        continue
                    sign[i] = -1 if (cum[pos] - cum[lo]) / cnt < 0.5 else 1
                    have[i] = True
                m = have.copy(); m[:cut] = False
                if m.sum() < 500:
                    continue
                pk = pL[m]; yk = yb[m]; sg = sign[m]
                pa = np.where(sg == 1, pk, 1 - pk)
                pick = (pa >= 0.5).astype(int)
                win = pick == yk
                a = auc(yk, pa); hit = win.mean()
                net = np.where(win, tp - fee, -sl - fee).mean()
                inv = (sg == -1).mean()
                log(f"{bf:>5.2f} {W:>5} {lag_m:>4} {int(m.sum()):>6} {a:>6.3f} {hit*100:>5.1f}% {net:>7.3f}% {inv*100:>4.0f}%")
                if best is None or net > best[0]:
                    best = (net, bf, W, lag_m, a, hit)
    log(f"\nBESTER Punkt (nach netto%/Trade): net={best[0]:.3f}%  base_frac={best[1]}  window={best[2]}  lag={best[3]}m  (AUC={best[4]:.3f}, hit={best[5]*100:.1f}%)")


if __name__ == "__main__":
    main()
