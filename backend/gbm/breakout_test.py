#!/usr/bin/env python3
"""Reaktions-/Ausbruch-Test (Volkers Idee): NICHT Richtung vorhersagen, sondern
auf den Move REAGIEREN. Hypothese: die Momentum-Richtung (Vorzeichen des jüngsten
Returns) trifft die Gewinner-Seite besser als das ~Münzwurf-Basis-Modell.

Nutzt den gecachten Datensatz (X,y,t). Walk-forward: nur die SPÄTE Hälfte (oos).
Pro Regel (Momentum bzw. Mean-Reversion über mehrere Fenster): Richtungstreffer
auf aufgelösten Fällen + EV/Entry (skip=-fee). Vergleich Momentum vs Reversion.
Reine Regel, kein Modell. Keine Defaults (Cache-Pfad aus settings)."""
import sys
sys.path.insert(0, "/opt/coin/backend/gbm")
import json
import numpy as np
from train_eval import load_cfg


def main():
    cfg = load_cfg()
    cache = json.load(open("/opt/coin/settings.json"))["gbm_predictor"]["paths"]["dataset_cache"]
    d = np.load(cache, allow_pickle=True)
    X, y, fc = d["X"], d["y"], list(d["feat_cols"])
    t = d["t"].astype("datetime64[ns]")
    o = np.argsort(t); X, y, t = X[o], y[o], t[o]
    n = len(y); split = int(n * 0.5)
    Xe, ye = X[split:], y[split:]           # oos späte Hälfte
    tp, sl, fee = cfg["tp"], cfg["sl"], cfg["fee"]
    res = ye != 0
    print(f"oos: {len(ye):,} Samples, davon {res.sum():,} aufgelöst (long-Anteil {(ye[res]==1).mean()*100:.1f}%)")

    def ev_for(side_pick):
        # side_pick: array 1(long)/2(short) je Sample
        win = side_pick == ye
        ev = np.where(win, tp - fee, np.where(ye == 0, -fee, -sl - fee))
        hit_res = (side_pick[res] == ye[res]).mean()
        return ev.mean(), hit_res, (side_pick[res] == ye[res]).sum()

    print(f"\n{'Regel':<22} {'dirHit(res)':>11} {'EV/Entry':>9}")
    for w in [5, 15, 30, 60, 120]:
        col = fc.index(f"ret_{w}")
        mom = np.where(Xe[:, col] >= 0, 1, 2)       # Momentum: rauf->long
        rev = np.where(Xe[:, col] >= 0, 2, 1)       # Reversion: rauf->short
        evm, hm, _ = ev_for(mom)
        evr, hr, _ = ev_for(rev)
        print(f"Momentum ret_{w:<3}        {hm*100:>10.1f}% {evm:>8.4f}%")
        print(f"Reversion ret_{w:<3}       {hr*100:>10.1f}% {evr:>8.4f}%")

    # Kombi: nur handeln wenn mehrere Fenster gleiche Richtung (starker Ausbruch)
    print("\n--- nur starke Ausbrüche (ret_15 & ret_60 gleiche Richtung) ---")
    c15, c60 = fc.index("ret_15"), fc.index("ret_60")
    up = (Xe[:, c15] >= 0) & (Xe[:, c60] >= 0)
    dn = (Xe[:, c15] < 0) & (Xe[:, c60] < 0)
    agree = up | dn
    pick = np.where(up, 1, 2)
    m = agree
    if m.sum():
        win = pick[m] == ye[m]
        ev = np.where(win, tp - fee, np.where(ye[m] == 0, -fee, -sl - fee))
        rr = ye[m] != 0
        print(f"  n={int(m.sum())} ({m.mean()*100:.0f}% d. Samples)  dirHit(res)={ (pick[m][rr]==ye[m][rr]).mean()*100:.1f}%  EV/Entry={ev.mean():.4f}%")


if __name__ == "__main__":
    main()
