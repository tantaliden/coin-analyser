#!/usr/bin/env python3
"""Hypothese: das Gate (P(Move kommt), hohe Vola) selektiert Setups, in denen der
enge 1,5%-SL durch Rauschen VOR dem 3%-TP getriggert wird -> Momentum-Win-Rate
auf GEGATETEN Setups schlechter als ungegatet. Erklärt warum live (~10% Win) <<
offline-ungegatet (~52%). Nutzt Cache + Gate-Modell. oos späte Hälfte."""
import sys
sys.path.insert(0, "/opt/coin/backend/gbm")
import json, numpy as np
from train_eval import load_cfg
import gbm_shadow_service as S
import xgboost as xgb


def main():
    cfg = load_cfg(); s = json.load(open("/opt/coin/settings.json")); g = s["gbm_predictor"]
    sv = g["service"]; paths = g["paths"]
    d = np.load(paths["dataset_cache"], allow_pickle=True)
    X, y, fc = d["X"], d["y"], list(d["feat_cols"]); t = d["t"].astype("datetime64[ns]")
    o = np.argsort(t); X, y = X[o], y[o]
    n = len(y); Xe, ye = X[n // 2:], y[n // 2:]
    gate = S.load_booster(paths["gate_model"])
    pres = gate.predict(xgb.DMatrix(Xe))
    mom = Xe[:, fc.index(f"ret_{sv['momentum_window']}")]
    gthr = sv["gate_threshold"]; minb = sv["min_breakout_pct"]
    tp, sl, fee = cfg["tp"], cfg["sl"], cfg["fee"]

    def stats(mask, label):
        idx = np.where(mask)[0]
        if len(idx) < 50:
            print(f"{label:<32} n={len(idx)} (zu wenig)"); return
        side = np.where(mom[idx] >= 0, 1, 2)
        yy = ye[idx]; res = yy != 0
        # SL-first-Rate unter aufgelösten: verliert die gewählte Seite?
        win = (side == yy)
        resolve = res.mean()
        dirhit = (side[res] == yy[res]).mean() if res.any() else float("nan")
        ev = np.where(win, tp - fee, np.where(yy == 0, -fee, -sl - fee)).mean()
        print(f"{label:<32} n={len(idx):>6} resolve={resolve*100:4.0f}% dirHit(res)={dirhit*100:4.1f}% EV/Entry={ev:+.4f}%")

    print("Momentum-Mitlaufen, oos:")
    stats(np.ones(len(ye), bool), "ALLE (ungegatet)")
    stats((pres >= gthr) & (np.abs(mom) >= minb), f"GEGATET (gate>={gthr},|mom|>={minb})")
    stats((pres < gthr), "NUR niedriges Gate (Vergleich)")
    # Vola-Schichten: dirHit nach Resolve-Wahrscheinlichkeit
    print("\ndirHit(res) nach Gate-Quantil:")
    qs = np.quantile(pres, [0, .25, .5, .75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (pres >= lo) & (pres < hi + 1e-9)
        side = np.where(mom[m] >= 0, 1, 2); yy = ye[m]; res = yy != 0
        if res.sum() < 30: continue
        print(f"  gate[{lo:.2f},{hi:.2f}) n={int(m.sum())} resolve={res.mean()*100:3.0f}% dirHit(res)={(side[res]==yy[res]).mean()*100:4.1f}%")


if __name__ == "__main__":
    main()
