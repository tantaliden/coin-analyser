#!/usr/bin/env python3
"""Schritt 3a: Produktiv-Modelle trainieren + speichern.

Trainiert auf der GESAMTEN verfügbaren Historie (Regime-Adaption übernimmt
später der Live-Detektor, daher darf das Basis-Modell alles sehen):
  - base : Richtung (binary long/short) auf aufgelösten Fällen
  - gate : P(sauberer Move kommt) = resolve vs skip, auf allen Fällen
Speichert beide als XGBoost-JSON + Feature-Reihenfolge (feat_meta).
Der Service lädt diese Dateien; periodisches Re-Training ruft dies erneut auf.

Liest Cache (oder baut ihn). Alles settings-getrieben, keine Defaults.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, make_xgb


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    cfg = load_cfg()
    s = json.load(open("/opt/coin/settings.json"))
    paths = s["gbm_predictor"]["paths"]
    cache = paths["dataset_cache"]
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y, fc = d["X"], d["y"], list(d["feat_cols"])
        log(f"Cache geladen: {len(y):,} Samples, {len(fc)} Features")
    else:
        from regime_test import build_dataset
        X, y, t, fc = build_dataset(cfg)
        np.savez_compressed(cache, X=X, y=y, t=t.astype("datetime64[ns]").astype("int64"),
                            feat_cols=np.array(fc, dtype=object))
        log(f"Cache gebaut: {len(y):,} Samples")

    res = y != 0
    log(f"Trainiere base (Richtung) auf {res.sum():,} aufgelösten ...")
    base = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    base.fit(X[res], (y[res] == 1).astype(int))
    base.get_booster().save_model(paths["base_model"])

    log(f"Trainiere gate (resolve vs skip) auf {len(y):,} ...")
    gate = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    gate.fit(X, (y != 0).astype(int))
    gate.get_booster().save_model(paths["gate_model"])

    json.dump({"feat_cols": fc, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "n_samples": int(len(y)), "n_resolved": int(res.sum())},
              open(paths["feat_meta"], "w"), indent=2)
    log(f"Gespeichert:\n  base -> {paths['base_model']}\n  gate -> {paths['gate_model']}\n  meta -> {paths['feat_meta']}")


if __name__ == "__main__":
    main()
