#!/usr/bin/env python3
"""Adaptives Vorzeichen — der Kern-Test gegen Regime-Inversion.

Befund (regime_test): Richtungs-AUC springt fold-weise 0.19..0.66 — das Signal
ist stark, nur das VORZEICHEN kippt (Momentum<->Mean-Reversion). Kennte man das
aktuelle Regime, läge die effektive AUC ~0.65.

Hier: Basis-Modell liefert P(long) "bis aufs Vorzeichen". Ein KAUSALER Regime-
Detektor misst die rollende Trefferquote der zuletzt AUFGELÖSTEN+BEKANNTEN Trades
(Entry + resolution_lag, kein Look-ahead) und invertiert das Signal, wenn das
Basis-Modell zuletzt unter 50% lag. Frage: zieht das die AUC über die Folds
stabil > 0.5?

Cache: baut den Datensatz einmal (paths.dataset_cache), danach Sekunden.
Alles aus settings.gbm_predictor. KEINE Defaults/Fallbacks.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, make_xgb
import json


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_dataset(cfg):
    s = json.load(open("/opt/coin/settings.json"))
    cache = s["gbm_predictor"]["paths"]["dataset_cache"]
    if os.path.exists(cache):
        log(f"Lade Cache {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["t"].astype("datetime64[ns]"), list(d["feat_cols"])
    log("Kein Cache — baue Datensatz (einmalig) ...")
    from regime_test import build_dataset
    X, y, t, fc = build_dataset(cfg)
    np.savez_compressed(cache, X=X, y=y,
                        t=t.astype("datetime64[ns]").astype("int64"),
                        feat_cols=np.array(fc, dtype=object))
    log(f"Cache geschrieben: {cache}")
    return X, y, t.astype("datetime64[ns]"), fc


def auc(yb, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else float("nan")


def main():
    cfg = load_cfg()
    s = json.load(open("/opt/coin/settings.json"))
    ad = s["gbm_predictor"]["adaptive"]
    W = int(ad["regime_window"]); lag = np.timedelta64(int(ad["resolution_lag_minutes"]), "m")
    minw = int(ad["min_window"]); btf = float(ad["base_train_frac"])

    X, y, t, fc = get_dataset(cfg)
    res = y != 0
    Xr = X[res]; yb = (y[res] == 1).astype(int); tr = t[res]
    o = np.argsort(tr); Xr, yb, tr = Xr[o], yb[o], tr[o]
    n = len(yb)
    log(f"Aufgelöst: {n:,} (long={yb.mean()*100:.1f}%)  W={W} lag={ad['resolution_lag_minutes']}m base_frac={btf}")

    # Basis-Modell auf erste btf
    cut = int(n * btf)
    base = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    base.fit(Xr[:cut], yb[:cut])
    pL = base.predict_proba(Xr)[:, 1]
    correct = ((pL >= 0.5).astype(int) == yb).astype(np.int8)

    # Kausaler Regime-Detektor: pro Test-Sample i die letzten W Trades, die bei t_i
    # schon bekannt sind (known = entry + lag <= t_i).
    known = tr + lag
    ko = np.argsort(known)
    kn_sorted = known[ko]; corr_by_known = correct[ko]
    cum = np.concatenate([[0], np.cumsum(corr_by_known)])  # Präfixsumme für schnelle Fenster-Mittel

    test_mask = np.arange(n) >= cut
    sign = np.ones(n, dtype=np.int8)
    have = np.zeros(n, dtype=bool)
    for i in np.where(test_mask)[0]:
        pos = np.searchsorted(kn_sorted, tr[i], side="right")  # Anzahl bei t_i bekannt
        lo = max(0, pos - W)
        cnt = pos - lo
        if cnt < minw:
            continue
        hit = (cum[pos] - cum[lo]) / cnt
        sign[i] = -1 if hit < 0.5 else 1
        have[i] = True

    te = test_mask & have
    pL_te = pL[te]; yb_te = yb[te]; sgn = sign[te]
    p_adj = np.where(sgn == 1, pL_te, 1.0 - pL_te)
    raw_hit = ((pL_te >= 0.5).astype(int) == yb_te).mean()
    adj_hit = ((p_adj >= 0.5).astype(int) == yb_te).mean()
    log(f"\nTest-resolved mit Detektor-History: {te.sum():,}")
    log(f"  RAW Basis : hit={raw_hit*100:5.1f}%  AUC={auc(yb_te,pL_te):.3f}")
    log(f"  ADAPTIV   : hit={adj_hit*100:5.1f}%  AUC={auc(yb_te,p_adj):.3f}   (invertiert {(sgn==-1).mean()*100:.0f}% der Zeit)")

    # Pro-Fold-Vergleich + Oracle-Decke
    NF = 8
    idx_te = np.where(te)[0]
    b = [int(len(idx_te) * k / NF) for k in range(NF + 1)]
    log(f"\n{'fold':>4} {'ab':>17} {'AUC_raw':>8} {'AUC_adapt':>9} {'AUC_oracle':>10} {'adapt_hit':>9}")
    for k in range(NF):
        sl = idx_te[b[k]:b[k + 1]]
        yk = yb[sl]; pk = pL[sl]; pak = np.where(sign[sl] == 1, pk, 1 - pk)
        a_raw = auc(yk, pk); a_ad = auc(yk, pak)
        a_or = max(a_raw, 1 - a_raw) if a_raw == a_raw else float("nan")  # perfektes Vorzeichen
        hk = ((pak >= 0.5).astype(int) == yk).mean()
        log(f"{k:>4} {str(tr[sl][0])[:16]:>17} {a_raw:>8.3f} {a_ad:>9.3f} {a_or:>10.3f} {hk*100:>8.1f}%")
    log("\n  ADAPT nahe ORACLE = Detektor fängt das Regime; ADAPT~RAW<0.5 = Detektor zu langsam.")


if __name__ == "__main__":
    main()
