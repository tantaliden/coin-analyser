#!/usr/bin/env python3
"""Schritt 1: Netto-$ nach Gebühren MIT Entry-Gate.

Kombiniert: adaptives Vorzeichen (Richtung, aus adaptive_sign-Logik) + ein
Entry-Gate (Resolve-Modell P(es kommt ein sauberer Move) — Volatilität ist
vorhersagbar). Nur wenn Gate offen UND Richtungs-Confidence hoch -> Trade.
Dann ehrliche $-Abrechnung über ALLE gesampelten Entries (inkl. Timeouts).

Payoff je Trade (size aus settings.gbm_predictor.dollar.trade_size_usd):
  pick richtig  -> +tp% - fee     | pick falsch -> -sl% - fee
  truth=skip    -> Timeout -> -fee (konservativ, kein Drift-Gewinn unterstellt)

Liest Cache (paths.dataset_cache). Alles settings-getrieben, keine Defaults.
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
    s = json.load(open("/opt/coin/settings.json"))
    cache = s["gbm_predictor"]["paths"]["dataset_cache"]
    if not os.path.exists(cache):
        from regime_test import build_dataset
        X, y, t, fc = build_dataset(cfg)
        np.savez_compressed(cache, X=X, y=y, t=t.astype("datetime64[ns]").astype("int64"),
                            feat_cols=np.array(fc, dtype=object))
        return X, y, t.astype("datetime64[ns]"), fc
    d = np.load(cache, allow_pickle=True)
    return d["X"], d["y"], d["t"].astype("datetime64[ns]"), list(d["feat_cols"])


def auc(yb, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else float("nan")


def main():
    cfg = load_cfg()
    s = json.load(open("/opt/coin/settings.json"))
    ad = s["gbm_predictor"]["adaptive"]
    W = int(ad["regime_window"]); lag = np.timedelta64(int(ad["resolution_lag_minutes"]), "m")
    minw = int(ad["min_window"]); btf = float(ad["base_train_frac"])
    if "dollar" not in s["gbm_predictor"] or "trade_size_usd" not in s["gbm_predictor"]["dollar"]:
        raise RuntimeError("settings.gbm_predictor.dollar.trade_size_usd fehlt — KEINE Defaults")
    size = float(s["gbm_predictor"]["dollar"]["trade_size_usd"])
    tp, sl, fee = cfg["tp"], cfg["sl"], cfg["fee"]

    X, y, t = get_dataset(cfg)[:3]
    o = np.argsort(t); X, y, t = X[o], y[o], t[o]
    N = len(y)
    cut = int(N * btf)
    log(f"Samples gesamt={N:,}  Train(erste {btf})={cut:,}  Test={N-cut:,}  size=${size}")

    # --- Richtung: Basis-Modell auf aufgelöste Train-Fälle ---
    res = y != 0
    base = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    tr_res = res[:cut]
    base.fit(X[:cut][tr_res], (y[:cut][tr_res] == 1).astype(int))
    pL = base.predict_proba(X)[:, 1]

    # --- Entry-Gate: Resolve-Modell P(y!=0) auf ALLE Train-Fälle ---
    gate = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    gate.fit(X[:cut], (y[:cut] != 0).astype(int))
    pRes = gate.predict_proba(X)[:, 1]
    log(f"Gate-Modell AUC (resolve vs skip, test) = {auc((y[cut:]!=0).astype(int), pRes[cut:]):.3f}")

    # --- Kausaler Regime-Detektor (Vorzeichen) auf aufgelösten Fällen ---
    ridx = np.where(res)[0]
    correct_r = ((pL[ridx] >= 0.5).astype(int) == (y[ridx] == 1).astype(int)).astype(np.int8)
    known_r = t[ridx] + lag
    ko = np.argsort(known_r); kn_sorted = known_r[ko]; corr_sorted = correct_r[ko]
    cum = np.concatenate([[0], np.cumsum(corr_sorted)])

    def sign_at(ti):
        pos = np.searchsorted(kn_sorted, ti, side="right")
        lo = max(0, pos - W); cnt = pos - lo
        if cnt < minw:
            return 0
        hit = (cum[pos] - cum[lo]) / cnt
        return -1 if hit < 0.5 else 1

    # --- $-Abrechnung über Test, Sweep über Gate- und Richtungs-Schwelle ---
    test = np.arange(cut, N)
    span_days = max(1e-9, (t[test[-1]] - t[test[0]]) / np.timedelta64(1, "D"))
    signs = np.array([sign_at(t[i]) for i in test], dtype=np.int8)
    pL_te = pL[test]; pRes_te = pRes[test]; y_te = y[test]
    p_adj = np.where(signs == 1, pL_te, 1.0 - pL_te)
    dir_conf = np.maximum(p_adj, 1 - p_adj)
    pick = np.where(p_adj >= 0.5, 1, 2)
    have = signs != 0

    log(f"\nTestspanne {span_days:.1f} Tage")
    log(f"{'gate>=':>7} {'dir>=':>6} {'n_tr':>6} {'tr/Tag':>7} {'hit%':>6} {'$/Tr':>7} {'$ ges':>9} {'$/Tag':>7}")
    for g in [0.40, 0.50, 0.60]:
        for dthr in [0.50, 0.55, 0.60]:
            m = have & (pRes_te >= g) & (dir_conf >= dthr)
            ns = int(m.sum())
            if ns == 0:
                log(f"{g:>7.2f} {dthr:>6.2f} {0:>6}"); continue
            pk = pick[m]; yt = y_te[m]
            win = pk == yt
            pnl_pct = np.where(win, tp - fee, np.where(yt == 0, -fee, -sl - fee))
            usd = size * pnl_pct / 100.0
            hit = win.mean()
            log(f"{g:>7.2f} {dthr:>6.2f} {ns:>6} {ns/span_days:>6.1f} {hit*100:>5.1f}% "
                f"{usd.mean():>6.3f} {usd.sum():>8.1f} {usd.sum()/span_days:>6.2f}")


if __name__ == "__main__":
    main()
