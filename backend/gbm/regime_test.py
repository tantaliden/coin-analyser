#!/usr/bin/env python3
"""Regime-Diagnose: Ist die Richtung REGIME-LOKAL vorhersagbar?

v2-Befund: globaler Walk-forward gibt AUC 0.42 (<0.5) -> Vorzeichen der
Richtungs-Signale kippt zwischen Train und Test (Nicht-Stationarität).

Dieser Test baut denselben Datensatz und prüft:
  1. Rolling-Walk-Forward: pro Zeit-Fold auf den VORHERGEHENDEN Folds trainieren,
     den NÄCHSTEN Fold testen -> AUC je Fold. Kippt das Vorzeichen über die Zeit?
  2. Kurzes Rolling-Fenster (nur die jüngsten Folds als Train) vs. expanding.
Wenn recent-trainiert die nahe Zukunft mit AUC>0.5 trifft -> adaptives Modell
ist der Weg. Wenn auch lokal ~0.5 -> Richtung echt nicht prognostizierbar.

Importiert die Daten-/Feature-Logik aus train_eval (eine Quelle der Wahrheit).
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, connect, fetch_coins, fetch_series, fetch_market, build_features, label_at, make_xgb


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_dataset(cfg):
    conn = connect(cfg["db"])
    coins = fetch_coins(conn, cfg)
    market = fetch_market(conn, cfg)
    feat_cols = None
    Xp, yp, tp_ = [], [], []
    for ci, sym in enumerate(coins):
        rows = fetch_series(conn, sym, cfg)
        if len(rows) < cfg["min_hist"] + int(cfg["horizon_h"] * 60) + 50:
            continue
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "bucket":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["bucket"] = pd.to_datetime(df["bucket"]); df = df.reset_index(drop=True)
        feats = build_features(df, cfg, market)
        idx = np.arange(cfg["min_hist"], len(df), cfg["step"])
        labels = label_at(df, idx, cfg)
        keep = labels != -1
        idx, labels = idx[keep], labels[keep]
        valid = feats.iloc[idx].notna().all(axis=1).to_numpy()
        idx, labels = idx[valid], labels[valid]
        if len(idx) == 0:
            continue
        if feat_cols is None:
            feat_cols = list(feats.columns)
        Xp.append(feats.iloc[idx][feat_cols].to_numpy(dtype=np.float32))
        yp.append(labels); tp_.append(df["bucket"].iloc[idx].to_numpy())
    conn.close()
    X = np.concatenate(Xp); y = np.concatenate(yp); t = np.concatenate(tp_)
    order = np.argsort(t)
    return X[order], y[order], t[order], feat_cols


def auc(yb, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else float("nan")


def main():
    cfg = load_cfg()
    log("Baue Datensatz ...")
    X, y, t, feat_cols = build_dataset(cfg)
    n = len(y)
    res = y != 0                      # nur aufgelöste Fälle (Richtungs-Frage)
    Xr, yr, tr_t = X[res], (y[res] == 1).astype(int), t[res]
    log(f"Datensatz: {n:,} gesamt, {len(yr):,} aufgelöst (long={yr.mean()*100:.1f}%)")

    NF = 10
    bnd = [int(len(yr) * k / NF) for k in range(NF + 1)]
    log(f"\n=== Rolling-Walk-Forward über {NF} Zeit-Folds (Binär long/short) ===")
    log(f"{'fold':>4} {'testzeit ab':>20} {'n_te':>6} {'AUC_exp':>8} {'AUC_roll2':>9} {'hit@0.5':>8}")
    aucs_exp, aucs_roll = [], []
    for i in range(2, NF):
        tr_lo_exp, tr_hi = 0, bnd[i]
        tr_lo_roll = bnd[i - 2]
        te_lo, te_hi = bnd[i], bnd[i + 1]
        Xte, yte = Xr[te_lo:te_hi], yr[te_lo:te_hi]
        # expanding
        m1 = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
        m1.fit(Xr[tr_lo_exp:tr_hi], yr[tr_lo_exp:tr_hi])
        p1 = m1.predict_proba(Xte)[:, 1]
        a1 = auc(yte, p1)
        # rolling (nur jüngste 2 Folds)
        m2 = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
        m2.fit(Xr[tr_lo_roll:tr_hi], yr[tr_lo_roll:tr_hi])
        p2 = m2.predict_proba(Xte)[:, 1]
        a2 = auc(yte, p2)
        hit = ((p1 >= 0.5).astype(int) == yte).mean()
        aucs_exp.append(a1); aucs_roll.append(a2)
        log(f"{i:>4} {str(pd.Timestamp(tr_t[te_lo]))[:19]:>20} {len(yte):>6} {a1:>8.3f} {a2:>9.3f} {hit*100:>7.1f}%")
    log(f"\n  mean AUC expanding={np.nanmean(aucs_exp):.3f}  rolling2={np.nanmean(aucs_roll):.3f}")
    log(f"  (>0.55 konsistent = regime-lokaler Edge; um 0.5 schwankend = kippt; <0.5 = invertiert)")

    # Sub-Perioden-AUC mit EINEM globalen Modell (zeigt Vorzeichen-Kippen explizit)
    log("\n=== Vorzeichen-Kippen: ein Modell (Train=erste 50%) auf spätere Sub-Perioden ===")
    half = n // 2
    res_tr = (y[:half] != 0)
    gm = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    gm.fit(X[:half][res_tr], (y[:half][res_tr] == 1).astype(int))
    rest_res = (y[half:] != 0)
    Xrest, yrest, trest = X[half:][rest_res], (y[half:][rest_res] == 1).astype(int), t[half:][rest_res]
    K = 6
    b2 = [int(len(yrest) * k / K) for k in range(K + 1)]
    for k in range(K):
        sl = slice(b2[k], b2[k + 1])
        log(f"  Sub {k} ab {str(pd.Timestamp(trest[b2[k]]))[:19]}  AUC={auc(yrest[sl], gm.predict_proba(Xrest[sl])[:,1]):.3f}")


if __name__ == "__main__":
    main()
