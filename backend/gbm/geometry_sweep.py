#!/usr/bin/env python3
"""TP/SL-Geometrie-Sweep — Volkers Frage: ist 3% TP zu hoch?

Features sind geometrie-UNABHÄNGIG (Vorboten vor dem Move) -> einmal bauen.
Nur die Labels hängen von TP/SL ab -> pro Geometrie neu labeln (billig).
Pro Geometrie: Auflöse-Quote, adaptive Richtungstreffer, Netto-%/aufgelöstem Trade,
und Erwartungswert PRO ENTRY (skip=-fee) — das berücksichtigt, dass hohes TP
seltener getroffen wird. Sortiert nach EV/Entry.

Grid inline (Research-Script wie tune.py). Restliches aus settings. Keine Defaults.
"""
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, connect, fetch_coins, fetch_series, fetch_market, build_features, make_xgb

TP_GRID = [1.5, 2.0, 2.5, 3.0, 4.0]
SL_GRID = [1.0, 1.5, 2.0]
HORIZON_H = None  # aus settings


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def label_np(high, low, close, idx, tp, sl, H):
    n = len(close); out = np.full(len(idx), -1, np.int8)
    tpf, slf = tp / 100.0, sl / 100.0
    for k, i in enumerate(idx):
        end = i + H
        if end >= n:
            continue
        e = close[i]
        if not np.isfinite(e) or e <= 0:
            continue
        fh = high[i + 1:end + 1]; fl = low[i + 1:end + 1]
        up = np.argmax(fh >= e * (1 + tpf)) if (fh >= e * (1 + tpf)).any() else -1
        dn = np.argmax(fl <= e * (1 - slf)) if (fl <= e * (1 - slf)).any() else -1
        lw = up != -1 and (dn == -1 or up < dn)
        dn2 = np.argmax(fl <= e * (1 - tpf)) if (fl <= e * (1 - tpf)).any() else -1
        up2 = np.argmax(fh >= e * (1 + slf)) if (fh >= e * (1 + slf)).any() else -1
        sw = dn2 != -1 and (up2 == -1 or dn2 < up2)
        out[k] = 1 if lw else (2 if sw else 0)
    return out


def adaptive_eval(X, y, t, cfg, tp, sl, fee):
    """Adaptives Vorzeichen (kausal) -> Kennzahlen für eine Geometrie."""
    from sklearn.metrics import roc_auc_score
    o = np.argsort(t); X, y, t = X[o], y[o], t[o]
    n = len(y); btf = 0.4; W = 500; lag = np.timedelta64(90, "m"); minw = 300
    res = y != 0
    Xr, yb, tr = X[res], (y[res] == 1).astype(int), t[res]
    nr = len(yb); cut = int(nr * btf)
    if nr - cut < 500:
        return None
    base = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    base.fit(Xr[:cut], yb[:cut])
    pL = base.predict_proba(Xr)[:, 1]
    correct = ((pL >= 0.5).astype(int) == yb).astype(np.int8)
    known = tr + lag; ko = np.argsort(known); kn = known[ko]; cs = correct[ko]
    cum = np.concatenate([[0], np.cumsum(cs)])
    sign = np.ones(nr, np.int8); have = np.zeros(nr, bool)
    for i in range(cut, nr):
        pos = np.searchsorted(kn, tr[i], side="right"); lo = max(0, pos - W); c = pos - lo
        if c < minw:
            continue
        sign[i] = -1 if (cum[pos] - cum[lo]) / c < 0.5 else 1; have[i] = True
    m = have.copy(); m[:cut] = False
    if m.sum() < 300:
        return None
    pk = pL[m]; yk = yb[m]; sg = sign[m]
    pa = np.where(sg == 1, pk, 1 - pk); pick = (pa >= 0.5).astype(int)
    hit = (pick == yk).mean()
    auc = roc_auc_score(yk, pa) if len(np.unique(yk)) == 2 else float("nan")
    net_res = np.where(pick == yk, tp - fee, -sl - fee).mean()
    # EV pro Entry über ALLE Test-Samples (skip=-fee), gleiche Zeitfenster wie resolved-test
    t_cut_time = tr[cut]
    test_all = t >= t_cut_time
    # adaptive Richtung für alle: nutze Basis-pL auf X[test_all] + nächstes bekanntes sign? -> Näherung: globales mean-sign
    resolve_rate = res[test_all].mean()
    return dict(auc=auc, hit=hit, net_res=net_res, resolve_rate=resolve_rate, n=int(m.sum()))


def main():
    cfg = load_cfg()
    H = int(round(cfg["horizon_h"] * 60)); fee = cfg["fee"]
    conn = connect(cfg["db"]); coins = fetch_coins(conn, cfg); market = fetch_market(conn, cfg)
    log(f"{len(coins)} Coins. Baue Features einmal, dann {len(TP_GRID)}x{len(SL_GRID)} Geometrien ...")
    feat_cols = None; X_parts = []; t_parts = []; price_store = []
    for sym in coins:
        rows = fetch_series(conn, sym, cfg)
        if len(rows) < cfg["min_hist"] + H + 50:
            continue
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "bucket":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["bucket"] = pd.to_datetime(df["bucket"]); df = df.reset_index(drop=True)
        feats = build_features(df, cfg, market)
        idx = np.arange(cfg["min_hist"], len(df), cfg["step"])
        valid = feats.iloc[idx].notna().all(axis=1).to_numpy()
        idx = idx[valid]
        if len(idx) == 0:
            continue
        if feat_cols is None:
            feat_cols = list(feats.columns)
        X_parts.append(feats.iloc[idx][feat_cols].to_numpy(np.float32))
        t_parts.append(df["bucket"].iloc[idx].to_numpy())
        price_store.append((df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), idx))
    conn.close()
    X = np.concatenate(X_parts)
    ti = pd.to_datetime(np.concatenate(t_parts))
    if ti.tz is not None:
        ti = ti.tz_localize(None)
    t = ti.to_numpy().astype("datetime64[ns]")
    log(f"Features fertig: {len(X):,} Samples. Sweep:")
    log(f"{'TP':>4} {'SL':>4} {'R:R':>5} {'resolve%':>8} {'dirHit%':>7} {'AUC':>6} {'net%/res':>8} {'EV/Entry%':>9}")

    results = []
    for tp in TP_GRID:
        for sl in SL_GRID:
            y_parts = []
            for (hi, lo, cl, idx) in price_store:
                y_parts.append(label_np(hi, lo, cl, idx, tp, sl, H))
            y = np.concatenate(y_parts)
            r = adaptive_eval(X, y, t, cfg, tp, sl, fee)
            if r is None:
                continue
            # EV pro Entry: resolve_rate*(hit*(tp-fee) + (1-hit)*(-sl-fee)) + (1-resolve_rate)*(-fee)
            ev = r["resolve_rate"] * (r["hit"] * (tp - fee) + (1 - r["hit"]) * (-sl - fee)) + (1 - r["resolve_rate"]) * (-fee)
            results.append((ev, tp, sl, r))
            log(f"{tp:>4.1f} {sl:>4.1f} {tp/sl:>5.2f} {r['resolve_rate']*100:>7.1f}% {r['hit']*100:>6.1f}% "
                f"{r['auc']:>6.3f} {r['net_res']:>7.3f}% {ev:>8.4f}%")
    results.sort(reverse=True)
    log("\n=== Beste Geometrien nach EV/Entry ===")
    for ev, tp, sl, r in results[:5]:
        log(f"  TP {tp}% / SL {sl}%  -> EV/Entry {ev:.4f}%  (resolve {r['resolve_rate']*100:.0f}%, hit {r['hit']*100:.0f}%, AUC {r['auc']:.3f})")
    cur = [x for x in results if x[1] == 3.0 and x[2] == 1.5]
    if cur:
        log(f"\n  Aktuell (TP3/SL1.5): EV/Entry {cur[0][0]:.4f}%  — Rang {results.index(cur[0])+1}/{len(results)}")


if __name__ == "__main__":
    main()
