#!/usr/bin/env python3
"""GBM-Predictor — Offline Wahrheitstest v2.

v1-Befund: Multi-Timeframe-Aggregate -> KEIN Richtungs-Edge, inverse Kalibrierung
(höhere Confidence = schlechter). Volatilität dominiert (richtungslos).

v2 testet die zwei stärksten ungetesteten Richtungs-Hebel:
  1. Cross-Coin-Lead-Lag: BTC/ETH-Returns über mehrere Fenster (Alts folgen Majors).
  2. Orderbuch-Imbalance-DYNAMIK (Änderung, nicht Snapshot).
Plus ein SAUBER GETRENNTER Binär-Test: gibt es — nur auf den aufgelösten Fällen
(long vs short) — überhaupt ein ausbeutbares Richtungssignal mit MONOTONER
Kalibrierung? Das ist der ehrliche Make-or-Break.

Label je Entry t:  long=+tp% vor -sl%, short=-tp% vor +sl%, skip=keines (Horizont).
Alles aus settings.gbm_predictor. KEINE Defaults, KEINE Fallbacks, fester Seed.
"""
import json
import time
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

SETTINGS = "/opt/coin/settings.json"
MARKET_COINS = ["BTC", "ETH"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def req(d, k, ctx):
    if k not in d:
        raise RuntimeError(f"settings.{ctx}.{k} fehlt — KEINE Defaults")
    return d[k]


def load_cfg():
    s = json.load(open(SETTINGS))
    c = req(s, "gbm_predictor", "root")
    lab = req(c, "label", "gbm_predictor"); feat = req(c, "features", "gbm_predictor")
    ds = req(c, "dataset", "gbm_predictor"); tr = req(c, "train", "gbm_predictor")
    return dict(
        db=s["databases"]["coins"],
        tp=float(req(lab, "tp_pct", "label")), sl=float(req(lab, "sl_pct", "label")),
        horizon_h=float(req(lab, "horizon_hours", "label")), fee=float(req(lab, "fee_roundtrip_pct", "label")),
        windows=list(req(feat, "windows_minutes", "features")),
        lookback_days=int(req(ds, "lookback_days", "dataset")), step=int(req(ds, "sample_step_minutes", "dataset")),
        min_hist=int(req(ds, "min_history_minutes", "dataset")), min_vol=float(req(ds, "min_dollar_volume_24h", "dataset")),
        test_frac=float(req(tr, "walkforward_test_frac", "train")),
        n_estimators=int(req(tr, "n_estimators", "train")), max_depth=int(req(tr, "max_depth", "train")),
        lr=float(req(tr, "learning_rate", "train")), min_child_weight=float(req(tr, "min_child_weight", "train")),
        subsample=float(req(tr, "subsample", "train")), colsample=float(req(tr, "colsample_bytree", "train")),
        seed=int(req(tr, "random_seed", "train")),
    )


def connect(db):
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def fetch_coins(conn, cfg):
    with conn.cursor() as cur:
        cur.execute("""SELECT symbol, sum(quote_asset_volume) AS dv
                       FROM agg_1m WHERE bucket > now() - interval '24 hours'
                       GROUP BY symbol HAVING sum(quote_asset_volume) >= %s
                       ORDER BY dv DESC""", (cfg["min_vol"],))
        return [r[0] for r in cur.fetchall()]


def fetch_series(conn, symbol, cfg):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT bucket, open, high, low, close, volume, taker_buy_base,
                              quote_asset_volume, funding, open_interest, premium,
                              spread_bps, book_imbalance_5, book_depth_5
                       FROM agg_1m WHERE symbol=%s AND bucket > now() - make_interval(days => %s)
                       ORDER BY bucket ASC""", (symbol, cfg["lookback_days"]))
        return cur.fetchall()


def fetch_market(conn, cfg):
    """BTC/ETH-Returns über Fenster, indexiert nach bucket (Lead-Lag-Features)."""
    out = None
    for sym in MARKET_COINS:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT bucket, close FROM agg_1m WHERE symbol=%s
                           AND bucket > now() - make_interval(days => %s) ORDER BY bucket ASC""",
                        (sym, cfg["lookback_days"]))
            rows = cur.fetchall()
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d["bucket"] = pd.to_datetime(d["bucket"])
        d = d.set_index("bucket")
        m = pd.DataFrame(index=d.index)
        pre = sym.lower()
        for w in cfg["windows"]:
            m[f"{pre}_ret_{w}"] = (d["close"] / d["close"].shift(w) - 1.0) * 100.0
        out = m if out is None else out.join(m, how="outer")
    return out


def build_features(df, cfg, market):
    f = pd.DataFrame(index=df.index)
    close = df["close"]; high = df["high"]; low = df["low"]
    vol = df["volume"]; taker = df["taker_buy_base"]
    ret_1m = close.pct_change() * 100.0
    for w in cfg["windows"]:
        f[f"ret_{w}"] = (close / close.shift(w) - 1.0) * 100.0
        vol_w = vol.rolling(w).sum(); vol_prev = vol.shift(w).rolling(w).sum()
        f[f"volr_{w}"] = vol_w / (vol_prev + 1e-9)
        f[f"takr_{w}"] = taker.rolling(w).sum() / (vol_w + 1e-9)
        f[f"rng_{w}"] = (((high - low) / close).rolling(w).mean()) * 100.0
        f[f"std_{w}"] = ret_1m.rolling(w).std()
        roll_max = high.rolling(w).max(); roll_min = low.rolling(w).min()
        f[f"pos_{w}"] = (close - roll_min) / (roll_max - roll_min + 1e-9)
    # Orderbuch-Imbalance-DYNAMIK (Änderung, nicht Snapshot)
    imb = df["book_imbalance_5"]
    f["imb5"] = imb
    f["imb5_chg15"] = imb - imb.shift(15)
    f["imb5_chg60"] = imb - imb.shift(60)
    f["imb5_slope"] = imb.rolling(30).mean() - imb.rolling(120).mean()
    # Taker-Beschleunigung (kurz vs lang)
    sw, lw = cfg["windows"][0], cfg["windows"][-1]
    f["taker_accel"] = f[f"takr_{sw}"] - f[f"takr_{lw}"]
    # Punkt-Features
    f["funding"] = df["funding"]
    f["funding_chg60"] = df["funding"] - df["funding"].shift(60)
    f["oi_chg60"] = (df["open_interest"] / df["open_interest"].shift(60) - 1.0) * 100.0
    f["premium"] = df["premium"]; f["spread_bps"] = df["spread_bps"]
    f["hour"] = df["bucket"].dt.hour; f["weekday"] = df["bucket"].dt.weekday
    # Cross-Coin Lead-Lag (per bucket gemappt)
    if market is not None:
        bk = pd.to_datetime(df["bucket"]).to_numpy()
        for col in market.columns:
            f[col] = market[col].reindex(pd.DatetimeIndex(bk)).to_numpy()
    return f


def label_at(df, idx, cfg):
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    n = len(close); H = int(round(cfg["horizon_h"] * 60))
    tp = cfg["tp"] / 100.0; sl = cfg["sl"] / 100.0
    out = np.full(len(idx), -1, dtype=np.int8)
    for k, i in enumerate(idx):
        end = i + H
        if end >= n:
            continue
        e = close[i]
        if not np.isfinite(e) or e <= 0:
            continue
        fh = high[i + 1:end + 1]; fl = low[i + 1:end + 1]
        up = np.argmax(fh >= e * (1 + tp)) if (fh >= e * (1 + tp)).any() else -1
        dn = np.argmax(fl <= e * (1 - sl)) if (fl <= e * (1 - sl)).any() else -1
        long_win = up != -1 and (dn == -1 or up < dn)
        dn2 = np.argmax(fl <= e * (1 - tp)) if (fl <= e * (1 - tp)).any() else -1
        up2 = np.argmax(fh >= e * (1 + sl)) if (fh >= e * (1 + sl)).any() else -1
        short_win = dn2 != -1 and (up2 == -1 or dn2 < up2)
        out[k] = 1 if long_win else (2 if short_win else 0)
    return out


def make_xgb(cfg, **kw):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["lr"],
        min_child_weight=cfg["min_child_weight"], subsample=cfg["subsample"], colsample_bytree=cfg["colsample"],
        random_state=cfg["seed"], n_jobs=-1, tree_method="hist", **kw)


def main():
    cfg = load_cfg()
    log(f"v2 CFG tp={cfg['tp']} sl={cfg['sl']} horizon={cfg['horizon_h']}h windows={cfg['windows']} step={cfg['step']}m")
    conn = connect(cfg["db"])
    coins = fetch_coins(conn, cfg)
    market = fetch_market(conn, cfg)
    log(f"{len(coins)} Coins; Market-Features: {list(market.columns) if market is not None else 'KEINE'}")

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
        sub = feats.iloc[idx]
        valid = sub.notna().all(axis=1).to_numpy()
        idx, labels = idx[valid], labels[valid]
        if len(idx) == 0:
            continue
        if feat_cols is None:
            feat_cols = list(feats.columns)
        Xp.append(feats.iloc[idx][feat_cols].to_numpy(dtype=np.float32))
        yp.append(labels)
        tp_.append(df["bucket"].iloc[idx].to_numpy())
        if (ci + 1) % 25 == 0:
            log(f"  {ci+1}/{len(coins)} Coins, {sum(len(p) for p in Xp):,} Samples")
    conn.close()

    X = np.concatenate(Xp); y = np.concatenate(yp); t = np.concatenate(tp_)
    order = np.argsort(t); X, y, t = X[order], y[order], t[order]
    n = len(y)
    log(f"\nDataset: {n:,} Samples, {len(feat_cols)} Features")
    for k, nm in [(1, "long"), (2, "short"), (0, "skip")]:
        log(f"  {nm:5}: {(y==k).sum():>8,} ({(y==k).mean()*100:5.1f}%)")
    split = int(n * (1 - cfg["test_frac"]))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    log(f"Walk-forward: train={len(ytr):,}, test={len(yte):,} (ab {pd.Timestamp(t[split])})")

    tp, sl, fee = cfg["tp"], cfg["sl"], cfg["fee"]

    # ---- Multiclass (deployment-realistisch, inkl. skip) ----
    clf = make_xgb(cfg, objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    log("Training multiclass ..."); clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte); classes = list(clf.classes_)
    pL = proba[:, classes.index(1)]; pS = proba[:, classes.index(2)]
    pick = np.where(pL >= pS, 1, 2); conf = np.maximum(pL, pS)
    log("\n=== Multiclass OOS-Selektion ===")
    log(f"{'conf>=':>7} {'n_sel':>7} {'%univ':>6} {'hit%':>6} {'net$/Tr':>9}")
    for thr in [0.50, 0.55, 0.60, 0.70, 0.80]:
        m = conf >= thr; ns = int(m.sum())
        if ns == 0:
            log(f"{thr:>7.2f} {0:>7}"); continue
        hit = (pick[m] == yte[m]).mean()
        pnl = np.where(pick[m] == yte[m], tp - fee, np.where(yte[m] == 0, -fee, -sl - fee))
        log(f"{thr:>7.2f} {ns:>7,} {ns/len(yte)*100:>5.1f}% {hit*100:>5.1f}% {pnl.mean():>8.3f}%")

    # ---- BINÄR: nur aufgelöste Fälle, gibt es überhaupt Richtungssignal? ----
    log("\n=== BINÄR-Test (nur aufgelöste long/short — ehrlicher Richtungs-Edge) ===")
    rtr = ytr != 0; rte = yte != 0
    yb_tr = (ytr[rtr] == 1).astype(int)  # 1=long, 0=short
    yb_te = (yte[rte] == 1).astype(int)
    base = yb_te.mean()
    log(f"  resolved train={rtr.sum():,} test={rte.sum():,} | Baserate long={base*100:.1f}%")
    bclf = make_xgb(cfg, objective="binary:logistic", eval_metric="logloss")
    bclf.fit(Xtr[rtr], yb_tr)
    pb = bclf.predict_proba(Xte[rte])[:, 1]  # P(long)
    cb = np.abs(pb - 0.5) + 0.5             # Richtungs-Confidence
    pickb = (pb >= 0.5).astype(int)
    log("  conf-Bucket -> reale Richtungs-Trefferquote (>50% = Edge, monoton = brauchbar):")
    for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.80),(0.80,1.01)]:
        m = (cb >= lo) & (cb < hi)
        if m.sum() < 50:
            log(f"    [{lo:.2f},{hi:.2f}) n={int(m.sum()):>6}  (zu wenig)"); continue
        acc = (pickb[m] == yb_te[m]).mean()
        log(f"    [{lo:.2f},{hi:.2f}) n={int(m.sum()):>6}  hit={acc*100:5.1f}%")
    # AUC als Gesamt-Trennschärfe
    try:
        from sklearn.metrics import roc_auc_score
        log(f"  ROC-AUC (long vs short) = {roc_auc_score(yb_te, pb):.4f}  (0.5=Zufall)")
    except Exception as e:
        log(f"  AUC n/a: {e}")

    imp = clf.feature_importances_; top = np.argsort(imp)[::-1][:18]
    log("\n=== Top-18 Features (multiclass gain) ===")
    for j in top:
        log(f"  {feat_cols[j]:16} {imp[j]:.4f}")


if __name__ == "__main__":
    main()
