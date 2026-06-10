#!/usr/bin/env python3
"""Konjunktions-Mining Pass 2: EXTREM-Bins (seltene hochpräzise Klasse).

Pass 1 (Terzile): 0 Kandidaten >=62% in-sample -> Präzision lebt nicht in breiten Bins.
Pass 2: nur Extreme (p5/p10/p90/p95 aus H1), Paare+Tripel, Mining H1 / Validierung H2.
Zusätzlich: Top-15 H1-Kandidaten IMMER ausgeben (ehrliche Decke, auch wenn unter Schwelle)
und für H2-Survivors die Dedup-Messung (nicht-überlappend). Speichert Panel als npz-Cache.
"""
import json, os, time
import numpy as np
import pandas as pd
import psycopg2

SETTINGS = "/opt/coin/settings.json"
CACHE = "/opt/coin/backend/gbm/conj_panel.npz"
TOP_N = 30
ANCHOR_STEP = 15
FWD = 240
TP = 1.0; SL = 1.0
H1_MIN_N = 150; H1_MIN_HIT = 0.60
H2_MIN_N = 80;  H2_MIN_HIT = 0.60
TOP_PAIRS_FOR_TRIPLES = 40

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def conn():
    d = json.load(open(SETTINGS))["databases"]["coins"]
    c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"],
                         user=d["user"], password=d["password"])
    c.autocommit = True
    return c

def top_coins(c, n):
    cur = c.cursor()
    cur.execute("""SELECT symbol, SUM(quote_asset_volume) v FROM agg_1m
                   WHERE bucket > now() - interval '24 hours'
                   GROUP BY symbol ORDER BY v DESC NULLS LAST LIMIT %s""", (n,))
    return [r[0] for r in cur.fetchall()]

def load_coin(c, sym):
    df = pd.read_sql("""SELECT bucket, open, high, low, close, volume, trades,
                               taker_buy_base, funding, open_interest, premium,
                               spread_bps, book_imbalance_5
                        FROM agg_1m WHERE symbol=%s ORDER BY bucket""", c, params=(sym,))
    if len(df) < 5000:
        return None
    df = df.set_index("bucket")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz=df.index.tz)
    return df.reindex(full)

def features(df, btc_ret30, breadth30):
    cl = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret_30"]  = (cl / cl.shift(30)  - 1) * 100
    f["ret_120"] = (cl / cl.shift(120) - 1) * 100
    vol = df["volume"]
    v2 = vol.rolling(30, min_periods=20).sum()
    v1 = v2.shift(30)
    f["vol_ratio_60"] = v2 / v1.where(v1 > 0)
    tb = df["taker_buy_base"].rolling(30, min_periods=20).sum()
    f["taker_30"] = (tb / v2.where(v2 > 0)) * 100
    oi = df["open_interest"]
    oi0 = oi.shift(60)
    f["oi_d_60"] = (oi / oi0.where(oi0 > 0) - 1) * 100
    fu = df["funding"]
    mu = fu.rolling(10080, min_periods=2880).mean().shift(1)
    sd = fu.rolling(10080, min_periods=2880).std().shift(1)
    f["funding_z"] = (fu - mu) / sd.where(sd > 0)
    f["book_imb_30"] = df["book_imbalance_5"].rolling(30, min_periods=15).mean()
    rng = ((df["high"] - df["low"]) / df["low"].where(df["low"] > 0)) * 100
    r2 = rng.rolling(30, min_periods=20).mean()
    r1 = r2.shift(30)
    f["range_exp_60"] = r2 / r1.where(r1 > 0)
    f["prem_30"] = df["premium"].rolling(30, min_periods=15).mean()
    f["spread_30"] = df["spread_bps"].rolling(30, min_periods=15).mean()
    f["btc_ret_30"] = btc_ret30.reindex(f.index)
    f["breadth_30"] = breadth30.reindex(f.index)
    return f

def outcome(df):
    cl = df["close"].to_numpy(); hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
    T = len(cl)
    up_j = np.full(T, np.inf); dn_j = np.full(T, np.inf)
    up_lvl = cl * (1 + TP / 100); dn_lvl = cl * (1 - SL / 100)
    valid = np.isfinite(cl) & (cl > 0)
    for j in range(1, FWD + 1):
        h = np.full(T, np.nan); l = np.full(T, np.nan)
        h[:-j] = hi[j:]; l[:-j] = lo[j:]
        hit_up = valid & np.isinf(up_j) & np.isfinite(h) & (h >= up_lvl)
        hit_dn = valid & np.isinf(dn_j) & np.isfinite(l) & (l <= dn_lvl)
        up_j[hit_up] = j; dn_j[hit_dn] = j
    resolved = np.isfinite(up_j) | np.isfinite(dn_j)
    return resolved, (up_j < dn_j), (dn_j < up_j)

def build_panel():
    c = conn()
    uni = top_coins(c, TOP_N)
    log(f"Universe ({len(uni)}): {','.join(uni)}")
    raw = {}
    for s in uni:
        df = load_coin(c, s)
        if df is None: log(f"skip {s}"); continue
        raw[s] = df
    btc_cl = raw["BTC"]["close"]
    btc_ret30 = (btc_cl / btc_cl.shift(30) - 1) * 100
    r30 = pd.DataFrame({s: (d["close"] / d["close"].shift(30) - 1) for s, d in raw.items()})
    nn = r30.notna().sum(axis=1)
    breadth30 = (r30 > 0).sum(axis=1) / nn.where(nn >= 10) * 100
    F_list, t_list, s_list = [], [], []
    Y = {"res": [], "yl": [], "ys": []}
    for s, df in raw.items():
        f = features(df, btc_ret30, breadth30)
        res, yl, ys = outcome(df)
        idx = np.arange(0, len(df), ANCHOR_STEP)
        idx = idx[idx < len(df) - FWD]
        F_list.append(f.iloc[idx].to_numpy())
        t_list.append(f.index[idx].astype("int64").to_numpy())
        s_list.append(np.array([s] * len(idx)))
        Y["res"].append(res[idx]); Y["yl"].append(yl[idx]); Y["ys"].append(ys[idx])
    feats = list(features(raw["BTC"], btc_ret30, breadth30).columns)
    np.savez_compressed(CACHE,
                        F=np.vstack(F_list), t=np.concatenate(t_list),
                        sym=np.concatenate(s_list), feats=np.array(feats),
                        RES=np.concatenate(Y["res"]), YL=np.concatenate(Y["yl"]),
                        YS=np.concatenate(Y["ys"]))
    log(f"Panel-Cache geschrieben: {CACHE}")

def main():
    t0 = time.time()
    if not os.path.exists(CACHE):
        build_panel()
    z = np.load(CACHE, allow_pickle=True)
    F = z["F"]; tvals = z["t"]; syms = z["sym"]; feats = list(z["feats"])
    RES = z["RES"]; YL = z["YL"]; YS = z["YS"]
    log(f"Panel: {F.shape[0]} anchors × {len(feats)} features, resolved {RES.mean():.2%}")
    tmid = tvals.min() + (tvals.max() - tvals.min()) // 2
    H1 = tvals <= tmid; H2 = ~H1
    log(f"H1={H1.sum()} H2={H2.sum()} | Basisrate long H1: {YL[RES & H1].mean():.3f}")

    # Extrem-Bins aus H1-Quantilen
    conds = []
    for ki, k in enumerate(feats):
        v = F[:, ki].astype(float)
        vh = v[H1 & np.isfinite(v)]
        if len(vh) < 1000: continue
        p5, p10, p90, p95 = np.percentile(vh, [5, 10, 90, 95])
        conds.append((f"{k}<=p5({p5:.3g})",  v <= p5,  ki))
        conds.append((f"{k}<=p10({p10:.3g})", v <= p10, ki))
        conds.append((f"{k}>=p90({p90:.3g})", v >= p90, ki))
        conds.append((f"{k}>=p95({p95:.3g})", v >= p95, ki))
    log(f"{len(conds)} Extrem-Bedingungen")

    def rate(mask, y):
        m = mask & RES
        n = int(m.sum())
        return n, (float(y[m].mean()) if n else 0.0)

    all_evald = []   # (desc, side, idxs, n1, h1) — ALLE mit n1>=H1_MIN_N (für Decke)
    for i, (ni, mi, fi) in enumerate(conds):
        for side, y in (("long", YL), ("short", YS)):
            n, h = rate(mi & H1, y)
            if n >= H1_MIN_N: all_evald.append((ni, side, (i,), n, h))
    for i in range(len(conds)):
        ni, mi, fi = conds[i]
        for j in range(i + 1, len(conds)):
            nj, mj, fj = conds[j]
            if fi == fj: continue
            m = mi & mj
            if (m & H1 & RES).sum() < H1_MIN_N: continue
            for side, y in (("long", YL), ("short", YS)):
                n, h = rate(m & H1, y)
                all_evald.append((f"{ni} & {nj}", side, (i, j), n, h))
    cands = [x for x in all_evald if x[4] >= H1_MIN_HIT]
    log(f"bewertet: {len(all_evald)} | H1-Kandidaten hit>={H1_MIN_HIT}: {len(cands)}")

    top_h1 = sorted(all_evald, key=lambda x: -x[4])[:15]
    log("--- DECKE: Top-15 H1 (in-sample!) ---")
    for desc, side, idxs, n, h in top_h1:
        log(f"  H1 {side:5s} hit={h:.3f} n={n}  <- {desc}")

    pairs = sorted([x for x in cands if len(x[2]) == 2], key=lambda x: -x[4])[:TOP_PAIRS_FOR_TRIPLES]
    for desc, side, ij, _, _ in pairs:
        i, j = ij
        y = YL if side == "long" else YS
        for k in range(len(conds)):
            if conds[k][2] in (conds[i][2], conds[j][2]): continue
            m = conds[i][1] & conds[j][1] & conds[k][1]
            n, h = rate(m & H1, y)
            if n >= H1_MIN_N and h >= H1_MIN_HIT + 0.03:
                cands.append((f"{desc} & {conds[k][0]}", side, (i, j, k), n, h))
    log(f"Kandidaten inkl. Tripel: {len(cands)}")

    survivors = []
    for desc, side, idxs, n1, h1 in cands:
        m = np.ones(len(F), bool)
        for i in idxs: m &= conds[i][1]
        y = YL if side == "long" else YS
        n2, h2 = rate(m & H2, y)
        if n2 >= H2_MIN_N and h2 >= H2_MIN_HIT:
            survivors.append((desc, side, idxs, n1, h1, n2, h2))
    survivors.sort(key=lambda x: -x[6])
    log(f"=== H2-SURVIVORS: {len(survivors)} von {len(cands)} ===")

    for desc, side, idxs, n1, h1, n2, h2 in survivors[:40]:
        m = np.ones(len(F), bool)
        for i in idxs: m &= conds[i][1]
        m &= RES
        y = YL if side == "long" else YS
        keep = np.zeros(len(F), bool); last = {}
        order = np.argsort(tvals)
        for ix in order:
            if not m[ix]: continue
            s = syms[ix]
            if s in last and (tvals[ix] - last[s]) < FWD * 60 * 1_000_000_000: continue
            keep[ix] = True; last[s] = tvals[ix]
        nu = int(keep.sum()); hu = float(y[keep].mean()) if nu else 0.0
        se = (hu * (1 - hu) / nu) ** 0.5 if nu > 1 else 1.0
        log(f"  {side.upper():5s} H1={h1:.3f}(n={n1}) H2={h2:.3f}(n={n2}) "
            f"DEDUP={hu:.3f}±{se:.3f}(n={nu})  <- {desc}")
    if not survivors:
        log("KEIN Survivor in den Extremen.")
    log(f"fertig in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
