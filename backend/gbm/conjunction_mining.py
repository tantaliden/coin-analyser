#!/usr/bin/env python3
"""Konjunktions-Mining: kleine HOCHPRÄZISE Klasse suchen (Volker 10.06.2026).

Frage: gibt es ein Zusammenspiel mehrerer Werte, bei dem — WENN es feuert —
die Richtung mit hoher Trefferquote stimmt? (Nicht jedes Event, nur die wenigen.)

Methodik (Lektionen der gbm-Saga eingebaut):
  - KAUSAL: alle Features nur aus der Vergangenheit, Funding-z rolling (kein global-z).
  - Mining NUR auf Hälfte 1, Validierung NUR auf Hälfte 2 (Bins aus H1-Quantilen).
  - Erfolg = symmetrisches Rennen: +1.0% VOR -1.0% (long) bzw. gespiegelt (short),
    first-crossing auf 1m-High/Low, Horizont 240min. Gleicher-Bar-Tie = Loss (konservativ,
    wie SL-first in den Shadow-Services). Unaufgelöste raus.
  - NaN matcht NIE (kein 0-Fill — RULE_6).
  - Survivors werden zusätzlich nicht-überlappend (>=240min Abstand je Symbol) nachgemessen,
    weil 15min-Anchors im 240min-Fenster seriell korrelieren.
Ausgabe: Klartext-Report nach stdout (nohup-Log).
"""
import json, sys, time
import numpy as np
import pandas as pd
import psycopg2

SETTINGS = "/opt/coin/settings.json"
TOP_N = 30
ANCHOR_STEP = 15          # Minuten
FWD = 240                 # Forward-Fenster Minuten
TP = 1.0; SL = 1.0        # symmetrisches Rennen (%)
H1_MIN_N = 400; H1_MIN_HIT = 0.62
H2_MIN_N = 150; H2_MIN_HIT = 0.60
TOP_PAIRS_FOR_TRIPLES = 25

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
    # auf 1min-Raster reindizieren (Lücken = NaN, KEIN ffill)
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
    """first-crossing Rennen je Minute: up_j/dn_j = Minuten bis +TP/-SL (np.inf wenn nie)."""
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
    y_long = (up_j < dn_j)      # Tie (==) -> False = Loss für beide (konservativ)
    y_short = (dn_j < up_j)
    return resolved, y_long, y_short

def main():
    t0 = time.time()
    c = conn()
    uni = top_coins(c, TOP_N)
    log(f"Universe ({len(uni)}): {','.join(uni)}")

    # --- Pass 1: BTC + ret_30 aller Coins (für Markt-Features) ---
    raw = {}
    for s in uni:
        df = load_coin(c, s)
        if df is None:
            log(f"skip {s}: zu wenig Daten"); continue
        raw[s] = df
    if "BTC" not in raw:
        log("FATAL: BTC fehlt"); sys.exit(1)
    btc_cl = raw["BTC"]["close"]
    btc_ret30 = (btc_cl / btc_cl.shift(30) - 1) * 100
    r30 = pd.DataFrame({s: (d["close"] / d["close"].shift(30) - 1) for s, d in raw.items()})
    breadth30 = (r30 > 0).sum(axis=1) / r30.notna().sum(axis=1).where(r30.notna().sum(axis=1) >= 10) * 100

    # --- Pass 2: Features + Outcomes je Coin, Anchors sammeln ---
    F_list, meta_list = [], []
    Y = {"res": [], "yl": [], "ys": []}
    for s, df in raw.items():
        f = features(df, btc_ret30, breadth30)
        res, yl, ys = outcome(df)
        idx = np.arange(0, len(df), ANCHOR_STEP)
        idx = idx[idx < len(df) - FWD]
        F_list.append(f.iloc[idx])
        meta_list.append(pd.DataFrame({"symbol": s, "t": f.index[idx]}))
        Y["res"].append(res[idx]); Y["yl"].append(yl[idx]); Y["ys"].append(ys[idx])
        log(f"{s}: {len(idx)} anchors, resolve={res[idx].mean():.2f}")
    F = pd.concat(F_list, ignore_index=True)
    META = pd.concat(meta_list, ignore_index=True)
    RES = np.concatenate(Y["res"]); YL = np.concatenate(Y["yl"]); YS = np.concatenate(Y["ys"])
    log(f"Anchors gesamt: {len(F)}, resolved: {RES.mean():.2%}")

    # --- Hälften (zeitlich) ---
    tmid = META["t"].min() + (META["t"].max() - META["t"].min()) / 2
    H1 = (META["t"] <= tmid).to_numpy(); H2 = ~H1
    base_l = YL[RES & H1].mean()
    log(f"Split @ {tmid} | H1={H1.sum()} H2={H2.sum()} | Basisrate long (H1, resolved): {base_l:.3f}")

    # --- Bins aus H1-Quantilen (kausal Richtung H2) ---
    feats = list(F.columns)
    conds = []  # (name, bool-array)
    for k in feats:
        v = F[k].to_numpy()
        q1, q2 = np.nanpercentile(v[H1], [33.3, 66.7])
        conds.append((f"{k}<lo({q1:.3g})", v <= q1))
        conds.append((f"{k}=mid",          (v > q1) & (v < q2)))
        conds.append((f"{k}>hi({q2:.3g})", v >= q2))
    log(f"{len(conds)} Einzel-Bedingungen -> {len(conds)*(len(conds)-1)//2} Paare × 2 Richtungen")

    def rate(mask, y):
        m = mask & RES
        n = int(m.sum())
        return n, (float(y[m].mean()) if n else 0.0)

    # --- Mining: Singles + Paare auf H1 ---
    cands = []  # (desc, dir, mask_fn-tuple)
    for i, (ni, mi) in enumerate(conds):
        for side, y in (("long", YL), ("short", YS)):
            n, h = rate(mi & H1, y)
            if n >= H1_MIN_N and h >= H1_MIN_HIT:
                cands.append((ni, side, (i,), n, h))
    for i in range(len(conds)):
        ni, mi = conds[i]
        for j in range(i + 1, len(conds)):
            if i // 3 == j // 3: continue  # gleiches Feature
            nj, mj = conds[j]
            m = mi & mj
            if (m & H1).sum() < H1_MIN_N: continue
            for side, y in (("long", YL), ("short", YS)):
                n, h = rate(m & H1, y)
                if h >= H1_MIN_HIT:
                    cands.append((f"{ni} & {nj}", side, (i, j), n, h))
    log(f"H1-Kandidaten (n>={H1_MIN_N}, hit>={H1_MIN_HIT}): {len(cands)}")

    # --- Tripel aus Top-Paaren ---
    pairs = sorted([x for x in cands if len(x[2]) == 2], key=lambda x: -x[4])[:TOP_PAIRS_FOR_TRIPLES]
    for desc, side, ij, _, _ in pairs:
        i, j = ij
        y = YL if side == "long" else YS
        for k in range(len(conds)):
            if k // 3 in (i // 3, j // 3): continue
            m = conds[i][1] & conds[j][1] & conds[k][1]
            n, h = rate(m & H1, y)
            if n >= H1_MIN_N and h >= H1_MIN_HIT + 0.03:
                cands.append((f"{desc} & {conds[k][0]}", side, (i, j, k), n, h))
    log(f"Kandidaten inkl. Tripel: {len(cands)}")

    # --- Validierung NUR auf H2 ---
    survivors = []
    for desc, side, idxs, n1, h1 in cands:
        m = np.ones(len(F), bool)
        for i in idxs: m &= conds[i][1]
        y = YL if side == "long" else YS
        n2, h2 = rate(m & H2, y)
        if n2 >= H2_MIN_N and h2 >= H2_MIN_HIT:
            survivors.append((desc, side, idxs, n1, h1, n2, h2))
    survivors.sort(key=lambda x: -x[6])
    log(f"=== H2-SURVIVORS: {len(survivors)} von {len(cands)} getestet ===")

    # --- Survivors nicht-überlappend nachmessen (Dedup je Symbol, >=FWD min Abstand) ---
    tvals = META["t"].to_numpy(); syms = META["symbol"].to_numpy()
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
            if s in last and (tvals[ix] - last[s]) / np.timedelta64(1, "m") < FWD: continue
            keep[ix] = True; last[s] = tvals[ix]
        nu = int(keep.sum()); hu = float(y[keep].mean()) if nu else 0.0
        # ungefährer Binomial-Stderr als Einordnung
        se = (hu * (1 - hu) / nu) ** 0.5 if nu > 1 else 1.0
        log(f"  {side.upper():5s} hit_H1={h1:.3f}(n={n1}) hit_H2={h2:.3f}(n={n2}) "
            f"DEDUP={hu:.3f}±{se:.3f}(n={nu})  <- {desc}")
    if not survivors:
        log("KEIN Survivor — kein Zusammenspiel mit >=60% Richtungs-Präzision OOS bei diesem Setup.")
    log(f"fertig in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
