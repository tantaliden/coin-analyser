#!/usr/bin/env python3
"""Verifikation des Pass-2-Survivors: SHORT bei taker_30<=p10 & breadth_30<=p10.
Robustheit: je Coin, je Woche, p5-Variante, EV nach Gebühr. Nutzt conj_panel.npz."""
import numpy as np
import pandas as pd

z = np.load("/opt/coin/backend/gbm/conj_panel.npz", allow_pickle=True)
F = z["F"]; tvals = z["t"]; syms = z["sym"]; feats = list(z["feats"])
RES = z["RES"]; YS = z["YS"]
ki_t = feats.index("taker_30"); ki_b = feats.index("breadth_30")
taker = F[:, ki_t].astype(float); breadth = F[:, ki_b].astype(float)
tmid = tvals.min() + (tvals.max() - tvals.min()) // 2
H1 = tvals <= tmid
FWD_NS = 240 * 60 * 1_000_000_000
FEE = 0.13

def dedup_mask(m):
    keep = np.zeros(len(F), bool); last = {}
    for ix in np.argsort(tvals):
        if not m[ix]: continue
        s = syms[ix]
        if s in last and (tvals[ix] - last[s]) < FWD_NS: continue
        keep[ix] = True; last[s] = tvals[ix]
    return keep

def report(name, q_t, q_b):
    p_t = np.percentile(taker[H1 & np.isfinite(taker)], q_t)
    p_b = np.percentile(breadth[H1 & np.isfinite(breadth)], q_b)
    rule = (taker <= p_t) & (breadth <= p_b) & RES
    d = dedup_mask(rule)
    n = int(d.sum()); h = float(YS[d].mean()) if n else 0
    ev = h * 1.0 - (1 - h) * 1.0 - FEE
    print(f"\n== {name}: taker<= {p_t:.1f} & breadth<= {p_b:.1f} | DEDUP n={n} hit={h:.3f} "
          f"EV/Trade={ev:+.3f}% (1/1-Rennen, Fee {FEE})")
    df = pd.DataFrame({"sym": syms[d], "t": pd.to_datetime(tvals[d]), "y": YS[d].astype(int)})
    bc = df.groupby("sym")["y"].agg(["count", "mean"]).sort_values("count", ascending=False)
    print("je Coin (top 12):"); print(bc.head(12).to_string())
    print(f"Konzentration: top3-Coins = {bc['count'].head(3).sum()}/{n}")
    df["woche"] = df["t"].dt.to_period("W")
    wk = df.groupby("woche")["y"].agg(["count", "mean"])
    print("je Woche:"); print(wk.to_string())

report("p10&p10 (Survivor)", 10, 10)
report("p5&p5 (schaerfer)", 5, 5)
report("p10 taker solo-Vergleich", 10, 100)
