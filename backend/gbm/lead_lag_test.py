#!/usr/bin/env python3
"""Lead-Lag-Test: BTC-Impuls -> folgen die Alts? (Volker 10.06.2026)

Mechanisches Event mit INTRINSISCHER Richtung (mitlaufen, nicht vorhersagen):
  Event = BTC bewegt sich >= imp_pct in 5min (dedup: 30min Mindestabstand).
  Je Alt ab Event-Zeit: Rennen +1.0% vor -1.0% in BTC-Richtung (240min, 1m-High/Low).
  Konditioniert auf "Alt hat noch NICHT mitgezogen" (Lag-Kandidat) vs "schon gezogen".
Kausal, Hälften-Split, Tie = Loss. Kein Fit, keine Parameter-Suche außer imp_pct-Sweep.
"""
import json, time
import numpy as np
import pandas as pd
import psycopg2

SETTINGS = "/opt/coin/settings.json"
TOP_N = 30
FWD = 240; TP = 1.0; SL = 1.0
IMP_PCTS = [0.4, 0.6, 0.8]
DEDUP_MIN = 30

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

def load(c, sym):
    df = pd.read_sql("SELECT bucket, high, low, close FROM agg_1m WHERE symbol=%s ORDER BY bucket",
                     c, params=(sym,))
    if len(df) < 5000: return None
    df = df.set_index("bucket")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz=df.index.tz)
    return df.reindex(full)

def race(df, t_idx, direction):
    """Erst +TP oder -SL (in `direction`) ab Index t_idx? 1=TP zuerst, 0=SL zuerst/Tie, None=unaufgelöst."""
    cl = df["close"].to_numpy(); hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
    e = cl[t_idx]
    if not np.isfinite(e) or e <= 0: return None
    if direction > 0: tp_l, sl_l = e * (1 + TP/100), e * (1 - SL/100)
    else:             tp_l, sl_l = e * (1 - TP/100), e * (1 + SL/100)
    for j in range(t_idx + 1, min(t_idx + 1 + FWD, len(cl))):
        h, l = hi[j], lo[j]
        if not (np.isfinite(h) and np.isfinite(l)): continue
        if direction > 0:
            sl_hit = l <= sl_l; tp_hit = h >= tp_l
        else:
            sl_hit = h >= sl_l; tp_hit = l <= tp_l
        if sl_hit: return 0          # Tie/SL -> Loss (konservativ)
        if tp_hit: return 1
    return None

def main():
    c = conn()
    uni = top_coins(c, TOP_N)
    data = {}
    for s in uni:
        d = load(c, s)
        if d is not None: data[s] = d
    btc = data.pop("BTC")
    log(f"BTC {btc.index.min()} .. {btc.index.max()}, Alts: {len(data)}")
    bcl = btc["close"]
    ret5 = (bcl / bcl.shift(5) - 1) * 100
    tmid = btc.index.min() + (btc.index.max() - btc.index.min()) / 2

    for imp in IMP_PCTS:
        ev_mask = ret5.abs() >= imp
        times = []
        last = None
        for t in btc.index[ev_mask.fillna(False)]:
            if last is not None and (t - last).total_seconds() < DEDUP_MIN * 60: continue
            times.append(t); last = t
        log(f"--- BTC-Impuls >= {imp}%/5min: {len(times)} Events (dedup {DEDUP_MIN}min) ---")
        res = {"lag": {1: [], 2: []}, "moved": {1: [], 2: []}}
        for t in times:
            dirn = 1 if ret5.loc[t] > 0 else -1
            half = 1 if t <= tmid else 2
            for s, df in data.items():
                if t not in df.index: continue
                ti = df.index.get_loc(t)
                cl = df["close"]
                if ti < 5: continue
                a0, a1 = cl.iloc[ti - 5], cl.iloc[ti]
                if not (np.isfinite(a0) and np.isfinite(a1)) or a0 <= 0: continue
                alt_ret = (a1 / a0 - 1) * 100 * dirn   # Mitzug in BTC-Richtung
                r = race(df, ti, dirn)
                if r is None: continue
                bucket = "lag" if alt_ret < imp * 0.25 else "moved"
                res[bucket][half].append(r)
        for b in ("lag", "moved"):
            for h in (1, 2):
                v = res[b][h]
                if v:
                    log(f"  {b:5s} H{h}: n={len(v):5d} folgt_BTC={np.mean(v):.3f}")
    log("fertig")

if __name__ == "__main__":
    main()
