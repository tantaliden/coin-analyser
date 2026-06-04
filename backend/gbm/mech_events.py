#!/usr/bin/env python3
"""Scope-c: mechanische Events mit INTRINSISCHER Richtung. Hypothese: extremes
Funding hat eine eingebaute Squeeze-Richtung, die NICHT regime-flippt:
  Funding << 0 (Shorts zahlen) -> überleveragte Shorts -> Squeeze HOCH
  Funding >> 0 (Longs zahlen)  -> überleveragte Longs  -> Abverkauf RUNTER
Test: Event = |Funding-zscore| extrem. Forward H: erreicht der Preis +T% in
Squeeze-Richtung VOR -T% dagegen? Precision je Hälfte (Stabilität) + OOS.
Read-only, coins-DB agg_1m. Coins mit konstantem Funding (std~0) raus."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
s = json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c = psycopg2.connect(host=s["host"], port=s["port"], dbname=s["name"], user=s["user"], password=s["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins = [r["symbol"] for r in cur.fetchall()]
H = 6*60; T = 1.5/100; ZTHR = 2.0; STEP = 10
ev = []  # (t, squeeze_dir(+1 up/-1 down), won(bool))
for sym in coins:
    cur.execute("""SELECT bucket,close,high,low,funding FROM agg_1m WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>40) ORDER BY bucket ASC""",(sym,))
    rows = cur.fetchall()
    if len(rows) < 2000: continue
    fund = np.array([float(r["funding"]) if r["funding"] is not None else np.nan for r in rows])
    if np.nanstd(fund) < 1e-9: continue   # konstantes Funding -> skip
    close=np.array([float(r["close"]) for r in rows]); high=np.array([float(r["high"]) for r in rows]); low=np.array([float(r["low"]) for r in rows])
    mu,sd=np.nanmean(fund),np.nanstd(fund); z=(fund-mu)/sd
    bk=[r["bucket"] for r in rows]; n=len(rows)
    for i in range(0,n-1,STEP):
        if not np.isfinite(z[i]) or abs(z[i])<ZTHR: continue
        e=close[i]
        if e<=0: continue
        sq = 1 if z[i]<0 else -1   # neg funding -> up
        end=min(i+H,n)
        fh=high[i+1:end]; fl=low[i+1:end]
        up=np.argmax(fh>=e*(1+T)) if (fh>=e*(1+T)).any() else 10**9
        dn=np.argmax(fl<=e*(1-T)) if (fl<=e*(1-T)).any() else 10**9
        if up==10**9 and dn==10**9: continue   # nicht aufgelöst
        up_first = up<dn
        won = (up_first and sq==1) or ((not up_first) and sq==-1)
        ev.append((bk[i], sq, won))
c.close()
ev.sort(key=lambda x:x[0])
m=len(ev); won=np.array([e[2] for e in ev])
print(f"Funding-Extrem-Events (|z|>{ZTHR}, aufgelöst): {m:,}")
if m:
    print(f"Squeeze-Richtung Precision GESAMT: {won.mean()*100:.1f}%  (>50% = mechanischer Edge, <50% = Funding-Momentum statt Reversion)")
    h=m//2
    print(f"  1. Hälfte: {won[:h].mean()*100:.1f}%   2. Hälfte: {won[h:].mean()*100:.1f}%  (stabil? = nicht regime-abhängig)")
    print(f"  Events/Tag ~ {m/40:.0f}")
