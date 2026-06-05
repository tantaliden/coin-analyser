#!/usr/bin/env python3
"""Look-ahead-Check: mech_strategy.py rechnete z GLOBAL über 40d (nutzt Zukunfts-
Funding zur Normalisierung). Der Live-Service rechnet KAUSAL (rolling 168h, nur
Vergangenheit) und zeigt live ~38% statt 65% WR. Hier: derselbe TP1.5/SL2.0-Test,
aber z KAUSAL rollend (W=z_lookback_hours), exakt wie live. Hält der Edge? Wenn
nein, war +0.10%/Event ein Look-ahead-Artefakt — dann ehrlich verwerfen.
Read-only agg_1m, 40d."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
S=json.load(open("/opt/coin/settings.json"))
s=S["databases"]["coins"]; f=S["funding_squeeze"]
fee=float(f["fee_roundtrip_pct"]); W=int(f["z_lookback_hours"])*60; ZTHR=float(f["z_threshold"])
TP=float(f["tp_pct"])/100; SL=float(f["sl_pct"])/100; H=int(f["timeout_hours"])*60
c=psycopg2.connect(host=s["host"],port=s["port"],dbname=s["name"],user=s["user"],password=s["password"])
cur=c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=%s""",(f["min_dollar_volume_24h"],))
coins=[r["symbol"] for r in cur.fetchall()]
def causal_z(fund, W):
    """z[i] = (f[i]-mean(f[i-W:i]))/std(f[i-W:i]), nur Vergangenheit. NaN wenn <W//2 Werte."""
    n=len(fund); out=np.full(n,np.nan)
    cs=np.concatenate([[0],np.cumsum(fund)]); cs2=np.concatenate([[0],np.cumsum(fund*fund)])
    for i in range(n):
        lo=max(0,i-W); k=i-lo
        if k<60: continue
        m=(cs[i]-cs[lo])/k; v=(cs2[i]-cs2[lo])/k-m*m
        sd=np.sqrt(v) if v>1e-18 else 0
        if sd>0: out[i]=(fund[i]-m)/sd
    return out
res=[]  # (t, won)
for sym in coins:
    cur.execute("""SELECT bucket,close,high,low,funding FROM agg_1m WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>40) ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)<2000: continue
    fund=np.array([float(x["funding"]) if x["funding"] is not None else np.nan for x in r])
    if np.nanstd(fund)<1e-9: continue
    fund=np.nan_to_num(fund, nan=np.nanmean(fund))
    close=np.array([float(x["close"]) for x in r]);high=np.array([float(x["high"]) for x in r]);low=np.array([float(x["low"]) for x in r])
    z=causal_z(fund,W); n=len(r); bk=[x["bucket"] for x in r]; INF=10**9
    i=0
    while i<n-1:
        if not np.isfinite(z[i]) or abs(z[i])<ZTHR: i+=1; continue
        e=close[i]; sq=1 if z[i]<0 else -1
        end=min(i+H,n); fh=high[i+1:end]; fl=low[i+1:end]
        if sq==1: tp,slv=e*(1+TP),e*(1-SL)
        else:     tp,slv=e*(1-TP),e*(1+SL)
        if sq==1:
            up=np.argmax(fh>=tp) if (fh>=tp).any() else INF; dn=np.argmax(fl<=slv) if (fl<=slv).any() else INF
        else:
            up=np.argmax(fl<=tp) if (fl<=tp).any() else INF; dn=np.argmax(fh>=slv) if (fh>=slv).any() else INF
        if up==INF and dn==INF: pnl=-fee
        else: pnl=(TP*100-fee) if up<dn else (-SL*100-fee)
        res.append((bk[i],pnl,up<dn and not(up==INF and dn==INF)))
        i+=60  # 60min Cooldown wie live (ungefähr)
c.close()
res.sort(key=lambda x:x[0]); m=len(res)
pnl=np.array([x[1] for x in res]); win=np.array([1 for x in res if x[2]]).sum()
resolved=(pnl!=-fee).sum()
print(f"KAUSALER z (W={W//60}h), |z|>{ZTHR}, TP{TP*100}/SL{SL*100}: {m} Events (~{m/40:.0f}/Tag)")
print(f"  resolve={resolved/m*100:.0f}%  WR(resolved)={win/max(1,resolved)*100:.1f}%  EV/Event={pnl.mean():+.3f}%")
h=m//2; print(f"  H1 EV={pnl[:h].mean():+.3f}%  H2 EV={pnl[h:].mean():+.3f}%")
print(f"  >>> Vergleich: GLOBAL-z (mech_strategy) war 65% WR / +0.101%/Event")
