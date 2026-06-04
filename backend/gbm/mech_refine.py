#!/usr/bin/env python3
"""Schärfung des Funding-Squeeze-Edges: z-Schwelle hoch + OI-Bestätigung →
kleinere, hochpräzise Teilmenge. Squeeze: neg Funding→hoch, pos→runter.
OI-Confirm: OI steigt (Positionen bauen sich auf) am Funding-Extrem = stärkerer
Squeeze. Precision je Hälfte (Stabilität) + Events/Tag. Read-only agg_1m."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
s = json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c = psycopg2.connect(host=s["host"], port=s["port"], dbname=s["name"], user=s["user"], password=s["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r["symbol"] for r in cur.fetchall()]
H=6*60; T=1.5/100; STEP=10
# sammle pro Event: (t, z, oi_chg60, won_up_first)  -> won je nach squeeze später ausgewertet
EV=[]
for sym in coins:
    cur.execute("""SELECT bucket,close,high,low,funding,open_interest FROM agg_1m WHERE symbol=%s
                   AND bucket>now()-make_interval(days=>40) ORDER BY bucket ASC""",(sym,))
    r=cur.fetchall()
    if len(r)<2000: continue
    fund=np.array([float(x["funding"]) if x["funding"] is not None else np.nan for x in r])
    if np.nanstd(fund)<1e-9: continue
    close=np.array([float(x["close"]) for x in r]); high=np.array([float(x["high"]) for x in r]); low=np.array([float(x["low"]) for x in r])
    oi=np.array([float(x["open_interest"]) if x["open_interest"] is not None else np.nan for x in r])
    mu,sd=np.nanmean(fund),np.nanstd(fund); z=(fund-mu)/sd; n=len(r); bk=[x["bucket"] for x in r]
    for i in range(60,n-1,STEP):
        if not np.isfinite(z[i]) or abs(z[i])<2.0: continue
        e=close[i]
        if e<=0: continue
        oichg=(oi[i]/oi[i-60]-1)*100 if (np.isfinite(oi[i]) and np.isfinite(oi[i-60]) and oi[i-60]>0) else np.nan
        end=min(i+H,n); fh=high[i+1:end]; fl=low[i+1:end]
        up=np.argmax(fh>=e*(1+T)) if (fh>=e*(1+T)).any() else 10**9
        dn=np.argmax(fl<=e*(1-T)) if (fl<=e*(1-T)).any() else 10**9
        if up==10**9 and dn==10**9: continue
        EV.append((bk[i], z[i], oichg, up<dn))
EV.sort(key=lambda x:x[0]); c.close()
EV=np.array([(z,oc,1 if uf else 0) for _,z,oc,uf in EV],dtype=float)  # z, oichg, up_first
zc=EV[:,0]; oic=EV[:,1]; upf=EV[:,2]
def prec(mask):
    m=mask&np.isfinite(zc)
    if m.sum()<50: return None
    sq_up = zc[m]<0   # squeeze hoch wenn funding neg
    won = (sq_up & (upf[m]==1)) | (~sq_up & (upf[m]==0))
    h=m.sum()//2; idx=np.where(m)[0]
    won_sorted=won  # schon zeitsortiert
    return won.mean(), won[:len(won)//2].mean(), won[len(won)//2:].mean(), int(m.sum())
print(f"Events gesamt: {len(EV):,}  (~{len(EV)/40:.0f}/Tag)")
print(f"{'Filter':<34} {'n':>6} {'/Tag':>5} {'Prec%':>6} {'H1%':>5} {'H2%':>5}")
for label,mask in [
    ("|z|>2 (alle)", np.abs(zc)>=2),
    ("|z|>2.5", np.abs(zc)>=2.5),
    ("|z|>3", np.abs(zc)>=3),
    ("|z|>4", np.abs(zc)>=4),
    ("|z|>2 + OI steigt(>0)", (np.abs(zc)>=2)&(oic>0)),
    ("|z|>3 + OI steigt(>0)", (np.abs(zc)>=3)&(oic>0)),
    ("|z|>3 + OI>+2%", (np.abs(zc)>=3)&(oic>2)),
    ("|z|>2 + OI FÄLLT(<0)", (np.abs(zc)>=2)&(oic<0)),
]:
    r=prec(mask)
    if r: print(f"{label:<34} {r[3]:>6} {r[3]/40:>5.0f} {r[0]*100:>5.1f} {r[1]*100:>5.1f} {r[2]*100:>5.1f}")
    else: print(f"{label:<34} zu wenig")
