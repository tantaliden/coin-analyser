#!/usr/bin/env python3
"""Entscheidender Symmetrie-Test: sind die Vorzeichen DROP-spezifisch oder feuern
sie genauso vor ANSTIEGEN (= nur Volatilitäts-Cluster, keine Richtung)?
Vergleicht Pre-Move-Features vor: großem DROP (fwd<=-T), großem RISE (fwd>=+T),
FLAT (|fwd|<0.2). Wenn Vola(drop)≈Vola(rise)>>flat → symmetrisch (nur Turbulenz).
Wenn ret60/OI/taker zwischen drop und rise UNTERSCHIEDLICH → echte Richtungs-Info.
Read-only BTC agg_1m 30d."""
import json, numpy as np, psycopg2
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("""SELECT bucket,close,quote_asset_volume,trades,taker_buy_quote,open_interest
               FROM agg_1m WHERE symbol='BTC' AND bucket>now()-interval '30 days' ORDER BY bucket""")
r=cur.fetchall(); c.close(); n=len(r)
close=np.array([float(x[1]) for x in r]); vol=np.array([float(x[2] or 0) for x in r])
trd=np.array([float(x[3] or 0) for x in r]); tbq=np.array([float(x[4] or 0) for x in r]); oi=np.array([float(x[5] or 0) for x in r])
taker=np.divide(tbq,vol,out=np.full(n,0.5),where=vol>0)
T=0.8
def feats(i):
    if i<120 or i+30>=n: return None
    fwd=(close[i+30]/close[i]-1)*100
    cls="drop" if fwd<=-T else ("rise" if fwd>=T else ("flat" if abs(fwd)<0.2 else "mid"))
    rr=np.diff(close[i-30:i+1])/close[i-30:i]
    return dict(cls=cls,
        vola=np.std(rr)*100,
        ret60=(close[i]/close[i-60]-1)*100,
        volr=vol[i-30:i].sum()/vol[i-60:i-30].sum() if vol[i-60:i-30].sum()>0 else np.nan,
        oichg=(oi[i]/oi[i-60]-1)*100 if oi[i-60]>0 else np.nan,
        tak=np.mean(taker[i-30:i]))
F=[feats(i) for i in range(n)]; F=[x for x in F if x]
def m(cls,key):
    v=[x[key] for x in F if x["cls"]==cls and np.isfinite(x[key])]; return np.mean(v) if v else float("nan")
nd=sum(1 for x in F if x["cls"]=="drop"); nr=sum(1 for x in F if x["cls"]=="rise"); nf=sum(1 for x in F if x["cls"]=="flat")
print(f"T={T}%  DROP={nd}  RISE={nr}  FLAT={nf}")
print(f"\n{'Feature':10}{'vor DROP':>10}{'vor RISE':>10}{'vor FLAT':>10}  Urteil")
for key,desc in [("vola","Vola 30m"),("ret60","Return 60m"),("volr","Vol-Ratio"),("oichg","OI-Change"),("tak","Taker-Buy")]:
    d,ri,fl=m("drop",key),m("rise",key),m("flat",key)
    # richtungsweisend wenn drop und rise deutlich verschieden; sonst symmetrisch
    if key in ("vola","volr"):
        urteil="SYMMETRISCH (nur Turbulenz)" if abs(d-ri)<0.15*abs(d-fl+1e-9) else "asymmetrisch"
    else:
        urteil="RICHTUNG (drop≠rise)" if (d-fl)*(ri-fl)<0 or abs(d-ri)>abs(0.3*(d-fl)) else "symmetrisch"
    print(f"{key:10}{d:>10.3f}{ri:>10.3f}{fl:>10.3f}  {desc}: {urteil}")
print("\n(Vola/Volumen symmetrisch = warnt nur vor Bewegung, nicht Richtung. ret60/OI/taker mit drop≠rise = echte Richtungs-Info.)")
