#!/usr/bin/env python3
"""Volkers Frage: Gibt es ein VORZEICHEN vor Trendwenden? Aufwärts-Strecke -> Abwärts-
Serie. Bricht Volumen/Trades ein, dünnt die Breadth aus, dreht Funding, bevor's kippt?
Wenn ja -> Gate: bei Warnung alle Events schließen + Predictoren pausieren/invertieren.

Markt-Index = equal-weight aller liquiden Coins (agg_1h, 42d). Wende (Top) = Trailing-24h
> +T UND Forward-24h < -T. Pro Indikator: Wert im Fenster VOR der Wende vs Gesamt-Baseline
(kausaler Perzentil). Plus Stichproben-Liste der Wenden. Read-only."""
import json, numpy as np, psycopg2
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r[0] for r in cur.fetchall()]
# stündliche Markt-Aggregate
cur.execute("""SELECT bucket,
                 sum(quote_asset_volume) vol, sum(trades) trd, sum(open_interest) oi,
                 sum(taker_buy_quote) tbq, avg(funding) fund, avg(spread_bps) spr,
                 count(*) ncoins
               FROM agg_1h WHERE bucket>now()-make_interval(days=>42) AND symbol=ANY(%s)
               GROUP BY bucket ORDER BY bucket""",(coins,))
M=cur.fetchall()
# Pro-Coin Returns für Index + Breadth
cur.execute("""SELECT symbol,bucket,close FROM agg_1h WHERE bucket>now()-make_interval(days=>42)
               AND symbol=ANY(%s) ORDER BY symbol,bucket""",(coins,))
px={}
for sym,b,cl in cur.fetchall():
    if cl is not None: px.setdefault(sym,{})[b]=float(cl)
c.close()
times=[r[0] for r in M]; T=len(times); idx={t:i for i,t in enumerate(times)}
vol=np.array([float(r[1] or 0) for r in M]); trd=np.array([float(r[2] or 0) for r in M])
oi=np.array([float(r[3] or 0) for r in M]); tbq=np.array([float(r[4] or 0) for r in M])
fund=np.array([float(r[5] or 0) for r in M]); spr=np.array([float(r[6] or 0) for r in M])
taker_ratio=np.divide(tbq,vol,out=np.full(T,0.5),where=vol>0)
# Markt-Index (equal-weight) + Breadth (Anteil Coins mit pos 6h-Return)
mret=np.zeros(T); cnt=np.zeros(T)
P={s:np.array([px[s].get(t,np.nan) for t in times]) for s in px}
for s,arr in P.items():
    r=np.full(T,np.nan); r[1:]=arr[1:]/arr[:-1]-1
    m=np.isfinite(r); mret[m]+=r[m]; cnt[m]+=1
mret=np.divide(mret,cnt,out=np.zeros(T),where=cnt>0)
mkt=np.cumprod(1+mret)  # Index
def breadth(t,L=6):
    up=tot=0
    for s,arr in P.items():
        if t-L>=0 and np.isfinite(arr[t]) and np.isfinite(arr[t-L]):
            tot+=1; up+= arr[t]>arr[t-L]
    return up/tot if tot else np.nan
# Wenden: Trailing24 > +T und Forward24 < -T  (Top)
W=24; Th=0.02
tops=[]
for i in range(W,T-W):
    tr=mkt[i]/mkt[i-W]-1; fw=mkt[i+W]/mkt[i]-1
    if tr>Th and fw<-Th: tops.append(i)
# dedup: nur lokale Maxima im 12h-Umkreis
ded=[]
for i in tops:
    if all(mkt[i]>=mkt[j] for j in tops if abs(j-i)<=12): ded.append(i)
tops=[]
for i in ded:
    if not tops or i-tops[-1]>W: tops.append(i)
print(f"Coins={len(P)} Stunden={T} (~{T/24:.0f}d)  Tops(Auf->Ab, |move|>{Th*100:.0f}%): {len(tops)}")

def pct_rank(series,i,lb=240):
    lo=max(0,i-lb); h=series[lo:i]
    return float((h<series[i]).mean()*100) if len(h)>10 else np.nan

print("\n=== Indikatoren im 6h-Fenster VOR jeder Wende (Perzentil vs vorherige 10d) ===")
print(f"{'Top @ ':>16} {'Vol%':>5} {'Trades%':>7} {'Breadth':>7} {'Funding%':>8} {'TakerBuy':>8} {'Vola%':>6}")
agg={k:[] for k in ["vol","trd","brd","fund","tak","vola"]}
for i in tops:
    w=range(max(0,i-6),i)
    vol_r=np.nanmean([pct_rank(vol,j) for j in w])
    trd_r=np.nanmean([pct_rank(trd,j) for j in w])
    brd=np.nanmean([breadth(j) for j in w])
    fund_r=np.nanmean([pct_rank(fund,j) for j in w])
    tak=np.nanmean([taker_ratio[j] for j in w])
    vola=np.nanstd(mret[max(0,i-6):i])*100
    vola_r=pct_rank(np.array([np.nanstd(mret[max(0,k-6):k]) for k in range(T)]),i)
    print(f"{str(times[i])[5:16]:>16} {vol_r:>5.0f} {trd_r:>7.0f} {brd:>7.2f} {fund_r:>8.0f} {tak:>8.3f} {vola_r:>6.0f}")
    agg["vol"].append(vol_r);agg["trd"].append(trd_r);agg["brd"].append(brd);agg["fund"].append(fund_r);agg["tak"].append(tak);agg["vola"].append(vola_r)
print("\n=== Mittel VOR Wenden vs neutrale Baseline (Perzentil 50 = kein Signal) ===")
for k,lab in [("vol","Volumen"),("trd","Trades"),("fund","Funding"),("vola","Vola")]:
    print(f"  {lab:10}: {np.nanmean(agg[k]):.0f}. Perzentil vor Wende (50=neutral)")
print(f"  {'Breadth':10}: {np.nanmean(agg['brd']):.2f} (Anteil Coins up; <0.5 = mehrheitlich schon schwach)")
print(f"  {'TakerBuy':10}: {np.nanmean(agg['tak']):.3f} (>0.5 = Käuferdominanz, <0.5 Verkäufer)")
