#!/usr/bin/env python3
"""Fehlalarm-Test des Wende-Vorzeichens. Ein Signal taugt nur, wenn P(Wende|Signal)
deutlich > Basisrate UND es nicht ständig grundlos feuert. Im Aufwärtstrend
(Trailing-24h>+2%): feuert Signal X -> folgt in 24h eine Wende (Forward-24h<-2%)?
Kandidaten: Seller-Divergenz (Taker<0.5), hohe Vola, Breadth-Schwund, Funding-Extrem.
Ausgabe je Signal: Treffer-P vs Basis, Feuer-Häufigkeit, Lift. Read-only agg_1h 42d."""
import json, numpy as np, psycopg2
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
               GROUP BY symbol HAVING sum(quote_asset_volume)>=5000000""")
coins=[r[0] for r in cur.fetchall()]
cur.execute("""SELECT bucket, sum(quote_asset_volume) vol, sum(trades) trd,
                 sum(taker_buy_quote) tbq, avg(funding) fund
               FROM agg_1h WHERE bucket>now()-make_interval(days=>42) AND symbol=ANY(%s)
               GROUP BY bucket ORDER BY bucket""",(coins,))
M=cur.fetchall()
cur.execute("""SELECT symbol,bucket,close FROM agg_1h WHERE bucket>now()-make_interval(days=>42)
               AND symbol=ANY(%s) ORDER BY symbol,bucket""",(coins,))
px={}
for sym,b,cl in cur.fetchall():
    if cl is not None: px.setdefault(sym,{})[b]=float(cl)
c.close()
times=[r[0] for r in M]; T=len(times)
vol=np.array([float(r[1] or 0) for r in M]); trd=np.array([float(r[2] or 0) for r in M])
tbq=np.array([float(r[3] or 0) for r in M]); fund=np.array([float(r[4] or 0) for r in M])
taker=np.divide(tbq,vol,out=np.full(T,0.5),where=vol>0)
P={s:np.array([px[s].get(t,np.nan) for t in times]) for s in px}
mret=np.zeros(T); cnt=np.zeros(T)
for s,arr in P.items():
    r=np.full(T,np.nan); r[1:]=arr[1:]/arr[:-1]-1
    m=np.isfinite(r); mret[m]+=r[m]; cnt[m]+=1
mret=np.divide(mret,cnt,out=np.zeros(T),where=cnt>0); mkt=np.cumprod(1+mret)
W=24; Th=0.02
def roll_pct(a,i,lb=240):
    lo=max(0,i-lb); h=a[lo:i]; return (h<a[i]).mean()*100 if len(h)>10 else 50.0
# Vola-Reihe + Taker-6h-Mittel + Breadth
vola=np.array([np.nanstd(mret[max(0,k-6):k]) for k in range(T)])
taker6=np.array([np.nanmean(taker[max(0,k-6):k]) for k in range(T)])
def breadth(i,L=6):
    up=tot=0
    for arr in P.values():
        if i-L>=0 and np.isfinite(arr[i]) and np.isfinite(arr[i-L]): tot+=1; up+=arr[i]>arr[i-L]
    return up/tot if tot else 0.5
brd=np.array([breadth(i) for i in range(T)])
# nur Punkte im Aufwärtstrend mit definierter Zukunft
elig=[i for i in range(W,T-W) if mkt[i]/mkt[i-W]-1>Th]
rev=np.array([ (mkt[i+W]/mkt[i]-1)<-Th for i in elig])  # Wende in naechsten 24h?
base=rev.mean()
print(f"Im Aufwärtstrend: {len(elig)} Stunden, davon Wende in 24h: {rev.sum()} = Basisrate {base*100:.0f}%")
print(f"\n{'Signal':28} {'feuert':>7} {'P(Wende|Sig)':>12} {'Lift':>6} {'Fehlalarm%':>10}")
def test(name, cond):
    c=np.array([cond(i) for i in elig])
    if c.sum()<5: print(f"{name:28} {c.sum():>7}  zu selten"); return
    p=rev[c].mean(); fa=(c & ~rev).sum()/max(1,c.sum())*100
    print(f"{name:28} {c.sum():>7} {p*100:>11.0f}% {p/base:>5.2f}x {fa:>9.0f}%")
test("Seller-Div (Taker6<0.48)", lambda i: taker6[i]<0.48)
test("Vola hoch (pct>80)",       lambda i: roll_pct(vola,i)>80)
test("Breadth schwach (<0.5)",   lambda i: brd[i]<0.5)
test("Funding extrem (pct>85)",  lambda i: roll_pct(fund,i)>85)
test("Volumen-Spike (pct>85)",   lambda i: roll_pct(vol,i)>85)
test("Seller+Vola kombi",        lambda i: taker6[i]<0.48 and roll_pct(vola,i)>70)
print(f"\n(Lift>1 = besser als Zufall; Lift~1 = wertlos. Fehlalarm% = wie oft Signal feuert OHNE Wende.)")
