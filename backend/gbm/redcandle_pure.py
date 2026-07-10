"""Volker-These PUR (10.07.2026): REINES Candle-Muster, KEINE Prozent-Schwellen.
 15 Candles (5m) in Folge: erste 10 = Misch (>=6 gruen), letzte 5 = alle rot.
 Entry = Close der 5. roten. Frage: kommt danach ein Hoch? Kein Filter ausser Farbe."""
import json, psycopg2, numpy as np
from collections import Counter

PRE_N=10; PRE_GREEN_MIN=6; N_RED=5; BUCKET_S=300
WINDOWS={'1h':12,'3h':36,'6h':72,'12h':144,'24h':288}
s=json.load(open('/opt/coin/settings.json')); d=s['databases']['coins']
c=psycopg2.connect(host=d['host'],port=d['port'],dbname=d['name'],user=d['user'],password=d['password'])
cur=c.cursor(); cur.execute("SELECT DISTINCT symbol FROM agg_5m"); syms=[r[0] for r in cur.fetchall()]

events=[]
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
    rows=cur.fetchall()
    if len(rows)<PRE_N+N_RED+300: continue
    ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
    h=np.array([float(r[2]) for r in rows]); lo=np.array([float(r[3]) for r in rows]); cl=np.array([float(r[4]) for r in rows])
    red=cl<o; green=cl>o; n=len(rows); i=PRE_N
    while i<=n-N_RED:
        # 5 rote in Folge, davor 10 Candles mit >=6 gruen, alles lueckenlos
        if red[i:i+N_RED].all() and green[i-PRE_N:i].sum()>=PRE_GREEN_MIN and o[i]>0 \
           and np.all(np.diff(ts[i-PRE_N:i+N_RED])==BUCKET_S):
            e=i+N_RED-1; entry=cl[e]
            if entry>0:
                ev={'sym':sym,'ts':ts[e],'drop':cl[e]/o[i]-1.0,'green':int(green[i-PRE_N:i].sum())}
                for name,W in WINDOWS.items():
                    sh=h[e+1:e+1+W]; slo=lo[e+1:e+1+W]
                    if not len(sh): ev[name]=None; continue
                    fmax=sh.max()/entry-1.0; fmin=slo.min()/entry-1.0
                    ft={}
                    for tp,sl in [(0.02,-0.04),(0.03,-0.03),(0.01,-0.02)]:
                        hit=0
                        for k in range(len(sh)):
                            up=sh[k]/entry-1>=tp; dn=slo[k]/entry-1<=sl
                            if up and dn: hit=0; break
                            if up: hit=1; break
                            if dn: hit=-1; break
                        ft[str(tp)]=hit
                    ev[name]={'fmax':fmax,'fmin':fmin,'ft':ft}
                events.append(ev)
            i+=N_RED; continue
        i+=1

print("=== These PUR: 10 Misch-Candles (>=6 gruen) + 5 rote -> Hoch? (KEINE %-Schwelle) ===")
print(f"Events: {len(events)} ueber {len(syms)} Coins (agg_5m, 24.04-10.07.2026)\n")
if not events: raise SystemExit
drops=np.array([e['drop'] for e in events])
print(f"Rutsch-Tiefe der 5 roten (Median {np.median(drops)*100:.1f}%, Spanne {drops.min()*100:.1f}..{drops.max()*100:.1f}%)\n")

def pct(x): return f"{100*x:5.1f}%"
for name in WINDOWS:
    evs=[e for e in events if e.get(name)]
    if not evs: continue
    fmax=np.array([e[name]['fmax'] for e in evs]); fmin=np.array([e[name]['fmin'] for e in evs])
    print(f"--- {name} (n={len(evs)}) ---")
    print(f"  Hoch >=+1%: {pct(np.mean(fmax>=0.01))} +2%: {pct(np.mean(fmax>=0.02))} +3%: {pct(np.mean(fmax>=0.03))} +5%: {pct(np.mean(fmax>=0.05))}")
    print(f"  Median Max-Hoch {np.median(fmax)*100:+.2f}% | Median Tiefer-Tief {np.median(fmin)*100:+.2f}% | faellt erst >2%: {pct(np.mean(fmin<=-0.02))}")
    for tp in ['0.02','0.03','0.01']:
        ft=np.array([e[name]['ft'][tp] for e in evs]); sl={'0.02':4,'0.03':3,'0.01':2}[tp]
        print(f"    +{float(tp)*100:.0f}%/-{sl}%: TP {pct(np.mean(ft==1))} SL {pct(np.mean(ft==-1))} keins {pct(np.mean(ft==0))}")
    print()
print("Coins (Top):", Counter(e['sym'] for e in events).most_common(12))
