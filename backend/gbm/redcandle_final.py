"""Volker-These FINAL (10.07.2026):
 VORLAUF 10 Kerzen: >=6 gruen (verteilt) UND jede Kerze Body <=1% (kein grosser Move).
 ABSTURZ 5 rote in Folge, mind. EINE Kerze Body >=3%.
 Zwei Absturz-Definitionen nebeneinander: MIT kumulativ<=-8% (Original) und OHNE (nur 1x>=3%).
 'Bewegung' = Kerzenkoerper close/open. Forward: kommt danach ein Hoch? 1:1 nach Volker."""
import json, psycopg2, numpy as np, datetime as dt
from collections import Counter

N_RED=5; DROP=-0.08; BIG_RED=0.03; PRE_N=10; PRE_GREEN_MIN=6; DIMPLE=0.01; BUCKET_S=300
WINDOWS={'1h':12,'3h':36,'6h':72,'12h':144,'24h':288}
s=json.load(open('/opt/coin/settings.json')); d=s['databases']['coins']
c=psycopg2.connect(host=d['host'],port=d['port'],dbname=d['name'],user=d['user'],password=d['password'])
cur=c.cursor(); cur.execute("SELECT DISTINCT symbol FROM agg_5m"); syms=[r[0] for r in cur.fetchall()]

def scan(require_drop):
    evs=[]
    for sym in syms:
        cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close,volume FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
        rows=cur.fetchall()
        if len(rows)<PRE_N+N_RED+300: continue
        ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
        h=np.array([float(r[2]) for r in rows]); lo=np.array([float(r[3]) for r in rows])
        cl=np.array([float(r[4]) for r in rows]); vol=np.array([float(r[5] or 0) for r in rows])
        with np.errstate(divide='ignore',invalid='ignore'): body=np.where(o>0,cl/o-1.0,0.0)
        red=cl<o; green=cl>o; n=len(rows); i=PRE_N
        while i<=n-N_RED:
            if red[i:i+N_RED].all() and o[i]>0 and np.all(np.diff(ts[i-PRE_N:i+N_RED])==BUCKET_S):
                cum=cl[i+N_RED-1]/o[i]-1.0
                big=bool(np.any(-body[i:i+N_RED]>=BIG_RED))
                pre=slice(i-PRE_N,i)
                preok=bool(green[pre].sum()>=PRE_GREEN_MIN and np.all(np.abs(body[pre])<=DIMPLE))
                dropok=(cum<=DROP) if require_drop else True
                if big and preok and dropok:
                    e=i+N_RED-1; entry=cl[e]
                    if entry>0:
                        ev={'sym':sym,'ts':ts[e],'cum':cum,'green':int(green[pre].sum())}
                        for name,W in WINDOWS.items():
                            sh=h[e+1:e+1+W]; sl=lo[e+1:e+1+W]
                            if not len(sh): ev[name]=None; continue
                            fmax=sh.max()/entry-1.0; fmin=sl.min()/entry-1.0
                            hit=0
                            for k in range(len(sh)):
                                up=sh[k]/entry-1>=0.02; dn=sl[k]/entry-1<=-0.04
                                if up and dn: hit=0; break
                                if up: hit=1; break
                                if dn: hit=-1; break
                            ev[name]={'fmax':fmax,'fmin':fmin,'ft':hit}
                        evs.append(ev)
                    i+=N_RED; continue
            i+=1
    return evs

for require_drop,label in [(True,'MIT kumulativ <=-8% (Original)'),(False,'OHNE -8% (nur 5 rote + 1x >=3%)')]:
    evs=scan(require_drop)
    print(f"\n########## Absturz-Def: {label}  ->  {len(evs)} Events ##########")
    if not evs:
        print("  keine Events."); continue
    for name in WINDOWS:
        e6=[e for e in evs if e.get(name)]
        if not e6: continue
        fmax=np.array([e[name]['fmax'] for e in e6]); ft=np.array([e[name]['ft'] for e in e6])
        fmin=np.array([e[name]['fmin'] for e in e6])
        print(f"  {name:4} n={len(e6):3} | +2%-Hoch je: {100*np.mean(fmax>=0.02):3.0f}% | Median-Hoch {np.median(fmax)*100:+5.2f}% | Tiefer-Tief med {np.median(fmin)*100:+5.2f}% | TP+2/-4: {100*np.mean(ft==1):3.0f}% SL {100*np.mean(ft==-1):3.0f}% keins {100*np.mean(ft==0):3.0f}%")
    print("  Coins:", Counter(e['sym'] for e in evs).most_common(8))
    print("  Events:")
    for e in sorted(evs,key=lambda x:x['ts']):
        r6=e.get('6h'); tag=('TP' if r6['ft']==1 else 'SL' if r6['ft']==-1 else '-') if r6 else '?'
        hoch=f"{r6['fmax']*100:+.1f}%" if r6 else '?'
        print(f"    {e['sym']:10} {dt.datetime.fromtimestamp(e['ts']).strftime('%Y-%m-%d %H:%M')} Absturz{e['cum']*100:+5.1f}% gruen={e['green']}/10 -> Hoch@6h {hoch} {tag}")
