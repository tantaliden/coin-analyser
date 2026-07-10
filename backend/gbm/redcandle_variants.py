"""Zwei plausible Lesarten der Vorlauf-Bedingung nebeneinander, mit Forward-Stats.
V1 = DUEMPELN-Prio: jede Vorlauf-Kerze <=+/-1% (grün-Zahl egal) + Absturz(>=3%).
V2 = GRUEN-Prio:    >=6 von 10 gruen (+/-1% egal) + Absturz(>=3%).
Volker entscheidet welche seiner Beobachtung entspricht. Keine Wertung."""
import json, psycopg2, numpy as np, datetime as dt
from collections import Counter

N_RED=5; DROP=-0.08; BIG_RED=0.03; PRE_N=10; PRE_GREEN_MIN=6; DIMPLE=0.01; BUCKET_S=300
W6=72  # 6h Fenster
s=json.load(open('/opt/coin/settings.json')); d=s['databases']['coins']
c=psycopg2.connect(host=d['host'],port=d['port'],dbname=d['name'],user=d['user'],password=d['password'])
cur=c.cursor(); cur.execute("SELECT DISTINCT symbol FROM agg_5m"); syms=[r[0] for r in cur.fetchall()]

def scan(mode):
    evs=[]
    for sym in syms:
        cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
        rows=cur.fetchall()
        if len(rows)<PRE_N+N_RED+W6+5: continue
        ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
        h=np.array([float(r[2]) for r in rows]); lo=np.array([float(r[3]) for r in rows]); cl=np.array([float(r[4]) for r in rows])
        with np.errstate(divide='ignore',invalid='ignore'): body=np.where(o>0,cl/o-1.0,0.0)
        red=cl<o; green=cl>o; n=len(rows); i=PRE_N
        while i<=n-N_RED:
            if red[i:i+N_RED].all() and o[i]>0 and np.all(np.diff(ts[i-PRE_N:i+N_RED])==BUCKET_S):
                cum=cl[i+N_RED-1]/o[i]-1.0
                if cum<=DROP and np.any(-body[i:i+N_RED]>=BIG_RED):
                    pre=slice(i-PRE_N,i)
                    ok = (np.all(np.abs(body[pre])<=DIMPLE)) if mode=='V1' else (green[pre].sum()>=PRE_GREEN_MIN)
                    if ok:
                        e=i+N_RED-1; entry=cl[e]
                        seg_h=h[e+1:e+1+W6]; seg_l=lo[e+1:e+1+W6]
                        if entry>0 and len(seg_h):
                            fmax=seg_h.max()/entry-1.0
                            hit=0
                            for k in range(len(seg_h)):
                                up=seg_h[k]/entry-1>=0.02; dn=seg_l[k]/entry-1<=-0.04
                                if up and dn: hit=0; break
                                if up: hit=1; break
                                if dn: hit=-1; break
                            evs.append({'sym':sym,'ts':ts[e],'cum':cum,'fmax':fmax,'ft':hit,
                                        'green':int(green[pre].sum())})
                    i+=N_RED; continue
            i+=1
    return evs

for mode,desc in [('V1','DUEMPELN-Prio: jede Vorlauf-Kerze <=1% (gruen egal)'),
                  ('V2','GRUEN-Prio: >=6/10 gruen (+/-1% egal)')]:
    evs=scan(mode)
    print(f"\n=== {mode} — {desc} ===")
    print(f"  Events: {len(evs)}")
    if not evs: continue
    ft=np.array([e['ft'] for e in evs]); fmax=np.array([e['fmax'] for e in evs])
    print(f"  +2%-Hoch je erreicht @6h: {100*np.mean(fmax>=0.02):.0f}%  |  Median Max-Hoch: {np.median(fmax)*100:+.2f}%")
    print(f"  First-touch +2%/-4% @6h: TP {100*np.mean(ft==1):.0f}%  SL {100*np.mean(ft==-1):.0f}%  keins {100*np.mean(ft==0):.0f}%")
    print(f"  Coins: {Counter(e['sym'] for e in evs).most_common(8)}")
    for e in sorted(evs,key=lambda x:x['ts']):
        print(f"    {e['sym']:10} {dt.datetime.fromtimestamp(e['ts']).strftime('%m-%d %H:%M')} Absturz{e['cum']*100:+.1f}% gruen={e['green']}/10 -> Hoch{e['fmax']*100:+.1f}% {'TP' if e['ft']==1 else 'SL' if e['ft']==-1 else '-'}")
