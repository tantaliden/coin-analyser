"""Beweis: der Edge sitzt in der RUTSCH-TIEFE, nicht in der Candle-Farbe.
 Gleiches Muster (10 Misch >=6 gruen + 5 rot), aber Ergebnis nach Tiefe des 5-Candle-Rutsches
 gebucketet. Erwartung: je tiefer, desto staerker der Bounce. TP=+2%/-4% first-touch @6h/@24h."""
import json, psycopg2, numpy as np

PRE_N=10; PRE_GREEN_MIN=6; N_RED=5; BUCKET_S=300
W={'6h':72,'24h':288}
s=json.load(open('/opt/coin/settings.json')); d=s['databases']['coins']
c=psycopg2.connect(host=d['host'],port=d['port'],dbname=d['name'],user=d['user'],password=d['password'])
cur=c.cursor(); cur.execute("SELECT DISTINCT symbol FROM agg_5m"); syms=[r[0] for r in cur.fetchall()]

def ftouch(sh,slo,entry,tp,sl):
    for k in range(len(sh)):
        up=sh[k]/entry-1>=tp; dn=slo[k]/entry-1<=sl
        if up and dn: return 0
        if up: return 1
        if dn: return -1
    return 0

rows_out=[]  # (drop, ft6, ft24, fmax6)
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
    rows=cur.fetchall()
    if len(rows)<PRE_N+N_RED+300: continue
    ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
    h=np.array([float(r[2]) for r in rows]); lo=np.array([float(r[3]) for r in rows]); cl=np.array([float(r[4]) for r in rows])
    red=cl<o; green=cl>o; n=len(rows); i=PRE_N
    while i<=n-N_RED:
        if red[i:i+N_RED].all() and green[i-PRE_N:i].sum()>=PRE_GREEN_MIN and o[i]>0 \
           and np.all(np.diff(ts[i-PRE_N:i+N_RED])==BUCKET_S):
            e=i+N_RED-1; entry=cl[e]
            if entry>0:
                drop=cl[e]/o[i]-1.0
                sh6=h[e+1:e+1+72]; sl6=lo[e+1:e+1+72]
                sh24=h[e+1:e+1+288]; sl24=lo[e+1:e+1+288]
                if len(sh6) and len(sh24):
                    ft6=ftouch(sh6,sl6,entry,0.02,-0.04); ft24=ftouch(sh24,sl24,entry,0.02,-0.04)
                    rows_out.append((drop, ft6, ft24, sh6.max()/entry-1.0))
            i+=N_RED; continue
        i+=1

arr=np.array(rows_out)
drop=arr[:,0]; ft6=arr[:,1]; ft24=arr[:,2]; fmax6=arr[:,3]
print(f"=== Bounce-Trefferquote nach Rutsch-Tiefe (n={len(arr)}) — TP +2%/-4% first-touch ===\n")
buckets=[( 0.00,-0.02,'  0 bis -2%'),(-0.02,-0.04,' -2 bis -4%'),(-0.04,-0.06,' -4 bis -6%'),
         (-0.06,-0.08,' -6 bis -8%'),(-0.08,-0.12,' -8 bis -12%'),(-0.12,-1.0,'  <= -12%')]
print(f"{'Rutsch-Tiefe':14} {'n':>6} {'+2%-Hoch@6h':>12} {'TP@6h':>7} {'SL@6h':>7} {'TP@24h':>8} {'SL@24h':>8}")
for hi,lohi,lab in buckets:
    m=(drop<=hi)&(drop>lohi)
    nn=m.sum()
    if nn==0: continue
    print(f"{lab:14} {nn:6} {100*np.mean(fmax6[m]>=0.02):11.1f}% {100*np.mean(ft6[m]==1):6.1f}% {100*np.mean(ft6[m]==-1):6.1f}% {100*np.mean(ft24[m]==1):7.1f}% {100*np.mean(ft24[m]==-1):7.1f}%")
print("\n-> Steigt die TP-Rate mit der Tiefe, ist der Edge die TIEFE (Kapitulation), nicht die Farbfolge.")
