"""Volker-These v3 (10.07.2026, exakt): GENAU 5 rote Candles in Folge, gruen umschlossen.
 - 10 Misch-Candles davor, >=6 gruen, die 10. (direkt vor den roten) GRUEN
 - genau 5 rote in Folge
 - die Candle direkt nach den 5 roten GRUEN (=> exakt 5, nicht 6+)
 Zwei Einstiege: A=Close 5. rote (Tief), B=Close erste gruene danach ('nicht gleich verbrannt').
 Ergebnis nach Rutsch-Tiefe gebucketet. Kein %-Filter. Forward TP=+2%/-4% first-touch."""
import json, psycopg2, numpy as np

PRE_N=10; N_RED=5; PRE_GREEN_MIN=6; BUCKET_S=300
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

out=[]  # (drop, A_ft6, A_ft24, A_fmax6, B_ft6, B_ft24, B_fmax6)
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
    rows=cur.fetchall()
    if len(rows)<PRE_N+N_RED+300: continue
    ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
    h=np.array([float(r[2]) for r in rows]); lo=np.array([float(r[3]) for r in rows]); cl=np.array([float(r[4]) for r in rows])
    red=cl<o; green=cl>o; n=len(rows); i=PRE_N
    while i<=n-N_RED-1:
        # genau 5 rot, gruen davor (10.) und gruen danach (16.), Vorlauf >=6 gruen, lueckenlos
        if red[i:i+N_RED].all() and green[i-1] and green[i+N_RED] \
           and green[i-PRE_N:i].sum()>=PRE_GREEN_MIN and o[i]>0 \
           and np.all(np.diff(ts[i-PRE_N:i+N_RED+1])==BUCKET_S):
            eA=i+N_RED-1; entryA=cl[eA]        # Close 5. rote
            eB=i+N_RED;   entryB=cl[eB]         # Close 1. gruene danach
            drop=entryA/o[i]-1.0
            shA6=h[eA+1:eA+1+72]; slA6=lo[eA+1:eA+1+72]; shA24=h[eA+1:eA+1+288]; slA24=lo[eA+1:eA+1+288]
            shB6=h[eB+1:eB+1+72]; slB6=lo[eB+1:eB+1+72]; shB24=h[eB+1:eB+1+288]; slB24=lo[eB+1:eB+1+288]
            if entryA>0 and entryB>0 and len(shA6) and len(shB6) and len(shA24) and len(shB24):
                out.append((drop,
                    ftouch(shA6,slA6,entryA,0.02,-0.04), ftouch(shA24,slA24,entryA,0.02,-0.04), shA6.max()/entryA-1,
                    ftouch(shB6,slB6,entryB,0.02,-0.04), ftouch(shB24,slB24,entryB,0.02,-0.04), shB6.max()/entryB-1))
            i+=N_RED; continue
        i+=1

arr=np.array(out)
print(f"=== These v3: genau 5 rote (gruen umschlossen) + Vorlauf >=6 gruen ===")
print(f"Events: {len(arr)} ueber {len(syms)} Coins (agg_5m 24.04-10.07)\n")
if not len(arr): raise SystemExit
drop=arr[:,0]
print(f"Rutsch-Tiefe Median {np.median(drop)*100:.1f}%, Spanne {drop.min()*100:.1f}..{drop.max()*100:.1f}%\n")
buckets=[(0.00,-0.02,' 0..-2%'),(-0.02,-0.04,'-2..-4%'),(-0.04,-0.06,'-4..-6%'),
         (-0.06,-0.08,'-6..-8%'),(-0.08,-0.12,'-8..-12%'),(-0.12,-1.0,'<=-12%')]
print("EINSTIEG A = Close der 5. roten (Tief):")
print(f"{'Tiefe':10}{'n':>6}{'+2%Hoch@6h':>12}{'TP@6h':>8}{'SL@6h':>8}{'TP@24h':>9}{'SL@24h':>9}")
for hi,lohi,lab in buckets:
    m=(drop<=hi)&(drop>lohi); nn=int(m.sum())
    if nn==0: continue
    print(f"{lab:10}{nn:6}{100*np.mean(arr[m,3]>=0.02):11.1f}%{100*np.mean(arr[m,1]==1):7.1f}%{100*np.mean(arr[m,1]==-1):7.1f}%{100*np.mean(arr[m,2]==1):8.1f}%{100*np.mean(arr[m,2]==-1):8.1f}%")
print("\nEINSTIEG B = Close der 1. gruenen danach ('nicht gleich verbrannt'):")
print(f"{'Tiefe':10}{'n':>6}{'+2%Hoch@6h':>12}{'TP@6h':>8}{'SL@6h':>8}{'TP@24h':>9}{'SL@24h':>9}")
for hi,lohi,lab in buckets:
    m=(drop<=hi)&(drop>lohi); nn=int(m.sum())
    if nn==0: continue
    print(f"{lab:10}{nn:6}{100*np.mean(arr[m,6]>=0.02):11.1f}%{100*np.mean(arr[m,4]==1):7.1f}%{100*np.mean(arr[m,4]==-1):7.1f}%{100*np.mean(arr[m,5]==1):8.1f}%{100*np.mean(arr[m,5]==-1):8.1f}%")
