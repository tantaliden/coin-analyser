"""Funnel: welche Bedingung der These v2 filtert wie stark? Zeigt Volker transparent,
wo die 72 Absturz-Events auf 1 zusammenschrumpfen. Keine Wertung, nur Zaehlung."""
import json, psycopg2, numpy as np

N_RED=5; DROP=-0.08; BIG_RED=0.03; PRE_N=10; PRE_GREEN_MIN=6; DIMPLE=0.01; BUCKET_S=300
s=json.load(open('/opt/coin/settings.json')); d=s['databases']['coins']
c=psycopg2.connect(host=d['host'],port=d['port'],dbname=d['name'],user=d['user'],password=d['password'])
cur=c.cursor(); cur.execute("SELECT DISTINCT symbol FROM agg_5m"); syms=[r[0] for r in cur.fetchall()]

cnt={'A_absturz':0,'B_big3pct':0,'C_pre6green':0,'D_predimple':0,'BC':0,'BD':0,'CD':0,'ALL':0}
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket),open,high,low,close FROM agg_5m WHERE symbol=%s ORDER BY bucket",(sym,))
    rows=cur.fetchall()
    if len(rows)<PRE_N+N_RED+50: continue
    ts=np.array([float(r[0]) for r in rows]); o=np.array([float(r[1]) for r in rows])
    cl=np.array([float(r[4]) for r in rows])
    with np.errstate(divide='ignore',invalid='ignore'):
        body=np.where(o>0,cl/o-1.0,0.0)
    red=cl<o; green=cl>o; n=len(rows); i=PRE_N
    while i<=n-N_RED:
        if red[i:i+N_RED].all() and o[i]>0 and np.all(np.diff(ts[i-PRE_N:i+N_RED])==BUCKET_S):
            cum=cl[i+N_RED-1]/o[i]-1.0
            if cum<=DROP:
                A=True
                B=bool(np.any(-body[i:i+N_RED]>=BIG_RED))
                pre=slice(i-PRE_N,i)
                C=bool(green[pre].sum()>=PRE_GREEN_MIN)
                D=bool(np.all(np.abs(body[pre])<=DIMPLE))
                cnt['A_absturz']+=1
                cnt['B_big3pct']+=B; cnt['C_pre6green']+=C; cnt['D_predimple']+=D
                cnt['BC']+=B and C; cnt['BD']+=B and D; cnt['CD']+=C and D
                cnt['ALL']+=B and C and D
                i+=N_RED; continue
        i+=1

print("=== FUNNEL: was ueberlebt jede Bedingung (Basis = 5 rote lueckenlos, kumulativ <=-8%) ===")
print(f"  A  Absturz (Basis)                  : {cnt['A_absturz']}")
print(f"  B  + mind. 1 rote Kerze >=3%        : {cnt['B_big3pct']}")
print(f"  C  + Vorlauf: >=6 von 10 gruen       : {cnt['C_pre6green']}")
print(f"  D  + Vorlauf: JEDE Kerze <=+/-1% Body: {cnt['D_predimple']}   <-- vermuteter Haupt-Filter")
print(f"  B&C: {cnt['BC']}   B&D: {cnt['BD']}   C&D: {cnt['CD']}")
print(f"  ALLE (B&C&D)                        : {cnt['ALL']}")
print("\n  D allein senkt am staerksten -> 'jede Vorlauf-Kerze <=1%' ist extrem selten,")
print("  weil real fast immer >=1 der 10 Kerzen einen Body >1% hat.")
