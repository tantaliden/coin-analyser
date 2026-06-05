#!/usr/bin/env python3
"""Volkers Frage: wie sieht der Pfad 0-30min NACH Beginn einer großen Bewegung aus?
Fake-Ruck (erst gegen, dann mit) oder gleich durchziehen? Und ab welcher Minute ist
die Richtung verlässlich (= so lange muss man warten)?
Onset = Übergang ruhig->Bewegung: |ret(t0..t0+30)|>=0.8% UND |ret(t0-30..t0)|<0.3%.
Pro Onset: Richtungs-Klarheit P(sign(ret t0..t0+m)==Final) je Minute m; Fake-Ruck-Quote;
mittlere Gegenbewegung. Final = Vorzeichen ret(t0..t0+30). Read-only BTC agg_1m 30d."""
import json, numpy as np, psycopg2
co=json.load(open("/opt/coin/settings.json"))["databases"]["coins"]
c=psycopg2.connect(host=co["host"],port=co["port"],dbname=co["name"],user=co["user"],password=co["password"]);cur=c.cursor()
cur.execute("""SELECT close FROM agg_1m WHERE symbol='BTC' AND bucket>now()-interval '30 days' ORDER BY bucket""")
close=np.array([float(x[0]) for x in cur.fetchall()]); c.close(); n=len(close)
def ret(i,j): return (close[j]/close[i]-1)*100 if close[i]>0 else 0.0
onsets=[]
i=60
while i<n-30:
    pre=abs(ret(i-30,i)); mv=ret(i,i+30)
    if abs(mv)>=0.8 and pre<0.3:
        onsets.append((i, 1 if mv>0 else -1)); i+=30   # dedupe: 30min Sprung
    else: i+=1
print(f"Onsets (ruhig->Bewegung >=0.8% in 30min): {len(onsets)}")
ups=[o for o in onsets if o[1]>0]; dns=[o for o in onsets if o[1]<0]
print(f"  davon Anstiege: {len(ups)}  Drops: {len(dns)}")
mins=[1,2,3,5,8,10,15,20,25,30]
def clarity(group):
    out={}
    for m in mins:
        hit=[(np.sign(ret(i,min(i+m,n-1)))==d) for i,d in group]
        out[m]=np.mean(hit)*100 if hit else float('nan')
    return out
def fakeout(group):
    # erster Ruck (min 1-3) GEGEN die finale Richtung?
    f=[]
    for i,d in group:
        early=ret(i,i+3)
        f.append(np.sign(early)!=d and abs(early)>0.05)
    return np.mean(f)*100 if f else float('nan')
def adverse(group):
    # mittlere maximale Gegenbewegung (gegen final) in ersten 10 min
    a=[]
    for i,d in group:
        path=[ret(i,i+k)*d for k in range(1,11)]  # >0 = mit final
        a.append(min(path))   # negativster = größte Gegenbewegung
    return np.mean(a) if a else float('nan')
print(f"\n=== Richtungs-Klarheit: P(Richtung nach m min == finale Richtung) ===")
print("min   ", "  ".join(f"{m:>4}" for m in mins))
for lab,g in [("ALLE",onsets),("Anstiege",ups),("Drops",dns)]:
    cl=clarity(g); print(f"{lab:9}", "  ".join(f"{cl[m]:>4.0f}" for m in mins))
print(f"\nFake-Ruck-Quote (erste 3min gegen finale Richtung): ALLE {fakeout(onsets):.0f}% | Anstiege {fakeout(ups):.0f}% | Drops {fakeout(dns):.0f}%")
print(f"Ø größte Gegenbewegung erste 10min (%, negativ=gegen): ALLE {adverse(onsets):+.3f} | Anstiege {adverse(ups):+.3f} | Drops {adverse(dns):+.3f}")
print("\n(Klarheit ~50% = Münzwurf, >80% = verlässlich. Wo die Zeile 80 überschreitet = X Minuten warten.)")
