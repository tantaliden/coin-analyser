"""Volker-These v2 (10.07.2026, praezisiert): Muster = Duempeln -> Absturz -> Hoch.
 VORLAUF: 10 Kerzen vor Absturz, mind. 6 gruen, JEDE Kerze |close/open-1|<=1% (kein grosser Move).
 ABSTURZ: 5 rote 5m-Kerzen, kumulativ <=-8%, mind. EINE Kerze mit >=3% Bewegung.
 DANACH: kommt ein Hoch? Ehrliche Pruefung auf agg_5m. 1:1 nach Volker, keine Interpretation."""
import json, psycopg2, numpy as np
from collections import Counter

N_RED = 5
DROP = -0.08
BIG_RED = 0.03        # mind. eine rote Kerze >= 3%
PRE_N = 10            # Vorlauf-Fenster
PRE_GREEN_MIN = 6     # mind. 6 gruen
DIMPLE = 0.01         # Duempeln: jede Vorlauf-Kerze Body <= +/-1%
BUCKET_S = 300
WINDOWS = {'1h': 12, '3h': 36, '6h': 72, '12h': 144, '24h': 288}

s = json.load(open('/opt/coin/settings.json'))
d = s['databases']['coins']
c = psycopg2.connect(host=d['host'], port=d['port'], dbname=d['name'], user=d['user'], password=d['password'])
cur = c.cursor()
cur.execute("SELECT DISTINCT symbol FROM agg_5m")
syms = [r[0] for r in cur.fetchall()]

events = []
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket), open, high, low, close, volume FROM agg_5m WHERE symbol=%s ORDER BY bucket", (sym,))
    rows = cur.fetchall()
    if len(rows) < PRE_N + N_RED + 300:
        continue
    ts = np.array([r[0] for r in rows])
    o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
    lo = np.array([float(r[3]) for r in rows]); cl = np.array([float(r[4]) for r in rows])
    vol = np.array([float(r[5] or 0) for r in rows])
    with np.errstate(divide='ignore', invalid='ignore'):
        body = np.where(o > 0, cl / o - 1.0, 0.0)   # Kerzen-Body close/open
    red = cl < o; green = cl > o
    n = len(rows)
    i = PRE_N
    while i <= n - N_RED:
        if red[i:i+N_RED].all() and o[i] > 0:
            cum = cl[i+N_RED-1] / o[i] - 1.0
            has_big = np.any(-body[i:i+N_RED] >= BIG_RED)     # eine rote >=3%
            # Vorlauf: 10 Kerzen davor
            pre = slice(i-PRE_N, i)
            pre_green = int(green[pre].sum())
            pre_dimple = bool(np.all(np.abs(body[pre]) <= DIMPLE))
            cont = bool(np.all(np.diff(ts[i-PRE_N:i+N_RED]) == BUCKET_S))
            if cum <= DROP and has_big and pre_green >= PRE_GREEN_MIN and pre_dimple and cont:
                e = i + N_RED - 1
                entry = cl[e]
                if entry > 0:
                    ev = {'sym': sym, 'cum': cum, 'entry': entry, 'pre_green': pre_green,
                          'vol5': float(vol[i:i+N_RED].sum()), 'ts': ts[e]}
                    for name, W in WINDOWS.items():
                        seg_h = h[e+1:e+1+W]; seg_l = lo[e+1:e+1+W]
                        if len(seg_h) == 0:
                            ev[name] = None; continue
                        fmax = seg_h.max() / entry - 1.0
                        fmin = seg_l.min() / entry - 1.0
                        ft = {}
                        for tp, sl in [(0.02, -0.04), (0.03, -0.03), (0.05, -0.05), (0.01, -0.02)]:
                            hit = 0
                            for k in range(len(seg_h)):
                                up = seg_h[k] / entry - 1.0 >= tp
                                dn = seg_l[k] / entry - 1.0 <= sl
                                if up and dn: hit = 0; break
                                if up: hit = 1; break
                                if dn: hit = -1; break
                            ft[f'{tp}'] = hit
                        ev[name] = {'fmax': fmax, 'fmin': fmin, 'ft': ft}
                    events.append(ev)
                i += N_RED
                continue
        i += 1

print("=== These v2: Duempeln(10K,>=6gruen,je<=1%) -> Absturz(5rot,<=-8%,>=1x3%) -> Hoch? ===")
print(f"Events: {len(events)} ueber {len(syms)} Coins (agg_5m, 24.04-10.07)\n")
if not events:
    print("KEIN einziges Event erfuellt alle Bedingungen. Muster zu streng fuer diese 2,5 Monate.")
    raise SystemExit

def pct(x): return f"{100*x:5.1f}%"
for name in WINDOWS:
    evs = [e for e in events if e.get(name)]
    if not evs: continue
    fmax = np.array([e[name]['fmax'] for e in evs]); fmin = np.array([e[name]['fmin'] for e in evs])
    print(f"--- Fenster {name} (n={len(evs)}) ---")
    print(f"  Hoch >= +1%: {pct(np.mean(fmax>=0.01))}  +2%: {pct(np.mean(fmax>=0.02))}  +3%: {pct(np.mean(fmax>=0.03))}  +5%: {pct(np.mean(fmax>=0.05))}")
    print(f"  Median Max-Hoch {np.median(fmax)*100:+.2f}% | Median Tiefer-Tief {np.median(fmin)*100:+.2f}% | faellt erst >2% weiter: {pct(np.mean(fmin<=-0.02))}")
    for tp in ['0.02','0.03','0.01']:
        ft = np.array([e[name]['ft'][tp] for e in evs]); sl={'0.02':4,'0.03':3,'0.01':2}[tp]
        print(f"    +{float(tp)*100:.0f}%/-{sl}%: TP {pct(np.mean(ft==1))} SL {pct(np.mean(ft==-1))} keins {pct(np.mean(ft==0))}")
    print()
print("Coin-Verteilung:", Counter(e['sym'] for e in events).most_common(10))
import datetime as dt
print("Events (Coin | Datum | Absturz% | Vorlauf-gruen):")
for e in sorted(events, key=lambda x:x['ts']):
    print(f"  {e['sym']:10} {dt.datetime.fromtimestamp(e['ts']).strftime('%Y-%m-%d %H:%M')}  {e['cum']*100:+.1f}%  gruen={e['pre_green']}/10")
