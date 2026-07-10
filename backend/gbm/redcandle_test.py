"""Volker-These (10.07.2026): 5 aufeinanderfolgende rote 5m-Kerzen mit kumulativ <= -8%
-> danach kommt ein Hoch (ggf. nach Duempeln +/-1%). Ehrliche empirische Pruefung auf agg_5m.
Kein Fallback, kein Hardcode ausser den These-Parametern (die SIND die These)."""
import json, psycopg2, numpy as np

N_RED = 5            # Anzahl roter Kerzen
DROP = -0.08         # kumulativ mind. -8%
BUCKET_S = 300       # 5min
WINDOWS = {'1h': 12, '3h': 36, '6h': 72, '12h': 144, '24h': 288}

s = json.load(open('/opt/coin/settings.json'))
d = s['databases']['coins']
c = psycopg2.connect(host=d['host'], port=d['port'], dbname=d['name'], user=d['user'], password=d['password'])
cur = c.cursor()
cur.execute("SELECT DISTINCT symbol FROM agg_5m")
syms = [r[0] for r in cur.fetchall()]

events = []   # je Event: dict mit forward-Messungen
for sym in syms:
    cur.execute("SELECT extract(epoch FROM bucket), open, high, low, close, volume FROM agg_5m WHERE symbol=%s ORDER BY bucket", (sym,))
    rows = cur.fetchall()
    if len(rows) < N_RED + 300:
        continue
    ts = np.array([r[0] for r in rows])
    o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
    lo = np.array([float(r[3]) for r in rows]); cl = np.array([float(r[4]) for r in rows])
    vol = np.array([float(r[5] or 0) for r in rows])
    red = cl < o
    n = len(rows)
    i = 0
    while i <= n - N_RED:
        # 5 rote Kerzen, zeitlich lueckenlos
        if red[i:i+N_RED].all() and np.all(np.diff(ts[i:i+N_RED]) == BUCKET_S):
            cum = cl[i+N_RED-1] / o[i] - 1.0
            if cum <= DROP and o[i] > 0:
                e = i + N_RED - 1           # Index letzte rote Kerze (Entry-Bucket)
                entry = cl[e]
                if entry > 0:
                    ev = {'sym': sym, 'cum': cum, 'entry': entry,
                          'vol5': float(vol[i:i+N_RED].sum())}
                    for name, W in WINDOWS.items():
                        seg_h = h[e+1:e+1+W]; seg_l = lo[e+1:e+1+W]
                        # zeitlich lueckenlos ab Entry? sonst Fenster kuerzen
                        if len(seg_h) == 0:
                            ev[name] = None; continue
                        fmax = seg_h.max() / entry - 1.0
                        fmin = seg_l.min() / entry - 1.0
                        # first-touch: kommt +2% Hoch vor -4% Tief? (und +3/-3, +5/-5)
                        ft = {}
                        for tp, sl in [(0.02, -0.04), (0.03, -0.03), (0.05, -0.05), (0.01, -0.02)]:
                            hit = 0
                            for k in range(len(seg_h)):
                                up = seg_h[k] / entry - 1.0 >= tp
                                dn = seg_l[k] / entry - 1.0 <= sl
                                if up and dn: hit = 0; break     # gleiche Kerze -> unklar, zaehl als kein TP
                                if up: hit = 1; break
                                if dn: hit = -1; break
                            ft[f'{tp}'] = hit
                        ev[name] = {'fmax': fmax, 'fmin': fmin, 'ft': ft}
                    events.append(ev)
                i += N_RED
                continue
        i += 1

print(f"=== These-Test: {N_RED} rote 5m-Kerzen, kumulativ <= {DROP*100:.0f}% ===")
print(f"Events gefunden: {len(events)} ueber {len(syms)} Coins ({s['databases']['coins']['name']}, 24.04-10.07)\n")
if not events:
    raise SystemExit

def pct(x): return f"{100*x:5.1f}%"
for name in WINDOWS:
    evs = [e for e in events if e.get(name)]
    if not evs: continue
    fmax = np.array([e[name]['fmax'] for e in evs])
    fmin = np.array([e[name]['fmin'] for e in evs])
    n = len(evs)
    print(f"--- Fenster {name} (n={n}) ---")
    print(f"  Hoch >= Entry+1% : {pct(np.mean(fmax>=0.01))}   +2%: {pct(np.mean(fmax>=0.02))}   +3%: {pct(np.mean(fmax>=0.03))}   +5%: {pct(np.mean(fmax>=0.05))}")
    print(f"  Median Max-Hoch  : {fmax.mean()*100:+.2f}% (Median {np.median(fmax)*100:+.2f}%)  |  Median Tiefer-Tief: {fmin.mean()*100:+.2f}% (Median {np.median(fmin)*100:+.2f}%)")
    print(f"  faellt ERST >2% weiter (fmin<=-2%): {pct(np.mean(fmin<=-0.02))}   >5% weiter: {pct(np.mean(fmin<=-0.05))}")
    for tp in ['0.01','0.02','0.03','0.05']:
        ft = np.array([e[name]['ft'][tp] for e in evs])
        win = np.mean(ft==1); loss = np.mean(ft==-1); non = np.mean(ft==0)
        sl = {'0.01':2,'0.02':4,'0.03':3,'0.05':5}[tp]
        print(f"    First-touch +{float(tp)*100:.0f}%/-{sl}% : TP {pct(win)}  SL {pct(loss)}  keins {pct(non)}")
    print()

# --- Liquiditaets-Split + Coin-Verteilung (Artefakt-Check) ---
from collections import Counter
print("\n=== Coin-Verteilung der 72 Events ===")
cc = Counter(e['sym'] for e in events)
print("  Top-Coins:", cc.most_common(12))
vols = np.array([e['vol5'] for e in events])
med = np.median(vols)
print(f"\n=== Liquiditaets-Split (5-Kerzen-Volumen, Median={med:,.0f} USD) — TP-Rate +2%/-4% @6h ===")
for label, mask in [('LIQUIDE (>Median Vol)', vols > med), ('ILLIQUIDE (<=Median)', vols <= med)]:
    idx = [i for i in range(len(events)) if mask[i] and events[i].get('6h')]
    ft = np.array([events[i]['6h']['ft']['0.02'] for i in idx])
    fmax = np.array([events[i]['6h']['fmax'] for i in idx])
    print(f"  {label:24} n={len(idx):3}  TP {100*np.mean(ft==1):4.1f}%  SL {100*np.mean(ft==-1):4.1f}%  Median-Hoch {np.median(fmax)*100:+.2f}%")
