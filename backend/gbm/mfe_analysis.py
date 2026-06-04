#!/usr/bin/env python3
"""Strukturelle Fehler-Analyse: Gehen die VERLIERER erst ins Plus und geben es
dann her? (peak_px/trough_px). Wenn ja -> Reißleine/Trailing/Breakeven-Stop
rettet sie. Plus: Max-Adverse-Excursion der GEWINNER (wie weit gegen sie, bevor
sie liefen) -> damit ein Stop die Gewinner nicht killt. Dann Simulation:
Breakeven-Stop nach +X% bzw. Trailing — wie viele Loser gerettet, Winner verloren?
Read-only auf open_predictions (Live)."""
import json, numpy as np, psycopg2
from psycopg2.extras import RealDictCursor
a = json.load(open("/opt/coin/settings.json"))["databases"]["app"]
c = psycopg2.connect(host=a["host"], port=a["port"], dbname=a["name"], user=a["user"], password=a["password"])
cur = c.cursor(cursor_factory=RealDictCursor)
cur.execute("""SELECT side, entry_px, peak_px, trough_px, tp_px, sl_px, status, pnl_pct,
                      predicted_up_pct, predicted_down_pct
               FROM open_predictions
               WHERE status IN ('win','loss') AND peak_px IS NOT NULL AND trough_px IS NOT NULL
                 AND entry_px>0""")
rows = cur.fetchall(); c.close()
print(f"n={len(rows):,}")

def mfe_mae(r):
    e = float(r["entry_px"]); pk = float(r["peak_px"]); tr = float(r["trough_px"])
    if r["side"] == "long":
        mfe = (pk - e) / e * 100; mae = (e - tr) / e * 100
    else:
        mfe = (e - tr) / e * 100; mae = (pk - e) / e * 100
    return mfe, mae

L_mfe = []; W_mae = []; L_tpfrac = []
for r in rows:
    mfe, mae = mfe_mae(r)
    if r["status"] == "loss":
        L_mfe.append(mfe)
        tgt = abs(float(r["predicted_up_pct"] or 0)) if r["side"]=="long" else abs(float(r["predicted_down_pct"] or 0))
        if tgt > 0: L_tpfrac.append(mfe / tgt)
    else:
        W_mae.append(mae)
L_mfe = np.array(L_mfe); W_mae = np.array(W_mae)

print(f"\n=== VERLIERER: Max-Favorable-Excursion (wie weit ins PLUS vor SL)? n={len(L_mfe):,} ===")
print(f"  median MFE={np.median(L_mfe):.2f}%  mean={L_mfe.mean():.2f}%")
for thr in [0.3,0.5,0.8,1.0,1.5,2.0,3.0]:
    print(f"  Loser, die >= +{thr}% im Plus waren: {(L_mfe>=thr).mean()*100:5.1f}%")

print(f"\n=== GEWINNER: Max-Adverse-Excursion (wie weit ins MINUS vor TP)? n={len(W_mae):,} ===")
print(f"  median MAE={np.median(W_mae):.2f}%  mean={W_mae.mean():.2f}%")
for thr in [0.3,0.5,0.8,1.0,1.5,2.0]:
    print(f"  Winner, die <= -{thr}% im Minus waren: {(W_mae>=thr).mean()*100:5.1f}%")

# Simulation: Breakeven-Stop nach +X% (arm bei MFE>=X -> Stop auf Entry).
# Loser mit MFE>=X -> wären als ~Breakeven (statt SL) geschlossen worden (gerettet vom SL-Verlust).
# Winner mit MAE>=X (gingen erst gegen uns, NACH +X)? -> Winner wird nur gekillt, wenn er +X erreichte UND dann auf Entry zurück fiel; das wissen wir nicht exakt -> Näherung: Winner mit MFE>=X (fast alle) bleiben Winner, da sie TP erreichten ohne vorher auf Entry zurückzufallen ist NICHT garantiert. Konservativ: schätze Winner-Verlust über MAE nach Peak (nicht verfügbar) -> wir berichten nur die Loser-Rettung + Winner-Gefährdung grob.
print("\n=== Breakeven-Stop-Sim: arm nach +X% Favorable -> dann Stop=Entry ===")
tp_mean = np.mean([abs(float(r['predicted_up_pct'] or 0)) if r['side']=='long' else abs(float(r['predicted_down_pct'] or 0)) for r in rows if (r['predicted_up_pct'] or r['predicted_down_pct'])])
sl_mean = np.mean([ (float(r['entry_px'])-float(r['sl_px']))/float(r['entry_px'])*100 if r['side']=='long' else (float(r['sl_px'])-float(r['entry_px']))/float(r['entry_px'])*100 for r in rows])
print(f"  (Kontext: mittl. TP-Ziel ~{tp_mean:.2f}%, mittl. SL-Abstand ~{sl_mean:.2f}%)")
nL, nW = len(L_mfe), len(W_mae)
for X in [0.5,0.8,1.0,1.5]:
    saved = (L_mfe>=X).sum()           # Loser, die +X erreichten -> statt -SL nun ~0
    risk = (W_mae>=X).sum()            # Winner, die >=X gegen sich hatten (grobe Obergrenze gefährdeter Winner)
    print(f"  X=+{X}%: Loser>=X (rettbar)={saved} ({saved/nL*100:.0f}% der Loser) | Winner mit MAE>=X (max gefährdet)={risk} ({risk/nW*100:.0f}%)")
