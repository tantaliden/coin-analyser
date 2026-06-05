#!/usr/bin/env python3
"""Sturm-Warner (Volker 05.06.) — BTC-basierter Turbulenz-Detektor + Zustandsautomat.

Belegt durch btc_drops.py / move_path.py: Vola/Volumen/OI zeigen an, DASS eine große
Bewegung kommt (nicht wohin); nach ~10 Min ist die Richtung zu ~88% klar.

Stateless: jeder Aufruf rechnet aus BTC agg_1m neu. Phasen:
  calm       — kein Sturm, normaler Betrieb
  waiting    — Sturm aktiv, läuft < wait_minutes -> Richtung noch unklar (Münzwurf)
  clarified  — Sturm aktiv, läuft >= wait_minutes -> clarified_dir (+1 hoch / -1 runter)

Alle Schwellen aus settings.predictor.storm_warner. KEINE Defaults, KEINE Fallbacks.
"""
import numpy as np
from psycopg2.extras import RealDictCursor


def _req(d, k):
    if k not in d:
        raise RuntimeError(f"settings.predictor.storm_warner.{k} fehlt — KEINE Defaults")
    return d[k]


def evaluate(coins_conn, sw_cfg):
    """coins_conn: offene Verbindung zur coins-DB. sw_cfg: settings.predictor.storm_warner.
    Returns dict(enabled, phase, clarified_dir, vola, volr, oichg, minutes_in_storm)."""
    enabled = bool(_req(sw_cfg, "enabled"))
    vola_win = int(_req(sw_cfg, "vola_window_min"))
    vola_thr = float(_req(sw_cfg, "vola_threshold"))
    volr_thr = float(_req(sw_cfg, "volr_threshold"))
    oi_thr = float(_req(sw_cfg, "oi_threshold_pct"))
    wait_min = int(_req(sw_cfg, "wait_minutes"))
    lookback = int(_req(sw_cfg, "lookback_min"))
    out = {"enabled": enabled, "phase": "calm", "clarified_dir": None,
           "vola": None, "volr": None, "oichg": None, "minutes_in_storm": 0}
    if not enabled:
        return out
    with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT close, quote_asset_volume, open_interest
                       FROM agg_1m WHERE symbol='BTC'
                       AND bucket > now() - make_interval(mins => %s)
                       ORDER BY bucket ASC""", (lookback,))
        rows = cur.fetchall()
    if len(rows) < vola_win + 65:
        return out  # zu wenig Daten -> calm (kein Eingriff)
    close = np.array([float(r["close"]) for r in rows])
    vol = np.array([float(r["quote_asset_volume"] or 0) for r in rows])
    oi = np.array([float(r["open_interest"] or 0) for r in rows])
    n = len(close)

    def vola_at(i):
        if i - vola_win < 0:
            return np.nan
        seg = close[i - vola_win:i + 1]
        r = np.diff(seg) / seg[:-1]
        return float(np.std(r) * 100)

    vola_now = vola_at(n - 1)
    volr_now = (vol[n - 30:n].sum() / vol[n - 60:n - 30].sum()) if vol[n - 60:n - 30].sum() > 0 else 0.0
    oichg_now = ((oi[n - 1] / oi[n - 61] - 1) * 100) if (n >= 61 and oi[n - 61] > 0) else 0.0
    out.update(vola=round(vola_now, 4), volr=round(volr_now, 3), oichg=round(oichg_now, 3))

    storm_now = (vola_now >= vola_thr) and ((volr_now >= volr_thr) or (oichg_now >= oi_thr))
    if not storm_now:
        return out  # calm

    # Onset: Anfang des aktuellen zusammenhängenden Sturm-Laufs (Vola >= Schwelle)
    run = 0
    for i in range(n - 1, vola_win - 1, -1):
        v = vola_at(i)
        if np.isfinite(v) and v >= vola_thr:
            run += 1
        else:
            break
    out["minutes_in_storm"] = run
    if run < wait_min:
        out["phase"] = "waiting"
        return out
    onset = n - 1 - run + 1
    move = close[n - 1] / close[max(onset, 0)] - 1.0
    out["phase"] = "clarified"
    out["clarified_dir"] = 1 if move > 0 else -1
    return out
