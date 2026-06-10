#!/usr/bin/env python3
"""Reach-Predictor PAPER-SHADOW-Service (09.06.2026).

Läuft live, ruft pro Coin reach_predictor.compute_signal (Richtung+TP+SL aus der
Coin-Historie, alle Features, kein Random) und öffnet bei Signal einen PAPER-Trade
im EIGENEN Buch (reach_paper_positions). Geschlossen wird mit TP/SL-First-Crossing
auf 10s-Klines + Horizont-Drift — identisch messbar.

KEIN Echtgeld. Alles aus settings.reach_model. Keine Fallbacks, kein Randomize.
"""
import json, time, sys, datetime as dt
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, "/opt/coin/backend")
from services.reach_predictor import compute_signal

SETTINGS = "/opt/coin/settings.json"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def req(d, k, c):
    if k not in d: raise RuntimeError(f"settings.{c}.{k} fehlt — KEINE Defaults")
    return d[k]

def load_cfg():
    s = json.load(open(SETTINGS)); c = s["reach_model"]
    lv = req(c, "live", "reach_model"); mt = req(c, "match", "reach_model"); tg = req(c, "target", "reach_model")
    flat = {}
    flat.update(mt); flat.update(tg)   # match ∪ target -> compute_signal-cfg
    return s, dict(
        scan_s=int(req(lv, "scan_interval_seconds", "live")),
        cooldown=int(req(lv, "cooldown_seconds_per_symbol", "live")),
        start_bal=float(req(lv, "paper_start_balance", "live")),
        size=float(req(lv, "trade_size_usd", "live")),
        top_n=int(req(lv, "top_n_coins", "live")),
        max_open=int(req(lv, "max_open", "live")),
        leverage_cap=int(req(lv, "leverage_cap", "live")),
        fee=float(req(tg, "fee_roundtrip_pct", "target")),
        # Halten = Messfenster: die Reach-Rate wurde über forward_window_minutes
        # gemessen -> genau so lange halten, sonst driftet der Trade aus der Statistik.
        hold_min=int(req(mt, "forward_window_minutes", "match")),
        sig_cfg=flat)

def dbc(s):
    d = s["databases"]["coins"]
    _c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"]); _c.autocommit = True; return _c
def dba(s):
    d = s["databases"]["app"]
    _c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"]); _c.autocommit = True; return _c

DDL = """
CREATE TABLE IF NOT EXISTS reach_paper_positions(
  id BIGSERIAL PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
  entry_px DOUBLE PRECISION NOT NULL, tp_px DOUBLE PRECISION NOT NULL, sl_px DOUBLE PRECISION NOT NULL,
  tp_pct DOUBLE PRECISION, sl_pct DOUBLE PRECISION, conf DOUBLE PRECISION NOT NULL, n_match INT,
  leverage INT, opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  status TEXT NOT NULL DEFAULT 'open', exit_px DOUBLE PRECISION, pnl_pct DOUBLE PRECISION,
  pnl_usd DOUBLE PRECISION, closed_at TIMESTAMPTZ, exit_reason TEXT);
ALTER TABLE reach_paper_positions ADD COLUMN IF NOT EXISTS leverage INT;
ALTER TABLE reach_paper_positions ADD COLUMN IF NOT EXISTS net_exp_pct DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS reach_pp_status ON reach_paper_positions(status);
CREATE TABLE IF NOT EXISTS reach_paper_meta(id INT PRIMARY KEY DEFAULT 1, start_balance DOUBLE PRECISION);
"""


def coin_leverage(coins, sym, cap):
    """Hebel = min(cap, max-verfügbar) aus hl_meta. RULE_6: kein Fallback —
    fehlt der Coin in hl_meta -> None -> Caller skippt (kein stiller Default)."""
    cur = coins.cursor()
    cur.execute("SELECT max_leverage FROM hl_meta WHERE symbol=%s", (sym,))
    r = cur.fetchone()
    if not r or r[0] is None:
        return None
    return min(cap, int(r[0]))

def top_coins(coins, n):
    """Top-N nach Dollar-Volumen der letzten 24h (agg_1m quote_asset_volume)."""
    cur = coins.cursor()
    cur.execute("""SELECT symbol, SUM(quote_asset_volume) v FROM agg_1m
                   WHERE bucket > now() - interval '24 hours'
                   GROUP BY symbol ORDER BY v DESC NULLS LAST LIMIT %s""", (n,))
    return [r[0] for r in cur.fetchall()]

def check_close(coins, pos, hold_minutes):
    """Erste TP/SL-Überschreitung seit opened_at auf 10s-Klines, sonst Timeout-Drift
    nach hold_minutes (= Messfenster der Reach-Rate)."""
    cur = coins.cursor()
    cur.execute("""SELECT high, low, close, open_time FROM klines WHERE symbol=%s AND interval='10s'
                   AND open_time > %s ORDER BY open_time""", (pos['symbol'], pos['opened_at']))
    rows = cur.fetchall()
    if not rows: return None
    for h, l, c, ot in rows:
        if h is None or l is None: continue
        if pos['side'] == 'long':
            if l <= pos['sl_px']: return pos['sl_px'], 'SL', ot
            if h >= pos['tp_px']: return pos['tp_px'], 'TP', ot
        else:
            if h >= pos['sl_px']: return pos['sl_px'], 'SL', ot
            if l <= pos['tp_px']: return pos['tp_px'], 'TP', ot
    age_min = (rows[-1][3] - pos['opened_at']).total_seconds() / 60
    if age_min >= hold_minutes:
        return float(rows[-1][2]), 'DRIFT', rows[-1][3]
    return None

def main():
    s, C = load_cfg()
    log(f"REACH-SHADOW start: scan {C['scan_s']}s top{C['top_n']} max_open={C['max_open']} "
        f"gate net>={C['sig_cfg']['min_net_expectancy_pct']}% reach>={C['sig_cfg']['min_reach_rate']} "
        f"size ${C['size']} lev≤{C['leverage_cap']}x hold={C['hold_min']}min")
    with dba(s) as app:
        with app.cursor() as cur:
            cur.execute(DDL)
            cur.execute("INSERT INTO reach_paper_meta(id,start_balance) VALUES(1,%s) ON CONFLICT(id) DO NOTHING", (C['start_bal'],))
        app.commit()
    while True:
        t0 = time.time()
        try:
            with dbc(s) as coins, dba(s) as app:
                # 1) offene Positionen prüfen/schließen
                with app.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM reach_paper_positions WHERE status='open'")
                    opens = cur.fetchall()
                closed = 0
                for pos in opens:
                    r = check_close(coins, pos, C["hold_min"])
                    if r is None: continue
                    px, reason, ct = r
                    lev = pos['leverage'] or 1
                    if pos['side'] == 'long': move = (px / pos['entry_px'] - 1) * 100
                    else:                     move = (1 - px / pos['entry_px']) * 100
                    # Gehebelt: Move & Gebühr (auf Notional) skalieren mit dem Hebel -> Rendite auf Margin.
                    pnl = (move - C["fee"]) * lev
                    usd = C["size"] * pnl / 100
                    with app.cursor() as cu:
                        cu.execute("""UPDATE reach_paper_positions SET status='closed',exit_px=%s,pnl_pct=%s,
                                      pnl_usd=%s,closed_at=%s,exit_reason=%s WHERE id=%s""",
                                   (px, pnl, usd, ct, reason, pos['id']))
                    app.commit(); closed += 1
                # 2) Slots/Cooldown
                with app.cursor() as cur:
                    cur.execute("SELECT symbol FROM reach_paper_positions WHERE status='open'")
                    held = {r[0] for r in cur.fetchall()}
                    cur.execute("""SELECT symbol,MAX(closed_at) FROM reach_paper_positions
                                   WHERE closed_at IS NOT NULL GROUP BY symbol""")
                    lastclose = {r[0]: r[1] for r in cur.fetchall()}
                free_slots = C["max_open"] - len(held)
                opened = 0
                if free_slots > 0:
                    uni = top_coins(coins, C["top_n"])
                    now = dt.datetime.now(dt.timezone.utc)
                    for sym in uni:
                        if opened >= free_slots: break
                        if sym in held: continue
                        lc = lastclose.get(sym)
                        if lc and (now - lc).total_seconds() < C["cooldown"]: continue
                        sig = compute_signal(coins, sym, C["sig_cfg"])
                        if not sig or not sig.get('side'): continue
                        side = sig['side']; tp_pct = sig['tp_pct']; sl_pct = sig['sl_pct']
                        # Entry = letzter agg_1m close
                        with coins.cursor() as cc:
                            cc.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
                            p = cc.fetchone()
                        if not p or p[0] is None: continue
                        entry = float(p[0])
                        lev = coin_leverage(coins, sym, C["leverage_cap"])
                        if lev is None: continue  # RULE_6: kein hl_meta -> kein stiller Default-Hebel
                        if side == 'long': tp = entry * (1 + tp_pct / 100); slx = entry * (1 - sl_pct / 100)
                        else:              tp = entry * (1 - tp_pct / 100); slx = entry * (1 + sl_pct / 100)
                        with app.cursor() as cu:
                            cu.execute("""INSERT INTO reach_paper_positions
                                          (symbol,side,entry_px,tp_px,sl_px,tp_pct,sl_pct,conf,n_match,leverage,net_exp_pct)
                                          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                                       (sym, side, entry, tp, slx, tp_pct, sl_pct, sig['conf'], sig.get('n_match'), lev, sig['net_exp']))
                        app.commit(); opened += 1
                        log(f"OPEN {side} {sym} @ {entry:.6f} reach={sig['conf']:.2f} n={sig.get('n_match')} "
                            f"{lev}x (tp +{tp_pct:.2f}/sl -{sl_pct:.2f})")
                if closed or opened:
                    log(f"scan: +{opened} open, {closed} closed, held={len(held)}")
        except Exception as e:
            log(f"loop error: {e}")
        dt_el = time.time() - t0
        time.sleep(max(1, C["scan_s"] - dt_el))

if __name__ == "__main__":
    main()
