#!/usr/bin/env python3
"""Funding-Squeeze Paper-Shadow — Forward-Validierung des ersten stabilen
mechanischen Edges (mech_strategy.py: TP1.5/SL2.0, +0.10%/Event, 65% WR in-sample).
KEIN Echtgeld. Eigene Paper-Welt (fsq_paper_*).

Regel: pro Coin rollender Funding-zscore über z_lookback_hours; bei |z|>=z_threshold
Event -> Entry in SQUEEZE-Richtung (Funding<0 => überleveragte Shorts => LONG;
Funding>0 => LONG-Squeeze => SHORT). TP/SL aus settings. Hebel pro Coin aus hl_meta.

Alles aus settings.funding_squeeze. KEINE Defaults, KEINE Fallbacks."""
import json, time, sys
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

SETTINGS = "/opt/coin/settings.json"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def req(d, k, ctx):
    if k not in d: raise RuntimeError(f"settings.{ctx}.{k} fehlt — KEINE Defaults")
    return d[k]


def load_cfg():
    s = json.load(open(SETTINGS))
    f = req(s, "funding_squeeze", "root")
    return s, dict(
        scan=int(req(f, "scan_interval_seconds", "funding_squeeze")),
        z_lookback_h=int(req(f, "z_lookback_hours", "funding_squeeze")),
        z_thr=float(req(f, "z_threshold", "funding_squeeze")),
        tp=float(req(f, "tp_pct", "funding_squeeze")),
        sl=float(req(f, "sl_pct", "funding_squeeze")),
        horizon_h=float(req(f, "timeout_hours", "funding_squeeze")),
        fee=float(req(f, "fee_roundtrip_pct", "funding_squeeze")),
        cooldown_min=int(req(f, "cooldown_minutes", "funding_squeeze")),
        max_open=int(req(f, "max_open", "funding_squeeze")),
        size=float(req(f, "trade_size_usd", "funding_squeeze")),
        lev_cap=int(req(f, "leverage", "funding_squeeze")),
        min_vol=float(req(f, "min_dollar_volume_24h", "funding_squeeze")),
    )


def dbc(s, which):
    d = s["databases"][which]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])


def universe(coins, cfg):
    with coins.cursor() as cur:
        cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
                       GROUP BY symbol HAVING sum(quote_asset_volume)>=%s""", (cfg["min_vol"],))
        return [r[0] for r in cur.fetchall()]


def max_lev_map(coins, syms):
    if not syms: return {}
    with coins.cursor() as cur:
        cur.execute("SELECT symbol,max_leverage FROM hl_meta WHERE symbol=ANY(%s)", (list(syms),))
        return {r[0]: int(r[1]) for r in cur.fetchall() if r[1]}


def funding_z(coins, sym, lookback_h):
    """Rollender z-score des jüngsten Funding-Werts + aktueller close. None wenn unbrauchbar."""
    with coins.cursor() as cur:
        cur.execute("""SELECT funding, close FROM agg_1m WHERE symbol=%s
                       AND bucket>now()-make_interval(hours=>%s) ORDER BY bucket ASC""", (sym, lookback_h))
        rows = cur.fetchall()
    if len(rows) < 120: return None
    fund = np.array([float(r[0]) for r in rows if r[0] is not None])
    if len(fund) < 120: return None
    sd = np.std(fund)
    if sd < 1e-12: return None   # konstantes Funding -> kein Event
    px = rows[-1][1]
    if px is None or float(px) <= 0: return None
    z = (fund[-1] - np.mean(fund)) / sd
    return z, float(px)


def wallet_apply(app, pnl_usd):
    with app.cursor() as cur:
        cur.execute("""UPDATE fsq_paper_wallet_state SET balance=balance+%s, n_trades=n_trades+1,
                       peak_balance=GREATEST(peak_balance, balance+%s), updated_at=now() WHERE id=1""",
                    (pnl_usd, pnl_usd))
    app.commit()


def watch(coins, app, cfg):
    import datetime as dt
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM fsq_paper_positions WHERE status='open'")
        opens = cur.fetchall()
    if not opens: return 0
    now = dt.datetime.now(dt.timezone.utc); H = cfg["horizon_h"]; closed = 0
    for p in opens:
        with coins.cursor() as cur:
            cur.execute("""SELECT high,low,close,bucket FROM agg_1m WHERE symbol=%s AND bucket>%s
                           ORDER BY bucket ASC""", (p["symbol"], p["opened_at"]))
            rows = cur.fetchall()
        exit_px = reason = ct = None
        for hi, lo, cl, bk in rows:
            if hi is None or lo is None: continue
            if p["side"] == "long":
                if lo <= p["sl_px"]: exit_px, reason, ct = p["sl_px"], "loss", bk; break
                if hi >= p["tp_px"]: exit_px, reason, ct = p["tp_px"], "win", bk; break
            else:
                if hi >= p["sl_px"]: exit_px, reason, ct = p["sl_px"], "loss", bk; break
                if lo <= p["tp_px"]: exit_px, reason, ct = p["tp_px"], "win", bk; break
        if exit_px is None:
            age = (now - p["opened_at"]).total_seconds() / 3600.0
            if age >= H and rows: exit_px, reason, ct = float(rows[-1][2]), "timeout", rows[-1][3]
        if exit_px is None: continue
        entry = p["entry_px"]; notional = float(p["qty"]) * entry
        gross = (exit_px - entry) / entry if p["side"] == "long" else (entry - exit_px) / entry
        pnl_usd = notional * (gross - cfg["fee"] / 100.0)
        pnl_pct = pnl_usd / p["margin_usd"] * 100.0 if p["margin_usd"] else 0.0
        with app.cursor() as cur:
            cur.execute("""UPDATE fsq_paper_positions SET status=%s,exit_px=%s,pnl_pct=%s,pnl_usd=%s,
                           closed_at=%s,close_reason=%s WHERE id=%s""",
                        (reason, exit_px, pnl_pct, pnl_usd, ct, reason, p["id"]))
        app.commit(); wallet_apply(app, pnl_usd); closed += 1
        log(f"CLOSE {p['side']} {p['symbol']} {reason} pnl={pnl_pct:.2f}% ${pnl_usd:.3f}")
    if closed: log(f"watch: {closed} geschlossen")
    return closed


def scan(coins, app, cfg):
    import datetime as dt
    with app.cursor() as cur:
        cur.execute("SELECT count(*) FROM fsq_paper_positions WHERE status='open'"); n_open = cur.fetchone()[0]
        cur.execute("SELECT symbol FROM fsq_paper_positions WHERE status='open'"); held = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT symbol,max(opened_at) FROM fsq_paper_positions GROUP BY symbol"); last = {r[0]: r[1] for r in cur.fetchall()}
    if n_open >= cfg["max_open"]: return 0
    now = dt.datetime.now(dt.timezone.utc); uni = universe(coins, cfg); mlev = max_lev_map(coins, uni)
    opened = 0
    for sym in uni:
        if n_open + opened >= cfg["max_open"]: break
        if sym in held: continue
        ml = mlev.get(sym)
        if ml is None: continue
        lo = last.get(sym)
        if lo and (now - lo).total_seconds() < cfg["cooldown_min"] * 60: continue
        r = funding_z(coins, sym, cfg["z_lookback_h"])
        if r is None: continue
        z, px = r
        if abs(z) < cfg["z_thr"]: continue
        side = "long" if z < 0 else "short"   # neg Funding -> Short-Squeeze -> LONG
        tp, sl = cfg["tp"], cfg["sl"]
        if side == "long": tp_px, sl_px = px * (1 + tp/100), px * (1 - sl/100)
        else:              tp_px, sl_px = px * (1 - tp/100), px * (1 + sl/100)
        eff_lev = min(cfg["lev_cap"], ml); qty = cfg["size"] * eff_lev / px
        with app.cursor() as cur:
            cur.execute("""INSERT INTO fsq_paper_positions
                (symbol,side,entry_px,tp_px,sl_px,qty,margin_usd,fund_z)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
                (sym, side, px, tp_px, sl_px, qty, cfg["size"], float(z)))
        app.commit(); opened += 1
        log(f"OPEN {side} {sym} @ {px:.6g} z={z:+.2f} lev={eff_lev} tp{tp}/sl{sl}")
    return opened


def main():
    s, cfg = load_cfg()
    log(f"Funding-Squeeze-Shadow start: |z|>={cfg['z_thr']} TP{cfg['tp']}/SL{cfg['sl']} "
        f"horizon={cfg['horizon_h']}h max_open={cfg['max_open']} scan={cfg['scan']}s")
    coins = dbc(s, "coins"); app = dbc(s, "app")
    while True:
        t0 = time.time()
        try:
            s, cfg = load_cfg()
            watch(coins, app, cfg)
            scan(coins, app, cfg)
        except Exception as e:
            log(f"loop error: {e}")
            try: coins.rollback(); app.rollback()
            except Exception: coins = dbc(s, "coins"); app = dbc(s, "app")
        time.sleep(max(1, cfg["scan"] - (time.time() - t0)))


if __name__ == "__main__":
    main()
