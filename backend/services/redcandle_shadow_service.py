#!/usr/bin/env python3
"""Kapitulations-Bounce Paper-Shadow (Volker-These 10.07.2026).
Muster (5m-Candles): GENAU n_red rote in Folge, gruen umschlossen (Candle davor UND danach gruen),
davor pre_n Candles mit >=pre_green_min gruen, und der Rutsch der roten Candles <= -min_drop_pct.
Einstieg = Close der 1. gruenen danach (Bestaetigung). Long, TP/SL/Timeout. Kein Echtgeld.
Config komplett aus settings.json (redcandle_shadow). Keine Fallbacks, keine Defaults, keine Hardcodes."""
import json, time, traceback, psycopg2, datetime as dt
from psycopg2.extras import RealDictCursor

SETTINGS = "/opt/coin/settings.json"


def load():
    return json.load(open(SETTINGS))


def req(cfg, key):
    if key not in cfg:
        raise KeyError(f"settings.redcandle_shadow['{key}'] fehlt")
    return cfg[key]


def db(name):
    s = load()["databases"][name]
    c = psycopg2.connect(host=s["host"], port=s["port"], dbname=s["name"], user=s["user"], password=s["password"])
    c.autocommit = True
    return c


def now():
    return dt.datetime.now(dt.timezone.utc)


def scan_patterns(coins, cfg):
    """Liefert je Coin die zuletzt geschlossene Signal-Candle, falls das Muster passt."""
    pre_n = int(req(cfg, "pre_n")); pre_green_min = int(req(cfg, "pre_green_min"))
    n_red = int(req(cfg, "n_red")); min_drop = float(req(cfg, "min_drop_pct"))
    need = pre_n + n_red + 1  # Vorlauf + rote + gruene Bestaetigung
    with coins.cursor(cursor_factory=RealDictCursor) as c:
        # alle geschlossenen 5m-Candles der letzten Stunden, ein Query fuer alle Coins
        c.execute("""SELECT symbol, bucket, open, high, low, close FROM agg_5m
                     WHERE bucket + interval '5 min' <= now()
                       AND bucket > now() - interval '3 hours'
                     ORDER BY symbol, bucket""")
        rows = c.fetchall()
    by = {}
    for r in rows:
        by.setdefault(r["symbol"], []).append(r)
    signals = []
    for sym, cs in by.items():
        if len(cs) < need:
            continue
        w = cs[-need:]  # genau: [pre_n Vorlauf][n_red rot][1 gruen]
        o = [float(x["open"]) for x in w]; cl = [float(x["close"]) for x in w]
        green = [cl[i] > o[i] for i in range(len(w))]
        red = [cl[i] < o[i] for i in range(len(w))]
        # zeitliche Lueckenlosigkeit
        ts = [x["bucket"] for x in w]
        if any((ts[i+1] - ts[i]).total_seconds() != 300 for i in range(len(ts)-1)):
            continue
        gi = len(w) - 1                 # Index gruene Bestaetigung (letzte, zuletzt geschlossen)
        red_slice = range(gi - n_red, gi)     # die n_red roten
        vor_slice = range(0, gi - n_red)      # die pre_n Vorlauf-Candles
        if not green[gi]:
            continue
        if not all(red[i] for i in red_slice):
            continue
        if not green[gi - n_red - 1]:   # Candle direkt vor den roten muss gruen sein (=> genau n_red)
            continue
        if sum(1 for i in vor_slice if green[i]) < pre_green_min:
            continue
        drop = cl[gi - 1] / o[gi - n_red] - 1.0   # Close letzte rote / Open erste rote
        if drop > -min_drop:
            continue
        signals.append({
            "symbol": sym, "signal_bucket": w[gi]["bucket"], "entry_px": cl[gi],
            "drop_pct": round(drop, 4), "pre_green": sum(1 for i in vor_slice if green[i]),
        })
    return signals


def open_positions(app, coins, cfg):
    tp_pct = float(req(cfg, "tp_pct")); sl_pct = float(req(cfg, "sl_pct"))
    size = float(req(cfg, "trade_size_usd")); lev = float(req(cfg, "leverage"))
    max_open = int(req(cfg, "max_open"))
    with app.cursor() as c:
        c.execute("SELECT count(*) FROM redcandle_paper_positions WHERE status='open'")
        n_open = c.fetchone()[0]
    if n_open >= max_open:
        return 0
    signals = scan_patterns(coins, cfg)
    opened = 0
    for sg in signals:
        if n_open + opened >= max_open:
            break
        entry = sg["entry_px"]
        if entry <= 0:
            continue
        tp = entry * (1 + tp_pct); sl = entry * (1 - sl_pct)
        notional = size * lev; qty = notional / entry
        with app.cursor() as c:
            c.execute("""INSERT INTO redcandle_paper_positions
                (symbol, side, signal_bucket, drop_pct, pre_green, entry_px, tp_px, sl_px,
                 qty, margin_usd, leverage, status)
                VALUES (%s,'long',%s,%s,%s,%s,%s,%s,%s,%s,%s,'open')
                ON CONFLICT (symbol, signal_bucket) DO NOTHING""",
                (sg["symbol"], sg["signal_bucket"], sg["drop_pct"], sg["pre_green"],
                 entry, tp, sl, qty, size, lev))
            if c.rowcount == 1:
                opened += 1
                print(f"[{now():%H:%M:%S}] OPEN {sg['symbol']} entry={entry:.6g} drop={sg['drop_pct']*100:.1f}% "
                      f"gruen={sg['pre_green']} TP={tp:.6g} SL={sl:.6g}", flush=True)
    return opened


def manage_positions(app, coins, cfg):
    fee = float(req(cfg, "fee_roundtrip_pct")); timeout_h = float(req(cfg, "timeout_hours"))
    with app.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("SELECT * FROM redcandle_paper_positions WHERE status='open'")
        pos = c.fetchall()
    closed = 0
    for p in pos:
        sym = p["symbol"]; entry = float(p["entry_px"]); tp = float(p["tp_px"]); sl = float(p["sl_px"])
        # chronologische agg_1m seit Eroeffnung -> erster Touch entscheidet
        with coins.cursor() as c:
            c.execute("""SELECT high, low, close FROM agg_1m
                         WHERE symbol=%s AND bucket > %s ORDER BY bucket""", (sym, p["opened_at"]))
            bars = c.fetchall()
        exit_px = None; reason = None
        for hi, lo, close in bars:
            hi = float(hi); lo = float(lo)
            if hi >= tp:
                exit_px = tp; reason = "TP"; break
            if lo <= sl:
                exit_px = sl; reason = "SL"; break
        if exit_px is None:
            age_h = (now() - p["opened_at"]).total_seconds() / 3600.0
            if age_h >= timeout_h and bars:
                exit_px = float(bars[-1][2]); reason = "TIMEOUT"
        if exit_px is None:
            continue
        qty = float(p["qty"]); margin = float(p["margin_usd"]); notional = qty * entry
        gross = qty * (exit_px - entry)                 # long
        fee_usd = notional * fee / 100.0
        pnl_usd = gross - fee_usd
        pnl_pct = pnl_usd / margin * 100.0 if margin else 0.0
        with app.cursor() as c:
            c.execute("""UPDATE redcandle_paper_positions
                SET status='closed', closed_at=now(), exit_px=%s, pnl_usd=%s, pnl_pct=%s, close_reason=%s
                WHERE id=%s""", (exit_px, pnl_usd, pnl_pct, reason, p["id"]))
            c.execute("""UPDATE redcandle_paper_wallet_state
                SET balance=balance+%s, n_trades=n_trades+1,
                    peak_balance=GREATEST(peak_balance, balance+%s), updated_at=now()
                WHERE id=1""", (pnl_usd, pnl_usd))
        closed += 1
        print(f"[{now():%H:%M:%S}] CLOSE {sym} {reason} exit={exit_px:.6g} pnl={pnl_usd:+.2f}$", flush=True)
    return closed


def main():
    print("redcandle-shadow gestartet", flush=True)
    while True:
        try:
            cfg = load()["redcandle_shadow"]
            app = db("app"); coins = db("coins")
            try:
                c = manage_positions(app, coins, cfg)
                o = open_positions(app, coins, cfg)
                if c or o:
                    print(f"[{now():%H:%M:%S}] scan: +{o} open, {c} closed", flush=True)
            finally:
                app.close(); coins.close()
        except Exception:
            print("LOOP-ERROR:\n" + traceback.format_exc(), flush=True)
        time.sleep(int(load()["redcandle_shadow"]["scan_interval_seconds"]))


if __name__ == "__main__":
    main()
