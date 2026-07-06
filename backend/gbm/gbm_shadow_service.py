#!/usr/bin/env python3
"""GBM-Predictor — Cross-Sectional Composite Paper-Shadow (marktneutral).

Loest die alte Momentum+Gate-Logik ab (5 Tage -$106, Muenzwurf-Edge, tot).
Kernbefund der GBM-Saga: kurzfristige RICHTUNG ist nicht vorhersagbar (4x belegt,
jedes Feature-Modell wird OOS schlechter als eine triviale Regel). Coins ziehen
fast immer mit BTC mit (Kopplung bricht an 0% der Tage) -> Richtung = unvorhersag-
bare BTC-Wette. Der EINZIGE robuste Edge (t>2, beide Haelften +): markt-NEUTRALER
Cross-Sectional-Spread auf einem MULTI-HORIZON-KOMPOSIT.

Volkers Prinzip "alle sinnvollen Werte, KEIN Rauspicken" operationalisiert:
  Signal je Coin = gleichgewichteter Mittelwert der cross-sectional z-Scores
  ueber ALLE Horizonte (lookback_hours_list) UND beide Komponenten:
    - ROH-Momentum     (log-Return ueber w)         -> Co-Movement-Persistenz
    - RESIDUAL-Momentum (BTC-Beta kausal rausgerechnet) -> idiosynkratische Staerke
  Kein Modell splittet/picks; jeder Horizont + jede Komponente traegt gleich bei.

Pro Rebalance (alle rebalance_hours): offenes Buch schliessen (PnL realisieren),
liquides Universum nach Komposit ranken, Top-k LONG + Bottom-k SHORT (gleiche
Dollar je Bein -> dollar-neutral). KEIN Echtgeld, eigene Paper-Welt (gbm_paper_*).

Alles aus settings.gbm_predictor.xsection. KEINE Defaults, KEINE Fallbacks.
"""
import json
import time
import datetime as dt
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

SETTINGS = "/opt/coin/settings.json"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def req(d, k, ctx):
    if k not in d:
        raise RuntimeError(f"settings.{ctx}.{k} fehlt — KEINE Defaults")
    return d[k]


def load_cfg():
    s = json.load(open(SETTINGS))
    g = req(s, "gbm_predictor", "root")
    x = req(g, "xsection", "gbm_predictor")
    cfg = dict(
        market=str(req(x, "market_coin", "xsection")),
        min_vol=float(req(x, "min_dollar_volume_24h", "xsection")),
        top_n=int(req(x, "universe_top_n", "xsection")),
        windows=[int(w) for w in req(x, "lookback_hours_list", "xsection")],
        beta_win=int(req(x, "beta_window_hours", "xsection")),
        w_roh=float(req(x, "weight_roh", "xsection")),
        w_res=float(req(x, "weight_residual", "xsection")),
        k=int(req(x, "k", "xsection")),
        rebal_h=float(req(x, "rebalance_hours", "xsection")),
        size=float(req(x, "trade_size_usd", "xsection")),
        lev=int(req(x, "leverage", "xsection")),
        fee=float(req(x, "fee_roundtrip_pct", "xsection")),
        scan=int(req(x, "scan_interval_seconds", "xsection")),
    )
    return s, cfg


def dbc(s, which):
    d = s["databases"][which]
    c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"],
                         user=d["user"], password=d["password"])
    c.autocommit = True
    return c


# ---------- Universum + Daten ----------
def top_universe(coins, cfg):
    with coins.cursor() as cur:
        cur.execute("""SELECT symbol, sum(quote_asset_volume) v FROM agg_1m
                       WHERE bucket>now()-interval '24 hours'
                       GROUP BY symbol HAVING sum(quote_asset_volume)>=%s
                       ORDER BY v DESC LIMIT %s""", (cfg["min_vol"], cfg["top_n"]))
        return [r[0] for r in cur.fetchall()]


def hourly_close(coins, syms, hours):
    """Stunden-Close-Panel (Zeilen=Stunden, Spalten=Coins), gemeinsame Achse."""
    out = {}
    with coins.cursor() as cur:
        for sym in syms:
            cur.execute("""SELECT bucket, close FROM agg_1h WHERE symbol=%s
                           AND bucket > now() - make_interval(hours => %s)
                           ORDER BY bucket ASC""", (sym, hours))
            rows = cur.fetchall()
            if len(rows) < hours // 2:
                continue
            out[sym] = pd.Series({pd.Timestamp(b): float(p) for b, p in rows if p is not None})
    if not out:
        return None
    px = pd.DataFrame(out).sort_index().ffill().dropna()
    return px


def price_now(coins, sym):
    with coins.cursor() as cur:
        cur.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
        r = cur.fetchone()
    return float(r[0]) if r and r[0] and float(r[0]) > 0 else None


# ---------- Komposit-Signal ----------
def zscore_last(M):
    """z-Score der LETZTEN Zeile ueber die Coins (Spalten)."""
    row = M.iloc[-1]
    mu = row.mean(); sd = row.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return (row - mu) / sd


def composite_scores(px, cfg):
    """Score je Coin = gleichgewichteter Mittelwert der cross-sectional z-Scores
    ueber ALLE Horizonte (roh + residual). Gibt dict sym->score (nur Alts)."""
    market = cfg["market"]
    if market not in px.columns:
        return None
    ret = np.log(px / px.shift(1))
    alts = [c for c in px.columns if c != market]
    if len(alts) < 2 * cfg["k"]:
        return None
    btc = ret[market]
    # kausale Residual-Returns (Beta aus Vergangenheit)
    res = pd.DataFrame(index=ret.index)
    for sym in alts:
        cov = ret[sym].rolling(cfg["beta_win"]).cov(btc).shift(1)
        var = btc.rolling(cfg["beta_win"]).var().shift(1)
        beta = (cov / var).clip(-5, 5)
        mb = btc.rolling(cfg["beta_win"]).mean().shift(1)
        my = ret[sym].rolling(cfg["beta_win"]).mean().shift(1)
        res[sym] = ret[sym] - ((my - beta * mb) + beta * btc)
    comps = []
    weights = []
    for w in cfg["windows"]:
        roh = np.log(px[alts] / px[alts].shift(w))          # roh-Momentum ueber w
        zr = zscore_last(roh)
        if zr is not None and cfg["w_roh"] > 0:
            comps.append(zr); weights.append(cfg["w_roh"])
        rm = res[alts].rolling(w).sum()                     # residual-Momentum ueber w
        zm = zscore_last(rm)
        if zm is not None and cfg["w_res"] > 0:
            comps.append(zm); weights.append(cfg["w_res"])
    if not comps:
        return None
    stacked = pd.concat(comps, axis=1)
    wv = np.array(weights, dtype=float)
    # RULE_6: keine silent 0-Fuellung. Coins mit fehlender Historie (NaN in irgend-
    # einer Komponente) werden ausgeschlossen, nicht neutral gerechnet.
    valid = stacked.notna().all(axis=1)
    dropped = [c for c in stacked.index if not valid[c]]
    if dropped:
        log(f"FALLBACK_TRIGGERED composite_scores: {dropped} unvollstaendige Historie -> ausgeschlossen")
    stacked = stacked[valid]
    if len(stacked) < 2 * cfg["k"]:
        return None
    score = (stacked.to_numpy() * wv).sum(axis=1) / wv.sum()
    return dict(zip(stacked.index, score))


# ---------- Paper-Buch ----------
def close_book(coins, app, cfg):
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM gbm_paper_positions WHERE status='open'")
        opens = cur.fetchall()
    if not opens:
        return 0
    total = 0.0
    for p in opens:
        px = price_now(coins, p["symbol"])
        if px is None:
            continue
        notional = float(p["qty"]) * p["entry_px"]
        gross = (px - p["entry_px"]) / p["entry_px"] if p["side"] == "long" else (p["entry_px"] - px) / p["entry_px"]
        net = gross - cfg["fee"] / 100.0
        pnl = notional * net
        pnl_pct = pnl / p["margin_usd"] * 100.0 if p["margin_usd"] else 0.0
        total += pnl
        st = "win" if pnl >= 0 else "loss"      # Bein-Trefferquote fuer die GBM-Kachel
        with app.cursor() as c2:
            c2.execute("""UPDATE gbm_paper_positions SET status=%s, exit_px=%s,
                          pnl_usd=%s, pnl_pct=%s, closed_at=now(), close_reason='rebalance'
                          WHERE id=%s""", (st, px, pnl, pnl_pct, p["id"]))
    with app.cursor() as c2:
        c2.execute("""UPDATE gbm_paper_wallet_state SET balance=balance+%s,
                      n_trades=n_trades+%s,
                      peak_balance=GREATEST(peak_balance, balance+%s), updated_at=now()
                      WHERE id=1""", (total, len(opens), total))
    log(f"REBALANCE close: {len(opens)} Positionen, realisiert ${total:.3f}")
    return len(opens)


def open_book(coins, app, cfg, scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    shorts = [s for s, _ in ranked[:cfg["k"]]]
    longs = [s for s, _ in ranked[-cfg["k"]:]]
    opened = 0
    for side, syms in [("long", longs), ("short", shorts)]:
        for sym in syms:
            px = price_now(coins, sym)
            if px is None:
                continue
            qty = cfg["size"] * cfg["lev"] / px
            with app.cursor() as c2:
                c2.execute("""INSERT INTO gbm_paper_positions
                    (symbol, side, entry_px, qty, margin_usd, dir_conf, regime_sign, status)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,'open')""",
                    (sym, side, px, qty, cfg["size"], float(scores[sym]), 0))
            opened += 1
    log(f"REBALANCE open: LONG {longs} | SHORT {shorts}")
    return opened


def last_rebalance(app):
    with app.cursor() as cur:
        cur.execute("SELECT max(opened_at) FROM gbm_paper_positions")
        r = cur.fetchone()
    return r[0] if r else None


def main():
    s, cfg = load_cfg()
    log(f"GBM Cross-Sectional Composite start: Top-{cfg['top_n']} K={cfg['k']} "
        f"windows={cfg['windows']}h roh={cfg['w_roh']} res={cfg['w_res']} "
        f"rebalance={cfg['rebal_h']}h lev={cfg['lev']}")
    coins = dbc(s, "coins"); app = dbc(s, "app")
    need_hours = max(cfg["windows"]) + cfg["beta_win"] + 12
    while True:
        t0 = time.time()
        try:
            s, cfg = load_cfg()
            need_hours = max(cfg["windows"]) + cfg["beta_win"] + 12
            lr = last_rebalance(app)
            now = dt.datetime.now(dt.timezone.utc)
            due = lr is None or (now - lr).total_seconds() >= cfg["rebal_h"] * 3600
            if due:
                uni = top_universe(coins, cfg)
                px = hourly_close(coins, uni, need_hours)
                if px is None or len([c for c in px.columns if c != cfg["market"]]) < 2 * cfg["k"]:
                    log("zu wenig Daten/Coins — Rebalance verschoben")
                else:
                    scores = composite_scores(px, cfg)
                    if scores is None:
                        log("Komposit nicht berechenbar — Rebalance verschoben")
                    else:
                        close_book(coins, app, cfg)
                        open_book(coins, app, cfg, scores)
        except Exception as e:
            log(f"loop error: {e}")
            try:
                coins.rollback(); app.rollback()
            except Exception:
                coins = dbc(s, "coins"); app = dbc(s, "app")
        time.sleep(max(5, cfg["scan"] - (time.time() - t0)))


if __name__ == "__main__":
    main()
