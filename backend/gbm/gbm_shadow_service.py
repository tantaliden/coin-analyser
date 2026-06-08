#!/usr/bin/env python3
"""GBM-Predictor — Live Paper-Shadow-Service.

Regime-adaptiver Predictor, läuft PARALLEL zum alten Predictor (eigene Paper-Welt
gbm_paper_*). KEIN Echtgeld. Beweist sich gegen den alten, bevor abgelöst wird.

Pro Scan, je Coin:
  base-Modell -> P(long)  ·  gate-Modell -> P(sauberer Move)
  aktuelles Regime-Vorzeichen invertiert P(long) bei Bedarf
  Gate offen + Richtungs-Confidence hoch  -> Paper-Trade (TP/SL aus settings)
Watch: TP/SL/Timeout schließt; jeder AUFGELÖSTE Close speist den Regime-Detektor
(rollende Korrektheit des BASIS-Modells -> Vorzeichen). Bei Start aus Historie
geseedet (kein Blind-Start, kein Randomize).

Alles aus settings.gbm_predictor. KEINE Defaults, KEINE Fallbacks.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from psycopg2.extras import RealDictCursor

sys.path.insert(0, "/opt/coin/backend/gbm")
from train_eval import load_cfg, build_features, label_at

SETTINGS = "/opt/coin/settings.json"
MARKET_COINS = ["BTC", "ETH"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def req(d, k, ctx):
    if k not in d:
        raise RuntimeError(f"settings.{ctx}.{k} fehlt — KEINE Defaults")
    return d[k]


def load_all():
    s = json.load(open(SETTINGS))
    g = req(s, "gbm_predictor", "root")
    cfg = load_cfg()
    sv = req(g, "service", "gbm_predictor")
    ad = req(g, "adaptive", "gbm_predictor")
    paths = req(g, "paths", "gbm_predictor")
    return s, g, cfg, sv, ad, paths


def dbc(s, which):
    d = s["databases"][which]
    _c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"],
                          user=d["user"], password=d["password"]); _c.autocommit = True; return _c


def load_booster(path):
    b = xgb.Booster(); b.load_model(path); return b


def predict_long(b, Xrow):
    return float(b.predict(xgb.DMatrix(Xrow))[0])


# ---------- Daten-Fetch (leichtgewichtig, nur jüngste Bars) ----------
RAW_SQL = """SELECT bucket, open, high, low, close, volume, taker_buy_base,
                    quote_asset_volume, funding, open_interest, premium,
                    spread_bps, book_imbalance_5, book_depth_5
             FROM agg_1m WHERE symbol=%s AND bucket > now() - make_interval(mins => %s)
             ORDER BY bucket ASC"""


def fetch_recent(coins, symbol, minutes):
    with coins.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(RAW_SQL, (symbol, minutes))
        return cur.fetchall()


def to_df(rows):
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "bucket":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bucket"] = pd.to_datetime(df["bucket"])
    return df.reset_index(drop=True)


def recent_market(coins, cfg, minutes):
    out = None
    for sym in MARKET_COINS:
        with coins.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT bucket, close FROM agg_1m WHERE symbol=%s
                           AND bucket > now() - make_interval(mins => %s) ORDER BY bucket ASC""",
                        (sym, minutes))
            rows = cur.fetchall()
        if not rows:
            continue
        d = pd.DataFrame(rows); d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d["bucket"] = pd.to_datetime(d["bucket"]); d = d.set_index("bucket")
        m = pd.DataFrame(index=d.index); pre = sym.lower()
        for w in cfg["windows"]:
            m[f"{pre}_ret_{w}"] = (d["close"] / d["close"].shift(w) - 1.0) * 100.0
        out = m if out is None else out.join(m, how="outer")
    return out


def coin_universe(coins, cfg):
    with coins.cursor() as cur:
        cur.execute("""SELECT symbol FROM agg_1m WHERE bucket > now() - interval '24 hours'
                       GROUP BY symbol HAVING sum(quote_asset_volume) >= %s""", (cfg["min_vol"],))
        return [r[0] for r in cur.fetchall()]


def max_leverage_map(coins, symbols):
    """Echter HL-Max-Hebel je Coin aus hl_meta (Paper soll Live spiegeln)."""
    if not symbols:
        return {}
    with coins.cursor() as cur:
        cur.execute("SELECT symbol, max_leverage FROM hl_meta WHERE symbol = ANY(%s)", (list(symbols),))
        return {r[0]: int(r[1]) for r in cur.fetchall() if r[1]}


def feat_vector(coins, symbol, cfg, market, feat_cols, minutes):
    rows = fetch_recent(coins, symbol, minutes)
    if len(rows) < max(cfg["windows"]) + 121:
        return None, None
    df = to_df(rows)
    feats = build_features(df, cfg, market)
    last = feats.iloc[-1]
    if last[feat_cols].isna().any():
        return None, None
    return last[feat_cols].to_numpy(dtype=np.float32).reshape(1, -1), float(df["close"].iloc[-1])


# ---------- Regime-State ----------
def load_regime(app):
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM gbm_regime_state WHERE id=1")
        return cur.fetchone()


def set_regime(app, sign, hit, n):
    with app.cursor() as cur:
        cur.execute("""UPDATE gbm_regime_state SET current_sign=%s, hit_rate=%s, n_closed=%s,
                       updated_at=now() WHERE id=1""", (sign, hit, n))
    app.commit()


def estimate_regime(coins, cfg, base, feat_cols, lookback_min, settle_min, min_samples):
    """Dichte, NICHT gate-verzerrte Regime-Schätzung (wie der Backtest):
    Basis-Modell auf das GANZE liquide Universum, Roh-Trefferquote auf kürzlich
    AUFGELÖSTEN Setups (TP-vor-SL, vorwärts bis jetzt gescannt). >0.5 = Modell
    direkt richtig -> sign +1; <0.5 -> invertieren (-1). Liefert (sign, hit, n)
    oder (None, None, n) bei zu wenig Evidenz."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    tp = cfg["tp"] / 100.0; sl = cfg["sl"] / 100.0
    market = recent_market(coins, cfg, lookback_min + cfg["min_hist"] + max(cfg["windows"]) + 5)
    correct = 0; total = 0
    for sym in coin_universe(coins, cfg):
        rows = fetch_recent(coins, sym, lookback_min + cfg["min_hist"] + 10)
        if len(rows) < cfg["min_hist"] + 10:
            continue
        df = to_df(rows)
        feats = build_features(df, cfg, market)
        close = df["close"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy()
        bk = df["bucket"]; n = len(df)
        for i in range(cfg["min_hist"], n - 1, cfg["step"]):
            age_min = (now - bk.iloc[i].to_pydatetime()).total_seconds() / 60.0
            if age_min < settle_min:   # zu jung -> evtl. noch nicht aufgelöst
                continue
            e = close[i]
            if not np.isfinite(e) or e <= 0:
                continue
            row = feats.iloc[i]
            if row[feat_cols].isna().any():
                continue
            fh = high[i + 1:]; fl = low[i + 1:]   # vorwärts bis jetzt
            up = np.argmax(fh >= e * (1 + tp)) if (fh >= e * (1 + tp)).any() else -1
            dn = np.argmax(fl <= e * (1 - sl)) if (fl <= e * (1 - sl)).any() else -1
            long_win = up != -1 and (dn == -1 or up < dn)
            dn2 = np.argmax(fl <= e * (1 - tp)) if (fl <= e * (1 - tp)).any() else -1
            up2 = np.argmax(fh >= e * (1 + sl)) if (fh >= e * (1 + sl)).any() else -1
            short_win = dn2 != -1 and (up2 == -1 or dn2 < up2)
            if not (long_win or short_win):
                continue   # (noch) nicht aufgelöst -> raus
            pl = predict_long(base, row[feat_cols].to_numpy(dtype=np.float32).reshape(1, -1))
            correct += 1 if ((pl >= 0.5) == long_win) else 0
            total += 1
    if total < min_samples:
        return None, None, total
    hit = correct / total
    return (-1 if hit < 0.5 else 1), hit, total


# ---------- Wallet ----------
def wallet_apply(app, pnl_usd):
    with app.cursor() as cur:
        cur.execute("""UPDATE gbm_paper_wallet_state
                       SET balance=balance+%s, n_trades=n_trades+1,
                           peak_balance=GREATEST(peak_balance, balance+%s), updated_at=now()
                       WHERE id=1""", (pnl_usd, pnl_usd))
    app.commit()


# ---------- Scan (öffnen) ----------
def scan(coins, app, cfg, sv, feat_cols, gate):
    market = recent_market(coins, cfg, sv["recent_fetch_minutes"] + max(cfg["windows"]) + 5)
    if market is None:
        log("kein Market-Frame — Scan übersprungen"); return 0
    with app.cursor() as cur:
        cur.execute("SELECT count(*) FROM gbm_paper_positions WHERE status='open'")
        n_open = cur.fetchone()[0]
        cur.execute("SELECT symbol FROM gbm_paper_positions WHERE status='open'")
        held = {r[0] for r in cur.fetchall()}
        cur.execute("""SELECT symbol, max(opened_at) FROM gbm_paper_positions
                       GROUP BY symbol""")
        last_open = {r[0]: r[1] for r in cur.fetchall()}
    if n_open >= sv["max_open"]:
        return 0
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    size = float(json.load(open(SETTINGS))["gbm_predictor"]["dollar"]["trade_size_usd"])
    lev_cap = sv["leverage"]; tp, sl = cfg["tp"], cfg["sl"]
    mom_idx = feat_cols.index(f"ret_{sv['momentum_window']}")
    min_brk = float(sv["min_breakout_pct"])
    uni = coin_universe(coins, cfg)
    maxlev = max_leverage_map(coins, uni)
    opened = 0
    for sym in uni:
        if n_open + opened >= sv["max_open"]:
            break
        if sym in held:
            continue
        ml = maxlev.get(sym)
        if ml is None:
            continue  # kein Hebel aus hl_meta bekannt -> kein Trade (kein Fallback)
        eff_lev = min(lev_cap, ml)
        lo = last_open.get(sym)
        if lo and (now - lo).total_seconds() < sv["cooldown_minutes"] * 60:
            continue
        Xrow, px = feat_vector(coins, sym, cfg, market, feat_cols, sv["recent_fetch_minutes"])
        if Xrow is None or px is None or px <= 0:
            continue
        # Gate (Modell) sagt OB ein Move kommt; Momentum-Richtung sagt WOHIN (mitlaufen)
        p_res = predict_long(gate, Xrow)
        mom = float(Xrow[0, mom_idx])           # Return über momentum_window in %
        if p_res < sv["gate_threshold"] or abs(mom) < min_brk:
            continue
        side = "long" if mom >= 0 else "short"
        if side == "long":
            tp_px = px * (1 + tp / 100); sl_px = px * (1 - sl / 100)
        else:
            tp_px = px * (1 - tp / 100); sl_px = px * (1 + sl / 100)
        qty = size * eff_lev / px
        with app.cursor() as cur:
            cur.execute("""INSERT INTO gbm_paper_positions
                (symbol,side,entry_px,tp_px,sl_px,qty,margin_usd,dir_conf,p_resolve,regime_sign)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (sym, side, px, tp_px, sl_px, qty, size, min(1.0, abs(mom) / 3.0), p_res, 0))
        app.commit(); opened += 1
        log(f"OPEN {side} {sym} @ {px:.6g} mom={mom:+.2f}% pres={p_res:.3f} lev={eff_lev}")
    return opened


# ---------- Watch (schließen + Regime füttern) ----------
def watch(coins, app, cfg, ad):
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM gbm_paper_positions WHERE status='open'")
        opens = cur.fetchall()
    if not opens:
        return 0
    H = cfg["horizon_h"]; closed = 0
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    for p in opens:
        with coins.cursor() as cur:
            cur.execute("""SELECT high,low,close,bucket FROM agg_1m WHERE symbol=%s
                           AND bucket > %s ORDER BY bucket ASC""", (p["symbol"], p["opened_at"]))
            rows = cur.fetchall()
        exit_px = None; reason = None; ct = None
        for hi, lo_, cl, bk in rows:
            if hi is None or lo_ is None:
                continue
            if p["side"] == "long":
                if lo_ <= p["sl_px"]:
                    exit_px, reason, ct = p["sl_px"], "loss", bk; break
                if hi >= p["tp_px"]:
                    exit_px, reason, ct = p["tp_px"], "win", bk; break
            else:
                if hi >= p["sl_px"]:
                    exit_px, reason, ct = p["sl_px"], "loss", bk; break
                if lo_ <= p["tp_px"]:
                    exit_px, reason, ct = p["tp_px"], "win", bk; break
        if exit_px is None:
            age_h = (now - p["opened_at"]).total_seconds() / 3600.0
            if age_h >= H and rows:
                exit_px, reason, ct = float(rows[-1][2]), "timeout", rows[-1][3]
        if exit_px is None:
            continue
        # qty trägt den (coin-spezifischen) Hebel -> PnL direkt aus Notional
        entry = p["entry_px"]; notional = float(p["qty"]) * entry
        gross = (exit_px - entry) / entry if p["side"] == "long" else (entry - exit_px) / entry
        net = gross - cfg["fee"] / 100.0
        pnl_usd = notional * net
        pnl_pct = pnl_usd / p["margin_usd"] * 100.0 if p["margin_usd"] else 0.0
        with app.cursor() as cur:
            cur.execute("""UPDATE gbm_paper_positions SET status=%s,exit_px=%s,pnl_pct=%s,
                           pnl_usd=%s,closed_at=%s,close_reason=%s WHERE id=%s""",
                        (reason, exit_px, pnl_pct, pnl_usd, ct, reason, p["id"]))
        app.commit()
        wallet_apply(app, pnl_usd)
        closed += 1
        log(f"CLOSE {p['side']} {p['symbol']} {reason} pnl={pnl_pct:.2f}% ${pnl_usd:.3f}")
    if closed:
        log(f"watch: {closed} geschlossen")
    return closed


def mtimes(paths):
    return (os.path.getmtime(paths["base_model"]), os.path.getmtime(paths["gate_model"]))


def main():
    s, g, cfg, sv, ad, paths = load_all()
    log(f"GBM-Shadow start: mode=momentum+gate gate>={sv['gate_threshold']} "
        f"mom_window={sv['momentum_window']} min_breakout={sv['min_breakout_pct']}% "
        f"max_open={sv['max_open']} scan={sv['scan_interval_seconds']}s")
    base = load_booster(paths["base_model"]); gate = load_booster(paths["gate_model"])
    feat_cols = json.load(open(paths["feat_meta"]))["feat_cols"]
    mt = mtimes(paths)
    coins = dbc(s, "coins"); app = dbc(s, "app")
    last_regime = 0.0
    while True:
        t0 = time.time()
        try:
            # Settings live nachladen (UI-editierbar, kein Restart nötig)
            s, g, cfg, sv, ad, paths = load_all()
            # Modelle neu laden, wenn Re-Training sie überschrieben hat
            if mtimes(paths) != mt:
                base = load_booster(paths["base_model"]); gate = load_booster(paths["gate_model"])
                feat_cols = json.load(open(paths["feat_meta"]))["feat_cols"]
                mt = mtimes(paths); log("Modelle neu geladen (Re-Training erkannt)")
            # Optional: Regime des Basis-Modells weiter mitloggen (treibt aber NICHT mehr den Handel)
            if ad.get("regime_estimate_enabled") and time.time() - last_regime >= ad["regime_refresh_seconds"]:
                sg, hit, ntot = estimate_regime(coins, cfg, base, feat_cols,
                                                ad["regime_lookback_minutes"], ad["regime_settle_minutes"],
                                                ad["regime_min_samples"])
                if sg is not None:
                    set_regime(app, sg, hit, ntot)
                    log(f"regime-info: sign={sg} hit={hit:.3f} n={ntot} (nur Info)")
                last_regime = time.time()
            watch(coins, app, cfg, ad)
            scan(coins, app, cfg, sv, feat_cols, gate)
        except Exception as e:
            log(f"loop error: {e}")
            try:
                coins.rollback(); app.rollback()
            except Exception:
                coins = dbc(s, "coins"); app = dbc(s, "app")
        dt_ = time.time() - t0
        time.sleep(max(1, sv["scan_interval_seconds"] - dt_))


if __name__ == "__main__":
    main()
