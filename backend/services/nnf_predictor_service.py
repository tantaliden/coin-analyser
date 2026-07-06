#!/usr/bin/env python3
"""nnf-Predictor PAPER-SHADOW-Service (18.06.2026, Volker-Vision: 2 Varianten live).

Betreibt ZWEI NN-Richtungs-Predictoren parallel als Paper-Shadow, jeder im EIGENEN Buch:
  Variante A = OHLCV (57 Features)        -> nnf_a_paper_positions
  Variante B = +HL/Funding (63 Features)  -> nnf_b_paper_positions
Pro Coin: Live-Features (EXAKT wie Training, via train_direction_hist/_staged) -> NN-Richtung+conf.
Bei conf>=min_confidence: Reach-TP/SL pro Coin aus Historie (peak_share*Median-Peak / Quantil-SL,
Messfenster = timeout_hours) -> Paper-Trade. Schliessen per TP/SL-First-Crossing (10s-klines) /
Timeout. ONLINE-LERNEND: nach Close (feat-Snapshot + post-hoc Reach-Label) periodisch nachtrainiert,
mit Replay-Anker gegen Direction-Collapse. KEIN Echtgeld. Alles aus settings.nnf_predictor.
"""
import json, time, sys, datetime as dt
import numpy as np
import torch
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, "/opt/training/scripts")
import train_direction_hist as H
import train_direction_staged as S
import train_direction_wf_reach as R
import test_ext_data as TE

SETTINGS = "/opt/coin/settings.json"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def dbc(s):
    d = s["databases"]["coins"]; c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"], cursor_factory=RealDictCursor); c.autocommit = True; return c
def dba(s):
    d = s["databases"]["app"]; c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"]); c.autocommit = True; return c

def ddl_for(table):
    return f"""
CREATE TABLE IF NOT EXISTS {table}(
  id BIGSERIAL PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
  entry_px DOUBLE PRECISION NOT NULL, tp_px DOUBLE PRECISION NOT NULL, sl_px DOUBLE PRECISION NOT NULL,
  tp_pct DOUBLE PRECISION, sl_pct DOUBLE PRECISION, conf DOUBLE PRECISION NOT NULL, leverage INT,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT now(), status TEXT NOT NULL DEFAULT 'open',
  exit_px DOUBLE PRECISION, pnl_pct DOUBLE PRECISION, pnl_usd DOUBLE PRECISION,
  closed_at TIMESTAMPTZ, exit_reason TEXT, learned BOOLEAN DEFAULT FALSE,
  feat DOUBLE PRECISION[], tp_l DOUBLE PRECISION, sl_l DOUBLE PRECISION, tp_s DOUBLE PRECISION, sl_s DOUBLE PRECISION);
CREATE INDEX IF NOT EXISTS {table}_status ON {table}(status);
"""

def load_variant(v):
    """Modell + Replay-Anker laden. Returns dict mit net, mu, sd, n_features, replay, buffer."""
    ck = torch.load(v["model_path"], map_location="cpu", weights_only=False)
    net = H.Net(int(ck["n_features"]), ck["hidden"], H.DROPOUT)
    net.load_state_dict(ck["state"]); net.eval()
    rp = np.load(v["replay_path"])
    return {"net": net, "mu": ck["mu"].astype(np.float32), "sd": ck["sd"].astype(np.float32),
            "n_features": int(ck["n_features"]), "use_hl": bool(v["use_hl"]), "use_ext": bool(v.get("use_ext", False)),
            "table": v["table"], "label": v["label"], "model_path": v["model_path"],
            "replay_X": rp["X"].astype(np.float32), "replay_Y": rp["Y"].astype(np.int64),
            "buf_X": [], "buf_Y": [], "closes_since_train": 0}

def infer(net, x_norm):
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(x_norm[None, :], dtype=torch.float32)), 1).numpy()[0]
    return int(p.argmax()), float(p.max())

def features_live(md, hd, ext, sym, now_n, var):
    """EXAKT wie Training: 57 OHLCV [+ 6 HL (B)] [+ 7 ext (C)]. None wenn HL fehlt (kein Fuellen)."""
    fa = H.features_at(md, sym, now_n)
    if var["use_ext"]:
        liq, oi, fund, dvol = ext
        return np.concatenate([fa, TE.ext_features(liq, oi, fund, dvol, sym, now_n)])
    if var["use_hl"]:
        hl = S.hl_features_at(hd, sym, now_n)
        if hl is None:
            return None
        return np.concatenate([fa, hl])
    return fa

def check_close(coins, pos, timeout_h):
    cur = coins.cursor()
    cur.execute("""SELECT high, low, close, open_time FROM klines WHERE symbol=%s AND interval='10s'
                   AND open_time > %s ORDER BY open_time""", (pos['symbol'], pos['opened_at']))
    rows = cur.fetchall()
    if not rows: return None
    for r in rows:
        h = r['high']; l = r['low']; ot = r['open_time']
        if h is None or l is None: continue
        if pos['side'] == 'long':
            if l <= pos['sl_px']: return pos['sl_px'], 'SL', ot
            if h >= pos['tp_px']: return pos['tp_px'], 'TP', ot
        else:
            if h >= pos['sl_px']: return pos['sl_px'], 'SL', ot
            if l <= pos['tp_px']: return pos['tp_px'], 'TP', ot
    age_h = (rows[-1]['open_time'] - pos['opened_at']).total_seconds() / 3600
    if age_h >= timeout_h:
        return float(rows[-1]['close']), 'TIMEOUT', rows[-1]['open_time']
    return None

def reach_label(coins, pos):
    """Post-hoc Reach-Label aus realisierten 10s-klines: welche Seite ihr TP VOR SL traf.
    Nutzt die beim Open gespeicherten beidseitigen TP/SL-%. -> 1 long / 2 short / 0 skip."""
    e = pos['entry_px']; tl, sl_, ts, ss = pos['tp_l'], pos['sl_l'], pos['tp_s'], pos['sl_s']
    if None in (tl, sl_, ts, ss) or e <= 0: return None
    tp_l = e*(1+tl/100); sll = e*(1-sl_/100); tp_s = e*(1-ts/100); sls = e*(1+ss/100)
    cur = coins.cursor()
    cur.execute("""SELECT high, low FROM klines WHERE symbol=%s AND interval='10s'
                   AND open_time > %s AND open_time <= %s ORDER BY open_time""",
                (pos['symbol'], pos['opened_at'], pos['closed_at']))
    long_hit = short_hit = None
    for r in cur.fetchall():
        h = r['high']; l = r['low']
        if h is None or l is None: continue
        if long_hit is None:
            if l <= sll: long_hit = False
            elif h >= tp_l: long_hit = True
        if short_hit is None:
            if h >= sls: short_hit = False
            elif l <= tp_s: short_hit = True
        if long_hit is not None and short_hit is not None: break
    lh = long_hit is True; sh = short_hit is True
    return 1 if (lh and not sh) else 2 if (sh and not lh) else 0

def online_train(var, L, dev):
    """Nach genug Closes: wenige Gradientenschritte ueber neue Samples + Replay-Anker."""
    if len(var["buf_Y"]) < L["min_new_samples"]: return False
    import torch.nn as nn, torch.optim as optim
    # WICHTIG: neue Samples GENAUSO normalisieren wie die Replay-Anker (die sind bereits
    # (x-mu)/sd). Sonst mischen sich rohe + normalisierte Skalen -> Netz kollabiert (Bug 19.06.).
    newX = (np.array(var["buf_X"], np.float32) - var["mu"]) / var["sd"]; newY = np.array(var["buf_Y"], np.int64)
    rb = min(int(L["replay_batch"]), len(var["replay_Y"]))
    ri = np.random.RandomState().choice(len(var["replay_Y"]), rb, replace=False)
    X = np.concatenate([newX, var["replay_X"][ri]]); Y = np.concatenate([newY, var["replay_Y"][ri]])
    from collections import Counter
    cnt = Counter(Y.tolist()); tot = sum(cnt.values())
    cw = torch.tensor([tot/(3*max(cnt.get(i,1),1)) for i in range(3)], dtype=torch.float32)
    net = var["net"].to(dev); net.train()
    opt = optim.Adam(net.parameters(), lr=float(L["lr"]), weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=cw.to(dev))
    Xt = torch.tensor(X).to(dev); Yt = torch.tensor(Y).to(dev)
    for _ in range(int(L["epochs"])):
        opt.zero_grad(); crit(net(Xt), Yt).backward(); opt.step()
    net.eval()
    torch.save({'state': net.state_dict(), 'mu': var["mu"], 'sd': var["sd"],
                'hidden': H.HIDDEN, 'n_features': var["n_features"]}, var["model_path"])
    n = len(var["buf_Y"]); var["buf_X"].clear(); var["buf_Y"].clear(); var["closes_since_train"] = 0
    log(f"  [{var['label']}] online-trained on {n} new + {rb} replay -> saved")
    return True

def main():
    s = json.load(open(SETTINGS)); C = s["nnf_predictor"]
    LV = C["live"]; RC = C["reach"]; L = C["learn"]
    timeout_h = int(LV["timeout_hours"]); fee = float(LV["fee_roundtrip_pct"])
    size = float(LV["trade_size_usd"]); cap = int(LV["leverage_cap"])
    max_open = int(LV["max_open"]); top_n = int(LV["top_n_coins"])
    min_conf = float(LV["min_confidence"]); cooldown = int(LV["cooldown_seconds_per_symbol"])
    scan_s = int(LV["scan_interval_seconds"])
    lookback_days = int(RC["lookback_days"]); OBS = C["observer"]
    # Reach-TP/SL-Fenster an nnf-timeout koppeln (statt reach_model): Konsistenz Live<->Modell
    R.FWD_N = timeout_h * 12
    R.PEAK_SHARE = float(RC["peak_share"]); R.SL_Q = float(RC["sl_survival_quantile"])
    R.MIN_TP = float(RC["min_tp_pct"]); R.MAX_TP = float(RC["max_tp_pct"])
    R.TP_STEP_BARS = max(1, int(RC["anchor_step_minutes"]) // 5)
    dev = "cpu"
    variants = [load_variant(C["variants"][k]) for k in C["variants"].keys()]
    need_ext = any(v["use_ext"] for v in variants)
    log(f"NNF-SHADOW start: {len(variants)} Varianten {[v['label'] for v in variants]} | "
        f"scan {scan_s}s top{top_n} max_open={max_open} conf>={min_conf} size ${size} lev<={cap}x timeout {timeout_h}h")
    with dba(s) as app:
        with app.cursor() as cur:
            for v in variants: cur.execute(ddl_for(v["table"]))

    while True:
        t0 = time.time()
        try:
            with dbc(s) as coins, dba(s) as app:
                now = dt.datetime.now(dt.timezone.utc); now_n = dt.datetime.now()
                uni = _top(coins, top_n)
                # Live-Daten EINMAL laden (alle Coins), Feature-/TP-SL-Quelle wie Training
                start = now_n - dt.timedelta(days=lookback_days + 1)
                md = S.preload_market_src(coins, uni, start, now_n,
                        [('agg_5m','agg_5m'),('agg_1h','agg_1h'),('agg_4h','agg_4h'),('agg_1d','agg_1d')])
                md['km'] = _load_km(coins, uni)
                kl = S.preload_1m_src(coins, uni, start, now_n, 'agg_5m')
                hd = S.preload_hl(coins, uni, start, now_n)
                ext = TE.load_ext(coins, uni, start, now_n) if need_ext else (None, None, None, None)
                tpsl = R.coin_tp_sl(kl, uni, now_n + dt.timedelta(minutes=1))
                lev_cache = _lev(coins, uni, cap)
                btc_shift = _btc_shift(md, int(OBS["btc_shift_minutes"]))
                for var in variants:
                    _observer_pass(coins, app, var, md, hd, ext, tpsl, btc_shift, now, now_n, min_conf, fee, size, OBS)
                    _close_pass(coins, app, var, timeout_h, fee, size, L, dev)
                    _open_pass(coins, app, var, uni, md, hd, ext, tpsl, lev_cache, now, now_n,
                               max_open, min_conf, cooldown, size)
        except Exception as e:
            log(f"loop error: {e}")
        time.sleep(max(1, scan_s - (time.time() - t0)))

def _load_km(coins, uni):
    kmc = ('pct_30m','pct_60m','pct_90m','pct_120m','pct_180m','pct_240m','pct_300m','pct_360m','pct_420m','pct_480m','pct_540m','pct_600m')
    cur = coins.cursor()
    cur.execute("SELECT DISTINCT ON (symbol) symbol, " + ", ".join(kmc) + " FROM kline_metrics WHERE symbol=ANY(%s) ORDER BY symbol, open_time DESC", (uni,))
    out = {}
    for r in cur.fetchall():
        out[r['symbol']] = {c: (float(r[c]) if r[c] is not None else 0.0) for c in kmc}
    return out

def _top(coins, n):
    cur = coins.cursor()
    cur.execute("""SELECT symbol, SUM(quote_asset_volume) v FROM agg_1m WHERE bucket > now()-interval '24 hours'
                   GROUP BY symbol ORDER BY v DESC NULLS LAST LIMIT %s""", (n,))
    return [r['symbol'] for r in cur.fetchall()]

def _lev(coins, uni, cap):
    cur = coins.cursor(); cur.execute("SELECT symbol,max_leverage FROM hl_meta WHERE symbol=ANY(%s)", (uni,))
    return {r['symbol']: min(cap, int(r['max_leverage'])) for r in cur.fetchall() if r['max_leverage'] is not None}

def _btc_shift(md, minutes):
    """BTC-Bewegung (%) ueber das Fenster (agg_5m). + = hoch."""
    d = md.get('agg_5m', {}).get('BTC')
    if not d or len(d.get('close', [])) < 2: return 0.0
    n = max(1, minutes // 5); cl = d['close']
    if len(cl) <= n: return 0.0
    return (cl[-1] / cl[-1 - n] - 1) * 100.0

def _observer_pass(coins, app, var, md, hd, ext, tpsl, btc_shift, now, now_n, min_conf, fee, size, OBS):
    """Observer-Schutz pro Scan: (c) Massen-Minus-Notausstieg, (b) BTC-Shift-Trigger,
    (a) Trailing-SL nachziehen, + Reversal-Close bei Richtungswechsel."""
    tbl = var["table"]; net, mu, sd = var["net"], var["mu"], var["sd"]
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT * FROM {tbl} WHERE status='open'")
        opens = cur.fetchall()
    if not opens: return
    prices = {}
    with coins.cursor() as cc:
        for sym in {p['symbol'] for p in opens}:
            cc.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
            r = cc.fetchone()
            if r and r['close'] is not None: prices[sym] = float(r['close'])
    def cmove(p):
        px = prices.get(p['symbol'])
        if not px: return None, None
        cm = (px/p['entry_px']-1)*100 if p['side'] == 'long' else (1-px/p['entry_px'])*100
        return cm, px
    def do_close(p, px, reason):
        lev = p['leverage'] or 1
        mv = (px/p['entry_px']-1)*100 if p['side'] == 'long' else (1-px/p['entry_px'])*100
        pnl = (mv - fee) * lev; usd = size * pnl / 100
        with app.cursor() as cu:
            cu.execute(f"UPDATE {tbl} SET status='closed',exit_px=%s,pnl_pct=%s,pnl_usd=%s,closed_at=%s,exit_reason=%s,learned=TRUE WHERE id=%s",
                       (px, pnl, usd, now, reason, p['id']))
        p['closed_at'] = now
        if p.get('feat'):
            y = reach_label(coins, p)
            if y is not None: var["buf_X"].append(np.array(p['feat'], np.float32)); var["buf_Y"].append(y)
        var["closes_since_train"] += 1

    # (c) Massen-Minus-Notausstieg: >= X% offen im Minus UND avg Coin-Verlust >= Schwelle (vor Hebel)
    minus = []
    for p in opens:
        cm, px = cmove(p)
        if cm is not None and cm < 0: minus.append((p, cm, px))
    if minus and len(minus) / len(opens) >= float(OBS["mass_exit_minus_fraction"]):
        avg_adv = sum(-cm for _, cm, _ in minus) / len(minus)
        if avg_adv >= float(OBS["mass_exit_avg_adverse_pct"]):
            for p, cm, px in minus: do_close(p, px, 'MASS_EXIT')
            log(f"  [{var['label']}] MASS-EXIT {len(minus)}/{len(opens)} Minus-Pos (avg {avg_adv:.2f}% Coin)")
            return

    trailing = bool(OBS.get("trailing_enabled"))
    for p in opens:
        cm, px = cmove(p)
        if px is None: continue
        # Reversal: Modell jetzt Gegenrichtung mit conf
        feat = features_live(md, hd, ext, p['symbol'], now_n, var)
        if feat is not None and len(feat) == var["n_features"]:
            pred, conf = infer(net, (feat - mu) / sd)
            cs = 'long' if pred == 1 else 'short' if pred == 2 else None
            if cs and cs != p['side'] and conf >= min_conf:
                do_close(p, px, 'REVERSE'); log(f"  [{var['label']}] REVERSE {p['side']}->{cs} {p['symbol']} conf={conf:.2f}"); continue
        # (b) BTC-Shift gegen die Position
        adverse = btc_shift if p['side'] == 'short' else -btc_shift
        if adverse >= float(OBS["btc_shift_pct"]):
            do_close(p, px, 'BTC_SHIFT'); log(f"  [{var['label']}] BTC-SHIFT {p['side']} {p['symbol']} (BTC {btc_shift:+.2f}%)"); continue
        # (a) Trailing-SL: Reach-SL relativ zum AKTUELLEN Preis, nur nachziehen (nie lockern)
        if trailing and px:
            ts = tpsl.get(p['symbol'])
            if ts:
                sl_pct = ts[p['side']][1]
                if p['side'] == 'long':
                    cand = px * (1 - sl_pct/100)
                    if cand > float(p['sl_px']):
                        with app.cursor() as cu: cu.execute(f"UPDATE {tbl} SET sl_px=%s WHERE id=%s", (cand, p['id']))
                else:
                    cand = px * (1 + sl_pct/100)
                    if cand < float(p['sl_px']):
                        with app.cursor() as cu: cu.execute(f"UPDATE {tbl} SET sl_px=%s WHERE id=%s", (cand, p['id']))

def _close_pass(coins, app, var, timeout_h, fee, size, L, dev):
    with app.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT * FROM {var['table']} WHERE status='open'")
        opens = cur.fetchall()
    for pos in opens:
        r = check_close(coins, pos, timeout_h)
        if r is None: continue
        px, reason, ct = r; lev = pos['leverage'] or 1
        move = (px/pos['entry_px']-1)*100 if pos['side'] == 'long' else (1-px/pos['entry_px'])*100
        pnl = (move - fee) * lev; usd = size * pnl / 100
        with app.cursor() as cu:
            cu.execute(f"""UPDATE {var['table']} SET status='closed',exit_px=%s,pnl_pct=%s,pnl_usd=%s,
                          closed_at=%s,exit_reason=%s WHERE id=%s""", (px, pnl, usd, ct, reason, pos['id']))
        # Online-Learning: post-hoc Reach-Label + feat -> Buffer
        pos['closed_at'] = ct
        if pos.get('feat'):
            y = reach_label(coins, pos)
            if y is not None:
                var["buf_X"].append(np.array(pos['feat'], np.float32)); var["buf_Y"].append(y)
        var["closes_since_train"] += 1
        with app.cursor() as cu:
            cu.execute(f"UPDATE {var['table']} SET learned=TRUE WHERE id=%s", (pos['id'],))
    if var["closes_since_train"] >= int(L["retrain_after_closes"]):
        online_train(var, L, dev)

def _open_pass(coins, app, var, uni, md, hd, ext, tpsl, lev_cache, now, now_n, max_open, min_conf, cooldown, size):
    with app.cursor() as cur:
        cur.execute(f"SELECT symbol FROM {var['table']} WHERE status='open'")
        held = {r[0] for r in cur.fetchall()}
        cur.execute(f"SELECT symbol,MAX(closed_at) FROM {var['table']} WHERE closed_at IS NOT NULL GROUP BY symbol")
        lastclose = {r[0]: r[1] for r in cur.fetchall()}
    free = max_open - len(held)
    if free <= 0: return
    net, mu, sd = var["net"], var["mu"], var["sd"]
    opened = 0
    for sym in uni:
        if opened >= free: break
        if sym in held or sym not in tpsl or sym not in lev_cache: continue
        lc = lastclose.get(sym)
        if lc and (now - lc).total_seconds() < cooldown: continue
        feat = features_live(md, hd, ext, sym, now_n, var)
        if feat is None or len(feat) != var["n_features"]: continue
        pred, conf = infer(net, (feat - mu) / sd)
        if pred == 0 or conf < min_conf: continue
        side = 'long' if pred == 1 else 'short'
        tp_l, sl_l = tpsl[sym]['long']; tp_s, sl_s = tpsl[sym]['short']
        tp_pct, sl_pct = (tp_l, sl_l) if side == 'long' else (tp_s, sl_s)
        with coins.cursor() as cc:
            cc.execute("SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT 1", (sym,))
            p = cc.fetchone()
        if not p or p['close'] is None: continue
        entry = float(p['close']); lev = lev_cache[sym]
        if side == 'long': tp = entry*(1+tp_pct/100); slx = entry*(1-sl_pct/100)
        else:              tp = entry*(1-tp_pct/100); slx = entry*(1+sl_pct/100)
        with app.cursor() as cu:
            cu.execute(f"""INSERT INTO {var['table']}
                (symbol,side,entry_px,tp_px,sl_px,tp_pct,sl_pct,conf,leverage,feat,tp_l,sl_l,tp_s,sl_s)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (sym, side, entry, tp, slx, tp_pct, sl_pct, conf, lev,
                 [float(x) for x in feat], tp_l, sl_l, tp_s, sl_s))
        opened += 1
        log(f"  [{var['label']}] OPEN {side} {sym} @ {entry:.6f} conf={conf:.2f} {lev}x (tp+{tp_pct:.2f}/sl-{sl_pct:.2f})")

if __name__ == "__main__":
    main()
