"""
Paper-Trading-Engine (Variante B, Volker 20.05.2026)
=====================================================
Pseudo-Wallet (paper_wallet_state) + virtuelle Positionen (paper_positions),
die mit den GLEICHEN Regeln wie HL befüllt/verkauft werden:
  - Slot-Limit = multi_head.max_open_predictions
  - Trade-Size = trading.default_size_usd (Volker bestimmt im Modal)
  - Hebel     = trading.default_leverage, gecappt gegen hl_meta.max_leverage
  - Close     = TP / SL / Timeout (get_timeout_hours), exakt wie der echte Trader

Welt-Trennung (CARDINAL RULE_1): eigene Tabellen, KEINE Berührung von
trader_positions / open_predictions ausser Lese-Referenz prediction_id.
"""
import logging

log = logging.getLogger("predictor.paper")


def _max_leverage(coins_conn, symbol):
    with coins_conn.cursor() as cur:
        cur.execute("SELECT max_leverage FROM hl_meta WHERE symbol=%s", (symbol,))
        r = cur.fetchone()
    return int(r[0]) if r and r[0] else None


def paper_count_open(app_conn):
    with app_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM paper_positions WHERE status='open'")
        return cur.fetchone()[0]


def paper_open(app_conn, coins_conn, pid, symbol, side, entry_px, tp_px, sl_px, cfg):
    """Öffnet eine virtuelle Paper-Position. entry_px = aktueller (frischer) Preis.
    tp_px/sl_px = Predictor-Werte (absolut, wie use_predictor_targets=True)."""
    t = cfg["trading"]
    margin = t.get("default_size_usd")
    lev_want = t.get("default_leverage")
    if margin is None or lev_want is None:
        log.error("FALLBACK_TRIGGERED paper_open: default_size_usd/default_leverage fehlen -> skip")
        return None
    margin = float(margin); lev_want = int(lev_want)
    max_lev = _max_leverage(coins_conn, symbol) or lev_want
    lev = max(1, min(lev_want, max_lev))
    notional = margin * lev
    if entry_px <= 0:
        return None
    qty = notional / entry_px
    with app_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO paper_positions
              (prediction_id, symbol, side, entry_px, qty, leverage, margin_usd,
               tp_px, sl_px, status, timeout_enabled)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'open',TRUE)
            RETURNING id
        """, (pid, symbol, side, entry_px, qty, lev, margin, tp_px, sl_px))
        ppid = cur.fetchone()[0]
    app_conn.commit()
    log.info("PAPER-OPEN %s %s ppid=%s entry=%.6g qty=%.6g lev=%dx margin=$%.0f tp=%.6g sl=%.6g",
             symbol, side, ppid, entry_px, qty, lev, margin, tp_px, sl_px)
    return ppid


def _wallet_apply(app_conn, pnl_usd):
    with app_conn.cursor() as cur:
        cur.execute("""
            UPDATE paper_wallet_state
            SET balance = balance + %s,
                peak_balance = GREATEST(peak_balance, balance + %s),
                n_trades = n_trades + 1,
                updated_at = now()
            WHERE id=1
            RETURNING balance
        """, (pnl_usd, pnl_usd))
        r = cur.fetchone()
    app_conn.commit()
    return float(r[0]) if r else None


def paper_watch(app_conn, coins_conn, cfg, timeout_h):
    """Prüft offene paper_positions via Klines auf TP/SL/Timeout, schließt,
    bucht PnL in paper_wallet_state. Gibt Anzahl Closes zurück."""
    slip_pct = float(cfg.get("paper_wallet", {}).get("slippage_pct_per_trade") or 0.0)
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT id, symbol, side, entry_px, qty, leverage, margin_usd, tp_px, sl_px,
                   opened_at, timeout_enabled,
                   EXTRACT(EPOCH FROM (now() - opened_at))/3600.0 AS age_h
            FROM paper_positions WHERE status='open'
        """)
        opens = cur.fetchall()
    if not opens:
        return 0

    symbols = list({o[1] for o in opens})
    with coins_conn.cursor() as cur_c:
        cur_c.execute("""
            SELECT DISTINCT ON (symbol) symbol, high, low, close
            FROM klines WHERE symbol = ANY(%s) AND interval='10s'
            ORDER BY symbol, open_time DESC
        """, (symbols,))
        px = {r[0]: (float(r[1]), float(r[2]), float(r[3])) for r in cur_c.fetchall()}

    n_closed = 0
    for (ppid, sym, side, entry, qty, lev, margin, tp, sl, opened, to_en, age_h) in opens:
        p = px.get(sym)
        if p is None:
            continue
        hi, lo, cl = p
        entry = float(entry); tp = float(tp); sl = float(sl)
        is_long = (side == 'long')
        status = None; exit_px = None; reason = None

        # TP/SL-Hit (intra-bucket via high/low)
        if is_long:
            if hi >= tp:
                status='win'; exit_px=tp; reason='tp'
            elif lo <= sl:
                status='loss'; exit_px=sl; reason='sl'
        else:
            if lo <= tp:
                status='win'; exit_px=tp; reason='tp'
            elif hi >= sl:
                status='loss'; exit_px=sl; reason='sl'
        # Timeout
        if status is None and to_en and age_h >= timeout_h:
            status='timeout'; exit_px=cl; reason='timeout'

        if status is None:
            continue

        pnl_coin_pct = ((exit_px - entry)/entry if is_long else (entry - exit_px)/entry) * 100.0
        notional = float(margin) * int(lev)
        pnl_usd = pnl_coin_pct/100.0 * notional
        slip_usd = notional * slip_pct/100.0
        net_usd = pnl_usd - slip_usd
        margin_ret_pct = pnl_coin_pct * int(lev)  # return auf margin (nach Hebel)

        with app_conn.cursor() as cur_u:
            cur_u.execute("""
                UPDATE paper_positions
                SET status=%s, closed_at=now(), exit_px=%s, pnl_pct=%s, pnl_usd=%s, close_reason=%s
                WHERE id=%s AND status='open'
            """, (status, exit_px, round(margin_ret_pct,4), round(net_usd,4), reason, ppid))
        app_conn.commit()
        bal = _wallet_apply(app_conn, net_usd)
        n_closed += 1
        log.info("PAPER-CLOSE %s %s ppid=%s %s pnl=%.3f%%margin net=$%.3f -> wallet=$%.2f",
                 sym, side, ppid, status, margin_ret_pct, net_usd, bal or -1)
    return n_closed
