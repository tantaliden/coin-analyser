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
    """Schliesst paper_positions GEKOPPELT an ihre Prediction (Volker 22.05.):
    sobald die zugehoerige open_predictions geschlossen ist (win/loss/timeout),
    wird das Paper mit DEMSELBEN Status + Exit-Preis geschlossen. Paper ist ein
    Schatten der Prediction → exakt synchron, KEIN eigener Klines-Check (vermeidet
    Versatz + Spike-Luecken). Paper rechnet nur sein $-PnL (Hebel + Slippage) aus
    dem geteilten Exit. coins_conn/timeout_h bleiben in der Signatur (Aufruf-Kompat),
    werden aber nicht mehr gebraucht. Gibt Anzahl Closes zurueck."""
    slip_pct = float(cfg.get("paper_wallet", {}).get("slippage_pct_per_trade") or 0.0)
    reason_map = {'win': 'tp', 'loss': 'sl', 'timeout': 'timeout'}
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT pp.id, pp.symbol, pp.side, pp.entry_px, pp.leverage, pp.margin_usd,
                   op.status, op.exit_px, pp.tp_px, pp.sl_px
            FROM paper_positions pp
            JOIN open_predictions op ON op.id = pp.prediction_id
            WHERE pp.status='open' AND op.status IN ('win','loss','timeout')
        """)
        rows = cur.fetchall()
    if not rows:
        return 0

    n_closed = 0
    for (ppid, sym, side, entry, lev, margin, pred_status, pred_exit, tp, sl) in rows:
        entry = float(entry)
        # Exit = realisierter Prediction-Exit; Fallback tp/sl je Status, sonst entry.
        if pred_exit is not None:
            exit_px = float(pred_exit)
        elif pred_status == 'win' and tp is not None:
            exit_px = float(tp)
        elif pred_status == 'loss' and sl is not None:
            exit_px = float(sl)
        else:
            exit_px = entry
        is_long = (side == 'long')
        pnl_coin_pct = ((exit_px - entry)/entry if is_long else (entry - exit_px)/entry) * 100.0
        notional = float(margin) * int(lev)
        pnl_usd = pnl_coin_pct/100.0 * notional
        slip_usd = notional * slip_pct/100.0
        net_usd = pnl_usd - slip_usd
        margin_ret_pct = pnl_coin_pct * int(lev)  # return auf margin (nach Hebel)
        reason = reason_map.get(pred_status, pred_status)

        with app_conn.cursor() as cur_u:
            cur_u.execute("""
                UPDATE paper_positions
                SET status=%s, closed_at=now(), exit_px=%s, pnl_pct=%s, pnl_usd=%s, close_reason=%s
                WHERE id=%s AND status='open'
            """, (pred_status, exit_px, round(margin_ret_pct,4), round(net_usd,4), reason, ppid))
        app_conn.commit()
        bal = _wallet_apply(app_conn, net_usd)
        n_closed += 1
        log.info("PAPER-CLOSE %s %s ppid=%s %s pnl=%.3f%%margin net=$%.3f -> wallet=$%.2f (gekoppelt an pred)",
                 sym, side, ppid, pred_status, margin_ret_pct, net_usd, bal or -1)
    return n_closed
