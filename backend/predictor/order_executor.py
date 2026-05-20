"""Gemeinsamer Order-Executor mit Failsafe-Pattern.

Genutzt von:
- predictor/routes.py — manuelle Order aus dem Frontend
- services/predictor_service.py — Auto-Trade nach jeder neuen Bandit-Prediction

Failsafe: Open + TP/SL strikt parallel; falls TP/SL nicht beide gesetzt -> 1x Retry
mit Cleanup -> sonst Auto-Close der ungeschuetzten Position. Eine Position kann
NIE ohne TP/SL offen bleiben.
"""
import logging
import time

log = logging.getLogger("predictor.order_executor")


def execute_order_with_failsafe(creds, symbol, is_long, leverage, size_usd,
                                  tp_px, sl_px, entry_px, slippage_pct=1.0,
                                  tp_pct=None, sl_pct=None):
    """Order ausfuehren, TP/SL setzen, bei Fehler Retry oder Auto-Close.

    size_usd ist die MARGIN (Volkers Invest). HL-Order erwartet die NOTIONAL
    (= margin x leverage), damit der Hebel sich auch im Profit/Loss niederschlaegt.

    Wenn tp_pct/sl_pct gegeben sind: TP/SL werden NACH der Open-Order
    auf Basis des TATSAECHLICHEN Fill-Preises (avg_price) neu berechnet —
    so verhindern wir dass Slippage zwischen Mid-Lookup und Order-Execution
    die TP/SL-Levels verfaelscht (sonst landen sie z.B. unter dem echten
    Open-Preis und werden sofort als Loss gefuellt).

    Returns dict:
      {success: True,  order: ..., tp_sl: ..., filled_qty: ..., effective_leverage: ...,
       actual_entry_px: ..., tp_px: ..., sl_px: ...}  # final verwendete TP/SL
      {success: False, ...}
    """
    from rl_agent.trader import (
        place_limit_ioc_order_hl, place_tp_sl_hl,
        close_position_hl, cancel_all_orders_for_coin_hl,
        _get_hl_coin_info,
    )

    # Hebel cappen gegen coin-spezifischen Maximum, dann Notional rechnen.
    coin_info = _get_hl_coin_info(symbol) or {}
    max_lev = int(coin_info.get("max_leverage") or leverage)
    effective_lev = max(1, min(int(leverage), max_lev))
    notional = float(size_usd) * effective_lev

    # === Schritt 1: Echte Limit-IOC-Order „Predictor-Preis oder besser" ===
    # Volker-Direktive 19.05.2026: kaufen zum Predictor-Preis ODER bis zu
    # max_slippage_pct schlechter — sonst gar nicht.
    #   Long:  limit_px = entry × (1 + slip/100)  (bereit bis +slip% teurer zu kaufen)
    #   Short: limit_px = entry × (1 - slip/100)  (bereit bis -slip% billiger zu verkaufen)
    # HL fillt zum BESTEN Match innerhalb dieses Limits — nie schlechter
    # → kein Post-Close, keine doppelten Fees.
    slip = float(slippage_pct) / 100.0
    if is_long:
        ioc_limit = float(entry_px) * (1.0 + slip)
    else:
        ioc_limit = float(entry_px) * (1.0 - slip)
    order_res = place_limit_ioc_order_hl(
        creds, symbol, is_buy=is_long, size_usd=notional,
        limit_px=ioc_limit, leverage=effective_lev,
    )
    if not order_res or not order_res.get("success"):
        err = order_res.get("error", order_res) if order_res else "no response"
        return {"success": False, "error": f"order failed: {err}",
                "order": order_res, "effective_leverage": effective_lev}

    if order_res.get("status") != "filled":
        return {"success": False, "error": f"order_not_filled (status={order_res.get('status')})",
                "order": order_res, "effective_leverage": effective_lev}

    actual_entry = float(order_res.get("avg_price") or entry_px)

    # WICHTIG: TP/SL gegen ACTUAL fill-price neu rechnen wenn tp_pct/sl_pct übergeben.
    if tp_pct is not None and sl_pct is not None:
        if is_long:
            tp_px = actual_entry * (1 + float(tp_pct) / 100.0)
            sl_px = actual_entry * (1 - float(sl_pct) / 100.0)
        else:
            tp_px = actual_entry * (1 - float(tp_pct) / 100.0)
            sl_px = actual_entry * (1 + float(sl_pct) / 100.0)
        log.info("TP/SL re-anchored to fill-price %s: tp=%.6g sl=%.6g (was based on mid=%.6g)",
                 actual_entry, tp_px, sl_px, entry_px)

    filled_qty = order_res.get("quantity") or (notional / actual_entry)

    # === Schritt 2: TP + SL sequentiell (place_tp_sl_hl macht das intern) ===
    # + Retry + Auto-Close-Failsafe als Sicherheitsnetz.
    tpsl = place_tp_sl_hl(creds, symbol, is_long=is_long,
                           quantity=filled_qty, tp_price=tp_px, sl_price=sl_px)
    if not tpsl.get("success"):
        log.warning("TP/SL v1 fehlgeschlagen %s (tp=%s sl=%s) -> Retry",
                    symbol, tpsl.get("tp_error") or tpsl.get("error"),
                    tpsl.get("sl_error") or tpsl.get("error"))
        try:
            cancel_all_orders_for_coin_hl(creds, symbol)
        except Exception as e:
            log.warning("cancel_all_orders %s failed (egal): %s", symbol, e)
        # Laengere Pause: gibt HL Zeit um Nonce + Cleanup zu verarbeiten,
        # bevor wir die TP/SL nochmal versuchen.
        time.sleep(0.5)
        tpsl_retry = place_tp_sl_hl(creds, symbol, is_long=is_long,
                                      quantity=filled_qty, tp_price=tp_px, sl_price=sl_px)
        if not tpsl_retry.get("success"):
            log.error("TP/SL Retry fehlgeschlagen %s -> Auto-Close ungeschuetzte Position", symbol)
            try:
                cancel_all_orders_for_coin_hl(creds, symbol)
            except Exception as e:
                log.warning("final cancel %s failed: %s", symbol, e)
            close_res = close_position_hl(creds, symbol, creds["wallet_address"])
            return {
                "success": False,
                "error": (f"TP/SL setup failed twice for {symbol} - position auto-closed. "
                          f"v1=tp:{tpsl.get('tp_error') or tpsl.get('error') or 'ok'} "
                          f"sl:{tpsl.get('sl_error') or tpsl.get('error') or 'ok'} | "
                          f"v2=tp:{tpsl_retry.get('tp_error') or tpsl_retry.get('error') or 'ok'} "
                          f"sl:{tpsl_retry.get('sl_error') or tpsl_retry.get('error') or 'ok'} | "
                          f"close={close_res.get('success', False)}"),
                "order": order_res, "tp_sl": tpsl_retry, "close": close_res,
                "effective_leverage": effective_lev,
                "failsafe_closed": True,
            }
        tpsl = tpsl_retry

    return {"success": True, "order": order_res, "tp_sl": tpsl,
            "filled_qty": filled_qty, "effective_leverage": effective_lev,
            "actual_entry_px": actual_entry, "tp_px": tp_px, "sl_px": sl_px}
