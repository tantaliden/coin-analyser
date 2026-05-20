"""Predictor API-Routes — Liste der Predictions, Status, Order aus Prediction.

settings.json -> 'predictor' wird live gelesen/geschrieben.
"""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

import psycopg2

router = APIRouter(prefix="/api/v1/predictor", tags=["predictor"])

SETTINGS_PATH = "/opt/coin/settings.json"

# REST-Fallback-Cache für mids (nur wenn WS-State noch nicht initialisiert).
_mids_cache = {"ts": 0.0, "data": {}}


def _live_mids() -> dict:
    """Aktuelle HL Mid-Preise — primaer aus WS-State (sub-200ms),
    Fallback: REST mit 1s-Cache."""
    try:
        from wallet.hl_ws_state import get_ws_state
        ws_mids = get_ws_state().get_mids()
        if ws_mids:
            return ws_mids
    except Exception as e:
        import logging
        logging.getLogger("predictor.routes").warning("ws_mids access failed: %s", e)
    # Fallback: REST
    now = time.time()
    if now - _mids_cache["ts"] < 1.0 and _mids_cache["data"]:
        return _mids_cache["data"]
    try:
        from rl_agent.trader import get_current_prices_hl
        _mids_cache["data"] = get_current_prices_hl()
        _mids_cache["ts"] = now
    except Exception as e:
        import logging
        logging.getLogger("predictor.routes").warning("allMids REST failed: %s", e)
    return _mids_cache["data"]


def _load_settings():
    with open(SETTINGS_PATH) as fp:
        return json.load(fp)


def _db_app():
    db = _load_settings()["databases"]["app"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def _db_coins():
    db = _load_settings()["databases"]["coins"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def _db_learner():
    db = _load_settings()["databases"]["learner"]
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["name"],
                            user=db["user"], password=db["password"])


def _service_running() -> bool:
    try:
        out = subprocess.run(["/usr/bin/systemctl", "is-active", "predictor"],
                             capture_output=True, text=True, timeout=5)
        s = (out.stdout or "").strip()
        return s == "active" or out.returncode == 0
    except Exception as e:
        import logging; logging.getLogger("predictor.routes").warning("is-active check failed: %s", e)
        return False


@router.get("/status")
def get_status():
    s = _load_settings()
    cfg = s.get("predictor", {})
    with _db_app() as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT threshold, model_version, closed_count, rolling_winrate,
                       last_calibration_at, universe, universe_refreshed_at, updated_at
                FROM predictor_state WHERE id=1
            """)
            st = cur.fetchone() or {}
            cur.execute("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE NOT auto_trade_skipped) AS visible
                FROM open_predictions WHERE status='open'
            """)
            _o = cur.fetchone()
            open_count = _o['total']
            open_count_visible = _o['visible']
            cur.execute("""
                SELECT
                  COUNT(*) FILTER (WHERE status='win') AS wins,
                  COUNT(*) FILTER (WHERE status='loss') AS losses,
                  COUNT(*) FILTER (WHERE status='timeout') AS timeouts,
                  COUNT(*) FILTER (WHERE status='timeout' AND pnl_pct > 0) AS timeouts_profit,
                  COUNT(*) FILTER (WHERE status IN ('win','loss','timeout')) AS total_real_closed,
                  COUNT(*) FILTER (WHERE status IN ('auto_trade_failed','auto_close_failsafe')) AS phantoms
                FROM open_predictions
            """)
            stats = cur.fetchone()
    wr = None
    # WR-Definition: profitable Closes / alle echten Closes.
    # - "Profitabel" = WIN status ODER TIMEOUT mit pnl_pct > 0
    # - "Echt" = win + loss + timeout (Phantoms ausgeschlossen — die haben keine
    #   reale HL-Realisierung gehabt, sind kein Bewertungs-Sample fuer den Bandit)
    if stats and stats['total_real_closed'] and stats['total_real_closed'] > 0:
        profitable = (stats['wins'] or 0) + (stats['timeouts_profit'] or 0)
        wr = round(profitable / stats['total_real_closed'] * 100, 2)
    return {
        "service_running": _service_running(),
        "enabled": cfg.get("enabled", False),
        "auto_trade": cfg.get("trading", {}).get("auto_trade", False),
        "threshold": float(st.get("threshold") or 0.0),
        "model_version": st.get("model_version") or 0,
        "closed_count": st.get("closed_count") or 0,
        "rolling_winrate": float(st["rolling_winrate"]) if st.get("rolling_winrate") is not None else None,
        "last_calibration_at": st.get("last_calibration_at"),
        "universe_size": len(st.get("universe") or []),
        "universe_refreshed_at": st.get("universe_refreshed_at"),
        "open_count": open_count,
        "open_count_visible": open_count_visible,
        "wins": stats['wins'] if stats else 0,
        "losses": stats['losses'] if stats else 0,
        "timeouts": stats['timeouts'] if stats else 0,
        "timeouts_profit": stats['timeouts_profit'] if stats else 0,
        "phantoms": stats['phantoms'] if stats else 0,
        "total_closed": stats['total_real_closed'] if stats else 0,
        "winrate_pct": wr,
    }


# Schmale Spalten fuer 'closed' und 'all' — features-jsonb (60+ keys) ist gross und nicht noetig fuer Anzeige.
_LIGHT_FIELDS = ("id, symbol, side, entry_px, sl_px, tp_px, score, rule_name, source, status, "
                 "last_px, peak_px, trough_px, predicted_up_pct, predicted_down_pct, "
                 "pnl_pct, exit_px, created_at, closed_at, learned")


@router.get("/predictions")
def list_predictions(scope: str = "visible", limit: int = 200, include_skipped: bool = False):
    """scope: 'visible' = open + < hide_after_hours; 'open' = alle offenen;
    'closed' = echte Closes (win/loss/timeout, OHNE Phantoms);
    'all' = alle echten (open + win/loss/timeout, OHNE Phantoms);
    'errors' = nur Phantoms (auto_trade_failed, auto_close_failsafe).

    include_skipped (default False): wenn False, Predictions mit auto_trade_skipped=TRUE
    werden in 'visible' und 'all' versteckt. 'open' zeigt sie immer (das ist der
    „alle offen"-Tab im Frontend).
    """
    cfg = _load_settings().get("predictor", {})
    hide_h = float(cfg.get("hide_after_hours", 1))
    PHANTOM = ('auto_trade_failed', 'auto_close_failsafe')
    skip_filter = "" if include_skipped else " AND NOT auto_trade_skipped"
    with _db_app() as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            if scope == "visible":
                cur.execute(f"""
                    SELECT * FROM open_predictions
                    WHERE status='open'{skip_filter} AND created_at >= now() - (%s || ' hours')::interval
                    ORDER BY created_at DESC LIMIT %s
                """, (hide_h, limit))
            elif scope == "open":
                # 'open' zeigt ALLE offenen Predictions inkl. skipped (= „alle offen"-Tab)
                cur.execute("SELECT * FROM open_predictions WHERE status='open' ORDER BY created_at DESC LIMIT %s", (limit,))
            elif scope == "closed":
                cur.execute(f"""
                    SELECT {_LIGHT_FIELDS} FROM open_predictions
                    WHERE status IN ('win','loss','timeout')
                    ORDER BY closed_at DESC NULLS LAST LIMIT %s
                """, (limit,))
            elif scope == "errors":
                cur.execute(f"""
                    SELECT {_LIGHT_FIELDS} FROM open_predictions
                    WHERE status = ANY(%s)
                    ORDER BY closed_at DESC NULLS LAST LIMIT %s
                """, (list(PHANTOM), limit))
            else:  # all
                cur.execute(f"""
                    SELECT {_LIGHT_FIELDS} FROM open_predictions
                    WHERE status NOT IN ('auto_trade_failed','auto_close_failsafe'){skip_filter}
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
    # Live-Mids nur fuer Scopes mit offenen Predictions (closed-Anzeige braucht keinen Live-Preis).
    if scope in ("visible", "open", "all"):
        mids = _live_mids()
        for r in rows:
            r["live_px"] = mids.get(r["symbol"])
    return {"predictions": rows, "scope": scope, "count": len(rows)}


@router.get("/predictions/{pred_id}")
def get_prediction(pred_id: int):
    with _db_app() as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM open_predictions WHERE id=%s", (pred_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"prediction {pred_id} not found")
    return row


class OrderRequest(BaseModel):
    leverage: int
    size_usd: float
    sl_px: float
    tp_px: float
    use_trailing: bool = False
    trailing_stop_pct: float = 1.0


@router.post("/predictions/{pred_id}/order")
def place_order_from_prediction(pred_id: int, req: OrderRequest):
    """Manuelle Order aus einer offenen Prediction. Nutzt den gemeinsamen
    Order-Executor (Failsafe-Pattern: Open + TP/SL parallel, Retry, Auto-Close)."""
    from rl_agent.trader import get_hl_credentials
    from predictor.order_executor import execute_order_with_failsafe

    with _db_app() as app:
        with app.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM open_predictions WHERE id=%s AND status='open'", (pred_id,))
            p = cur.fetchone()
    if not p:
        raise HTTPException(404, "prediction not found or already closed")

    creds = get_hl_credentials(user_id=1)
    if not creds:
        raise HTTPException(400, "Hyperliquid credentials missing")

    mids = _live_mids()
    live_px = mids.get(p["symbol"])
    if not live_px:
        raise HTTPException(500, f"no live price for {p['symbol']} (HL allMids leer)")
    entry = float(live_px)

    cfg = _load_settings().get("predictor", {})
    slippage_pct = float(cfg.get("trading", {}).get("order_slippage_pct", 1.0))

    res = execute_order_with_failsafe(
        creds, p["symbol"], is_long=(p["side"] == "long"),
        leverage=req.leverage, size_usd=req.size_usd,
        tp_px=req.tp_px, sl_px=req.sl_px,
        entry_px=entry, slippage_pct=slippage_pct,
    )
    eff_lev = res.get("effective_leverage")
    if not res["success"]:
        # CARDINAL RULE: HL-Fehler beruehrt open_predictions NICHT. Die Predictor-Row
        # laeuft via watch_pass_v4 weiter und wird via Klines (TP/SL/Timeout) geschlossen
        # — unabhaengig davon ob die HL-Order erfolgreich war oder nicht.
        raise HTTPException(500, res["error"])
    # Erfolgsfall: NUR effective_leverage. tp_px/sl_px bleiben die Bandit-Werte
    # (fuer Bandit-Lernen). Manual-Order kann andere HL-TP/SL nutzen, die DB-Row
    # behaelt aber die Bandit-Vorgaben — sonst verfaelscht das den Hindsight-Replay.
    try:
        with _db_app() as app2:
            with app2.cursor() as cur2:
                cur2.execute("UPDATE open_predictions SET effective_leverage=%s WHERE id=%s",
                             (eff_lev, pred_id))
            app2.commit()
    except Exception as e:
        import logging as _log
        _log.getLogger("predictor.order").warning(
            "DB effective_leverage pid=%s failed: %s", pred_id, e)
    return {"status": "ok", "order": res["order"], "tp_sl": res["tp_sl"],
            "prediction_id": pred_id, "entry_px_live": entry,
            "effective_leverage": eff_lev}


@router.post("/service/{action}")
def control_service(action: str):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start/stop/restart")
    try:
        r = subprocess.run(["/usr/bin/systemctl", action, "predictor"],
                            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            raise HTTPException(500, f"systemctl {action} failed (rc={r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        return {"status": "ok", "action": action, "running": _service_running()}
    except FileNotFoundError as e:
        raise HTTPException(500, f"systemctl binary not found: {e}")
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"systemctl {action} timeout")


@router.get("/config")
def get_config():
    s = _load_settings()
    return s.get("predictor", {})


class ConfigUpdate(BaseModel):
    config: dict


@router.put("/config")
def put_config(payload: ConfigUpdate):
    """Komplettes 'predictor' Block in settings.json ersetzen."""
    s = _load_settings()
    s["predictor"] = payload.config
    bak = SETTINGS_PATH + ".bak"
    Path(bak).write_text(json.dumps(_load_settings(), indent=2))
    Path(SETTINGS_PATH).write_text(json.dumps(s, indent=2))
    return {"status": "ok"}


# Cache fuer coin-meta (sz_decimals, price_decimals, max_leverage), 30s TTL.
_coin_meta_cache = {"ts": 0.0, "data": {}}


@router.get("/coin-meta")
def get_coin_meta():
    """Liefert pro Symbol die Rundungs-Decimals + max-Leverage aus hl_meta.
    Frontend nutzt das fuer konsistente Preis-Anzeige (entry/sl/tp/peak/trough/live)."""
    now = time.time()
    if now - _coin_meta_cache["ts"] < 30 and _coin_meta_cache["data"]:
        return _coin_meta_cache["data"]
    with _db_coins() as coins:
        with coins.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, sz_decimals, price_decimals, max_leverage
                FROM hl_meta WHERE sz_decimals IS NOT NULL
            """)
            rows = cur.fetchall()
    out = {r["symbol"]: {
        "sz_decimals": r["sz_decimals"],
        "price_decimals": r["price_decimals"],
        "max_leverage": r["max_leverage"],
    } for r in rows}
    _coin_meta_cache["data"] = out
    _coin_meta_cache["ts"] = now
    return out


@router.get("/feedback")
def list_feedback(limit: int = 100):
    with _db_learner() as ldb:
        with ldb.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT feedback_id, prediction_id, symbol, direction, entry_price,
                       detected_at, resolved_at, status, was_correct, actual_result_pct,
                       duration_minutes, score, rule_name
                FROM prediction_feedback
                WHERE scanner_type='predictor'
                ORDER BY resolved_at DESC LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    return {"feedback": rows, "count": len(rows)}
