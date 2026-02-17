"""MOMENTUM ROUTES - Scanner Config, Predictions, Stats"""
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Query
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db, get_coins_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/momentum", tags=["momentum"])

class ConfigUpdate(BaseModel):
    is_active: Optional[bool] = None
    coin_group_id: Optional[int] = None
    scan_all_symbols: Optional[bool] = None
    idle_seconds: Optional[int] = None
    min_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    min_confidence: Optional[int] = None
    tp_sl_mode: Optional[str] = None
    fixed_tp_pct: Optional[float] = None
    fixed_sl_pct: Optional[float] = None
    range_tp_min: Optional[float] = None
    range_tp_max: Optional[float] = None
    range_sl_min: Optional[float] = None
    range_sl_max: Optional[float] = None

@router.get("/config")
async def get_config(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM momentum_scan_config WHERE user_id = %s", (current_user['user_id'],))
            config = cur.fetchone()
            if not config:
                cur.execute("""
                    INSERT INTO momentum_scan_config (user_id, is_active, scan_all_symbols, idle_seconds, min_target_pct, stop_loss_pct, min_confidence)
                    VALUES (%s, false, true, 60, 5.0, 2.0, 60) RETURNING *
                """, (current_user['user_id'],))
                config = cur.fetchone()
                conn.commit()
            return dict(config)

@router.put("/config")
async def update_config(data: ConfigUpdate, current_user: dict = Depends(get_current_user)):
    updates = {k: v for k, v in data.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Keine Änderungen")
    set_clauses = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [current_user['user_id']]
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE momentum_scan_config SET {set_clauses}, updated_at = NOW() WHERE user_id = %s RETURNING *", values)
            config = cur.fetchone()
            conn.commit()
            if not config:
                raise HTTPException(404, "Config nicht gefunden")
            return dict(config)

def _enrich_with_prices(predictions):
    """Aktuellen Kurs + pct_change für jede Prediction holen"""
    if not predictions:
        return predictions

    symbols = list(set(p['symbol'] for p in predictions))
    prices = {}

    with get_coins_db() as conn:
        with conn.cursor() as cur:
            # Letzter 1m-Kurs für jedes Symbol
            cur.execute("""
                SELECT DISTINCT ON (symbol) symbol, close, open_time
                FROM klines
                WHERE symbol = ANY(%s) AND interval = '1m'
                ORDER BY symbol, open_time DESC
            """, (symbols,))
            for row in cur.fetchall():
                prices[row['symbol']] = {
                    'current_price': row['close'],
                    'price_time': row['open_time']
                }

    for p in predictions:
        sym = p['symbol']
        if sym in prices:
            current = prices[sym]['current_price']
            entry = p['entry_price']
            if p['direction'] == 'long':
                pct = ((current - entry) / entry) * 100 if entry else 0
            else:
                pct = ((entry - current) / entry) * 100 if entry else 0
            p['current_price'] = current
            p['current_pct'] = round(pct, 2)
            p['price_time'] = prices[sym]['price_time'].isoformat() if prices[sym]['price_time'] else None
        else:
            p['current_price'] = None
            p['current_pct'] = None
            p['price_time'] = None

    return predictions

@router.get("/predictions")
async def get_predictions(
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            where = ["user_id = %s"]
            params = [current_user['user_id']]
            if status:
                where.append("status = %s")
                params.append(status)
            if symbol:
                where.append("symbol = %s")
                params.append(symbol)
            where_sql = " AND ".join(where)
            cur.execute(f"SELECT COUNT(*) as total FROM momentum_predictions WHERE {where_sql}", params)
            total = cur.fetchone()['total']
            cur.execute(f"""
                SELECT * FROM momentum_predictions WHERE {where_sql}
                ORDER BY detected_at DESC LIMIT %s OFFSET %s
            """, params + [limit, offset])
            predictions = [dict(r) for r in cur.fetchall()]

    # Aktuelle Kurse anreichern
    predictions = _enrich_with_prices(predictions)

    return {"predictions": predictions, "total": total}

@router.delete("/predictions/{prediction_id}")
async def cancel_prediction(prediction_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE momentum_predictions SET status = 'invalidated', resolved_at = NOW()
                WHERE prediction_id = %s AND user_id = %s AND status = 'active'
                RETURNING prediction_id
            """, (prediction_id, current_user['user_id']))
            result = cur.fetchone()
            conn.commit()
            if not result:
                raise HTTPException(404, "Prediction nicht gefunden oder nicht aktiv")
            return {"message": "Prediction invalidiert", "prediction_id": prediction_id}

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM momentum_stats WHERE user_id = %s", (current_user['user_id'],))
            stats = {r['period']: dict(r) for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) as c FROM momentum_predictions WHERE user_id = %s AND status = 'active'", (current_user['user_id'],))
            active = cur.fetchone()['c']
            return {"stats": stats, "active_predictions": active}

@router.get("/corrections")
async def get_corrections(limit: int = Query(20, ge=1, le=100), current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT mc.*, mp.symbol, mp.direction, mp.confidence, mp.actual_result_pct
                FROM momentum_corrections mc
                JOIN momentum_predictions mp ON mp.prediction_id = mc.prediction_id
                WHERE mp.user_id = %s ORDER BY mc.created_at DESC LIMIT %s
            """, (current_user['user_id'], limit))
            return [dict(r) for r in cur.fetchall()]

@router.get("/optimizations")
async def get_optimizations(limit: int = Query(10, ge=1, le=50), current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM momentum_optimization_log
                WHERE user_id = %s ORDER BY run_at DESC LIMIT %s
            """, (current_user['user_id'], limit))
            return [dict(r) for r in cur.fetchall()]
