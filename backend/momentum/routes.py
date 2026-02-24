"""MOMENTUM ROUTES - Scanner Config, Predictions, Stats, Trade Execution"""
import json, math
from pathlib import Path
from typing import Optional
from decimal import Decimal, ROUND_DOWN
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Query
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db, get_coins_db
from auth.auth import get_current_user
from wallet.routes import get_user_binance_client

router = APIRouter(prefix="/api/v1/momentum", tags=["momentum"])

# === MODELS ===

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
    long_fixed_tp_pct: Optional[float] = None
    long_fixed_sl_pct: Optional[float] = None
    short_fixed_tp_pct: Optional[float] = None
    short_fixed_sl_pct: Optional[float] = None

class TradeRequest(BaseModel):
    take_profit_pct: float
    stop_loss_pct: float

# === HELPERS ===

def _round_price(price: float, precision: int) -> str:
    """Preis auf Binance-Precision runden"""
    d = Decimal(str(price))
    fmt = Decimal(10) ** -precision
    return str(d.quantize(fmt, rounding=ROUND_DOWN))

def _round_qty(qty: float, precision: int, min_qty: float) -> str:
    """Menge auf Binance-Precision runden, min_qty beachten"""
    d = Decimal(str(qty))
    fmt = Decimal(10) ** -precision
    rounded = d.quantize(fmt, rounding=ROUND_DOWN)
    if float(rounded) < min_qty:
        return None
    return str(rounded)

def _enrich_with_prices(predictions):
    """Aktuellen Kurs + pct_change für jede Prediction holen"""
    if not predictions:
        return predictions
    symbols = list(set(p['symbol'] for p in predictions))
    prices = {}
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (symbol) symbol, close, open_time
                FROM klines WHERE symbol = ANY(%s) AND interval = '1m'
                ORDER BY symbol, open_time DESC
            """, (symbols,))
            for row in cur.fetchall():
                prices[row['symbol']] = {'current_price': row['close'], 'price_time': row['open_time']}
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

# === CONFIG ===

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

# === PREDICTIONS ===

@router.get("/predictions")
async def get_predictions(
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    hide_traded: Optional[bool] = Query(None),
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
            if direction:
                where.append("direction = %s")
                params.append(direction)
            if hide_traded:
                where.append("traded = false")
            where_sql = " AND ".join(where)
            cur.execute(f"SELECT COUNT(*) as total FROM momentum_predictions WHERE {where_sql}", params)
            total = cur.fetchone()['total']
            cur.execute(f"""
                SELECT * FROM momentum_predictions WHERE {where_sql}
                ORDER BY detected_at DESC LIMIT %s OFFSET %s
            """, params + [limit, offset])
            predictions = [dict(r) for r in cur.fetchall()]
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

# === TRADE EXECUTION ===

@router.post("/trade/{prediction_id}")
async def execute_trade(prediction_id: int, req: TradeRequest, current_user: dict = Depends(get_current_user)):
    """
    Prediction an Tradebot übergeben:
    1. Market Buy in Höhe von amount_per_trade
    2. Free Balance des Coins prüfen (Binance zieht Fees ab)
    3. Auf Binance-Rundungswerte runden (coin_info)
    4. LIMIT Sell Order (TP) + STOP_LOSS_LIMIT Order (SL) als OCO
    5. Prediction als traded markieren
    """
    user_id = current_user['user_id']

    # Bot Config holen
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT amount_per_trade FROM trading_bot_config WHERE user_id = %s", (user_id,))
            bot_cfg = cur.fetchone()
    if not bot_cfg:
        raise HTTPException(400, "Bot nicht konfiguriert - Amount per Trade fehlt")
    amount_usdt = float(bot_cfg['amount_per_trade'])

    # Prediction holen
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT prediction_id, symbol, direction, entry_price, status, traded
                FROM momentum_predictions WHERE prediction_id = %s AND user_id = %s
            """, (prediction_id, user_id))
            pred = cur.fetchone()
    if not pred:
        raise HTTPException(404, "Prediction nicht gefunden")
    if pred['traded']:
        raise HTTPException(400, "Prediction wurde bereits gehandelt")
    if pred['status'] != 'active':
        raise HTTPException(400, f"Prediction Status ist '{pred['status']}', nicht 'active'")
    if pred['direction'] != 'long':
        raise HTTPException(400, "Nur Long-Predictions können auf Spot gehandelt werden")

    symbol = pred['symbol']

    # Coin Info für Rundungswerte
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT price_precision, qty_precision, min_qty, min_notional FROM coin_info WHERE symbol = %s", (symbol,))
            coin = cur.fetchone()
    if not coin:
        raise HTTPException(400, f"Keine coin_info für {symbol} - Rundungswerte fehlen")

    price_prec = coin['price_precision']
    qty_prec = coin['qty_precision']
    min_qty = float(coin['min_qty'])
    min_notional = float(coin['min_notional'])

    if amount_usdt < min_notional:
        raise HTTPException(400, f"Amount {amount_usdt} USDC unter min_notional {min_notional} für {symbol}")

    # Binance Client
    client = get_user_binance_client(user_id)
    if not client:
        raise HTTPException(400, "Kein gültiger Binance API Key")

    # === SCHRITT 1: Market Buy ===
    try:
        buy_order = client.create_order(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quoteOrderQty=str(amount_usdt)
        )
    except Exception as e:
        raise HTTPException(500, f"Market Buy fehlgeschlagen: {str(e)}")

    filled_qty = float(buy_order.get('executedQty', 0))
    filled_quote = float(buy_order.get('cummulativeQuoteQty', 0))
    avg_price = filled_quote / filled_qty if filled_qty > 0 else 0

    if filled_qty <= 0:
        raise HTTPException(500, "Buy Order hat keine Menge gefüllt")

    # === SCHRITT 2: Free Balance prüfen (nach Fee-Abzug) ===
    base_asset = symbol.replace('USDC', '')
    try:
        balance = client.get_asset_balance(asset=base_asset)
        free_qty = float(balance.get('free', 0)) if balance else 0
    except Exception:
        free_qty = filled_qty  # Fallback

    # Die tatsächlich verkaufbare Menge ist das Minimum
    sell_qty = min(free_qty, filled_qty)

    # === SCHRITT 3: Rundung auf Binance-Werte ===
    rounded_qty = _round_qty(sell_qty, qty_prec, min_qty)
    if not rounded_qty:
        raise HTTPException(500, f"Gerundete Menge {sell_qty} unter min_qty {min_qty}")

    tp_price_raw = avg_price * (1 + req.take_profit_pct / 100)
    sl_price_raw = avg_price * (1 - req.stop_loss_pct / 100)
    sl_limit_raw = sl_price_raw * 0.995  # SL Limit etwas unter Stop Price

    tp_price = _round_price(tp_price_raw, price_prec)
    sl_price = _round_price(sl_price_raw, price_prec)
    sl_limit = _round_price(sl_limit_raw, price_prec)

    # === SCHRITT 4: OCO Order (TP + SL) ===
    try:
        oco = client.create_oco_order(
            symbol=symbol,
            side='SELL',
            quantity=rounded_qty,
            aboveType='LIMIT_MAKER',
            abovePrice=tp_price,
            belowType='STOP_LOSS_LIMIT',
            belowPrice=sl_limit,
            belowStopPrice=sl_price,
            belowTimeInForce='GTC'
        )
        oco_order_id = str(oco.get('orderListId', ''))
    except Exception as e:
        # OCO fehlgeschlagen - Market Sell als Notfall, damit kein offener Bestand bleibt
        try:
            client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=rounded_qty)
        except Exception:
            pass
        raise HTTPException(500, f"OCO Order fehlgeschlagen: {str(e)}. Position wurde per Market verkauft.")

    # === SCHRITT 5: DB Updates ===
    with get_app_db() as conn:
        with conn.cursor() as cur:
            # Prediction als traded markieren
            cur.execute("UPDATE momentum_predictions SET traded = true WHERE prediction_id = %s", (prediction_id,))

            # Trade History
            cur.execute("""
                INSERT INTO trade_history (user_id, prediction_id, symbol, side, price, quantity, quote_amount, order_id, is_bot_trade, executed_at)
                VALUES (%s, %s, %s, 'buy', %s, %s, %s, %s, TRUE, NOW())
            """, (user_id, prediction_id, symbol, avg_price, float(rounded_qty), filled_quote, oco_order_id))

            # Bot stats
            cur.execute("UPDATE trading_bot_config SET today_trades = today_trades + 1 WHERE user_id = %s", (user_id,))
            conn.commit()

    return {
        "status": "executed",
        "symbol": symbol,
        "buy_qty": filled_qty,
        "sell_qty": float(rounded_qty),
        "buy_price": round(avg_price, price_prec),
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_pct": req.take_profit_pct,
        "sl_pct": req.stop_loss_pct,
        "amount_usdt": round(filled_quote, 2),
        "oco_order_id": oco_order_id,
        "prediction_id": prediction_id
    }

# === STATS ===

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM momentum_stats WHERE user_id = %s", (current_user['user_id'],))
            stats = {r['period']: dict(r) for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) as c FROM momentum_predictions WHERE user_id = %s AND status = 'active'", (current_user['user_id'],))
            active = cur.fetchone()['c']
            return {"stats": stats, "active_predictions": active}

# === CORRECTIONS & OPTIMIZATIONS ===

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
