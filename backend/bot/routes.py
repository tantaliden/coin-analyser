"""BOT ROUTES - Trading Bot Configuration"""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from binance.exceptions import BinanceAPIException
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user
from wallet.routes import get_user_binance_client

router = APIRouter(prefix="/api/v1/bot", tags=["bot"])

class BotConfigUpdate(BaseModel):
    is_active: Optional[bool] = None
    amount_per_trade: Optional[float] = None
    max_trades_enabled: Optional[bool] = None
    max_trades_limit: Optional[int] = None

@router.get("/config")
async def get_bot_config(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_active, amount_per_trade, today_trades, today_profit_loss, max_trades_enabled, max_trades_limit FROM trading_bot_config WHERE user_id = %s", (user_id,))
            config = cur.fetchone()
            if not config:
                cur.execute("INSERT INTO trading_bot_config (user_id, is_active, amount_per_trade) VALUES (%s, FALSE, 50.0) ON CONFLICT DO NOTHING", (user_id,))
                conn.commit()
                config = {'is_active': False, 'amount_per_trade': 50.0, 'today_trades': 0, 'today_profit_loss': 0}
    return {"is_active": config['is_active'], "amount_per_trade": float(config['amount_per_trade']),
        "today_trades": config.get('today_trades') or 0, "today_profit_loss": float(config.get('today_profit_loss') or 0),
        "max_trades_enabled": config.get('max_trades_enabled', False), "max_trades_limit": config.get('max_trades_limit', 10)}

@router.put("/config")
async def update_bot_config(request: BotConfigUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    if request.is_active:
        client = get_user_binance_client(user_id)
        if not client:
            return {"error": "Kein API Key - Bot kann nicht aktiviert werden"}
    updates, values = [], []
    if request.is_active is not None:
        updates.append("is_active = %s"); values.append(request.is_active)
    if request.amount_per_trade is not None:
        if request.amount_per_trade < 10:
            return {"error": "Mindestbetrag 10 USDC"}
        updates.append("amount_per_trade = %s"); values.append(request.amount_per_trade)
    if request.max_trades_enabled is not None:
        updates.append("max_trades_enabled = %s"); values.append(request.max_trades_enabled)
    if request.max_trades_limit is not None:
        updates.append("max_trades_limit = %s"); values.append(request.max_trades_limit)
    if not updates:
        return {"error": "Keine Änderungen"}
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO trading_bot_config (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
            cur.execute(f"UPDATE trading_bot_config SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = %s", values + [user_id])
            conn.commit()
    return {"status": "updated"}

@router.get("/upcoming")
async def get_upcoming_events(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT ue.event_id, ue.symbol, ue.expected_start, ue.expected_duration_min, ue.expected_target_pct,
                ue.take_profit_pct, ue.stop_loss_pct, ue.status, iset.name as set_name, iset.current_accuracy
                FROM upcoming_events ue LEFT JOIN indicator_sets iset ON ue.indicator_set_id = iset.set_id
                WHERE ue.status IN ('waiting', 'active') AND ue.expected_start >= NOW() - INTERVAL '30 minutes'
                ORDER BY ue.expected_start ASC LIMIT 50""")
            events = [dict(row) for row in cur.fetchall()]
    return {"events": events}

@router.post("/execute/{event_id}")
async def execute_bot_trade(event_id: int, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_active, amount_per_trade FROM trading_bot_config WHERE user_id = %s", (user_id,))
            config = cur.fetchone()
    if not config or not config['is_active']:
        return {"error": "Bot nicht aktiv"}
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, take_profit_pct, stop_loss_pct, status FROM upcoming_events WHERE event_id = %s", (event_id,))
            event = cur.fetchone()
    if not event or event['status'] not in ('waiting', 'active'):
        return {"error": "Event nicht handelbar"}
    client = get_user_binance_client(user_id)
    if not client:
        return {"error": "Kein API Key"}
    try:
        buy_order = client.create_order(symbol=event['symbol'], side='BUY', type='MARKET', quoteOrderQty=float(config['amount_per_trade']))
        filled_qty = float(buy_order.get('executedQty', 0))
        filled_quote = float(buy_order.get('cummulativeQuoteQty', 0))
        avg_price = filled_quote / filled_qty if filled_qty > 0 else 0
        tp_pct = float(event['take_profit_pct'] or 5.0)
        sl_pct = float(event['stop_loss_pct'] or 3.0)
        tp_price = round(avg_price * (1 + tp_pct / 100), 8)
        sl_price = round(avg_price * (1 - sl_pct / 100), 8)
        oco = client.create_oco_order(symbol=event['symbol'], side='SELL', quantity=filled_qty, price=tp_price, stopPrice=sl_price, stopLimitPrice=round(sl_price * 0.995, 8), stopLimitTimeInForce='GTC')
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO trade_history (user_id, event_id, symbol, side, price, quantity, quote_amount, is_bot_trade, executed_at) VALUES (%s, %s, %s, 'buy', %s, %s, %s, TRUE, NOW())",
                    (user_id, event_id, event['symbol'], avg_price, filled_qty, filled_quote))
                cur.execute("UPDATE upcoming_events SET status = 'active' WHERE event_id = %s", (event_id,))
                cur.execute("UPDATE trading_bot_config SET today_trades = today_trades + 1 WHERE user_id = %s", (user_id,))
                conn.commit()
        return {"status": "executed", "symbol": event['symbol'], "buy_qty": filled_qty, "buy_price": avg_price, "tp_price": tp_price, "sl_price": sl_price}
    except BinanceAPIException as e:
        return {"error": e.message}
