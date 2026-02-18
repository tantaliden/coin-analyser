"""WALLET ROUTES - Binance Account Integration (vollständig aus altem Analyser)"""
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user, decrypt_value

router = APIRouter(prefix="/api/v1/wallet", tags=["wallet"])


def get_user_binance_client(user_id: int):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT binance_api_key_encrypted, binance_api_secret_encrypted, binance_api_valid FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
    if not user or not user['binance_api_key_encrypted'] or not user['binance_api_valid']:
        return None
    try:
        return BinanceClient(decrypt_value(user['binance_api_key_encrypted']), decrypt_value(user['binance_api_secret_encrypted']))
    except Exception as e:
        print(f"[WALLET] Error creating Binance client for user {user_id}: {e}")
        return None


@router.get("/status")
async def get_wallet_status(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT binance_api_valid FROM users WHERE user_id = %s", (current_user['user_id'],))
            user = cur.fetchone()
    return {"configured": bool(user and user['binance_api_valid'])}


@router.get("/balance")
async def get_wallet_balance(current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        account = client.get_account()
        usdc_balance, usdc_free, positions_value = 0.0, 0.0, 0.0
        for asset in account.get('balances', []):
            total = float(asset['free']) + float(asset['locked'])
            if total > 0:
                if asset['asset'] == 'USDC':
                    usdc_balance = total
                    usdc_free = float(asset['free'])
                else:
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{asset['asset']}USDC")
                        positions_value += total * float(ticker['price'])
                    except:
                        pass
        return {
            "usdc_balance": round(usdc_balance, 2),
            "usdc_free": round(usdc_free, 2),
            "positions_value": round(positions_value, 2),
            "total_portfolio": round(usdc_balance + positions_value, 2)
        }
    except BinanceAPIException as e:
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Error getting balance: {e}")
        return {"error": str(e)}


@router.get("/positions")
async def get_wallet_positions(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    client = get_user_binance_client(user_id)
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        account = client.get_account()
        positions = []
        for asset in account.get('balances', []):
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            if total > 0 and asset['asset'] not in ['USDC']:
                symbol = f"{asset['asset']}USDC"
                try:
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    value_usdt = total * current_price
                    if value_usdt < 1.0:
                        continue
                    # Durchschnittlichen Einstiegspreis aus Trade History
                    trades = client.get_my_trades(symbol=symbol, limit=50)
                    buy_qty, buy_cost = 0.0, 0.0
                    for trade in trades:
                        if trade['isBuyer']:
                            qty = float(trade['qty'])
                            price = float(trade['price'])
                            buy_qty += qty
                            buy_cost += qty * price
                    avg_entry = buy_cost / buy_qty if buy_qty > 0 else current_price
                    unrealized_pnl = (current_price - avg_entry) * total
                    pnl_percent = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
                    # Bot-Trade Info aus DB
                    bot_info = None
                    with get_app_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT is_bot_trade, indicator_set_name, indicator_set_accuracy
                                FROM trade_history
                                WHERE user_id = %s AND symbol = %s AND side = 'buy'
                                ORDER BY executed_at DESC LIMIT 1
                            """, (user_id, symbol))
                            bot_info = cur.fetchone()
                    positions.append({
                        "asset": asset['asset'],
                        "symbol": symbol,
                        "quantity": total,
                        "free": free,
                        "locked": locked,
                        "avg_entry_price": round(avg_entry, 8),
                        "current_price": current_price,
                        "value_usdt": round(value_usdt, 2),
                        "unrealized_pnl": round(unrealized_pnl, 2),
                        "pnl_percent": round(pnl_percent, 2),
                        "is_bot_trade": bot_info['is_bot_trade'] if bot_info else False,
                        "indicator_set_name": bot_info['indicator_set_name'] if bot_info else None,
                        "indicator_set_accuracy": float(bot_info['indicator_set_accuracy']) if bot_info and bot_info['indicator_set_accuracy'] else None
                    })
                except Exception as e:
                    print(f"[WALLET] Error processing position {asset['asset']}: {e}")
                    continue
        positions.sort(key=lambda x: -x['value_usdt'])
        return {"positions": positions}
    except BinanceAPIException as e:
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Error getting positions: {e}")
        return {"error": str(e)}


@router.get("/orders")
async def get_wallet_orders(current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        open_orders = client.get_open_orders()
        orders = []
        for order in open_orders:
            orders.append({
                "order_id": order['orderId'],
                "symbol": order['symbol'],
                "type": order['type'],
                "side": order['side'],
                "price": float(order['price']) if order['price'] else None,
                "stop_price": float(order['stopPrice']) if order.get('stopPrice') else None,
                "quantity": float(order['origQty']),
                "executed_qty": float(order['executedQty']),
                "status": order['status'],
                "time": order['time']
            })
        return {"orders": orders}
    except BinanceAPIException as e:
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Error getting orders: {e}")
        return {"error": str(e)}


@router.delete("/orders/{symbol}/{order_id}")
async def cancel_order(symbol: str, order_id: int, current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        result = client.cancel_order(symbol=symbol, orderId=order_id)
        print(f"[WALLET] User {current_user['user_id']} cancelled order {order_id} on {symbol}")
        return {"status": "cancelled", "order_id": order_id}
    except BinanceAPIException as e:
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Error cancelling order: {e}")
        return {"error": str(e)}


class CreateOrderRequest(BaseModel):
    symbol: str
    side: str
    type: str
    price: Optional[float] = None
    quantity: float


@router.post("/orders")
async def create_order(request: CreateOrderRequest, current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        info = client.get_symbol_info(request.symbol)
        if not info:
            return {"error": f"Symbol {request.symbol} nicht gefunden"}
        price_precision, qty_precision = 8, 8
        for f in info.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick_size = f['tickSize']
                price_precision = len(tick_size.rstrip('0').split('.')[-1]) if '.' in tick_size else 0
            elif f['filterType'] == 'LOT_SIZE':
                step_size = f['stepSize']
                qty_precision = len(step_size.rstrip('0').split('.')[-1]) if '.' in step_size else 0
        quantity = round(request.quantity, qty_precision)
        price = round(request.price, price_precision) if request.price else None
        print(f"[WALLET] Creating {request.type} {request.side} order for {request.symbol}: {quantity} @ {price}")
        if request.type == 'LIMIT':
            order = client.create_order(
                symbol=request.symbol, side=request.side, type='LIMIT',
                timeInForce='GTC', quantity=quantity, price=price
            )
        elif request.type == 'MARKET':
            order = client.create_order(
                symbol=request.symbol, side=request.side, type='MARKET', quantity=quantity
            )
        else:
            return {"error": f"Order-Typ {request.type} nicht unterstützt"}
        print(f"[WALLET] Order created: {order['orderId']}")
        return {"status": "created", "order_id": order['orderId'], "symbol": order['symbol']}
    except BinanceAPIException as e:
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Error creating order: {e}")
        return {"error": str(e)}


@router.get("/history")
async def get_trade_history(days: int = 7, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, side, price, quantity, quote_amount,
                       is_bot_trade, indicator_set_name, indicator_set_accuracy, executed_at
                FROM trade_history
                WHERE user_id = %s AND executed_at >= NOW() - INTERVAL '%s days'
                ORDER BY executed_at DESC LIMIT 100
            """, (user_id, days))
            trades = cur.fetchall()
    return {"trades": [{
        "id": t['id'], "symbol": t['symbol'], "side": t['side'],
        "price": float(t['price']), "quantity": float(t['quantity']),
        "quote_amount": float(t['quote_amount']),
        "is_bot_trade": t['is_bot_trade'],
        "indicator_set_name": t['indicator_set_name'],
        "indicator_set_accuracy": float(t['indicator_set_accuracy']) if t['indicator_set_accuracy'] else None,
        "executed_at": t['executed_at'].isoformat() if t['executed_at'] else None
    } for t in trades]}


@router.get("/realized-pnl")
async def get_realized_pnl(days: int = 7, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN side = 'sell' THEN quote_amount ELSE 0 END) as total_sells,
                    SUM(CASE WHEN side = 'buy' THEN quote_amount ELSE 0 END) as total_buys
                FROM trade_history
                WHERE user_id = %s AND executed_at >= NOW() - INTERVAL '%s days'
            """, (user_id, days))
            result = cur.fetchone()
    total_sells = float(result['total_sells'] or 0)
    total_buys = float(result['total_buys'] or 0)
    return {
        "realized_pnl": round(total_sells - total_buys, 2),
        "total_sells": round(total_sells, 2),
        "total_buys": round(total_buys, 2),
        "period_days": days
    }



class ConvertRequest(BaseModel):
    amount: Optional[float] = None  # None = max


@router.post("/convert-usdc")
async def convert_usdc_to_usdt(request: ConvertRequest, current_user: dict = Depends(get_current_user)):
    """USDC → USDT per Binance Convert API (gebührenfrei, EU-kompatibel)"""
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein gültiger API Key konfiguriert"}
    try:
        # Free USDC Balance holen
        balance = client.get_asset_balance(asset='USDC')
        free_usdc = float(balance.get('free', 0)) if balance else 0

        if free_usdc < 5:
            return {"error": f"Zu wenig USDC verfügbar ({free_usdc:.2f}). Minimum: 5 USDC"}

        # Menge bestimmen
        if request.amount and request.amount > 0:
            convert_amount = min(request.amount, free_usdc)
        else:
            convert_amount = free_usdc  # Max

        if convert_amount < 5:
            return {"error": f"Menge {convert_amount:.2f} USDC unter Minimum (5)"}

        # Schritt 1: Quote anfordern
        quote = client.convert_request_quote(
            fromAsset='USDC',
            toAsset='USDT',
            fromAmount=f"{convert_amount:.2f}"
        )

        quote_id = quote.get('quoteId')
        if not quote_id:
            error_msg = quote.get('msg', quote.get('message', str(quote)))
            return {"error": f"Convert Quote fehlgeschlagen: {error_msg}"}

        to_amount = float(quote.get('toAmount', 0))
        ratio = quote.get('ratio', '1')

        print(f"[WALLET] Convert Quote: {convert_amount:.2f} USDC → {to_amount:.2f} USDT (ratio: {ratio}, quoteId: {quote_id})")

        # Schritt 2: Quote akzeptieren
        result = client.convert_accept_quote(quoteId=quote_id)

        order_status = result.get('orderStatus', 'UNKNOWN')
        order_id = result.get('orderId', '')

        if order_status in ('SUCCESS', 'ACCEPT_SUCCESS', 'PROCESS'):
            print(f"[WALLET] User {current_user['user_id']} converted {convert_amount:.2f} USDC → {to_amount:.2f} USDT (orderId: {order_id})")
            return {
                "status": "success",
                "usdc_sold": round(convert_amount, 2),
                "usdt_received": round(to_amount, 2),
                "order_id": str(order_id),
                "ratio": ratio
            }
        else:
            return {"error": f"Convert fehlgeschlagen: Status={order_status}"}

    except BinanceAPIException as e:
        # Fallback: Spot Trade versuchen falls Convert nicht verfügbar
        if 'not authorized' in str(e.message).lower() or 'not permitted' in str(e.message).lower():
            return {"error": f"Convert API nicht verfügbar für diesen Account: {e.message}"}
        return {"error": f"Binance API Fehler: {e.message}"}
    except Exception as e:
        print(f"[WALLET] Convert error: {e}")
        return {"error": str(e)}
