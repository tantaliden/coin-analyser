"""WALLET ROUTES - Binance Account Integration"""
from pathlib import Path
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
    except:
        return None

@router.get("/status")
async def get_wallet_status(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT binance_api_valid FROM users WHERE user_id = %s", (current_user['user_id'],))
            user = cur.fetchone()
    return {"configured": user and user['binance_api_valid']}

@router.get("/balance")
async def get_wallet_balance(current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein API Key"}
    try:
        account = client.get_account()
        usdt_balance, positions_value = 0.0, 0.0
        for asset in account.get('balances', []):
            total = float(asset['free']) + float(asset['locked'])
            if total > 0:
                if asset['asset'] == 'USDT':
                    usdt_balance = total
                elif asset['asset'] != 'USDC':
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{asset['asset']}USDT")
                        positions_value += total * float(ticker['price'])
                    except:
                        pass
        return {"usdt_balance": round(usdt_balance, 2), "positions_value": round(positions_value, 2), "total_portfolio": round(usdt_balance + positions_value, 2)}
    except BinanceAPIException as e:
        return {"error": e.message}

@router.get("/positions")
async def get_wallet_positions(current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein API Key"}
    try:
        account = client.get_account()
        positions = []
        for asset in account.get('balances', []):
            total = float(asset['free']) + float(asset['locked'])
            if total > 0 and asset['asset'] not in ['USDT', 'USDC']:
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{asset['asset']}USDT")
                    price = float(ticker['price'])
                    if total * price >= 1.0:
                        positions.append({"asset": asset['asset'], "quantity": total, "current_price": price, "value_usdt": round(total * price, 2)})
                except:
                    pass
        return {"positions": sorted(positions, key=lambda x: -x['value_usdt'])}
    except BinanceAPIException as e:
        return {"error": e.message}

@router.get("/orders")
async def get_wallet_orders(current_user: dict = Depends(get_current_user)):
    client = get_user_binance_client(current_user['user_id'])
    if not client:
        return {"error": "Kein API Key"}
    try:
        return {"orders": [{"order_id": o['orderId'], "symbol": o['symbol'], "side": o['side'], "price": float(o['price']) if o['price'] else None, "quantity": float(o['origQty'])} for o in client.get_open_orders()]}
    except BinanceAPIException as e:
        return {"error": e.message}

@router.get("/history")
async def get_trade_history(days: int = 7, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM trade_history WHERE user_id = %s AND executed_at >= NOW() - INTERVAL '%s days' ORDER BY executed_at DESC LIMIT 100", (current_user['user_id'], days))
            return {"trades": [dict(row) for row in cur.fetchall()]}
