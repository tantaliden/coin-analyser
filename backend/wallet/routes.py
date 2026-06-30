"""WALLET ROUTES - Binance + Hyperliquid Account Integration

HL-Daten kommen primaer aus dem WS-Live-State (hl_ws_state.py) — sub-200ms
Latenz, kein Rate-Limit. Bei stale WS-Daten Fallback auf REST mit Stale-Cache.
"""
import json
import time
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from hyperliquid.info import Info as HLInfo
from hyperliquid.utils import constants as hl_constants
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db, get_coins_db
from auth.auth import get_current_user, decrypt_value
from wallet.hl_ws_state import get_ws_state, ensure_started

router = APIRouter(prefix="/api/v1/wallet", tags=["wallet"])

# REST-Fallback-Caches — TTLs werden bei jedem Cache-Lookup live aus settings.json
# gelesen. So sind Frontend-Änderungen sofort wirksam, OHNE Service-Restart.
# KEIN Default — fehlt der Wert raise (keine silent fallbacks).
import logging
_wallet_log = logging.getLogger("wallet")

def _wallet_cfg() -> dict:
    with open('/opt/coin/settings.json') as f:
        return json.load(f).get('wallet', {})

def _get_state_ttl() -> float:
    v = _wallet_cfg().get('hl_cache_ttl_seconds')
    if v is None:
        _wallet_log.error("FALLBACK_TRIGGERED wallet.routes: wallet.hl_cache_ttl_seconds fehlt -> raise")
        raise RuntimeError("settings.wallet.hl_cache_ttl_seconds missing")
    return float(v)

def _get_fills_ttl() -> float:
    v = _wallet_cfg().get('hl_fills_ttl_seconds')
    if v is None:
        _wallet_log.error("FALLBACK_TRIGGERED wallet.routes: wallet.hl_fills_ttl_seconds fehlt -> raise")
        raise RuntimeError("settings.wallet.hl_fills_ttl_seconds missing")
    return float(v)

_hl_user_state_cache: dict = {}
_hl_open_orders_cache: dict = {}
_hl_spot_state_cache: dict = {}
_hl_user_fills_cache: dict = {}


def _cached_user_state(address: str) -> dict:
    """Primary: WS-Live-State. Fallback: REST mit Stale-Cache."""
    ensure_started(address)
    ws_data = get_ws_state().get_user_state_view()
    if ws_data is not None:
        return ws_data
    # Fallback: REST
    now = time.time()
    c = _hl_user_state_cache.get(address)
    if c and now - c["ts"] < _get_state_ttl():
        return c["data"]
    try:
        data = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True).user_state(address)
        _hl_user_state_cache[address] = {"ts": now, "data": data}
        return data
    except Exception as e:
        print(f"[WALLET-HL] user_state REST fallback failed: {e}")
        if c:
            return c["data"]
        raise


def _cached_open_orders(address: str) -> list:
    """Primary: WS-Live-State (openOrders aus webData2). Fallback: REST."""
    ensure_started(address)
    ws_data = get_ws_state().get_open_orders_view()
    if ws_data is not None:
        return ws_data
    now = time.time()
    c = _hl_open_orders_cache.get(address)
    if c and now - c["ts"] < _get_state_ttl():
        return c["data"]
    try:
        data = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True).open_orders(address)
        _hl_open_orders_cache[address] = {"ts": now, "data": data}
        return data
    except Exception as e:
        print(f"[WALLET-HL] open_orders REST fallback failed: {e}")
        if c:
            return c["data"]
        raise


def _cached_spot_user_state(address: str) -> dict:
    """Primary: WS-Live-State. Fallback: REST."""
    ensure_started(address)
    ws_data = get_ws_state().get_spot_state_view()
    if ws_data is not None:
        return ws_data
    now = time.time()
    c = _hl_spot_state_cache.get(address)
    if c and now - c["ts"] < _get_state_ttl():
        return c["data"]
    try:
        data = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True).spot_user_state(address)
        _hl_spot_state_cache[address] = {"ts": now, "data": data}
        return data
    except Exception as e:
        print(f"[WALLET-HL] spot_user_state REST fallback failed: {e}")
        if c:
            return c["data"]
        return {}


def _cached_user_fills(address: str) -> list:
    """Primary: WS-Live-State (userFills push). Fallback: REST."""
    ensure_started(address)
    ws = get_ws_state()
    if ws.is_fills_fresh():
        fills = ws.get_fills()
        if fills:
            return fills
    # Fallback: REST (besonders nach Server-Start solange WS noch keinen Push hatte)
    now = time.time()
    c = _hl_user_fills_cache.get(address)
    if c and now - c["ts"] < _get_fills_ttl():
        return c["data"]
    try:
        data = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True).user_fills(address)
        _hl_user_fills_cache[address] = {"ts": now, "data": data}
        return data
    except Exception as e:
        print(f"[WALLET-HL] user_fills REST fallback failed: {e}")
        if c:
            return c["data"]
        return []


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


@router.get("/config")
async def get_wallet_config(current_user: dict = Depends(get_current_user)):
    """Liefert den 'wallet'-Block aus settings.json — für UI-Settings."""
    with open('/opt/coin/settings.json') as f:
        return json.load(f).get('wallet', {})


class WalletConfigUpdate(BaseModel):
    config: dict


@router.put("/config")
async def put_wallet_config(payload: WalletConfigUpdate,
                              current_user: dict = Depends(get_current_user)):
    """Ersetzt den 'wallet'-Block in settings.json. Live-Reload — wirkt sofort."""
    settings_path = '/opt/coin/settings.json'
    with open(settings_path) as f:
        s = json.load(f)
    s['wallet'] = payload.config
    with open(settings_path + '.tmp', 'w') as f:
        json.dump(s, f, indent=2)
    Path(settings_path + '.tmp').rename(settings_path)
    return {"status": "ok"}


@router.get("/status")
async def get_wallet_status(current_user: dict = Depends(get_current_user)):
    # Binance deaktiviert — nur HL aktiv
    return {"configured": False}


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
def get_wallet_positions(current_user: dict = Depends(get_current_user)):
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
def get_trade_history(days: int = 30, limit: int = 500, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    from datetime import datetime, timezone
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    trades = []

    # 1) HL-Fills — nur Close-Fills (closedPnl != 0) als History-Eintraege.
    #    Open-Fills haben definitionsgemaess keinen Verkaufswert (Position ist noch offen).
    # Fragmente mit gleicher oid werden zu einem Trade aggregiert (Fix 10.05.2026).
    # Hebel-Lookup mit 4-Stufen-Fallback (Fix 10.05.2026: trader_positions als 2. Stufe):
    #   1. open_predictions.effective_leverage (Predictor-Welt)
    #   2. trader_positions.leverage (Trader-Welt) — definitiver Hebel pro Trade
    #   3. Aktuell offene HL-Position fuer den Coin (manuelle Trades)
    #   4. settings.predictor.trading.default_leverage
    pred_index = []
    trader_index = []
    try:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol, created_at,
                           COALESCE(closed_at, NOW() + INTERVAL '30 seconds') AS end_at,
                           effective_leverage
                    FROM open_predictions
                    WHERE created_at >= NOW() - (%s || ' days')::interval
                      AND effective_leverage IS NOT NULL
                    ORDER BY created_at DESC
                """, (days,))
                pred_index = cur.fetchall()
                cur.execute("""
                    SELECT symbol, opened_at,
                           COALESCE(closed_at, NOW() + INTERVAL '30 seconds') AS end_at,
                           leverage
                    FROM trader_positions
                    WHERE opened_at >= NOW() - (%s || ' days')::interval
                      AND leverage IS NOT NULL
                    ORDER BY opened_at DESC
                """, (days,))
                trader_index = cur.fetchall()
    except Exception:
        pred_index = []
        trader_index = []

    # Aktuelle HL-Positionen als Hebel-Quelle fuer manuelle Trades, die nicht
    # ueber den Predictor liefen (also keine open_predictions-Row haben).
    hl_position_levs = {}
    try:
        hl_address_for_lookup = get_user_hl_address(user_id)
        if hl_address_for_lookup:
            state_lookup = _cached_user_state(hl_address_for_lookup)
            for ap in state_lookup.get("assetPositions", []):
                p = ap.get("position", {})
                coin = p.get("coin")
                lev_v = p.get("leverage", {}).get("value")
                if coin and lev_v:
                    hl_position_levs[coin] = int(lev_v)
    except Exception:
        pass

    # Settings-Default als finaler Fallback (Volker tradet i.d.R. mit 5x).
    try:
        with open("/opt/coin/settings.json") as fp:
            _s = json.load(fp)
        default_lev = int(_s.get("predictor", {}).get("trading", {}).get("default_leverage", 5))
    except Exception:
        default_lev = 5

    # Coin-spezifischer HL-Max-Leverage aus coins.hl_meta (TimescaleDB) als
    # finaler Cap. HL akzeptiert nicht jeden Hebel fuer jeden Coin (POPCAT/AERO
    # = 3x max, BTC = 40x). Wenn DB/Default-Wert ueber dem HL-Max liegt, war
    # der Trade in Realitaet auf hl_max gekappt -> Anzeige korrigieren.
    hl_max_lev = {}
    try:
        with get_coins_db() as cconn:
            with cconn.cursor() as ccur:
                ccur.execute("SELECT symbol, max_leverage FROM hl_meta WHERE max_leverage IS NOT NULL")
                hl_max_lev = {r['symbol']: int(r['max_leverage']) for r in ccur.fetchall()}
    except Exception as e:
        print(f"[WALLET-HL] hl_meta max_leverage lookup failed: {e}")

    def _lookup_leverage(coin, fill_dt):
        from datetime import timedelta
        lev = default_lev
        # 1. open_predictions (Predictor-Welt)
        for sym, start, end, l in pred_index:
            if sym == coin and start <= fill_dt <= end + timedelta(seconds=30):
                lev = int(l) if l else 1
                break
        else:
            # 2. trader_positions (Trader-Welt) — fix fuer Faelle wo open_predictions
            #    den Trade nicht im Window hat (z.B. Predictor closed schon gewatcht
            #    aber Fragment-Close kam nach end_at). trader_positions ist Auto-Trade-Welt.
            for sym, start, end, l in trader_index:
                if sym == coin and start <= fill_dt <= end + timedelta(seconds=30):
                    lev = int(l) if l else 1
                    break
            else:
                # 3. Aktuelle HL-Position fuer diesen Coin (manuelle Trades, noch offen)
                if coin in hl_position_levs:
                    lev = hl_position_levs[coin]
                # 4. sonst bleibt default_lev
        # Final-Cap gegen coin-spezifisches HL-Maximum (stetig statt hardcoded).
        cap = hl_max_lev.get(coin)
        return min(lev, cap) if cap else lev

    hl_address = get_user_hl_address(user_id)
    if hl_address:
        from collections import defaultdict
        fills = _cached_user_fills(hl_address)
        # Fragment-Aggregation (Fix 10.05.2026): HL fuellt Orders manchmal in mehreren
        # Stuecken gegen Order-Book-Levels. Pro oid wird nur EIN History-Eintrag erzeugt
        # mit aufsummierter qty / quote / pnl / fee und volume-gewichtetem avg-Preis.
        groups = defaultdict(list)
        for f in fills:
            t_ms = int(f.get("time", 0))
            if t_ms < cutoff_ms:
                continue
            if float(f.get("closedPnl", 0) or 0) == 0.0:
                continue  # Open-Fill, nicht in History
            oid = f.get("oid")
            if oid is None:
                # Sehr seltener Fall — als Einzeltrade behalten (eindeutiger Key per t_ms+coin)
                oid = f"_solo_{t_ms}_{f.get('coin','')}"
            groups[oid].append(f)

        for oid, frags in groups.items():
            sz_total = sum(float(f.get("sz", 0)) for f in frags)
            quote_total = sum(float(f.get("sz", 0)) * float(f.get("px", 0)) for f in frags)
            pnl_total = sum(float(f.get("closedPnl", 0) or 0) for f in frags)
            fee_total = sum(float(f.get("fee", 0) or 0) for f in frags)
            avg_px = (quote_total / sz_total) if sz_total > 0 else 0.0
            t_ms_min = min(int(f.get("time", 0)) for f in frags)
            fill_dt = datetime.fromtimestamp(t_ms_min / 1000, tz=timezone.utc)
            coin = frags[0].get("coin", "")
            side_raw = frags[0].get("side", "")
            side = "buy" if side_raw == "B" else "sell"
            d = frags[0].get("dir", "")
            sold_for = quote_total + pnl_total
            lev = _lookup_leverage(coin, fill_dt)
            margin = quote_total / lev if lev > 0 else quote_total
            pnl_pct = (pnl_total / margin * 100.0) if margin > 0 else 0.0
            if "Close Long" in d or "Long >" in d:
                position_side = "long"
            elif "Close Short" in d or "Short >" in d:
                position_side = "short"
            else:
                position_side = side
            trades.append({
                "id": str(oid),
                "symbol": coin,
                "side": position_side,
                "direction": d,
                "price": avg_px,
                "quantity": sz_total,
                "quote_amount": round(quote_total, 2),
                "sold_for": round(sold_for, 2),
                "pnl_usd": round(pnl_total, 4),
                "pnl_percent": round(pnl_pct, 2),
                "leverage": lev,
                "margin_usd": round(margin, 2),
                "fee": round(fee_total, 4),
                "exit_reason": d,
                "exchange": "hyperliquid",
                "is_bot_trade": False,
                "source": "hl_fills",
                "executed_at": fill_dt.isoformat(),
                "hash": frags[0].get("hash"),
                "n_fragments": len(frags),
            })

    with get_app_db() as conn:
        with conn.cursor() as cur:
            # RL-Agent Positions (closed) als History
            cur.execute("""
                SELECT id, symbol, direction, leverage, entry_price, exit_price,
                       position_size_usd, pnl_percent, pnl_usd, exit_reason,
                       exchange, entry_time, exit_time, duration_minutes
                FROM rl_positions
                WHERE status = 'closed' AND exit_time >= NOW() - INTERVAL '%s days'
                ORDER BY exit_time DESC LIMIT %s
            """, (days, limit))
            rl_trades = cur.fetchall()

            # Alte trade_history (Binance Rocket-Button) dazu
            cur.execute("""
                SELECT id, symbol, side, price, quantity, quote_amount,
                       is_bot_trade, indicator_set_name, indicator_set_accuracy, executed_at
                FROM trade_history
                WHERE user_id = %s AND executed_at >= NOW() - INTERVAL '%s days'
                ORDER BY executed_at DESC LIMIT 100
            """, (user_id, days))
            old_trades = cur.fetchall()

    # RL-Positions als Trades formatieren
    for t in rl_trades:
        size = float(t['position_size_usd']) if t['position_size_usd'] else 0
        pnl = float(t['pnl_usd']) if t['pnl_usd'] else 0
        trades.append({
            "id": t['id'], "symbol": t['symbol'],
            "side": t['direction'],
            "price": float(t['entry_price']) if t['entry_price'] else 0,
            "exit_price": float(t['exit_price']) if t['exit_price'] else None,
            "quantity": 0,
            "quote_amount": round(size, 2),
            "sold_for": round(size + pnl, 2),
            "leverage": t['leverage'],
            "pnl_percent": round(pnl / size * 100, 2) if size > 0 else 0,
            "pnl_usd": pnl,
            "exit_reason": t['exit_reason'],
            "exchange": t['exchange'] or 'hyperliquid',
            "is_bot_trade": True,
            "source": "rl_agent",
            "executed_at": t['exit_time'].isoformat() if t['exit_time'] else None,
            "entry_at": t['entry_time'].isoformat() if t['entry_time'] else None,
            "duration_minutes": t['duration_minutes'],
        })
    # Alte Trades
    for t in old_trades:
        trades.append({
            "id": t['id'], "symbol": t['symbol'], "side": t['side'],
            "price": float(t['price']), "quantity": float(t['quantity']),
            "quote_amount": float(t['quote_amount']),
            "is_bot_trade": t['is_bot_trade'],
            "source": "binance",
            "executed_at": t['executed_at'].isoformat() if t['executed_at'] else None
        })
    # Nach Datum sortieren
    trades.sort(key=lambda x: x.get('executed_at') or '', reverse=True)
    return {"trades": trades[:limit]}


@router.get("/realized-pnl")
async def get_realized_pnl(days: int = 7, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    cutoff_ms = int((time.time() - days * 86400) * 1000)

    # HL realized PnL aus user_fills (closedPnl je Close-Trade, abzueglich fees)
    hl_pnl = 0.0
    hl_fees = 0.0
    hl_trades_count = 0
    hl_address = get_user_hl_address(user_id)
    if hl_address:
        for f in _cached_user_fills(hl_address):
            t_ms = int(f.get("time", 0))
            if t_ms < cutoff_ms:
                continue
            hl_pnl += float(f.get("closedPnl", 0) or 0)
            hl_fees += float(f.get("fee", 0) or 0)
            hl_trades_count += 1

    return {
        "realized_pnl": round(hl_pnl - hl_fees, 2),
        "gross_pnl": round(hl_pnl, 2),
        "total_fees": round(hl_fees, 4),
        "trades_count": hl_trades_count,
        "period_days": days,
        "exchange": "hyperliquid"
    }



class ConvertRequest(BaseModel):
    amount: Optional[float] = None  # None = max


@router.post("/convert-usdc")
def convert_usdc_to_usdt(request: ConvertRequest, current_user: dict = Depends(get_current_user)):
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


# ========== HYPERLIQUID ==========

def get_user_hl_address(user_id: int):
    """Wallet-Adresse für Read-Zugriff. Sub-Account (vault_address) hat Vorrang."""
    try:
        with open("/opt/coin/settings.json") as f:
            s = json.load(f)
        vault = s.get("hyperliquid", {}).get("vault_address")
        if vault:
            return vault
    except Exception:
        pass
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT hyperliquid_wallet_address, hyperliquid_api_valid FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
    if not user or not user['hyperliquid_wallet_address'] or not user['hyperliquid_api_valid']:
        return None
    return user['hyperliquid_wallet_address']


def get_hl_info():
    return HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True)


@router.get("/hl/status")
async def get_hl_status(current_user: dict = Depends(get_current_user)):
    address = get_user_hl_address(current_user['user_id'])
    return {"configured": address is not None}


@router.get("/hl/balance")
async def get_hl_balance(current_user: dict = Depends(get_current_user)):
    address = get_user_hl_address(current_user['user_id'])
    if not address:
        return {"error": "Kein gültiger Hyperliquid Key konfiguriert"}
    try:
        state = _cached_user_state(address)
        margin = state.get("marginSummary", {})
        account_value = float(margin.get("accountValue", 0))
        margin_used = float(margin.get("totalMarginUsed", 0))
        notional_pos = float(margin.get("totalNtlPos", 0))
        withdrawable = float(state.get("withdrawable", 0))
        # Spot-Balance zusaetzlich (USDC auf Spot-Account) — eigener Cache
        # spot_total = alle USDC im Wallet, spot_hold = davon als Perp-Margin gebunden
        spot_usdc = 0.0
        spot_hold = 0.0
        spot = _cached_spot_user_state(address)
        for b in spot.get("balances", []):
            if b.get("coin") == "USDC":
                spot_usdc = float(b.get("total", 0))
                spot_hold = float(b.get("hold", 0))
                break
        spot_free = max(0.0, spot_usdc - spot_hold)
        # equity = Perp accountValue (inkl. uPnL) + freier Spot. Vorher addierte
        # die Formel spot_total und zaehlte damit spot_hold (= Perp-Margin) doppelt.
        equity = account_value + spot_free
        available = max(withdrawable, spot_free)
        return {
            "account_value": round(account_value, 2),
            "margin_used": round(margin_used, 2),
            "notional_positions": round(notional_pos, 2),
            "withdrawable": round(withdrawable, 2),
            "spot_usdc": round(spot_usdc, 2),
            "total_combined": round(equity, 2),
            "equity": round(equity, 2),
            "available": round(available, 2),
        }
    except Exception as e:
        print(f"[WALLET-HL] Error getting balance: {e}")
        return {"error": str(e)}


@router.get("/hl/positions")
def get_hl_positions(current_user: dict = Depends(get_current_user)):
    address = get_user_hl_address(current_user['user_id'])
    if not address:
        return {"error": "Kein gültiger Hyperliquid Key konfiguriert"}
    try:
        state = _cached_user_state(address)
        # Predictor-DB-Lookup: peak/trough fuer offene Predictions
        # → werden bei matching coin+side an die HL-Position angehaengt.
        pred_lookup = {}  # key: (symbol, side) -> {peak_pct, trough_pct, tp_px, sl_px, entry_px}
        try:
            with get_app_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, side, entry_px, peak_px, trough_px, tp_px, sl_px
                        FROM open_predictions
                        WHERE status='open' AND peak_px IS NOT NULL AND trough_px IS NOT NULL
                    """)
                    for r in cur.fetchall():
                        sym = r['symbol']; side = r['side']
                        entry = float(r['entry_px'])
                        if entry <= 0: continue
                        peak = float(r['peak_px']); trough = float(r['trough_px'])
                        # Profit-Richtung: long = peak (oben), short = trough (unten)
                        if side == 'long':
                            peak_pct = (peak - entry) / entry * 100.0
                            trough_pct = (trough - entry) / entry * 100.0
                        else:
                            peak_pct = (entry - trough) / entry * 100.0
                            trough_pct = (entry - peak) / entry * 100.0
                        pred_lookup[(sym, side)] = {
                            'peak_pct': round(peak_pct, 2),
                            'trough_pct': round(trough_pct, 2),
                            'tp_px': float(r['tp_px']) if r['tp_px'] else None,
                            'sl_px': float(r['sl_px']) if r['sl_px'] else None,
                        }
        except Exception as e:
            print(f"[WALLET-HL] predictor lookup failed (peak/trough): {e}")

        # Trader-DB Lookup: timeout_enabled + opened_at fuer offene HL-Positionen
        trader_lookup = {}  # key: (symbol, side) -> {timeout_enabled, opened_at}
        try:
            with get_app_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, side, timeout_enabled, opened_at
                        FROM trader_positions WHERE status='open'
                    """)
                    for r in cur.fetchall():
                        trader_lookup[(r['symbol'], r['side'])] = {
                            'timeout_enabled': bool(r['timeout_enabled']),
                            'opened_at': r['opened_at'].isoformat() if r['opened_at'] else None,
                        }
        except Exception as e:
            print(f"[WALLET-HL] trader lookup failed: {e}")

        # Settings-Timeout-Hours fuer Frontend (kein Hardcode)
        try:
            with open("/opt/coin/settings.json") as fp:
                _s = json.load(fp)
            timeout_hours = float(_s.get("predictor", {}).get("bandit", {}).get("timeout_hours", 2))
        except Exception:
            timeout_hours = 2.0

        positions = []
        for asset in state.get("assetPositions", []):
            pos = asset.get("position", {})
            szi = float(pos.get("szi", 0))
            if szi == 0:
                continue
            entry_px = float(pos.get("entryPx", 0))
            position_value = float(pos.get("positionValue", 0))
            unrealized_pnl = float(pos.get("unrealizedPnl", 0))
            leverage_info = pos.get("leverage", {})
            leverage = leverage_info.get("value", 1)
            leverage_type = leverage_info.get("type", "cross")
            liquidation_px = pos.get("liquidationPx")
            margin_used = float(pos.get("marginUsed", 0))
            roe = float(pos.get("returnOnEquity", 0))
            current_price = position_value / abs(szi) if abs(szi) > 0 else 0
            coin = pos.get("coin", "?")
            direction = "long" if szi > 0 else "short"
            pred_data = pred_lookup.get((coin, direction), {})
            peak_pct = pred_data.get("peak_pct")
            trough_pct = pred_data.get("trough_pct")
            # Fallback: HL-Position ohne Predictor-Row -> peak/trough aus klines+fills
            if peak_pct is None or trough_pct is None:
                try:
                    fills = _cached_user_fills(address)
                    target_dir = f"Open {'Long' if direction == 'long' else 'Short'}"
                    open_ts_ms = None
                    for fl in fills:  # neueste zuerst
                        if fl.get('coin') == coin and fl.get('dir') == target_dir:
                            open_ts_ms = int(fl.get('time', 0))
                            break
                    if open_ts_ms and entry_px > 0:
                        with get_coins_db() as cconn:
                            with cconn.cursor() as ccur:
                                ccur.execute("""
                                    SELECT MAX(high) AS mh, MIN(low) AS ml
                                    FROM klines
                                    WHERE symbol=%s AND interval='10s'
                                      AND open_time >= to_timestamp(%s/1000.0)
                                """, (coin, open_ts_ms))
                                row = ccur.fetchone()
                        if row and row["mh"] is not None and row["ml"] is not None:
                            mh = float(row["mh"]); ml = float(row["ml"])
                            if direction == 'long':
                                peak_pct = round((mh - entry_px) / entry_px * 100.0, 2)
                                trough_pct = round((ml - entry_px) / entry_px * 100.0, 2)
                            else:
                                peak_pct = round((entry_px - ml) / entry_px * 100.0, 2)
                                trough_pct = round((entry_px - mh) / entry_px * 100.0, 2)
                except Exception as e:
                    print(f"[WALLET-HL] peak/trough fallback {coin}/{direction} failed: {e}")
            t_info = trader_lookup.get((coin, direction), {})
            positions.append({
                "coin": coin,
                "direction": direction,
                "size": abs(szi),
                "entry_price": entry_px,
                "current_price": round(current_price, 6),
                "position_value": round(position_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 4),
                "roe_percent": round(roe * 100, 2),
                "leverage": leverage,
                "leverage_type": leverage_type,
                "liquidation_price": float(liquidation_px) if liquidation_px else None,
                "margin_used": round(margin_used, 2),
                # peak/trough aus Predictor-DB oder Fallback aus klines
                "peak_pct": peak_pct,
                "trough_pct": trough_pct,
                "predictor_tp": pred_data.get("tp_px"),
                "predictor_sl": pred_data.get("sl_px"),
                # Trader-Welt: Timeout-Status pro Position + Opened-Zeitpunkt
                "timeout_enabled": t_info.get('timeout_enabled', None),
                "opened_at": t_info.get('opened_at', None),
            })
        positions.sort(key=lambda x: -abs(x['position_value']))
        return {"positions": positions, "timeout_hours": timeout_hours}
    except Exception as e:
        print(f"[WALLET-HL] Error getting positions: {e}")
        return {"error": str(e)}


class HLTimeoutToggle(BaseModel):
    enabled: bool


@router.post("/hl/positions/{coin}/{side}/timeout")
def toggle_hl_position_timeout(coin: str, side: str, body: HLTimeoutToggle,
                                       current_user: dict = Depends(get_current_user)):
    """Toggelt timeout_enabled der offenen trader_positions-Row fuer (coin, side).
    Trader-Welt only — keine Auswirkung auf open_predictions oder HL-Orders."""
    if side not in ('long', 'short'):
        return {"error": f"invalid side: {side}"}
    try:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trader_positions
                    SET timeout_enabled=%s
                    WHERE symbol=%s AND side=%s AND status='open'
                """, (bool(body.enabled), coin, side))
                n = cur.rowcount
            conn.commit()
        return {"ok": True, "updated": n, "timeout_enabled": bool(body.enabled)}
    except Exception as e:
        print(f"[WALLET-HL] timeout toggle {coin}/{side} failed: {e}")
        return {"error": str(e)}


@router.get("/hl/orders")
async def get_hl_orders(current_user: dict = Depends(get_current_user)):
    address = get_user_hl_address(current_user['user_id'])
    if not address:
        return {"error": "Kein gültiger Hyperliquid Key konfiguriert"}
    try:
        open_orders = _cached_open_orders(address)
        orders = []
        for order in open_orders:
            orders.append({
                "order_id": order.get("oid"),
                "coin": order.get("coin", "?"),
                "side": "BUY" if order.get("side") == "B" else "SELL",
                "price": float(order.get("limitPx", 0)),
                "size": float(order.get("sz", 0)),
                "timestamp": order.get("timestamp")
            })
        return {"orders": orders}
    except Exception as e:
        print(f"[WALLET-HL] Error getting orders: {e}")
        return {"error": str(e)}


class HLCloseRequest(BaseModel):
    coins: list[str]


class HLUpdateTPRequest(BaseModel):
    coins: list[str]
    tp_pct: float


@router.get("/hl/quick-tp-percentages")
async def get_quick_tp_percentages(current_user: dict = Depends(get_current_user)):
    """Liefert die konfigurierten Quick-TP-Werte aus settings.json."""
    try:
        cfg = json.load(open('/opt/coin/settings.json'))
        return {"values": cfg.get('wallet', {}).get('quick_tp_percentages', [0.5, 1.0, 2.0, 3.0])}
    except Exception as e:
        return {"values": [0.5, 1.0, 2.0, 3.0], "error": str(e)}


@router.put("/hl/quick-tp-percentages")
async def set_quick_tp_percentages(request: dict, current_user: dict = Depends(get_current_user)):
    """Speichert die Quick-TP-Werte in settings.json."""
    try:
        values = request.get('values', [])
        cleaned = sorted({round(float(v), 2) for v in values if float(v) > 0})
        path = '/opt/coin/settings.json'
        cfg = json.load(open(path))
        cfg.setdefault('wallet', {})['quick_tp_percentages'] = cleaned
        with open(path, 'w') as fp:
            json.dump(cfg, fp, indent=2, ensure_ascii=False)
        return {"success": True, "values": cleaned}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/hl/update-tp")
async def update_hl_tp(request: HLUpdateTPRequest, current_user: dict = Depends(get_current_user)):
    """Setzt einheitlichen TP (% vor Hebel = Coin-%-Bewegung) auf markierte HL-Positionen.
    Aendert NUR HL-Orders, DB-Predictor-Rows bleiben unberuehrt (Lernautonomie)."""
    from rl_agent.trader import get_hl_credentials, get_hl_open_positions, update_tp_only_hl

    creds = get_hl_credentials()
    positions = get_hl_open_positions(creds['wallet_address'])
    pos_map = {p['coin']: p for p in positions}
    pct = float(request.tp_pct)
    results = []

    for coin in request.coins:
        pos = pos_map.get(coin)
        if not pos:
            results.append({"coin": coin, "success": False, "error": "no open position"})
            continue
        try:
            entry = float(pos['entry_price'])
            is_long = pos['direction'] == 'long'
            new_tp = entry * (1 + pct / 100.0) if is_long else entry * (1 - pct / 100.0)
            r = update_tp_only_hl(creds, coin, is_long, pos['size'], new_tp)
            results.append({"coin": coin, "success": r.get('success', False),
                             "tp_price": r.get('tp_price'), "cancelled": r.get('cancelled'),
                             "error": r.get('error')})
        except Exception as e:
            results.append({"coin": coin, "success": False, "error": str(e)})

    return {"results": results}


@router.post("/hl/close")
def close_hl_positions(request: HLCloseRequest, current_user: dict = Depends(get_current_user)):
    """Schließt ausgewählte HL-Positionen manuell — alle Coins parallel.

    Nutzt safe_close_position_hl (10.05.2026): close + cancel_all_orders + verify.
    Damit bleiben nach manuellem Close keine TP/SL-Phantom-Orders zurueck.
    """
    import sys
    if '/opt/coin/backend' not in sys.path:
        sys.path.insert(0, '/opt/coin/backend')
    from services.predictor_service import safe_close_position_hl
    from rl_agent.trader import get_hl_credentials
    import concurrent.futures

    creds = get_hl_credentials()
    wallet = creds['wallet_address']

    def _close_one(coin):
        try:
            result = safe_close_position_hl(creds, coin, wallet)
            return coin, {"coin": coin, "success": result.get("success", False),
                          "price": result.get("avg_price"),
                          "orders_cancelled": result.get("orders_cancelled", 0),
                          "error": result.get("error") if not result.get("success") else None}
        except Exception as e:
            return coin, {"coin": coin, "success": False, "error": str(e)}

    # Alle Closes parallel statt sequentiell
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(request.coins) or 1, 8)) as ex:
        futures = {ex.submit(_close_one, coin): coin for coin in request.coins}
        for fut in concurrent.futures.as_completed(futures):
            try:
                _, res = fut.result(timeout=15)
                results.append(res)
            except Exception as e:
                coin = futures[fut]
                results.append({"coin": coin, "success": False, "error": f"timeout/exception: {e}"})

    # DB-Updates fuer erfolgreiche Closes (sequentiell, ist eh schnell)
    with get_app_db() as conn:
        with conn.cursor() as cur:
            for r in results:
                if not r.get("success"):
                    continue
                coin = r["coin"]
                exit_price = r.get("price") or 0
                cur.execute("""
                    SELECT id, entry_price, direction, position_size_usd
                    FROM rl_positions WHERE symbol = %s AND status = 'open'
                """, (coin + "USDC",))
                pos = cur.fetchone()
                pnl_pct = None; pnl_usd = None
                if pos and exit_price:
                    ep = float(pos['entry_price'])
                    if ep > 0:
                        pnl_pct = ((float(exit_price) - ep) / ep * 100) if pos['direction'] == 'long' else ((ep - float(exit_price)) / ep * 100)
                        size = float(pos['position_size_usd'] or 20)
                        pnl_usd = size * pnl_pct / 100
                cur.execute("""
                    UPDATE rl_positions SET status = 'closed', exit_reason = 'manual_close',
                           exit_time = NOW(), exit_price = %s, pnl_percent = %s,
                           pnl_usd = %s, duration_minutes = EXTRACT(EPOCH FROM (NOW() - entry_time))::int / 60
                    WHERE symbol = %s AND status = 'open'
                """, (exit_price, pnl_pct, pnl_usd, coin + "USDC"))
        conn.commit()

    return {"results": results}
