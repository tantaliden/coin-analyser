"""
RL-Agent Trader — Order Execution auf Hyperliquid (bevorzugt) mit Binance Fallback.
Limit Orders, keine Market Orders.
"""
import json
import time
from pathlib import Path
from decimal import Decimal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import decrypt_value

from eth_account import Account as EthAccount
from hyperliquid.info import Info as HLInfo
from hyperliquid.exchange import Exchange as HLExchange
from hyperliquid.utils import constants as hl_constants

SETTINGS_PATH = "/opt/coin/settings.json"


def get_hl_credentials(user_id: int = 1):
    """Holt Hyperliquid API-Wallet-Secret + Main-Wallet-Adresse."""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hyperliquid_api_key_encrypted, hyperliquid_api_secret_encrypted, hyperliquid_wallet_address FROM users WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    if not row or not row["hyperliquid_wallet_address"]:
        return None
    return {
        "api_wallet": decrypt_value(row["hyperliquid_api_key_encrypted"]),
        "secret_key": decrypt_value(row["hyperliquid_api_secret_encrypted"]),
        "wallet_address": row["hyperliquid_wallet_address"],
    }


def _hl_account(creds: dict):
    """Erstellt ein eth_account.Account aus dem Secret Key für HLExchange."""
    return EthAccount.from_key(creds["secret_key"])


def get_hl_info():
    return HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True)


def get_hl_exchange(creds):
    """Erstellt eine Exchange-Instanz für Trading."""
    return HLExchange(
        wallet=None,
        base_url=hl_constants.MAINNET_API_URL,
        account_address=creds["wallet_address"],
        vault_address=None,
    )


def get_hl_balance(wallet_address: str) -> float:
    """Aktuelles Perp-Guthaben auf Hyperliquid."""
    info = get_hl_info()
    state = info.user_state(wallet_address)
    return float(state.get("marginSummary", {}).get("accountValue", 0))


def get_hl_open_positions(wallet_address: str) -> list:
    """Offene Positionen auf Hyperliquid."""
    info = get_hl_info()
    state = info.user_state(wallet_address)
    positions = []
    for asset in state.get("assetPositions", []):
        pos = asset.get("position", {})
        szi = float(pos.get("szi", 0))
        if szi != 0:
            positions.append({
                "coin": pos.get("coin"),
                "size": szi,
                "direction": "long" if szi > 0 else "short",
                "entry_price": float(pos.get("entryPx", 0)),
                "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                "leverage": pos.get("leverage", {}).get("value", 1),
            })
    return positions


def get_available_coins_hl() -> set:
    """Alle handelbaren Coins auf Hyperliquid."""
    info = get_hl_info()
    meta = info.meta()
    return {asset["name"] for asset in meta.get("universe", [])}


def calculate_position_size(balance: float, min_size: float = 25.0, max_fraction: float = 0.05) -> dict:
    """
    Position-Sizing nach Volkers Regeln:
    - Minimum: $25
    - Maximum: 1/20 (5%) des Guthabens
    - Wenn 1/20 < $25: trotzdem $25 (bis genug Kapital da ist)
    """
    max_size = balance * max_fraction
    if max_size >= min_size:
        return {"min": min_size, "max": round(max_size, 2), "balance": round(balance, 2)}
    else:
        return {"min": min_size, "max": min_size, "balance": round(balance, 2)}


def _get_hl_coin_info(coin: str) -> dict:
    """Holt HL-spezifische Coin-Info aus der DB (szDecimals, maxLeverage, priceDecimals).
    coin kann 'BTC' oder 'BTCUSDC' sein."""
    symbol = coin if coin.endswith("USDC") else coin + "USDC"
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hl_sz_decimals, hl_max_leverage, hl_price_decimals FROM coin_info WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
    if row and row["hl_sz_decimals"] is not None:
        return {
            "sz_decimals": row["hl_sz_decimals"],
            "max_leverage": row["hl_max_leverage"],
            "price_decimals": row["hl_price_decimals"],
        }
    return None


def refresh_hl_coin_info():
    """Aktualisiert HL-Metadaten (szDecimals, maxLeverage, priceDecimals) in coin_info.
    Einmal beim Service-Start aufrufen."""
    try:
        info = get_hl_info()
        meta = info.meta_and_asset_ctxs()
        universe = meta[0]["universe"]
        ctxs = meta[1]

        with get_app_db() as conn:
            with conn.cursor() as cur:
                updated = 0
                for i, asset in enumerate(universe):
                    if i >= len(ctxs):
                        break
                    mark = ctxs[i].get("markPx", "0")
                    price_dec = len(mark.split(".")[1]) if "." in mark else 0
                    symbol = asset["name"] + "USDC"
                    cur.execute(
                        "UPDATE coin_info SET hl_price_decimals = %s, hl_sz_decimals = %s, hl_max_leverage = %s WHERE symbol = %s",
                        (price_dec, asset["szDecimals"], asset["maxLeverage"], symbol),
                    )
                    if cur.rowcount > 0:
                        updated += 1
            conn.commit()
        print(f"[RL-TRADER] HL Coin-Info aktualisiert: {updated} Coins")
    except Exception as e:
        print(f"[RL-TRADER] HL Coin-Info Refresh fehlgeschlagen: {e}")


def place_limit_order_hl(creds: dict, coin: str, is_buy: bool, size_usd: float,
                         price: float, leverage: int = 1) -> dict:
    """
    Platziert eine Limit Order auf Hyperliquid.
    Returns: {"success": True, "order_id": ..., "status": ...} oder {"success": False, "error": ...}
    """
    try:
        # Coin-Info aus DB
        coin_info = _get_hl_coin_info(coin)
        if coin_info is None:
            return {"success": False, "error": f"Coin {coin} nicht in coin_info (HL-Daten fehlen)"}

        sz_decimals = coin_info["sz_decimals"]

        # Quantity berechnen aus USD-Size und Preis
        quantity = round(size_usd / price, sz_decimals)
        if quantity <= 0:
            return {"success": False, "error": f"Quantity zu klein: {quantity}"}

        # Exchange mit API-Wallet Secret
        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        # Leverage setzen (max aus DB)
        max_lev = coin_info.get("max_leverage", 5)
        leverage = min(leverage, max_lev)
        if leverage > 1:
            exchange.update_leverage(leverage, coin, is_cross=True)

        # Preis auf HL-Precision runden (aus coin_info)
        price_dec = coin_info.get("price_decimals", 5)
        price = round(price, price_dec)

        # Limit Order
        order_result = exchange.order(
            coin,
            is_buy,
            quantity,
            price,
            {"limit": {"tif": "Gtc"}},  # Good till cancelled
        )

        if order_result.get("status") == "ok":
            statuses = order_result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "resting" in statuses[0]:
                oid = statuses[0]["resting"]["oid"]
                return {"success": True, "order_id": oid, "quantity": quantity, "status": "resting"}
            elif statuses and "filled" in statuses[0]:
                fill = statuses[0]["filled"]
                return {"success": True, "order_id": fill.get("oid"), "quantity": quantity,
                        "avg_price": float(fill.get("avgPx", price)), "status": "filled"}
            else:
                print(f"[RL-TRADER] Unbekannter Order-Status: {statuses}")
                return {"success": False, "order_id": None, "quantity": quantity, "status": "unknown",
                        "error": f"Unbekannter Status: {statuses}", "raw": statuses}
        else:
            return {"success": False, "error": str(order_result)}

    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_order_hl(creds: dict, coin: str, order_id: int) -> dict:
    """Storniert eine Order auf Hyperliquid."""
    try:
        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )
        result = exchange.cancel(coin, order_id)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def close_position_hl(creds: dict, coin: str, wallet_address: str) -> dict:
    """Schließt eine offene Position komplett via Market Order."""
    try:
        info = get_hl_info()
        state = info.user_state(wallet_address)

        # Finde die Position
        position = None
        for asset in state.get("assetPositions", []):
            pos = asset.get("position", {})
            if pos.get("coin") == coin and float(pos.get("szi", 0)) != 0:
                position = pos
                break

        if not position:
            return {"success": False, "error": f"Keine offene Position für {coin}"}

        szi = float(position["szi"])
        is_buy = szi < 0  # Short schließen = kaufen, Long schließen = verkaufen
        quantity = abs(szi)

        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        # Market close via aggressiver Limit Order
        mid = info.all_mids()
        current_price = float(mid.get(coin, 0))
        if current_price == 0:
            return {"success": False, "error": f"Kein Preis für {coin}"}

        # Slippage-Toleranz: 0.5% über/unter Mid
        ci = _get_hl_coin_info(coin)
        pdec = ci["price_decimals"] if ci else 5
        close_price = round(current_price * (1.005 if is_buy else 0.995), pdec)

        result = exchange.order(
            coin, is_buy, quantity, close_price,
            {"limit": {"tif": "Ioc"}},  # Immediate or Cancel
            reduce_only=True,
        )
        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_current_prices_hl() -> dict:
    """Alle aktuellen Mid-Preise von Hyperliquid."""
    info = get_hl_info()
    mids = info.all_mids()
    return {k: float(v) for k, v in mids.items()}


def place_tp_sl_hl(creds: dict, coin: str, is_long: bool, quantity,
                   tp_price, sl_price) -> dict:
    """
    Setzt TP + SL Orders auf Hyperliquid nach Entry.
    TP = Limit Order (reduce_only), SL = Trigger Order (Stop Market).
    """
    try:
        coin_info = _get_hl_coin_info(coin)
        price_dec = coin_info["price_decimals"] if coin_info else 5
        quantity = float(quantity)
        tp_price = round(float(tp_price), price_dec)
        sl_price = round(float(sl_price), price_dec)
        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        results = {"tp": None, "sl": None}

        # TP: Limit Order auf Gegenseite (reduce_only)
        tp_is_buy = not is_long  # Long → Sell bei TP, Short → Buy bei TP
        tp_result = exchange.order(
            coin, tp_is_buy, quantity, tp_price,
            {"limit": {"tif": "Gtc"}},
            reduce_only=True,
        )
        if tp_result.get("status") == "ok":
            statuses = tp_result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "resting" in statuses[0]:
                results["tp"] = statuses[0]["resting"]["oid"]

        # SL: Trigger Order (Stop Market)
        sl_is_buy = not is_long  # Long → Buy-to-close bei SL (nein, Sell!), Short → Buy bei SL
        # Trigger: wenn Preis SL erreicht → Market Order zum Schließen
        trigger_above = not is_long  # Long: SL unter Entry → trigger wenn Preis FÄLLT (triggerPx > marketPx → isMarket=True)
        sl_result = exchange.order(
            coin, sl_is_buy, quantity, sl_price,
            {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
        )
        if sl_result.get("status") == "ok":
            statuses = sl_result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "resting" in statuses[0]:
                results["sl"] = statuses[0]["resting"]["oid"]

        return {"success": True, "tp_oid": results["tp"], "sl_oid": results["sl"]}

    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_all_orders_for_coin_hl(creds: dict, coin: str) -> dict:
    """Storniert ALLE offenen Orders für einen Coin auf Hyperliquid."""
    try:
        info = get_hl_info()
        address = creds["wallet_address"]

        # Normale Orders
        open_orders = info.open_orders(address)
        coin_orders = [o for o in open_orders if o.get("coin") == coin]

        # Auch Frontend-Orders (inkl. Trigger/TP/SL)
        try:
            frontend_orders = info.frontend_open_orders(address)
            coin_frontend = [o for o in frontend_orders if o.get("coin") == coin]
        except:
            coin_frontend = []

        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        cancelled = 0
        for order in coin_orders:
            try:
                exchange.cancel(coin, order["oid"])
                cancelled += 1
            except:
                pass

        # Trigger Orders separat canceln
        for order in coin_frontend:
            oid = order.get("oid")
            if oid and oid not in [o.get("oid") for o in coin_orders]:
                try:
                    exchange.cancel(coin, oid)
                    cancelled += 1
                except:
                    pass

        return {"success": True, "cancelled": cancelled}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== BINANCE SPOT ==========

def get_binance_client(user_id: int = 1):
    """Binance Client für Spot-Trading."""
    from binance.client import Client as BinanceClient
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT binance_api_key_encrypted, binance_api_secret_encrypted, binance_api_valid FROM users WHERE user_id = %s",
                (user_id,),
            )
            user = cur.fetchone()
    if not user or not user["binance_api_key_encrypted"] or not user["binance_api_valid"]:
        return None
    try:
        return BinanceClient(
            decrypt_value(user["binance_api_key_encrypted"]),
            decrypt_value(user["binance_api_secret_encrypted"]),
        )
    except Exception as e:
        print(f"[TRADER] Binance Client Fehler: {e}")
        return None


def get_binance_balance(user_id: int = 1) -> float:
    """Gesamtguthaben auf Binance (USDC + Positionen)."""
    client = get_binance_client(user_id)
    if not client:
        return 0
    try:
        account = client.get_account()
        total = 0
        for asset in account.get("balances", []):
            amt = float(asset["free"]) + float(asset["locked"])
            if amt > 0:
                if asset["asset"] == "USDC":
                    total += amt
                else:
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{asset['asset']}USDC")
                        total += amt * float(ticker["price"])
                    except:
                        pass
        return total
    except:
        return 0


def buy_spot_binance(symbol: str, size_usd: float, user_id: int = 1) -> dict:
    """Market Buy auf Binance Spot."""
    client = get_binance_client(user_id)
    if not client:
        return {"success": False, "error": "Kein Binance Client"}
    try:
        order = client.create_order(
            symbol=symbol, side="BUY", type="MARKET",
            quoteOrderQty=size_usd,
        )
        qty = float(order.get("executedQty", 0))
        quote = float(order.get("cummulativeQuoteQty", 0))
        avg_price = quote / qty if qty > 0 else 0
        return {
            "success": True,
            "quantity": qty,
            "avg_price": avg_price,
            "quote_amount": quote,
            "order_id": order.get("orderId"),
            "exchange": "binance",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def set_oco_binance(symbol: str, quantity: float, tp_price: float, sl_price: float,
                    user_id: int = 1) -> dict:
    """OCO Sell Order auf Binance (TP + SL in einer Order)."""
    client = get_binance_client(user_id)
    if not client:
        return {"success": False, "error": "Kein Binance Client"}
    try:
        # Precision ermitteln
        info = client.get_symbol_info(symbol)
        price_prec, qty_prec = 8, 8
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick = f["tickSize"]
                price_prec = len(tick.rstrip("0").split(".")[-1]) if "." in tick else 0
            elif f["filterType"] == "LOT_SIZE":
                step = f["stepSize"]
                qty_prec = len(step.rstrip("0").split(".")[-1]) if "." in step else 0

        qty = round(quantity, qty_prec)
        tp = round(tp_price, price_prec)
        sl = round(sl_price, price_prec)
        sl_limit = round(sl_price * 0.995, price_prec)  # Leicht unter SL

        oco = client.create_oco_order(
            symbol=symbol, side="SELL", quantity=qty,
            price=tp, stopPrice=sl, stopLimitPrice=sl_limit,
            stopLimitTimeInForce="GTC",
        )
        return {
            "success": True,
            "order_list_id": oco.get("orderListId"),
            "orders": [o.get("orderId") for o in oco.get("orders", [])],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_orders_binance(symbol: str, user_id: int = 1) -> dict:
    """Alle offenen Orders für ein Symbol auf Binance stornieren."""
    client = get_binance_client(user_id)
    if not client:
        return {"success": False, "error": "Kein Binance Client"}
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        cancelled = 0
        for order in open_orders:
            try:
                client.cancel_order(symbol=symbol, orderId=order["orderId"])
                cancelled += 1
            except:
                pass
        return {"success": True, "cancelled": cancelled}
    except Exception as e:
        return {"success": False, "error": str(e)}


def sell_market_binance(symbol: str, quantity: float, user_id: int = 1) -> dict:
    """Market Sell auf Binance Spot (für Early Exit)."""
    client = get_binance_client(user_id)
    if not client:
        return {"success": False, "error": "Kein Binance Client"}
    try:
        # Precision
        info = client.get_symbol_info(symbol)
        qty_prec = 8
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = f["stepSize"]
                qty_prec = len(step.rstrip("0").split(".")[-1]) if "." in step else 0

        qty = round(quantity, qty_prec)
        order = client.create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qty)
        quote = float(order.get("cummulativeQuoteQty", 0))
        avg_price = quote / qty if qty > 0 else 0
        return {
            "success": True,
            "quantity": qty,
            "avg_price": avg_price,
            "quote_amount": quote,
            "order_id": order.get("orderId"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_binance_position(symbol: str, user_id: int = 1) -> dict:
    """Prüft ob eine Binance Spot Position für ein Symbol existiert."""
    client = get_binance_client(user_id)
    if not client:
        return None
    try:
        base = symbol.replace("USDC", "").replace("USDT", "")
        balance = client.get_asset_balance(asset=base)
        qty = float(balance.get("free", 0)) + float(balance.get("locked", 0))
        if qty > 0:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            return {"quantity": qty, "current_price": price, "value": qty * price}
    except:
        pass
    return None
