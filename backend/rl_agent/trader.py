"""
RL-Agent Trader — Order Execution auf Hyperliquid (bevorzugt) mit Binance Fallback.
Entry: IOC (Immediate or Cancel), Close: IOC (reduce_only), TP/SL: GTC/Trigger.
"""
import json
import time
import logging
from pathlib import Path
from decimal import Decimal
import sys

close_logger = logging.getLogger('rl_closes')

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
    Platziert eine Market Order auf Hyperliquid (SDK market_open mit 5% Slippage).
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

        # Leverage setzen (IMMER, auch bei 1x — HL merkt sich alten Wert)
        max_lev = coin_info.get("max_leverage", 5)
        leverage = min(leverage, max_lev)
        exchange.update_leverage(leverage, coin, is_cross=True)

        # Market Order via SDK (aggressive IOC, max 0.5% Slippage)
        order_result = exchange.market_open(coin, is_buy, quantity, slippage=0.005)

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
    """Schließt eine offene Position via market_close (SDK). 3 Versuche, 10s Timeout."""
    import concurrent.futures

    for attempt in range(1, 4):
        try:
            close_logger.info(f"HL_CLOSE_TRY {coin} | Versuch {attempt}/3")

            def _do_close():
                # Prüfen ob Position existiert
                info = get_hl_info()
                state = info.user_state(wallet_address)
                position = None
                for asset in state.get("assetPositions", []):
                    pos = asset.get("position", {})
                    if pos.get("coin") == coin and float(pos.get("szi", 0)) != 0:
                        position = pos
                        break
                if not position:
                    return {"success": False, "error": f"Keine offene Position für {coin}"}

                szi = float(position["szi"])
                exchange = HLExchange(
                    wallet=_hl_account(creds),
                    base_url=hl_constants.MAINNET_API_URL,
                    account_address=creds["wallet_address"],
                )

                # SDK market_close — handhabt Tick-Size + Lot-Size korrekt
                close_logger.info(f"HL_MARKET_CLOSE {coin} | szi={szi} | slippage=0.2%")
                result = exchange.market_close(coin, slippage=0.002)
                return {"raw_result": result}

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_close)
                result = future.result(timeout=10)

            if "raw_result" not in result:
                close_logger.warning(f"HL_CLOSE_ERROR {coin} | {result.get('error')}")
                return result

            raw = result["raw_result"]

            if raw.get("status") == "ok":
                statuses = raw.get("response", {}).get("data", {}).get("statuses", [])
                if statuses and "filled" in statuses[0]:
                    fill = statuses[0]["filled"]
                    avg_px = float(fill.get("avgPx", 0))
                    # Bestätigung: Position wirklich weg?
                    try:
                        info2 = get_hl_info()
                        state2 = info2.user_state(wallet_address)
                        still_open = False
                        for asset in state2.get("assetPositions", []):
                            p = asset.get("position", {})
                            if p.get("coin") == coin and float(p.get("szi", 0)) != 0:
                                still_open = True
                                break
                        if still_open:
                            close_logger.error(f"HL_FILL_BUT_STILL_OPEN {coin} | avgPx={avg_px} — Position existiert noch!")
                            if attempt < 3:
                                time.sleep(1)
                                continue
                            return {"success": False, "error": f"Fill gemeldet aber Position noch offen"}
                        close_logger.info(f"HL_CONFIRMED {coin} | avgPx={avg_px} | Position bestätigt geschlossen")
                    except Exception as ve:
                        close_logger.warning(f"HL_VERIFY_FAILED {coin} | {ve} — nehme Fill als bestätigt")
                    return {"success": True, "avg_price": avg_px}
                else:
                    close_logger.warning(f"HL_NOT_FILLED {coin} | statuses={statuses} | Versuch {attempt}")
                    if attempt < 3:
                        time.sleep(1)
                        continue
                    return {"success": False, "error": f"Nicht gefüllt nach 3 Versuchen: {statuses}"}
            else:
                close_logger.error(f"HL_ORDER_FAILED {coin} | result={raw} | Versuch {attempt}")
                if attempt < 3:
                    time.sleep(1)
                    continue
                return {"success": False, "error": str(raw)}

        except concurrent.futures.TimeoutError:
            close_logger.error(f"HL_TIMEOUT {coin} | Versuch {attempt}/3 — 10s Timeout")
            if attempt < 3:
                time.sleep(1)
                continue
            return {"success": False, "error": f"Timeout nach 3 Versuchen"}

        except Exception as e:
            close_logger.error(f"HL_EXCEPTION {coin} | {e} | Versuch {attempt}/3")
            if attempt < 3:
                time.sleep(1)
                continue
            return {"success": False, "error": str(e)}

    return {"success": False, "error": "3 Versuche fehlgeschlagen"}


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
