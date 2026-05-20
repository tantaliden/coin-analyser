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
from shared.database import get_app_db, get_coins_db
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
    """Aktueller HL-Guthaben (Unified-aware): Perp accountValue + Spot-USDC.
    Im Unified-Modus liegt das Geld auf Spot und dient als Cross-Margin fuer Perp."""
    info = get_hl_info()
    state = info.user_state(wallet_address)
    perp_av = float(state.get("marginSummary", {}).get("accountValue", 0))
    spot_usdc = 0.0
    try:
        spot = info.spot_user_state(wallet_address)
        for b in spot.get("balances", []):
            if b.get("coin") == "USDC":
                spot_usdc = float(b.get("total", 0) or 0)
                break
    except Exception:
        pass
    return perp_av + spot_usdc


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


# Cache fuer price_decimals aus HL allMids (60s) — sz_decimals/max_leverage kommen aus hl_meta.
_price_dec_cache = {"ts": 0.0, "data": {}}


def _refresh_price_decimals():
    """Liest aktuelle markPx von HL und leitet price_decimals ab. 60s gecached."""
    now = time.time()
    if now - _price_dec_cache["ts"] < 60 and _price_dec_cache["data"]:
        return _price_dec_cache["data"]
    try:
        info = get_hl_info()
        meta = info.meta_and_asset_ctxs()
        universe, ctxs = meta[0]["universe"], meta[1]
        d = {}
        for i, asset in enumerate(universe):
            if i >= len(ctxs):
                break
            mark = str(ctxs[i].get("markPx", "0"))
            d[asset["name"]] = len(mark.split(".")[1]) if "." in mark else 0
        _price_dec_cache["data"] = d
        _price_dec_cache["ts"] = now
    except Exception as e:
        logging.getLogger("rl_trader").warning("HL price_decimals refresh failed: %s", e)
    return _price_dec_cache["data"]


def round_hl_price(px, sz_decimals: int) -> float:
    """HL-konforme Preis-Rundung. HL hat ZWEI Constraints fuer Perp-Preise:
      1) max (6 - szDecimals) Decimals
      2) max 5 signifikante Stellen (sig figs)
    Beide muessen erfuellt sein, sonst lehnt HL ab mit 'Price must be divisible by tick size' / 'Invalid TP/SL price'.

    Beispiele:
      BTC szDec=5 -> max 1 Decimal: 78325.567 -> 78325.6 (1 dec, 5 sig figs OK)
      WIF szDec=1 -> max 5 Decimals: 0.876543 -> 0.87654 (5 dec, 5 sig figs OK)
      MEME szDec=0 -> max 6 Decimals: 0.0123456 -> 0.012346 (5 sig figs, dann 6 dec)"""
    if px is None: return px
    px = float(px)
    if px <= 0: return px
    import math
    max_dec = max(0, 6 - int(sz_decimals))
    sig_figs = 5
    # Erst auf 5 sig figs runden
    d_for_sig = sig_figs - int(math.floor(math.log10(abs(px)))) - 1
    px_rounded = round(px, max(0, d_for_sig))
    # Dann auf max-Decimals begrenzen
    return round(px_rounded, max_dec)


def _get_hl_coin_info(coin: str) -> dict:
    """Holt HL-spezifische Coin-Info aus coins.hl_meta (single source: hl-ingestor pflegt sz/price/lev).
    Fallback: bei fehlendem price_decimals -> live HL markPx (60s gecached). Akzeptiert 'BTC' oder 'BTCUSDC'."""
    symbol = coin.replace("USDC", "") if coin.endswith("USDC") else coin
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sz_decimals, max_leverage, price_decimals FROM hl_meta WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
    if not row or row["sz_decimals"] is None:
        return None
    price_dec = row["price_decimals"]
    if price_dec is None:
        # Fallback: hl_meta hat (noch) kein price_decimals -> live aus HL parsen
        price_dec = _refresh_price_decimals().get(symbol)
    if price_dec is None:
        return None
    return {
        "sz_decimals": row["sz_decimals"],
        "max_leverage": row["max_leverage"],
        "price_decimals": price_dec,
    }


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


# Leverage-Cache: (wallet, coin) -> (timestamp, leverage_value), 5min TTL.
_leverage_cache = {}
_LEVERAGE_TTL = 300


def _maybe_update_leverage(exchange, coin: str, leverage: int, wallet: str = "") -> None:
    """Setzt leverage nur wenn Wert nicht schon kuerzlich gesetzt wurde (spart 300-500ms pro Order)."""
    now = time.time()
    key = (wallet, coin)
    cached = _leverage_cache.get(key)
    if cached and (now - cached[0] < _LEVERAGE_TTL) and (cached[1] == leverage):
        return  # bereits aktuell, skip update
    try:
        exchange.update_leverage(leverage, coin, is_cross=True)
        _leverage_cache[key] = (now, leverage)
    except Exception as e:
        # Fehler beim leverage-Update darf Order nicht blockieren — HL behaelt evtl alten Wert
        logging.getLogger("rl_trader").warning("update_leverage %s failed: %s", coin, e)


def place_limit_order_hl(creds: dict, coin: str, is_buy: bool, size_usd: float,
                         price: float, leverage: int = 1, slippage_pct: float = 0.5) -> dict:
    """
    Platziert eine Market Order auf Hyperliquid (SDK market_open).
    slippage_pct: max. Slippage in Prozent (default 0.5 %).
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

        # Leverage setzen — gecached pro Coin/User (5min) damit nicht bei jeder Order ein
        # Extra-RTT zur HL-API gemacht wird (spart ~300-500ms pro Order).
        max_lev = coin_info.get("max_leverage")
        if max_lev is None:
            err = f"FALLBACK_TRIGGERED place_limit_order_hl {coin}: max_leverage in coin_info fehlt -> order rejected"
            print(f"[RL-TRADER] {err}")
            return {"success": False, "error": err}
        leverage = min(leverage, int(max_lev))
        _maybe_update_leverage(exchange, coin, leverage, creds.get("wallet_address", ""))

        # Market Order via SDK (aggressive IOC, max slippage_pct % Slippage)
        order_result = exchange.market_open(coin, is_buy, quantity, slippage=slippage_pct / 100.0)

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


def place_limit_ioc_order_hl(creds: dict, coin: str, is_buy: bool, size_usd: float,
                              limit_px: float, leverage: int = 1) -> dict:
    """Echte Limit-IOC-Order (Immediate-or-Cancel) zum übergebenen Preis.

    Im Unterschied zu place_limit_order_hl (das intern market_open mit
    Slippage-Cap vom AKTUELLEN mid macht und unseren `price` ignoriert):

      - HL versucht zum exakten limit_px (oder besser) zu fillen
      - matched nichts → unfilled, nichts geöffnet, keine Fees
      - matched → filled mit avg_price, behalten

    Volker-Direktive 19.05.2026: „kaufen zum Predictor-Preis oder besser,
    sonst gar nicht — sonst zahle ich für nichts" (Open+Close-Fees bei
    Post-Slippage-Close).
    """
    try:
        coin_info = _get_hl_coin_info(coin)
        if coin_info is None:
            return {"success": False, "error": f"Coin {coin} nicht in coin_info"}
        sz_decimals = coin_info["sz_decimals"]
        price_decimals = coin_info.get("price_decimals", 6)
        quantity = round(size_usd / limit_px, sz_decimals)
        if quantity <= 0:
            return {"success": False, "error": f"Quantity zu klein: {quantity}"}

        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        max_lev = coin_info.get("max_leverage")
        if max_lev is None:
            return {"success": False,
                    "error": f"FALLBACK_TRIGGERED place_limit_ioc_order_hl {coin}: max_leverage fehlt"}
        leverage = min(leverage, int(max_lev))
        _maybe_update_leverage(exchange, coin, leverage, creds.get("wallet_address", ""))

        # HL-konforme Preis-Rundung (tick size + max 5 sig figs), NICHT simples
        # round(px, decimals) — sonst "Price must be divisible by tick size".
        rounded_px = round_hl_price(float(limit_px), sz_decimals)
        order_result = exchange.order(
            coin, is_buy, quantity, rounded_px,
            {"limit": {"tif": "Ioc"}}, reduce_only=False,
        )

        if order_result.get("status") != "ok":
            return {"success": False, "error": str(order_result)}
        statuses = order_result.get("response", {}).get("data", {}).get("statuses", [])
        if not statuses:
            return {"success": False, "error": "empty statuses"}
        st0 = statuses[0]
        if "filled" in st0:
            fill = st0["filled"]
            # quantity = echte gefüllte Größe (totalSz), aber Plausibilitäts-Check:
            # totalSz muss nahe der berechneten quantity sein. Sonst (HL-Format-Eigenheit
            # oder kumulativer Wert) -> berechnete quantity nutzen, sonst landet eine
            # absurd große qty in der TP-Order -> "Order value too large".
            ts = fill.get("totalSz")
            try:
                q_fill = float(ts)
            except (TypeError, ValueError):
                q_fill = None
            if q_fill is None or q_fill <= 0 or q_fill > quantity * 1.5:
                print(f"[RL-TRADER] place_limit_ioc {coin}: totalSz={ts!r} unplausibel "
                      f"(calc={quantity}) -> nutze calc")
                q_final = quantity
            else:
                q_final = q_fill
            return {"success": True, "order_id": fill.get("oid"), "quantity": q_final,
                    "avg_price": float(fill.get("avgPx", rounded_px)), "status": "filled"}
        if "error" in st0:
            return {"success": False, "error": st0["error"], "status": "rejected"}
        # IOC: alles was nicht gefilled wird ist auto-cancelled von HL Seite, also resting heißt eher "partial cancelled"
        return {"success": False, "error": f"unfilled: {st0}", "status": "unfilled"}

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
    """Schließt eine offene Position via market_close (SDK).

    Schlank: kein Pre-Check (HL meldet selbst wenn keine Position),
    kein Post-Verify (fill-status im SDK-Response reicht), Slippage 1%
    (greift auch bei volatilen Spikes), maximal 1 Retry. Erwartete Latenz:
    ~500ms-1s pro Aufruf.
    """
    import concurrent.futures

    exchange = HLExchange(
        wallet=_hl_account(creds),
        base_url=hl_constants.MAINNET_API_URL,
        account_address=creds["wallet_address"],
    )

    def _do_close():
        return exchange.market_close(coin, slippage=0.01)  # 1% slippage

    for attempt in range(1, 3):  # max 2 attempts (1 retry)
        try:
            close_logger.info(f"HL_CLOSE {coin} | Versuch {attempt}/2 | slippage=1%")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                raw = ex.submit(_do_close).result(timeout=8)

            if not raw or raw.get("status") != "ok":
                close_logger.warning(f"HL_CLOSE_FAIL {coin} | Versuch {attempt} | {raw}")
                if attempt < 2:
                    time.sleep(0.3)
                    continue
                return {"success": False, "error": str(raw)}

            statuses = raw.get("response", {}).get("data", {}).get("statuses", [])
            if not statuses:
                if attempt < 2:
                    time.sleep(0.3)
                    continue
                return {"success": False, "error": f"empty statuses: {raw}"}

            st0 = statuses[0]
            # Manchmal meldet HL "no_position" wenn die Pos. schon weg ist (z.B. via TP/SL)
            if isinstance(st0, dict) and st0.get("error", "").lower().startswith("no position"):
                close_logger.info(f"HL_CLOSE {coin} | Position bereits geschlossen (no_position)")
                return {"success": True, "avg_price": 0, "note": "no_position"}

            if "filled" in st0:
                fill = st0["filled"]
                avg_px = float(fill.get("avgPx", 0))
                close_logger.info(f"HL_CLOSE_OK {coin} | avgPx={avg_px}")
                return {"success": True, "avg_price": avg_px}

            close_logger.warning(f"HL_NOT_FILLED {coin} | statuses={statuses}")
            if attempt < 2:
                time.sleep(0.3)
                continue
            return {"success": False, "error": f"nicht gefüllt: {statuses}"}

        except concurrent.futures.TimeoutError:
            close_logger.error(f"HL_TIMEOUT {coin} | Versuch {attempt}/2")
            if attempt < 2:
                time.sleep(0.3)
                continue
            return {"success": False, "error": "timeout"}
        except Exception as e:
            close_logger.error(f"HL_EXCEPTION {coin} | {e} | Versuch {attempt}/2")
            if attempt < 2:
                time.sleep(0.3)
                continue
            return {"success": False, "error": str(e)}

    return {"success": False, "error": "2 Versuche fehlgeschlagen"}


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
        if not coin_info:
            err = f"FALLBACK_TRIGGERED place_tp_sl_hl {coin}: coin_info None -> TP/SL rejected"
            print(f"[RL-TRADER] {err}")
            return {"success": False, "error": err}
        sz_dec = int(coin_info["sz_decimals"])
        quantity = float(quantity)
        # HL-konforme Rundung: max (6-szDec) Decimals UND max 5 sig figs
        tp_price = round_hl_price(tp_price, sz_dec)
        sl_price = round_hl_price(sl_price, sz_dec)
        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
        )

        # TP + SL SEQUENTIELL feuern, NICHT parallel.
        # Grund: HL-API nutzt monoton steigenden Nonce pro API-Wallet. Bei
        # parallelen Calls greifen beide Threads den gleichen Nonce-Wert ab,
        # HL lehnt einen ab mit "Invalid nonce: duplicate nonce". Sequentiell
        # mit kleiner Pause loest das. Latenz-Kosten: ~250-400ms, dafuer
        # Failsafe-Auslosungen quasi 0.
        tp_is_buy = not is_long  # Long → Sell bei TP
        sl_is_buy = not is_long  # symmetrisch fuer SL

        try:
            tp_result = exchange.order(coin, tp_is_buy, quantity, tp_price,
                                        {"limit": {"tif": "Gtc"}}, reduce_only=True)
        except Exception as e:
            tp_result = {"status": "error", "error": str(e)}

        # Mini-Pause damit HL den Nonce sicher inkrementiert hat
        time.sleep(0.08)

        try:
            sl_result = exchange.order(coin, sl_is_buy, quantity, sl_price,
                                        {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
                                        reduce_only=True)
        except Exception as e:
            sl_result = {"status": "error", "error": str(e)}

        def _parse_oid(res):
            if res.get("status") != "ok":
                return None, str(res)
            statuses = res.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "resting" in statuses[0]:
                return statuses[0]["resting"]["oid"], None
            if statuses and "filled" in statuses[0]:
                # Sofort gefuellt — auch ok
                return statuses[0]["filled"].get("oid"), None
            return None, f"unexpected response: {statuses}"

        tp_oid, tp_err = _parse_oid(tp_result)
        sl_oid, sl_err = _parse_oid(sl_result)

        # Verify-Loop (10.05.2026): nach Platzierung gegen frontend_open_orders pruefen
        # ob beide Orders wirklich resting sind. Falls nicht -> success=False, Caller
        # entscheidet (place_tp_sl-Caller macht failsafe-close).
        if tp_oid is not None and sl_oid is not None:
            time.sleep(1.0)  # HL-OrderBook-Propagation
            try:
                info = HLInfo(hl_constants.MAINNET_API_URL, skip_ws=True)
                fe_orders = info.frontend_open_orders(creds["wallet_address"]) or []
                live_oids = {int(o.get("oid", 0)) for o in fe_orders if o.get("coin") == coin}
                tp_live = int(tp_oid) in live_oids
                sl_live = int(sl_oid) in live_oids
                if not tp_live:
                    tp_err = f"verify_failed: tp_oid {tp_oid} not in open orders after 1s"
                if not sl_live:
                    sl_err = f"verify_failed: sl_oid {sl_oid} not in open orders after 1s"
            except Exception as ve:
                # Verify-Fehler ist kein harter Fail (OrderBook-Read kann auch failen),
                # aber wird im Log dokumentiert
                tp_err = tp_err or f"verify_exception: {ve}"

        # STRIKT: success NUR wenn BEIDE platziert UND beide live im OrderBook
        success = (tp_oid is not None and sl_oid is not None
                   and tp_err is None and sl_err is None)
        return {
            "success": success,
            "tp_oid": tp_oid, "sl_oid": sl_oid,
            "tp_error": tp_err, "sl_error": sl_err,
        }

    except Exception as e:
        # tp_error/sl_error explizit setzen, damit Reporting im Caller konsistent ist
        return {"success": False, "error": str(e), "tp_oid": None, "sl_oid": None,
                "tp_error": f"exception: {e}", "sl_error": f"exception: {e}"}


def update_tp_only_hl(creds: dict, coin: str, is_long: bool, quantity, new_tp_price) -> dict:
    """Aendert NUR die TP-Limit-Order: bestehende reduce-only Limit-Orders auf der
    Gegenseite canceln, neue TP-Limit-Order platzieren. SL-Trigger-Orders bleiben.
    DB wird nicht beruehrt (Predictor-Lernautonomie)."""
    try:
        coin_info = _get_hl_coin_info(coin)
        if not coin_info:
            err = f"FALLBACK_TRIGGERED update_tp_only_hl {coin}: coin_info None -> rejected"
            print(f"[RL-TRADER] {err}")
            return {"success": False, "error": err}
        price_dec = coin_info["price_decimals"]
        quantity = float(quantity)
        tp_price = round(float(new_tp_price), price_dec)

        info = get_hl_info()
        address = creds["wallet_address"]

        tp_is_buy = not is_long
        target_side = "B" if tp_is_buy else "A"

        try:
            fe_orders = info.frontend_open_orders(address)
        except Exception:
            fe_orders = []
        existing_tps = [o for o in fe_orders
                         if o.get("coin") == coin
                         and o.get("reduceOnly")
                         and o.get("orderType") == "Limit"
                         and o.get("side") == target_side]

        exchange = HLExchange(
            wallet=_hl_account(creds),
            base_url=hl_constants.MAINNET_API_URL,
            account_address=creds["wallet_address"],
            vault_address=creds.get("vault_address"),
        )

        cancelled = 0
        for o in existing_tps:
            try:
                exchange.cancel(coin, o["oid"])
                cancelled += 1
            except Exception:
                pass

        result = exchange.order(
            coin, tp_is_buy, quantity, tp_price,
            {"limit": {"tif": "Gtc"}},
            reduce_only=True,
        )
        if result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "resting" in statuses[0]:
                return {"success": True, "tp_oid": statuses[0]["resting"]["oid"],
                        "tp_price": tp_price, "cancelled": cancelled}
            if statuses and "filled" in statuses[0]:
                return {"success": True, "tp_oid": None, "filled": True,
                        "tp_price": tp_price, "cancelled": cancelled}
        return {"success": False, "error": str(result), "cancelled": cancelled}

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
