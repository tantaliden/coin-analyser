#!/usr/bin/env python3
"""
RL-Agent Service — Hauptloop.

Beobachtet neue Predictions, entscheidet über Trades, führt aus,
monitort offene Positionen, lernt aus Ergebnissen.

Logik:
1. Neue Predictions aus DB abholen
2. Observation zusammenbauen (CNN + Sentiment + Account-State)
3. Agent entscheidet: traden oder skip
4. Wenn traden: Limit Order auf Hyperliquid (bevorzugt) oder Binance (Fallback)
5. Offene Trades monitoren, bei Exit Reward berechnen
6. Periodisch trainieren
"""
import json
import time
import signal
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.agent import RLAgent, OBS_DIM
from rl_agent.trader import (
    get_hl_credentials,
    get_hl_balance,
    get_hl_open_positions,
    get_available_coins_hl,
    get_current_prices_hl,
    calculate_position_size,
    place_limit_order_hl,
    close_position_hl,
    place_tp_sl_hl,
    cancel_all_orders_for_coin_hl,
    cancel_order_hl,
    refresh_hl_coin_info,
    # Binance Spot Fallback
    get_binance_client,
    get_binance_balance,
    buy_spot_binance,
    set_oco_binance,
    cancel_orders_binance,
    sell_market_binance,
    get_binance_position,
)

COINS_DB_SETTINGS = None  # Lazy loaded


def get_coins_conn():
    """Verbindung zur coins-DB für Klines."""
    global COINS_DB_SETTINGS
    if COINS_DB_SETTINGS is None:
        s = json.load(open(SETTINGS_PATH))
        COINS_DB_SETTINGS = s["databases"]["coins"]
    db = COINS_DB_SETTINGS
    return psycopg2.connect(
        dbname=db["name"], user=db["user"], password=db["password"],
        host=db["host"], port=db["port"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )

SETTINGS_PATH = "/opt/coin/settings.json"
POLL_INTERVAL = 30  # Sekunden zwischen Checks
POSITION_CHECK_INTERVAL = 10  # Sekunden für Position-Monitoring
TRAIN_INTERVAL = 3600  # Jede Stunde Training versuchen
TRAIN_MIN_SAMPLES = 30  # Mindestens 30 Erfahrungen für Training

running = True
last_processed_id = 0


def signal_handler(sig, frame):
    global running
    print(f"\n[RL-AGENT] Signal {sig} empfangen, stoppe...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_conn():
    s = json.load(open(SETTINGS_PATH))
    db = s["databases"]["app"]
    return psycopg2.connect(
        dbname=db["name"],
        user=db["user"],
        password=db["password"],
        host=db["host"],
        port=db["port"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_agent_config(conn, user_id: int = 1) -> dict:
    """Liest RL-Agent Konfiguration aus DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM rl_agent_config WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        return {
            "is_active": False,
            "min_trade_size": 25.0,
            "max_capital_fraction": 0.05,
            "max_leverage": 5.0,
            "max_concurrent_positions": 5,
        }
    return {
        "is_active": row["is_active"],
        "min_trade_size": float(row["min_trade_size"]),
        "max_capital_fraction": float(row["max_capital_fraction"]),
        "max_leverage": float(row["max_leverage"]),
        "max_concurrent_positions": row["max_concurrent_positions"],
    }


def get_new_predictions(conn, after_id: int) -> list:
    """Neue aktive Predictions die der Agent noch nicht gesehen hat."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT prediction_id, symbol, direction, confidence,
                   take_profit_price as tp_price, stop_loss_price as sl_price,
                   detected_at, entry_price
            FROM momentum_predictions
            WHERE prediction_id > %s
            AND status = 'active'
            AND (scanner_type = 'default' OR scanner_type IS NULL)
            ORDER BY prediction_id ASC
            """,
            (after_id,),
        )
        return cur.fetchall()


def get_sentiment(conn, symbol: str) -> tuple:
    """Sentiment Score und Fear&Greed für ein Symbol."""
    sentiment = 0.0
    fear_greed = 50

    # Coin-spezifischer Sentiment
    base = symbol.replace("USDC", "").replace("USDT", "")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT score FROM sentiment_scores
            WHERE symbol = %s AND expires_at > NOW()
            ORDER BY updated_at DESC LIMIT 1
            """,
            (base,),
        )
        row = cur.fetchone()
        if row:
            sentiment = float(row["score"])

        # Fear & Greed
        cur.execute(
            """
            SELECT score FROM sentiment_scores
            WHERE source = 'fear_greed' AND expires_at > NOW()
            ORDER BY updated_at DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        if row:
            fear_greed = int(row["score"])

    return sentiment, fear_greed


def get_coin_hit_rate(conn, symbol: str) -> float:
    """Historische Hit-Rate für einen Coin."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'hit_tp') as tp,
                COUNT(*) FILTER (WHERE status = 'hit_sl') as sl
            FROM momentum_predictions
            WHERE symbol = %s AND status IN ('hit_tp', 'hit_sl')
            """,
            (symbol,),
        )
        row = cur.fetchone()
        tp, sl = row["tp"] or 0, row["sl"] or 0
        if tp + sl < 10:
            return 0.75  # Default wenn wenig Daten
        return tp / (tp + sl)


def get_1m_klines_features(symbol: str) -> dict:
    """
    Holt 1m Klines der letzten 15 Minuten und berechnet Features.
    Returns dict mit allen 1m-basierten Observation-Features.
    """
    defaults = {
        "return_1m": 0.0, "return_5m": 0.0, "return_15m": 0.0,
        "volatility_15m": 0.0, "volume_trend": 1.0,
        "taker_ratio_1m": 0.5, "taker_ratio_5m": 0.5,
        "trades_trend": 1.0, "last_close": 0.0,
    }
    try:
        coins_conn = get_coins_conn()
        with coins_conn.cursor() as cur:
            cur.execute(
                """
                SELECT open_time, open, high, low, close, volume, trades,
                       taker_buy_base, quote_asset_volume
                FROM klines
                WHERE symbol = %s AND interval = '1m'
                ORDER BY open_time DESC LIMIT 15
                """,
                (symbol,),
            )
            rows = cur.fetchall()
        coins_conn.close()

        if len(rows) < 2:
            return defaults

        # Sortiere chronologisch (älteste zuerst)
        rows = list(reversed(rows))

        # Preise
        closes = [float(r["close"]) for r in rows]
        opens = [float(r["open"]) for r in rows]
        highs = [float(r["high"]) for r in rows]
        lows = [float(r["low"]) for r in rows]
        volumes = [float(r["volume"]) for r in rows]
        trades_list = [int(r["trades"] or 0) for r in rows]
        taker_buys = [float(r["taker_buy_base"] or 0) for r in rows]

        last_close = closes[-1]
        if last_close == 0:
            return defaults

        # Returns
        return_1m = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
        return_5m = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else return_1m
        return_15m = (closes[-1] - closes[0]) / closes[0] * 100

        # Volatilität: (max_high - min_low) / last_close
        max_high = max(highs)
        min_low = min(lows)
        volatility_15m = (max_high - min_low) / last_close * 100

        # Volume Trend: letzte Minute vs Durchschnitt
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        volume_trend = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        # Taker Ratio (letzte 1 Min)
        last_vol = volumes[-1]
        last_taker = taker_buys[-1]
        taker_ratio_1m = last_taker / last_vol if last_vol > 0 else 0.5

        # Taker Ratio (letzte 5 Min)
        vol_5 = sum(volumes[-5:])
        taker_5 = sum(taker_buys[-5:])
        taker_ratio_5m = taker_5 / vol_5 if vol_5 > 0 else 0.5

        # Trades Trend: letzte Minute vs Durchschnitt
        avg_trades = sum(trades_list) / len(trades_list) if trades_list else 1
        trades_trend = trades_list[-1] / avg_trades if avg_trades > 0 else 1.0

        return {
            "return_1m": return_1m,
            "return_5m": return_5m,
            "return_15m": return_15m,
            "volatility_15m": volatility_15m,
            "volume_trend": volume_trend,
            "taker_ratio_1m": taker_ratio_1m,
            "taker_ratio_5m": taker_ratio_5m,
            "trades_trend": trades_trend,
            "last_close": last_close,
        }

    except Exception as e:
        print(f"[RL-AGENT] 1m Klines Fehler für {symbol}: {e}")
        return defaults


def build_observation(
    prediction: dict,
    sentiment: float,
    fear_greed: int,
    hit_rate: float,
    balance: float,
    open_count: int,
    unrealized_pnl: float,
    klines_1m: dict = None,
    hl_price: float = 0.0,
) -> np.ndarray:
    """Baut den Observation-Vektor für den Agent (24 Features)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    entry = float(prediction["entry_price"] or 0)
    tp = float(prediction["tp_price"] or 0)
    sl = float(prediction["sl_price"] or 0)

    tp_pct = abs(tp - entry) / entry * 100 if entry > 0 else 3.0
    sl_pct = abs(sl - entry) / entry * 100 if entry > 0 else 2.0

    now = datetime.now()
    if klines_1m is None:
        klines_1m = {}

    # --- CNN Prediction ---
    obs[0] = float(prediction["confidence"] or 0.5)
    obs[1] = 1.0 if prediction["direction"] == "long" else 0.0
    obs[2] = min(tp_pct / 10.0, 1.0)
    obs[3] = min(sl_pct / 10.0, 1.0)

    # --- Sentiment ---
    obs[4] = np.clip(sentiment, -1.0, 1.0)
    obs[5] = fear_greed / 100.0

    # --- Coin Stats ---
    obs[6] = hit_rate

    # --- Account State ---
    obs[7] = min(balance / 10000.0, 1.0)
    obs[8] = min(open_count / 10.0, 1.0)
    obs[9] = np.clip(unrealized_pnl / max(balance, 1.0), -1.0, 1.0)

    # --- Zeitkontext ---
    obs[10] = now.hour / 24.0
    obs[11] = now.weekday() / 7.0

    # --- 1m Klines ---
    obs[12] = np.clip(klines_1m.get("return_1m", 0) / 5.0, -1.0, 1.0)
    obs[13] = np.clip(klines_1m.get("return_5m", 0) / 5.0, -1.0, 1.0)
    obs[14] = np.clip(klines_1m.get("return_15m", 0) / 10.0, -1.0, 1.0)
    obs[15] = np.clip(klines_1m.get("volatility_15m", 0) / 5.0, -1.0, 1.0)
    obs[16] = np.clip(klines_1m.get("volume_trend", 1.0) / 5.0, 0.0, 1.0)
    obs[17] = np.clip(klines_1m.get("taker_ratio_1m", 0.5) * 2 - 1, -1.0, 1.0)  # 0-1 → -1..1
    obs[18] = np.clip(klines_1m.get("taker_ratio_5m", 0.5) * 2 - 1, -1.0, 1.0)
    obs[19] = np.clip(klines_1m.get("trades_trend", 1.0) / 5.0, 0.0, 1.0)

    # --- Live-Preis ---
    spot_close = klines_1m.get("last_close", 0)
    if entry > 0 and hl_price > 0:
        obs[20] = np.clip((hl_price - entry) / entry * 100 / 5.0, -1.0, 1.0)
    if spot_close > 0 and hl_price > 0:
        obs[21] = np.clip((hl_price - spot_close) / spot_close * 100 / 2.0, -1.0, 1.0)

    # 22, 23: reserviert (bleiben 0)

    return obs


def calculate_reward(pnl_pct: float, fees_covered: bool, duration_min: int) -> float:
    """
    Reward-Funktion:
    - Profit → positive Belohnung (proportional)
    - Verlust → negative Strafe (proportional)
    - Fees nicht gedeckt → Extra-Strafe
    - Schnelle profitable Trades → Bonus
    """
    reward = pnl_pct / 100.0 * 10  # Skalierung: 1% = 0.1 Reward

    if not fees_covered:
        reward -= 0.5  # Extra-Strafe wenn Fees nicht mal gedeckt

    # Zeitbonus: Schnelle profitable Trades belohnen
    if pnl_pct > 0 and duration_min < 60:
        reward *= 1.2

    return reward


def calculate_counterfactual_reward(prediction: dict, conn) -> float:
    """Was wäre passiert wenn man die Prediction genommen/nicht genommen hätte."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM momentum_predictions WHERE prediction_id = %s",
            (prediction["prediction_id"],),
        )
        row = cur.fetchone()
    if not row:
        return 0.0

    status = row["status"]
    if status == "hit_tp":
        entry = float(prediction["entry_price"] or 0)
        tp = float(prediction["tp_price"] or 0)
        if entry > 0:
            return abs(tp - entry) / entry * 100  # Entgangener Gewinn in %
    elif status == "hit_sl":
        entry = float(prediction["entry_price"] or 0)
        sl = float(prediction["sl_price"] or 0)
        if entry > 0:
            return -abs(sl - entry) / entry * 100  # Vermiedener Verlust in %

    return 0.0


def log_observation(conn, prediction: dict, action: str, obs_data: dict):
    """Speichert eine Agent-Observation in die DB."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rl_observations
            (prediction_id, symbol, direction, confidence, sentiment_score, fear_greed,
             coin_hit_rate, agent_action, position_size_usd, leverage)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                prediction["prediction_id"],
                prediction["symbol"],
                prediction["direction"],
                min(float(prediction["confidence"] or 0) / 100.0, 1.0),
                float(obs_data.get("sentiment", 0)),
                int(obs_data.get("fear_greed", 50)),
                float(obs_data.get("hit_rate", 0.75)),
                action,
                float(obs_data["size_usd"]) if obs_data.get("size_usd") is not None else None,
                float(obs_data["leverage"]) if obs_data.get("leverage") is not None else None,
            ),
        )
        conn.commit()


def log_trade(conn, prediction: dict, trade_data: dict) -> int:
    """Loggt einen neuen Trade in die DB. Returns trade_id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rl_trades
            (prediction_id, symbol, direction, leverage, entry_price, entry_time,
             position_size_usd, quantity, hl_order_id, status, exchange)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, 'open', %s)
            RETURNING id
            """,
            (
                prediction["prediction_id"],
                prediction["symbol"],
                prediction["direction"],
                float(trade_data["leverage"]),
                float(trade_data["entry_price"]),
                float(trade_data["size_usd"]),
                float(trade_data["quantity"]),
                str(trade_data.get("order_id") or ""),
                trade_data.get("exchange", "hyperliquid"),
            ),
        )
        trade_id = cur.fetchone()["id"]
        conn.commit()
        return trade_id


def close_trade_hl(creds: dict, coin: str) -> bool:
    """Schließt HL-Position: erst alle Orders canceln, dann Position schließen."""
    cancel_all_orders_for_coin_hl(creds, coin)
    result = close_position_hl(creds, coin, creds["wallet_address"])
    if not result.get("success"):
        print(f"[RL-AGENT] HL Close fehlgeschlagen für {coin}: {result.get('error')}")
        return False
    return True


def close_trade_binance(symbol: str, quantity: float) -> bool:
    """Schließt Binance Spot Position: Orders canceln + Market Sell."""
    cancel_orders_binance(symbol)
    result = sell_market_binance(symbol, quantity)
    if not result.get("success"):
        print(f"[RL-AGENT] Binance Close fehlgeschlagen für {symbol}: {result.get('error')}")
        return False
    return True


def check_open_trades(conn, creds: dict, agent: RLAgent):
    """
    Prüft offene Trades und schließt wenn nötig.
    Unterstützt Hyperliquid UND Binance Spot Trades.
    Agent kann aktiv Positionen managen (early exit, cancel TP/SL).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.id, t.symbol, t.direction, t.leverage, t.entry_price, t.entry_time,
                   t.position_size_usd, t.quantity, t.prediction_id, t.exchange,
                   p.take_profit_price as tp_price, p.stop_loss_price as sl_price,
                   p.status as pred_status
            FROM rl_trades t
            LEFT JOIN momentum_predictions p ON t.prediction_id = p.prediction_id
            WHERE t.status = 'open'
            """
        )
        open_trades = cur.fetchall()

    if not open_trades:
        return

    # Aktuelle Preise (HL hat die meisten)
    try:
        prices = get_current_prices_hl()
    except Exception as e:
        print(f"[RL-AGENT] Preise holen fehlgeschlagen: {e}")
        return

    for trade in open_trades:
        coin = trade["symbol"].replace("USDC", "").replace("USDT", "")
        exchange = trade.get("exchange", "hyperliquid")
        current_price = prices.get(coin, 0)

        # Für Binance-Trades: Preis direkt von Binance holen wenn nicht in HL
        if current_price == 0 and exchange == "binance":
            try:
                pos = get_binance_position(trade["symbol"])
                if pos:
                    current_price = pos["current_price"]
            except:
                pass

        if current_price == 0:
            continue

        entry = float(trade["entry_price"] or 0)
        if entry == 0:
            continue

        # PnL berechnen
        if trade["direction"] == "long":
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100

        pnl_usd = float(trade["position_size_usd"]) * float(trade["leverage"]) * pnl_pct / 100

        # --- Agent-basierte Entscheidung: soll die Position gehalten werden? ---
        # Observation für die aktive Position
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = float(trade.get("confidence", 0.5) or 0.5)
        obs[1] = 1.0 if trade["direction"] == "long" else 0.0
        obs[7] = 0.5  # Placeholder balance
        obs[9] = np.clip(pnl_pct / 10.0, -1.0, 1.0)  # Aktueller PnL in Observation

        # Zeitkontext
        now = datetime.now()
        obs[10] = now.hour / 24.0
        obs[11] = now.weekday() / 7.0

        # 1m Klines für aktuelle Marktsituation
        try:
            klines_1m = get_1m_klines_features(trade["symbol"])
            obs[12] = np.clip(klines_1m.get("return_1m", 0) / 5.0, -1.0, 1.0)
            obs[13] = np.clip(klines_1m.get("return_5m", 0) / 5.0, -1.0, 1.0)
            obs[14] = np.clip(klines_1m.get("return_15m", 0) / 10.0, -1.0, 1.0)
            obs[15] = np.clip(klines_1m.get("volatility_15m", 0) / 5.0, -1.0, 1.0)
        except:
            pass

        # --- Automatische Exit-Gründe ---
        pred_status = trade.get("pred_status")
        should_close = False
        exit_reason = None

        if pred_status == "hit_tp":
            should_close = True
            exit_reason = "tp"
        elif pred_status == "hit_sl":
            should_close = True
            exit_reason = "sl"
        elif pred_status == "expired":
            should_close = True
            exit_reason = "expired"

        # Zeitlimit: 72h wie Prediction
        if trade["entry_time"]:
            age = datetime.now(timezone.utc) - trade["entry_time"]
            if age > timedelta(hours=72):
                should_close = True
                exit_reason = "expired"

        # --- Agent-basierter Early Exit ---
        # DEAKTIVIERT bis genug Erfahrung gesammelt (min. 100 Trades im Buffer).
        # TP/SL regeln den Exit. Agent soll erstmal lernen, nicht mit
        # zufälligen Aktionen echtes Geld verbrennen.
        # TODO: Aktivieren wenn agent.buffer.size() > 100
        # if not should_close and trade["entry_time"]:
        #     age_min = (datetime.now(timezone.utc) - trade["entry_time"]).total_seconds() / 60
        #     if age_min > 60 and agent.buffer.size() > 100:
        #         decision = agent.decide(obs)
        #         if float(decision["raw_action"][0]) < -0.7:
        #             should_close = True
        #             exit_reason = "agent_exit"

        if should_close:
            print(f"[RL-AGENT] Schließe Trade {trade['id']} ({coin} {trade['direction']} @{exchange}): {exit_reason}, PnL: {pnl_pct:.2f}%")

            # Position schließen je nach Exchange — NUR als closed markieren wenn Sell erfolgreich
            if exchange == "binance":
                closed_ok = close_trade_binance(trade["symbol"], float(trade["quantity"]))
            else:
                closed_ok = close_trade_hl(creds, coin)

            if not closed_ok:
                print(f"[RL-AGENT] WARNUNG: Close fehlgeschlagen für Trade {trade['id']}, bleibt open!")
                continue

            # Trade in DB als closed markieren
            entry_time = trade["entry_time"]
            duration = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60) if entry_time else 0
            fees = float(trade["position_size_usd"]) * 0.001  # ~0.1% Fees geschätzt

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rl_trades SET
                        exit_price = %s, exit_time = NOW(), exit_reason = %s,
                        pnl_usd = %s, pnl_percent = %s, fees_usd = %s,
                        duration_minutes = %s, status = 'closed'
                    WHERE id = %s
                    """,
                    (current_price, exit_reason, round(pnl_usd, 2), round(pnl_pct, 4),
                     round(fees, 4), duration, trade["id"]),
                )
                conn.commit()

            # Reward berechnen und speichern
            fees_covered = pnl_usd > fees
            reward = calculate_reward(pnl_pct, fees_covered, duration)

            action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Took trade
            agent.record_outcome(obs, action, reward)

            print(f"[RL-AGENT] Trade {trade['id']} closed: PnL ${pnl_usd:.2f} ({pnl_pct:.2f}%), Reward: {reward:.3f}")


def check_skipped_predictions(conn, agent: RLAgent):
    """Prüft übersprungene Predictions für Counterfactual Learning."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.id, o.prediction_id, o.confidence, o.sentiment_score, o.fear_greed,
                   o.coin_hit_rate, p.status as pred_status, p.entry_price,
                   p.take_profit_price as tp_price, p.stop_loss_price as sl_price
            FROM rl_observations o
            JOIN momentum_predictions p ON o.prediction_id = p.prediction_id
            WHERE o.agent_action = 'skipped'
            AND o.counterfactual_pnl IS NULL
            AND p.status IN ('hit_tp', 'hit_sl', 'expired')
            LIMIT 50
            """,
        )
        rows = cur.fetchall()

    for row in rows:
        entry = float(row["entry_price"] or 0)
        tp = float(row["tp_price"] or 0)
        sl = float(row["sl_price"] or 0)

        if row["pred_status"] == "hit_tp" and entry > 0:
            cf_pnl = abs(tp - entry) / entry * 100
        elif row["pred_status"] == "hit_sl" and entry > 0:
            cf_pnl = -abs(sl - entry) / entry * 100
        else:
            cf_pnl = 0.0

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rl_observations SET counterfactual_pnl = %s WHERE id = %s",
                (round(cf_pnl, 2), row["id"]),
            )
            conn.commit()

        # Reward für Skipping: Entgangener Gewinn ist negativ, vermiedener Verlust positiv
        reward = -cf_pnl / 100.0 * 5  # Invertiert: Guten Trade verpasst = negativ

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = float(row["confidence"] or 0.5)
        obs[4] = float(row["sentiment_score"] or 0)
        obs[5] = float(row["fear_greed"] or 50) / 100.0
        obs[6] = float(row["coin_hit_rate"] or 0.75)

        action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # Skipped
        agent.record_outcome(obs, action, reward)


def main():
    global running, last_processed_id

    print("[RL-AGENT] Service gestartet")

    # Agent initialisieren
    agent = RLAgent()

    # Credentials
    creds = get_hl_credentials()
    if not creds:
        print("[RL-AGENT] FEHLER: Keine Hyperliquid Credentials gefunden!")
        return

    # HL Coin-Info in DB aktualisieren (szDecimals, priceDecimals, maxLeverage)
    refresh_hl_coin_info()

    # Verfügbare Coins auf Hyperliquid
    try:
        hl_coins = get_available_coins_hl()
        print(f"[RL-AGENT] {len(hl_coins)} Coins auf Hyperliquid verfügbar")
    except Exception as e:
        print(f"[RL-AGENT] HL Coins laden fehlgeschlagen: {e}")
        hl_coins = set()

    # Letzte verarbeitete Prediction-ID
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(prediction_id) as max_id FROM rl_observations"
        )
        row = cur.fetchone()
        last_processed_id = row["max_id"] or 0
    conn.close()
    print(f"[RL-AGENT] Starte ab Prediction-ID {last_processed_id}")

    last_train_time = time.time()
    iteration = 0

    while running:
        try:
            conn = get_conn()

            # Config lesen
            config = get_agent_config(conn)
            if not config["is_active"]:
                conn.close()
                time.sleep(POLL_INTERVAL)
                continue

            # 1. Offene Trades checken
            check_open_trades(conn, creds, agent)

            # 2. Skipped Predictions checken (Counterfactual)
            if iteration % 10 == 0:  # Alle 5 Minuten
                check_skipped_predictions(conn, agent)

            # 3. Neue Predictions verarbeiten
            predictions = get_new_predictions(conn, last_processed_id)

            if predictions:
                # Account-State (HL + Binance)
                hl_bal = get_hl_balance(creds["wallet_address"])
                bin_bal = get_binance_balance()
                balance = hl_bal + bin_bal
                hl_positions = get_hl_open_positions(creds["wallet_address"])
                open_count = len(hl_positions)
                total_upnl = sum(p["unrealized_pnl"] for p in hl_positions)

                # Alle HL-Preise einmal holen (für Observation + Order)
                try:
                    prices = get_current_prices_hl()
                except Exception as e:
                    print(f"[RL-AGENT] Preise holen fehlgeschlagen: {e}")
                    prices = {}

                # Position-Sizing
                sizing = calculate_position_size(
                    balance, config["min_trade_size"], config["max_capital_fraction"]
                )

                for pred in predictions:
                    last_processed_id = pred["prediction_id"]

                    # Max concurrent positions check
                    if open_count >= config["max_concurrent_positions"]:
                        print(f"[RL-AGENT] Max Positionen erreicht ({open_count}), skip {pred['symbol']}")
                        log_observation(conn, pred, "skipped", {"sentiment": 0, "fear_greed": 50})
                        continue

                    coin = pred["symbol"].replace("USDC", "").replace("USDT", "")

                    # Exchange bestimmen: HL bevorzugt, Binance Spot als Fallback
                    use_exchange = None
                    if coin in hl_coins:
                        use_exchange = "hyperliquid"
                    else:
                        # Binance Fallback: nur long möglich (Spot)
                        if pred["direction"] == "long":
                            use_exchange = "binance"
                        else:
                            print(f"[RL-AGENT] {coin} nicht auf HL + short → kein Binance Spot, skip")
                            log_observation(conn, pred, "skipped", {"sentiment": 0, "fear_greed": 50})
                            continue

                    if use_exchange is None:
                        log_observation(conn, pred, "skipped", {"sentiment": 0, "fear_greed": 50})
                        continue

                    # Balance zu niedrig?
                    if balance < config["min_trade_size"]:
                        print(f"[RL-AGENT] Balance zu niedrig (${balance:.2f}), skip")
                        log_observation(conn, pred, "skipped", {"sentiment": 0, "fear_greed": 50})
                        continue

                    # Observation bauen
                    sentiment, fear_greed = get_sentiment(conn, pred["symbol"])
                    hit_rate = get_coin_hit_rate(conn, pred["symbol"])

                    # 1m Klines Features
                    klines_1m = get_1m_klines_features(pred["symbol"])

                    # Aktueller Hyperliquid-Preis
                    hl_price = prices.get(coin, 0)

                    obs = build_observation(
                        pred, sentiment, fear_greed, hit_rate,
                        balance, open_count, total_upnl,
                        klines_1m=klines_1m, hl_price=hl_price,
                    )

                    # Agent entscheidet
                    decision = agent.decide(obs)

                    if not decision["take"]:
                        print(f"[RL-AGENT] Skip {pred['symbol']} {pred['direction']} (conf={pred['confidence']:.3f})")
                        log_observation(conn, pred, "skipped", {
                            "sentiment": sentiment, "fear_greed": fear_greed,
                            "hit_rate": hit_rate,
                        })
                        continue

                    # Position Size berechnen
                    size_usd = sizing["min"] + decision["size_factor"] * (sizing["max"] - sizing["min"])
                    size_usd = round(min(size_usd, sizing["max"]), 2)

                    # Leverage (nur HL, Binance Spot = immer 1x)
                    if use_exchange == "hyperliquid":
                        leverage = max(1, round(1 + decision["leverage_factor"] * (config["max_leverage"] - 1)))
                    else:
                        leverage = 1  # Binance Spot = kein Hebel

                    is_buy = pred["direction"] == "long"
                    tp_price = float(pred["tp_price"] or 0)
                    sl_price = float(pred["sl_price"] or 0)

                    # ========== ENTRY: Hyperliquid ==========
                    if use_exchange == "hyperliquid":
                        current_price = prices.get(coin, 0)
                        if current_price == 0:
                            print(f"[RL-AGENT] Kein Preis für {coin}")
                            continue

                        # Limit Order: leicht besser als Mid
                        if is_buy:
                            limit_price = round(current_price * 0.999, 6)
                        else:
                            limit_price = round(current_price * 1.001, 6)

                        print(
                            f"[RL-AGENT] TRADE HL: {pred['symbol']} {pred['direction']} "
                            f"${size_usd:.2f} @ ${limit_price:.4f} {leverage}x"
                        )

                        result = place_limit_order_hl(
                            creds, coin, is_buy, size_usd, limit_price, leverage
                        )

                        if result["success"]:
                            entry_price = result.get("avg_price", limit_price)
                            quantity = result["quantity"]

                            trade_id = log_trade(conn, pred, {
                                "leverage": leverage,
                                "entry_price": entry_price,
                                "size_usd": size_usd,
                                "quantity": quantity,
                                "order_id": str(result.get("order_id", "")),
                                "exchange": "hyperliquid",
                            })
                            open_count += 1

                            # TP/SL setzen (bei filled sofort, bei resting nach kurzer Wartezeit)
                            if tp_price > 0 and sl_price > 0:
                                if result.get("status") == "resting":
                                    # Limit Order wartet noch — TP/SL können erst nach Fill
                                    # TODO: In check_open_trades TP/SL nachsetzen wenn gefüllt
                                    print(f"[RL-AGENT] Entry resting, TP/SL wird nach Fill gesetzt")
                                else:
                                    tpsl = place_tp_sl_hl(creds, coin, is_buy, quantity, tp_price, sl_price)
                                    if tpsl.get("success"):
                                        print(f"[RL-AGENT] TP/SL gesetzt: TP={tp_price}, SL={sl_price}")
                                    else:
                                        print(f"[RL-AGENT] TP/SL setzen fehlgeschlagen: {tpsl.get('error')}")

                            print(f"[RL-AGENT] Trade {trade_id} platziert @HL: {result['status']}")

                            log_observation(conn, pred, "taken", {
                                "sentiment": sentiment, "fear_greed": fear_greed,
                                "hit_rate": hit_rate, "size_usd": size_usd,
                                "leverage": leverage,
                            })
                        else:
                            print(f"[RL-AGENT] HL Order fehlgeschlagen: {result['error']}")
                            log_observation(conn, pred, "skipped", {
                                "sentiment": sentiment, "fear_greed": fear_greed,
                                "hit_rate": hit_rate,
                            })

                    # ========== ENTRY: Binance Spot Fallback ==========
                    elif use_exchange == "binance":
                        print(
                            f"[RL-AGENT] TRADE BIN: {pred['symbol']} long "
                            f"${size_usd:.2f} (Spot, kein Hebel)"
                        )

                        result = buy_spot_binance(pred["symbol"], size_usd)

                        if result["success"]:
                            entry_price = result["avg_price"]
                            quantity = result["quantity"]

                            trade_id = log_trade(conn, pred, {
                                "leverage": 1,
                                "entry_price": entry_price,
                                "size_usd": size_usd,
                                "quantity": quantity,
                                "order_id": str(result.get("order_id", "")),
                                "exchange": "binance",
                            })
                            open_count += 1

                            # OCO Sell (TP + SL) setzen
                            if tp_price > 0 and sl_price > 0:
                                oco = set_oco_binance(pred["symbol"], quantity, tp_price, sl_price)
                                if oco.get("success"):
                                    print(f"[RL-AGENT] OCO gesetzt: TP={tp_price}, SL={sl_price}")
                                else:
                                    print(f"[RL-AGENT] OCO fehlgeschlagen: {oco.get('error')}")

                            print(f"[RL-AGENT] Trade {trade_id} platziert @Binance")

                            log_observation(conn, pred, "taken", {
                                "sentiment": sentiment, "fear_greed": fear_greed,
                                "hit_rate": hit_rate, "size_usd": size_usd,
                                "leverage": 1,
                            })
                        else:
                            print(f"[RL-AGENT] Binance Buy fehlgeschlagen: {result['error']}")
                            log_observation(conn, pred, "skipped", {
                                "sentiment": sentiment, "fear_greed": fear_greed,
                                "hit_rate": hit_rate,
                            })

            # 4. Periodisches Training
            if time.time() - last_train_time > TRAIN_INTERVAL:
                agent.train_on_buffer(TRAIN_MIN_SAMPLES)
                agent.save()
                last_train_time = time.time()

            conn.close()
            iteration += 1

        except Exception as e:
            print(f"[RL-AGENT] Fehler im Hauptloop: {e}")
            traceback.print_exc()
            try:
                conn.close()
            except:
                pass

        time.sleep(POLL_INTERVAL)

    # Cleanup
    agent.save()
    print("[RL-AGENT] Service gestoppt, Modell gespeichert")


if __name__ == "__main__":
    main()
