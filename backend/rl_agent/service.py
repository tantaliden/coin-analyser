#!/usr/bin/env python3
"""
RL-Agent Service V4 — PPO Discrete(21), Live Trading.

Agent bekommt CNN-Predictions als Trigger, entscheidet SELBST:
- Richtung (long/short)
- Hebel (1x-10x)
- Exit-Timing (alle 5 Min: halten oder schließen)

Agent sieht KEIN Geld — nur Marktdaten und Reward-Punkte.
Portfolio-Management läuft unsichtbar im Hintergrund.

Discrete(21): skip, long 1x-10x, short 1x-10x
3-Stufen-Reward: Verlust-Penalty, Early-Exit-Penalty, Win-Bonus

Usage:
    systemctl start rl-agent
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
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v4 import LEVERAGE_MAP, _reward
from rl_agent.features import compute_observation_live, N_FEATURES
from rl_agent.trader import (
    get_hl_credentials,
    get_hl_balance,
    get_hl_open_positions,
    get_available_coins_hl,
    get_current_prices_hl,
    place_limit_order_hl,
    close_position_hl,
    cancel_all_orders_for_coin_hl,
    refresh_hl_coin_info,
    get_binance_balance,
    buy_spot_binance,
    cancel_orders_binance,
    sell_market_binance,
    get_binance_position,
)

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = "/opt/coin/database/data/models/rl_ppo_trading_v4.zip"
STATE_PATH = "/opt/coin/database/data/models/rl_agent_state.json"

POLL_INTERVAL = 30          # Hauptloop-Takt (Sekunden)
MGMT_INTERVAL = 300         # Positions-Management alle 5 Min (Sekunden)
SL_PERCENT = 5.0
STEP_MINUTES = 5
FEE_RATE = 0.001            # 0.1% pro Seite
MAX_CONCURRENT = 50
MAX_POSITION_AGE = 6 * 60   # 6h in Minuten

# Progressive Trade-Size (identisch zum Backtest)
BASE_TRADE_SIZE = 15.0


def _calc_trade_size(wallet_balance, base_size=BASE_TRADE_SIZE):
    """Progressive Trade-Size (Notional): base_size flat → 1/100 ab $1500 → ... → 1/200 ab $14000+.
    Basiert auf echtem Wallet-Guthaben der jeweiligen Exchange."""
    if wallet_balance < 1500:
        return base_size
    if wallet_balance < 5000:
        return wallet_balance / 100
    step = int((wallet_balance - 5000) / 1000)
    divisor = 110 + step * 10
    if divisor > 200:
        divisor = 200
    return wallet_balance / divisor


def _get_wallet_balance(creds, exchange):
    """Echtes Wallet-Guthaben von der jeweiligen Exchange holen."""
    try:
        if exchange == 'hyperliquid':
            return get_hl_balance(creds['wallet_address'])
        elif exchange == 'binance':
            return get_binance_balance()
    except Exception as e:
        print(f"[RL-AGENT] Balance-Abfrage {exchange} fehlgeschlagen: {e}")
    return 0


# Punktesystem
def _point_bonus(total_points):
    """Bonus-Multiplikator auf Rewards basierend auf Punktestand."""
    if total_points >= 5000:
        return 2.0
    if total_points >= 2000:
        return 1.5
    if total_points >= 500:
        return 1.2
    return 1.0


def _check_day_rollover(state):
    """Tageswechsel: Prüft Verlust-Serie und zieht ggf. Punkte ab."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if state['current_day'] == today:
        return  # Gleicher Tag

    if state['current_day']:
        # Tageswechsel — alten Tag auswerten
        if state['day_pnl'] < 0:
            state['losing_streak_days'] += 1
            if state['losing_streak_days'] >= 2:
                # Ab Tag 2: 1%, Tag 3: 2%, Tag 4: 4%, ... gedeckelt bei 20%
                penalty_pct = min(2 ** (state['losing_streak_days'] - 2), 20) / 100
                penalty = state['total_points'] * penalty_pct
                state['total_points'] -= penalty
                print(f"[RL-AGENT] VERLUST-SERIE Tag {state['losing_streak_days']}: "
                      f"-{penalty_pct*100:.0f}% = -{penalty:.0f} Punkte "
                      f"(neu: {state['total_points']:.0f})")
        else:
            if state['losing_streak_days'] > 0:
                print(f"[RL-AGENT] Verlust-Serie beendet nach {state['losing_streak_days']} Tagen")
            state['losing_streak_days'] = 0

    state['current_day'] = today
    state['day_pnl'] = 0.0


def _check_week_rollover(state):
    """Wochenwechsel: Prüft Wochen-Serie und gibt Bonus auf Wochenpunkte."""
    now = datetime.now(timezone.utc)
    current_week = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
    if state['current_week'] == current_week:
        return  # Gleiche Woche

    if state['current_week']:
        # Wochenwechsel — alte Woche auswerten
        if state['week_points'] > 0:
            state['winning_streak_weeks'] += 1
            multiplier = 1.20 + (state['winning_streak_weeks'] - 1) * 0.05
            bonus_points = state['week_points'] * (multiplier - 1.0)
            state['total_points'] += bonus_points
            print(f"[RL-AGENT] WOCHEN-BONUS Woche {state['winning_streak_weeks']}: "
                  f"x{multiplier:.2f} auf {state['week_points']:.0f} Punkte = "
                  f"+{bonus_points:.0f} Bonus (neu: {state['total_points']:.0f})")
        else:
            if state['winning_streak_weeks'] > 0:
                print(f"[RL-AGENT] Wochen-Serie beendet nach {state['winning_streak_weeks']} Wochen")
            state['winning_streak_weeks'] = 0

    state['current_week'] = current_week
    state['week_points'] = 0.0


running = True


def signal_handler(sig, frame):
    global running
    print(f"\n[RL-AGENT] Signal {sig} empfangen, stoppe...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# DB Connections
# ============================================================

def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


# ============================================================
# State Persistence (Punkte + Portfolio)
# ============================================================

def load_state():
    """Lade persistenten Agent-State (Punkte, Profit-Tracking, Serien)."""
    if Path(STATE_PATH).exists():
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            # Migration: Alte Felder entfernen/ergänzen
            state.setdefault('total_points', 103645.0)
            state.setdefault('total_profit', 0.0)
            state.setdefault('last_processed_id', 0)
            state.setdefault('losing_streak_days', 0)
            state.setdefault('winning_streak_weeks', 0)
            state.setdefault('current_day', '')
            state.setdefault('current_week', '')
            state.setdefault('day_pnl', 0.0)
            state.setdefault('week_points', 0.0)
            state.setdefault('last_trade_times', {})
            # Alte portfolio-Felder ignorieren (wird jetzt live von Wallets gelesen)
            state.pop('portfolio', None)
            state.pop('peak_portfolio', None)
            print(f"[RL-AGENT] State geladen: {state['total_points']:.0f} Punkte, "
                  f"Profit: ${state['total_profit']:.2f}")
            return state
        except Exception as e:
            print(f"[RL-AGENT] State laden fehlgeschlagen: {e}")

    return {
        'total_points': 103645.0,
        'total_profit': 0.0,
        'last_processed_id': 0,
        'losing_streak_days': 0,
        'winning_streak_weeks': 0,
        'current_day': '',
        'current_week': '',
        'day_pnl': 0.0,
        'week_points': 0.0,
        'last_trade_times': {},
    }


def save_state(state):
    """Speichere Agent-State."""
    try:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[RL-AGENT] State speichern fehlgeschlagen: {e}")


# ============================================================
# DB Queries
# ============================================================

def get_agent_config(conn):
    """Agent-Konfiguration laden."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM rl_agent_config WHERE user_id = 1")
        row = cur.fetchone()
    if not row:
        return {'is_active': False, 'max_concurrent_positions': MAX_CONCURRENT, 'max_leverage': 10}
    return {
        'is_active': row['is_active'],
        'max_concurrent_positions': row['max_concurrent_positions'],
        'max_leverage': row['max_leverage'],
        'base_trade_size': float(row['base_trade_size']) if row.get('base_trade_size') else BASE_TRADE_SIZE,
    }


def get_new_predictions(conn, after_id):
    """Neue CNN-Predictions abholen."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT prediction_id, symbol, direction, confidence,
                   take_profit_price as tp_price, stop_loss_price as sl_price,
                   take_profit_pct, stop_loss_pct,
                   detected_at, entry_price
            FROM momentum_predictions
            WHERE prediction_id > %s AND status = 'active'
            AND (scanner_type = 'default' OR scanner_type IS NULL)
            ORDER BY prediction_id ASC
        """, (after_id,))
        return cur.fetchall()


def get_open_positions_db(conn):
    """Offene RL-Positionen aus DB laden."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, symbol, direction, entry_price, entry_time,
                   position_size_usd, leverage, exchange, prediction_id
            FROM rl_positions
            WHERE status = 'open'
            ORDER BY entry_time ASC
        """)
        return cur.fetchall()


def log_position_open(conn, symbol, direction, entry_price, size_usd, leverage,
                      exchange, prediction_id):
    """Neue Position in DB loggen."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rl_positions
            (symbol, direction, entry_price, entry_time, position_size_usd,
             leverage, status, mode, exchange, prediction_id)
            VALUES (%s, %s, %s, NOW(), %s, %s, 'open', 'live', %s, %s)
            RETURNING id
        """, (symbol, direction, entry_price, size_usd, leverage, exchange, prediction_id))
        pos_id = cur.fetchone()['id']
        conn.commit()
        return pos_id


def log_position_close(conn, pos_id, exit_price, exit_reason, pnl_pct, pnl_usd, duration_min):
    """Position als geschlossen loggen."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE rl_positions SET
                exit_price = %s, exit_time = NOW(), exit_reason = %s,
                pnl_percent = %s, pnl_usd = %s, duration_minutes = %s,
                status = 'closed'
            WHERE id = %s
        """, (exit_price, exit_reason, round(pnl_pct, 4), round(pnl_usd, 2),
              duration_min, pos_id))
        conn.commit()


def log_decision(conn, pos_id, symbol, action, reward, in_position, unrealized_pnl):
    """Agent-Entscheidung loggen."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rl_decisions
            (position_id, timestamp, symbol, action, reward, in_position, unrealized_pnl)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s)
        """, (pos_id, symbol, action, round(reward, 4) if reward else None,
              in_position, round(unrealized_pnl, 4) if unrealized_pnl else None))
        conn.commit()


# ============================================================
# Position Management
# ============================================================

def get_current_price(symbol, prices_hl, exchange):
    """Aktuellen Preis holen (HL oder Binance)."""
    coin = symbol.replace('USDC', '').replace('USDT', '')

    if exchange == 'hyperliquid' and coin in prices_hl:
        return prices_hl[coin]

    if exchange == 'binance':
        try:
            pos = get_binance_position(symbol)
            if pos and pos.get('current_price'):
                return pos['current_price']
        except:
            pass

    # Fallback: HL Preis
    return prices_hl.get(coin, 0)


def calc_pnl_pct(direction, entry_price, current_price):
    """PnL% berechnen."""
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    if direction == 'long':
        return (current_price - entry_price) / entry_price * 100
    else:
        return (entry_price - current_price) / entry_price * 100


def check_stop_loss(direction, entry_price, current_price):
    """SL-Check: Wurde -5% erreicht?"""
    pnl = calc_pnl_pct(direction, entry_price, current_price)
    return pnl <= -SL_PERCENT


def close_position_exchange(creds, pos, hl_coins):
    """Position auf Exchange schließen."""
    symbol = pos['symbol']
    coin = symbol.replace('USDC', '').replace('USDT', '')
    exchange = pos.get('exchange', 'hyperliquid')

    if exchange == 'binance':
        cancel_orders_binance(symbol)
        # Binance: Menge aus Position holen
        binance_pos = get_binance_position(symbol)
        if binance_pos and binance_pos.get('quantity', 0) > 0:
            result = sell_market_binance(symbol, binance_pos['quantity'])
            return result.get('success', False), result.get('avg_price', 0)
        return False, 0
    else:
        cancel_all_orders_for_coin_hl(creds, coin)
        result = close_position_hl(creds, coin, creds['wallet_address'])
        return result.get('success', False), result.get('avg_price', 0)


def _finalize_close(app_conn, state, pos, current_price, exit_price, pnl_pct, exit_reason):
    """Close bestätigt: Punkte verbuchen und DB auf closed setzen."""
    symbol = pos['symbol']
    direction = pos['direction']
    leverage = int(pos['leverage'])
    position_size = float(pos['position_size_usd'])
    entry_time = pos['entry_time']
    pos_id = pos['id']
    now = datetime.now(timezone.utc)
    duration_min = int((now - entry_time).total_seconds() / 60) if entry_time else 0

    reward = _reward(pnl_pct, leverage)

    # Wiederholungs-Penalty: Gleicher Coin <1h → Reward x0.8 (nur bei Gewinn)
    last_trade_times = state.get('last_trade_times', {})
    last_time = last_trade_times.get(symbol)
    if last_time and reward > 0:
        try:
            last_dt = datetime.fromisoformat(last_time)
            if (now - last_dt).total_seconds() < 3600:
                reward *= 0.8
        except:
            pass

    bonus = _point_bonus(state['total_points'])
    state['total_points'] += reward * bonus
    state['week_points'] += reward * bonus

    pnl_dollar = position_size * pnl_pct / 100
    fees = position_size * FEE_RATE * 2
    net_pnl = pnl_dollar - fees
    state['total_profit'] += net_pnl
    state['day_pnl'] += net_pnl

    log_position_close(app_conn, pos_id, exit_price or current_price,
                     exit_reason, pnl_pct, net_pnl, duration_min)
    log_decision(app_conn, pos_id, symbol, -1, reward, False, pnl_pct)

    print(f"[RL-AGENT] {exit_reason.upper()} {symbol} {direction} {leverage}x | "
          f"PnL: {pnl_pct:+.2f}% ${net_pnl:+.2f} | "
          f"Punkte: {state['total_points']:+.0f} (x{bonus})")


def _set_pending_close(app_conn, state, pos_id):
    """Position auf pending_close setzen."""
    with app_conn.cursor() as cur:
        cur.execute("UPDATE rl_positions SET status = 'pending_close' WHERE id = %s", (pos_id,))
        app_conn.commit()
    # Timestamp im State merken
    state.setdefault('pending_closes', {})[str(pos_id)] = {
        'requested_at': datetime.now(timezone.utc).isoformat(),
        'retried': False,
    }


def _set_back_to_open(app_conn, state, pos_id, symbol):
    """Pending_close zurück auf open setzen (Close fehlgeschlagen)."""
    with app_conn.cursor() as cur:
        cur.execute("UPDATE rl_positions SET status = 'open' WHERE id = %s", (pos_id,))
        app_conn.commit()
    state.get('pending_closes', {}).pop(str(pos_id), None)
    print(f"[RL-AGENT] {symbol} zurück auf OPEN (Close fehlgeschlagen, Agent versucht erneut)")


def _close_and_log(app_conn, state, creds, hl_coins, pos, current_price, pnl_pct, exit_reason):
    """Position schließen — bei Erfolg sofort closed, bei Fehler pending_close."""
    symbol = pos['symbol']
    pos_id = pos['id']

    success, exit_price = close_position_exchange(creds, pos, hl_coins)

    if success:
        # Sofort abgeschlossen — Punkte verbuchen
        _finalize_close(app_conn, state, pos, current_price, exit_price, pnl_pct, exit_reason)
    else:
        # Close fehlgeschlagen → pending_close, KEINE Punkte noch
        print(f"[RL-AGENT] Close fehlgeschlagen für {symbol} → pending_close")
        _set_pending_close(app_conn, state, pos_id)
        save_state(state)


def check_pending_closes(app_conn, state, creds, hl_coins):
    """
    Pending-Close Positionen prüfen (jede 30s Iteration).
    - Position/Order nicht mehr auf HL → closed + Punkte
    - 15s pending → Nochmal Sell-Order senden
    - 60s pending → Cancel, prüfen, ggf. zurück auf open
    """
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT id, symbol, direction, entry_price, entry_time,
                   position_size_usd, leverage, exchange, prediction_id
            FROM rl_positions
            WHERE status = 'pending_close'
            ORDER BY entry_time ASC
        """)
        pending = cur.fetchall()

    if not pending:
        return

    # HL State einmal holen für alle pending Positionen
    try:
        hl_pos_list = get_hl_open_positions(creds['wallet_address'])
        hl_coins_open = {p['coin'] for p in hl_pos_list}
    except:
        hl_coins_open = set()

    now = datetime.now(timezone.utc)
    pending_state = state.get('pending_closes', {})

    for pos in pending:
        pos_id = pos['id']
        symbol = pos['symbol']
        coin = symbol.replace('USDC', '').replace('USDT', '')
        direction = pos['direction']
        entry_price = float(pos['entry_price'])

        info = pending_state.get(str(pos_id), {})
        requested_at_str = info.get('requested_at')
        retried = info.get('retried', False)

        if requested_at_str:
            try:
                requested_at = datetime.fromisoformat(requested_at_str)
            except:
                requested_at = now
        else:
            requested_at = now
            pending_state[str(pos_id)] = {'requested_at': now.isoformat(), 'retried': False}

        elapsed = (now - requested_at).total_seconds()

        # Prüfen ob Position noch auf HL existiert
        coin_on_hl = coin in hl_coins_open

        if not coin_on_hl:
            # Position weg von HL → Close bestätigt
            try:
                prices_hl = get_current_prices_hl()
                current_price = prices_hl.get(coin, 0)
            except:
                current_price = 0
            if current_price <= 0:
                current_price = entry_price

            pnl_pct = calc_pnl_pct(direction, entry_price, current_price)
            _finalize_close(app_conn, state, pos, current_price, current_price, pnl_pct, 'agent_exit')
            pending_state.pop(str(pos_id), None)
            print(f"[RL-AGENT] Pending {symbol} bestätigt: Position nicht mehr auf HL")
            continue

        # Position noch auf HL — Retry/Timeout-Logik
        if elapsed >= 60:
            # 60s vergangen → Cancel orders, prüfen, zurück auf open
            try:
                cancel_all_orders_for_coin_hl(creds, coin)
            except:
                pass
            _set_back_to_open(app_conn, state, pos_id, symbol)
            continue

        if elapsed >= 15 and not retried:
            # 15s vergangen → Nochmal Sell-Order senden
            print(f"[RL-AGENT] Pending {symbol}: Retry sell nach {elapsed:.0f}s")
            try:
                close_position_hl(creds, coin, creds['wallet_address'])
            except Exception as e:
                print(f"[RL-AGENT] Retry sell {symbol} fehlgeschlagen: {e}")
            info['retried'] = True

    state['pending_closes'] = pending_state
    save_state(state)


def check_sl_and_timeout(app_conn, state, creds, hl_coins):
    """
    SL + Timeout Check — läuft alle 30s im Hauptloop mit Live-Preisen.
    Reagiert schnell auf Stop-Loss und Timeout, ohne Agent-Entscheidung.
    """
    open_positions = get_open_positions_db(app_conn)
    if not open_positions:
        return

    try:
        prices_hl = get_current_prices_hl()
    except:
        prices_hl = {}

    now = datetime.now(timezone.utc)

    for pos in open_positions:
        symbol = pos['symbol']
        direction = pos['direction']
        entry_price = float(pos['entry_price'])
        exchange = pos.get('exchange', 'hyperliquid')
        entry_time = pos['entry_time']

        current_price = get_current_price(symbol, prices_hl, exchange)
        if current_price <= 0:
            continue

        pnl_pct = calc_pnl_pct(direction, entry_price, current_price)
        duration_min = int((now - entry_time).total_seconds() / 60) if entry_time else 0

        # === SL Check ===
        if check_stop_loss(direction, entry_price, current_price):
            _close_and_log(app_conn, state, creds, hl_coins, pos, current_price, -SL_PERCENT, 'sl')
            continue

        # === Timeout Check (6h) ===
        if duration_min >= MAX_POSITION_AGE:
            _close_and_log(app_conn, state, creds, hl_coins, pos, current_price, pnl_pct, 'timeout')

    save_state(state)


def manage_positions(model, app_conn, coins_conn, state, creds, hl_coins):
    """
    Agent-Entscheidung alle 5 Min: halten oder schließen.
    SL/Timeout werden separat im 30s-Loop geprüft.
    """
    open_positions = get_open_positions_db(app_conn)
    if not open_positions:
        return

    try:
        prices_hl = get_current_prices_hl()
    except:
        prices_hl = {}

    now = datetime.now(timezone.utc)
    n_open = len(open_positions)

    for pos in open_positions:
        symbol = pos['symbol']
        direction = pos['direction']
        entry_price = float(pos['entry_price'])
        leverage = int(pos['leverage'])
        exchange = pos.get('exchange', 'hyperliquid')
        entry_time = pos['entry_time']
        pos_id = pos['id']

        current_price = get_current_price(symbol, prices_hl, exchange)
        if current_price <= 0:
            continue

        pnl_pct = calc_pnl_pct(direction, entry_price, current_price)
        duration_min = int((now - entry_time).total_seconds() / 60) if entry_time else 0

        # === Agent-Entscheidung: halten oder schließen? ===
        pos_state = {
            'in_position': True,
            'direction': 1 if direction == 'long' else -1,
            'unrealized_pnl': pnl_pct,
            'duration_min': duration_min,
        }

        obs = compute_observation_live(
            coins_conn, symbol, now.replace(tzinfo=None),
            position_state=pos_state,
            n_open_positions=n_open,
        )
        mgmt_action, _ = model.predict(obs, deterministic=True)
        mgmt_action = int(mgmt_action)

        if mgmt_action != 0:
            # Agent will schließen
            _close_and_log(app_conn, state, creds, hl_coins, pos, current_price, pnl_pct, 'agent_exit')
        else:
            # Agent will halten
            log_decision(app_conn, pos_id, symbol, 0, None, True, pnl_pct)

    save_state(state)


# ============================================================
# Neue Predictions verarbeiten
# ============================================================

def process_predictions(model, app_conn, coins_conn, state, creds, hl_coins):
    """
    Neue CNN-Predictions verarbeiten.
    Agent entscheidet SELBST: Richtung + Hebel.
    CNN liefert nur Symbol + Zeitpunkt als Trigger.
    """
    predictions = get_new_predictions(app_conn, state['last_processed_id'])
    if not predictions:
        return

    config = get_agent_config(app_conn)
    open_positions = get_open_positions_db(app_conn)
    n_open = len(open_positions)
    now = datetime.now(timezone.utc)

    try:
        prices_hl = get_current_prices_hl()
    except:
        prices_hl = {}

    for pred in predictions:
        state['last_processed_id'] = pred['prediction_id']
        symbol = pred['symbol']
        coin = symbol.replace('USDC', '').replace('USDT', '')

        # Max Positionen erreicht?
        if n_open >= config['max_concurrent_positions']:
            continue

        # Observation bauen (OHNE Position — Entry-Entscheidung)
        obs = compute_observation_live(
            coins_conn, symbol, now.replace(tzinfo=None),
            position_state=None,
            n_open_positions=n_open,
        )
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        if action == 0:
            # Skip
            log_decision(app_conn, None, symbol, 0, None, False, None)
            continue

        # Agent will traden: Richtung + Hebel aus Action
        agent_direction = 'long' if action <= 10 else 'short'
        leverage = LEVERAGE_MAP[action]

        # Hebel-Limit prüfen
        if leverage > config['max_leverage']:
            leverage = config['max_leverage']

        # Drawdown-Modus: Woche im Minus → max 3x
        if state['week_points'] < 0 and leverage > 3:
            leverage = 3

        # Exchange bestimmen (Binance-Trading aktuell deaktiviert)
        use_exchange = None
        if coin in hl_coins:
            use_exchange = 'hyperliquid'
        else:
            # Coin nicht auf HL verfügbar → skip
            log_decision(app_conn, None, symbol, action, None, False, None)
            continue

        # Preis
        current_price = get_current_price(symbol, prices_hl, use_exchange)
        if current_price <= 0:
            continue

        # Trade-Size = Notional, basiert auf echtem Wallet-Guthaben
        wallet_balance = _get_wallet_balance(creds, use_exchange)
        position_size = _calc_trade_size(wallet_balance, config['base_trade_size'])
        margin = position_size / leverage

        # === TRADE AUSFÜHREN ===
        is_buy = agent_direction == 'long'
        trade_success = False
        order_quantity = 0
        entry_price = current_price

        if use_exchange == 'hyperliquid':
            # Frischen Preis direkt vor Order holen
            try:
                fresh_prices = get_current_prices_hl()
                fresh_price = fresh_prices.get(coin, 0)
                if fresh_price > 0:
                    current_price = fresh_price
                    entry_price = fresh_price
            except:
                pass
            limit_price = round(current_price * (0.999 if is_buy else 1.001), 6)
            result = place_limit_order_hl(creds, coin, is_buy, position_size,
                                          limit_price, leverage)
            if result.get('success'):
                trade_success = True
                entry_price = result.get('avg_price', limit_price)
                order_quantity = result.get('quantity', 0)
            else:
                print(f"[RL-AGENT] HL Order fehlgeschlagen: {result.get('error')}")

        elif use_exchange == 'binance':
            # Binance: Spot, kein Hebel (leverage wird nur für Reward-Rechnung genutzt)
            result = buy_spot_binance(symbol, position_size)
            if result.get('success'):
                trade_success = True
                entry_price = result.get('avg_price', current_price)
                order_quantity = result.get('quantity', 0)
            else:
                print(f"[RL-AGENT] Binance Buy fehlgeschlagen: {result.get('error')}")

        if not trade_success:
            log_decision(app_conn, None, symbol, action, None, False, None)
            continue

        # Position in DB loggen (position_size_usd = Notional)
        pos_id = log_position_open(
            app_conn, symbol, agent_direction, entry_price,
            position_size, leverage, use_exchange, pred['prediction_id'],
        )
        log_decision(app_conn, pos_id, symbol, action, None, True, 0.0)

        # Wiederholungs-Tracking: Letzten Trade-Zeitpunkt merken
        state.setdefault('last_trade_times', {})[symbol] = now.isoformat()
        # Alte Einträge aufräumen (>2h)
        state['last_trade_times'] = {
            s: t for s, t in state['last_trade_times'].items()
            if (now - datetime.fromisoformat(t)).total_seconds() < 7200
        }

        n_open += 1

        print(f"[RL-AGENT] TRADE {symbol} {agent_direction} {leverage}x | "
              f"${position_size:.2f} (Margin ${margin:.2f}) @{entry_price:.4f} ({use_exchange}) | "
              f"Wallet: ${wallet_balance:.2f} | "
              f"Punkte: {state['total_points']:+.0f} (x{_point_bonus(state['total_points'])})")

    save_state(state)


# ============================================================
# Main Loop
# ============================================================

def main():
    global running

    print("=" * 70)
    print("  RL-Agent Service V4 — PPO Discrete(21), Live Trading")
    print("  3-Stufen-Reward, Punktesystem, Drawdown-Modus, Wiederholungs-Penalty")
    print("=" * 70)

    # PPO V3 Modell laden
    if not Path(MODEL_PATH).exists():
        print(f"[RL-AGENT] FEHLER: Modell nicht gefunden: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH)
    print(f"[RL-AGENT] PPO V4 Modell geladen: {MODEL_PATH}")

    # State laden
    state = load_state()

    # Hyperliquid Credentials
    creds = get_hl_credentials()
    if not creds:
        print("[RL-AGENT] FEHLER: Keine Hyperliquid Credentials!")
        return

    refresh_hl_coin_info()

    try:
        hl_coins = get_available_coins_hl()
        print(f"[RL-AGENT] {len(hl_coins)} Coins auf Hyperliquid")
    except Exception as e:
        print(f"[RL-AGENT] HL Coins fehlgeschlagen: {e}")
        hl_coins = set()

    # Letzte verarbeitete ID aus DB prüfen (falls State veraltet)
    app_conn = get_conn('app')
    with app_conn.cursor() as cur:
        cur.execute("SELECT MAX(prediction_id) as max_id FROM rl_positions WHERE status IN ('open', 'closed')")
        row = cur.fetchone()
        db_max = row['max_id'] or 0
    if db_max > state['last_processed_id']:
        state['last_processed_id'] = db_max
    app_conn.close()

    print(f"[RL-AGENT] Starte ab Prediction-ID {state['last_processed_id']}")
    print(f"[RL-AGENT] Punkte: {state['total_points']:.0f} (x{_point_bonus(state['total_points'])}) | "
          f"Profit: ${state['total_profit']:.2f} | "
          f"Verlust-Serie: {state['losing_streak_days']}d | "
          f"Wochen-Serie: {state['winning_streak_weeks']}w")
    print()

    last_mgmt_time = 0
    iteration = 0

    while running:
        try:
            # Tag/Wochen-Rollover prüfen
            _check_day_rollover(state)
            _check_week_rollover(state)

            app_conn = get_conn('app')

            # Aktiv?
            config = get_agent_config(app_conn)
            if not config['is_active']:
                app_conn.close()
                time.sleep(POLL_INTERVAL)
                continue

            # 1. SL + Timeout Check (jede Iteration, 30s, Live-Preise)
            check_sl_and_timeout(app_conn, state, creds, hl_coins)

            # 1b. Pending-Close Positionen prüfen (Retry/Bestätigung)
            check_pending_closes(app_conn, state, creds, hl_coins)

            coins_conn = get_conn('coins')

            # 2. Agent-Entscheidung (alle 5 Min)
            now_ts = time.time()
            if now_ts - last_mgmt_time >= MGMT_INTERVAL:
                manage_positions(model, app_conn, coins_conn, state, creds, hl_coins)
                last_mgmt_time = now_ts

            # 3. Neue Predictions verarbeiten
            process_predictions(model, app_conn, coins_conn, state, creds, hl_coins)

            coins_conn.close()
            app_conn.close()

            iteration += 1
            if iteration % 60 == 0:  # Alle ~30 Min Status
                n_open = 0
                try:
                    app_conn = get_conn('app')
                    positions = get_open_positions_db(app_conn)
                    n_open = len(positions)
                    app_conn.close()
                except:
                    pass
                hl_bal = _get_wallet_balance(creds, 'hyperliquid')
                bnb_bal = _get_wallet_balance(creds, 'binance')
                print(f"[RL-AGENT] Status | HL: ${hl_bal:.2f} BNB: ${bnb_bal:.2f} | "
                      f"Profit: ${state['total_profit']:.2f} | "
                      f"Punkte: {state['total_points']:.0f} (x{_point_bonus(state['total_points'])}) | "
                      f"Offen: {n_open} | "
                      f"Verlust: {state['losing_streak_days']}d Wochen: {state['winning_streak_weeks']}w")

        except Exception as e:
            print(f"[RL-AGENT] Fehler im Hauptloop: {e}")
            traceback.print_exc()
            try:
                app_conn.close()
            except:
                pass
            try:
                coins_conn.close()
            except:
                pass

        time.sleep(POLL_INTERVAL)

    save_state(state)
    print("[RL-AGENT] Service gestoppt, State gespeichert")


if __name__ == '__main__':
    main()
