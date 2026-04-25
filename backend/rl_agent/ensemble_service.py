#!/usr/bin/env python3
"""
RL Independent Dual-Agent Trading Service — CNN Pre-Filter + Independent Portfolios.

Architektur:
1. V10 und V11 sind KOMPLETT unabhaengige Trader
2. Jeder hat eigenes virtuelles 2000-Punkte-Konto und 15 virtuelle Slots
3. NUR 15 ECHTE Positionen auf HL insgesamt
4. Agent mit MEHR Punkten = Leader → bekommt die echten Slots
5. Follower handelt rein virtuell (mit Slippage)
6. Bei Konsens (gleicher Coin, gleiche Richtung) → beide buchen die Position
7. Leader-Wechsel nach jedem Verkauf (Punktestand-Vergleich)
8. Experience-Learning: Nur echte Trades (alle 100)

Kein SL, kein Timeout — Agenten entscheiden Exit komplett autonom.

Usage:
    systemctl start rl-ensemble
"""
import json
import time
import signal
import sys
import logging
import random
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from copy import deepcopy

import numpy as np
import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

# Close-Logger
close_logger = logging.getLogger('rl_closes')
close_logger.setLevel(logging.DEBUG)
_cl_handler = logging.FileHandler('/opt/coin/logs/rl_closes.log')
_cl_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
close_logger.addHandler(_cl_handler)
close_logger.propagate = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v8 import LEVERAGE_MAP as LEVERAGE_MAP_V10, TradingEnvV6 as TradingEnvV10
from rl_agent.env_v11 import LEVERAGE_MAP as LEVERAGE_MAP_V11, _reward, TradingEnvV11
from rl_agent.features import compute_observation_live, compute_observation_hl, fetch_hl_ref_candles, N_FEATURES
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
)
from momentum.cnn_predict import get_cnn_model, analyze_symbol_cnn

# ============================================
# KONFIGURATION
# ============================================

SETTINGS_PATH = "/opt/coin/settings.json"
V10_MODEL_PATH = "/opt/coin/database/data/models/rl_ppo_trading_v10.zip"
V11_MODEL_PATH = "/opt/coin/database/data/models/rl_ppo_trading_v11.zip"
V10_STATE_PATH = "/opt/coin/database/data/models/rl_v10_state.json"
V11_STATE_PATH = "/opt/coin/database/data/models/rl_v11_state.json"
FRONTEND_STATE_PATH = "/opt/coin/database/data/models/rl_agent_state.json"

POLL_INTERVAL = 30          # Hauptloop-Takt (Sekunden)
ENTRY_INTERVAL = 300        # CNN-Scan alle 5 Min
MGMT_INTERVAL = 30          # Position-Management alle 30s

MAX_REAL_POSITIONS = 15     # Echte HL-Positionen insgesamt
MAX_VIRTUAL_SLOTS = 15      # Virtuelle Slots pro Agent
CNN_CONFIDENCE_THRESHOLD = 0

FEE_RATE = 0.00035          # 0.035% pro Seite (Hyperliquid Taker)
MAX_TRADES_PER_DAY = 2000
MIN_24H_VOLUME = 20_000
LEARN_BATCH_SIZE = 100      # Pro Agent
LEARN_STEPS = 2000

BASE_TRADE_SIZE = 20.0

VIRTUAL_SLIPPAGE = 0.001    # 0.1% Slippage fuer virtuelle Trades

STABLECOINS = {'USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC'}

# Experience Buffer pro Agent (nur echte Trades)
v10_experience_buffer = []
v11_experience_buffer = []

# Skip-Tracker: Offene Skips die auf +/-1% warten
skip_tracker = []


# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/coin/logs/rl_ensemble.log'),
    ]
)
logger = logging.getLogger('rl_ensemble')

# ============================================
# SIGNAL HANDLER
# ============================================

running = True


def signal_handler(sig, frame):
    global running
    logger.info(f"Signal {sig} empfangen, stoppe...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ============================================
# DB CONNECTIONS
# ============================================

def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


# ============================================
# TELEGRAM
# ============================================

def _send_telegram_alarm(message):
    """Alarm ueber Telegram senden (best-effort)."""
    try:
        s = json.load(open(SETTINGS_PATH))
        bot_token = s.get('telegram', {}).get('bot_token')
        chat_id = s.get('telegram', {}).get('chat_id')
        if not bot_token or not chat_id:
            return
        import urllib.request
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = json.dumps({'chat_id': chat_id, 'text': f"⚠️ RL-Dual: {message}"}).encode()
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ============================================
# STATE PERSISTENCE — PRO AGENT
# ============================================

def _default_agent_state():
    return {
        'total_points': 2000.0,
        'total_profit': 0.0,
        'losing_streak_days': 0,
        'winning_streak_weeks': 0,
        'current_day': '',
        'current_week': '',
        'day_pnl': 0.0,
        'week_points': 0.0,
        'week_points_raw': 0.0,
        'prev_week_raw_points': 0,
        'last_trade_times': {},
        'total_trades': 0,
        'total_skips': 0,
        'total_wins': 0,
        'total_losses': 0,
        'cnn_scans': 0,
        'cnn_signals': 0,
        'trades_today': 0,
        'trade_day': '',
        'pending_closes': {},
        'is_leader': False,
        'virtual_positions': [],
    }


def load_agent_state(version):
    """State fuer einen Agent laden (v10 oder v11)."""
    path = V10_STATE_PATH if version == 'v10' else V11_STATE_PATH
    if Path(path).exists():
        try:
            with open(path) as f:
                state = json.load(f)
            # Defaults setzen fuer fehlende Keys
            defaults = _default_agent_state()
            for k, v in defaults.items():
                state.setdefault(k, v)
            logger.info(f"[{version.upper()}] State geladen: {state['total_points']:.0f} Punkte, "
                        f"Profit: ${state['total_profit']:.2f}, "
                        f"{len(state['virtual_positions'])} virtuelle Positionen")
            return state
        except Exception as e:
            logger.error(f"[{version.upper()}] State laden fehlgeschlagen: {e}")
    return _default_agent_state()


def save_agent_state(version, state):
    """State fuer einen Agent speichern."""
    path = V10_STATE_PATH if version == 'v10' else V11_STATE_PATH
    try:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"[{version.upper()}] State speichern fehlgeschlagen: {e}")


def get_leader(v10_state, v11_state):
    """Welcher Agent hat mehr Punkte? Gibt ('v10'/'v11', leader_state, follower_version, follower_state) zurueck."""
    if v10_state['total_points'] > v11_state['total_points']:
        return 'v10', v10_state, 'v11', v11_state
    else:
        return 'v11', v11_state, 'v10', v10_state


def update_frontend_state(v10_state, v11_state):
    """Gesamtstand fuer Frontend in rl_agent_state.json schreiben."""
    leader_ver, leader_state, follower_ver, follower_state = get_leader(v10_state, v11_state)

    # Echte Positionen zaehlen (aus DB)
    try:
        ac = get_conn('app')
        with ac.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM rl_positions WHERE status = 'open' AND is_real = true")
            real_count = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM rl_positions WHERE status = 'open' AND agent_version = 'v10'")
            v10_real_count = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM rl_positions WHERE status = 'open' AND agent_version = 'v11'")
            v11_real_count = cur.fetchone()['cnt']
        ac.close()
    except:
        real_count = 0
        v10_real_count = 0
        v11_real_count = 0

    combined = {
        'total_points': leader_state['total_points'],
        'total_profit': v10_state['total_profit'] + v11_state['total_profit'],
        'total_trades': v10_state['total_trades'] + v11_state['total_trades'],
        'total_skips': v10_state['total_skips'] + v11_state['total_skips'],
        'total_wins': v10_state['total_wins'] + v11_state['total_wins'],
        'total_losses': v10_state['total_losses'] + v11_state['total_losses'],
        'day_pnl': v10_state['day_pnl'] + v11_state['day_pnl'],
        'week_points': v10_state['week_points'] + v11_state['week_points'],
        'leader': leader_ver.upper(),
        'leader_points': leader_state['total_points'],
        'follower': follower_ver.upper(),
        'follower_points': follower_state['total_points'],
        'v10_points': v10_state['total_points'],
        'v11_points': v11_state['total_points'],
        'v10_trades': v10_state['total_trades'],
        'v11_trades': v11_state['total_trades'],
        'v10_wins': v10_state['total_wins'],
        'v11_wins': v11_state['total_wins'],
        'v10_losses': v10_state['total_losses'],
        'v11_losses': v11_state['total_losses'],
        'v10_profit': v10_state['total_profit'],
        'v11_profit': v11_state['total_profit'],
        'v10_virtual_positions': len(v10_state['virtual_positions']),
        'v11_virtual_positions': len(v11_state['virtual_positions']),
        'v10_real_positions': v10_real_count,
        'v11_real_positions': v11_real_count,
        'v10_total_positions': len(v10_state['virtual_positions']) + v10_real_count,
        'v11_total_positions': len(v11_state['virtual_positions']) + v11_real_count,
        'real_positions': real_count,
        'open_positions': real_count,
        'trades_today': v10_state.get('trades_today', 0) + v11_state.get('trades_today', 0),
        'mode': 'independent_dual',
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(FRONTEND_STATE_PATH, 'w') as f:
            json.dump(combined, f, indent=2)
    except Exception as e:
        logger.error(f"Frontend-State speichern fehlgeschlagen: {e}")


# ============================================
# DAY / WEEK ROLLOVER
# ============================================

def _check_day_rollover(state):
    today = datetime.now().strftime('%Y-%m-%d')  # Berliner Zeit (Server-TZ)
    if state['current_day'] == today:
        return
    state['current_day'] = today
    state['day_pnl'] = 0.0
    state['trades_today'] = 0
    state['trade_day'] = today


def _check_week_rollover(state):
    now = datetime.now(timezone.utc)
    current_week = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
    if state['current_week'] == current_week:
        return
    state['prev_week_raw_points'] = state.get('week_points_raw', 0)
    state['current_week'] = current_week
    state['week_points'] = 0.0
    state['week_points_raw'] = 0.0


# ============================================
# TRADE SIZE
# ============================================

def _calc_trade_size(wallet_balance, base_size=BASE_TRADE_SIZE):
    """Trade-Size: $20 Basis, ab $1000 Wallet 1/50, in 5er Schritten."""
    if wallet_balance < 1000:
        return base_size
    raw = wallet_balance / 50.0
    stepped = int(raw / 5) * 5
    return max(base_size, float(stepped))


# ============================================
# LIQUIDITAETS-FILTER
# ============================================

def update_liquid_coins(tradeable_coins):
    try:
        from hyperliquid.info import Info
        info = Info(skip_ws=True)
        metas = info.meta_and_asset_ctxs()
        assets = metas[0]['universe']
        ctxs = metas[1]

        volumes = {}
        for asset, ctx in zip(assets, ctxs):
            volumes[asset['name']] = float(ctx.get('dayNtlVlm', 0))

        liquid = []
        blocked = []
        for symbol in tradeable_coins:
            coin = symbol.replace('USDC', '').replace('USDT', '')
            vol = volumes.get(coin, 0)
            if vol >= MIN_24H_VOLUME:
                liquid.append(symbol)
            else:
                blocked.append((symbol, vol))

        if blocked:
            logger.info(f"Liquiditaets-Filter: {len(blocked)} Coins gesperrt (< ${MIN_24H_VOLUME:,})")
        logger.info(f"{len(liquid)} liquide Coins von {len(tradeable_coins)}")
        return liquid
    except Exception as e:
        logger.warning(f"Liquiditaets-Check fehlgeschlagen: {e} — verwende alle Coins")
        return tradeable_coins


# ============================================
# DB QUERIES
# ============================================

def get_agent_config(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM rl_agent_config WHERE user_id = 1")
        row = cur.fetchone()
    if not row:
        return {'is_active': False, 'max_concurrent_positions': MAX_REAL_POSITIONS, 'max_leverage': 10}
    return {
        'is_active': row['is_active'],
        'max_concurrent_positions': row['max_concurrent_positions'],
        'max_leverage': row['max_leverage'],
        'base_trade_size': float(row['base_trade_size']) if row.get('base_trade_size') else BASE_TRADE_SIZE,
    }


def get_tradeable_coins(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM coin_info WHERE 'hyperliquid' = ANY(exchanges)")
        coins = [r['symbol'] for r in cur.fetchall()]
    return [c for c in coins if c not in STABLECOINS]


def get_open_positions_db(conn, agent_version=None, is_real=None):
    """Offene Positionen aus DB holen, optional gefiltert."""
    query = """
        SELECT id, symbol, direction, entry_price, entry_time,
               position_size_usd, leverage, agent_version, is_real
        FROM rl_positions
        WHERE status = 'open'
    """
    params = []
    if agent_version is not None:
        query += " AND agent_version = %s"
        params.append(agent_version)
    if is_real is not None:
        query += " AND is_real = %s"
        params.append(is_real)
    query += " ORDER BY entry_time ASC"

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def get_real_position_count(conn):
    """Anzahl echter offener Positionen auf HL."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM rl_positions WHERE status = 'open' AND is_real = true")
        return cur.fetchone()['cnt']


def log_position_open(conn, symbol, direction, entry_price, size_usd, leverage, agent_version, is_real):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rl_positions
            (symbol, direction, entry_price, entry_time, position_size_usd,
             leverage, status, mode, exchange, agent_version, is_real)
            VALUES (%s, %s, %s, NOW(), %s, %s, 'open', 'live', 'hyperliquid', %s, %s)
            RETURNING id
        """, (symbol, direction, entry_price, size_usd, leverage, agent_version, is_real))
        pos_id = cur.fetchone()['id']
        conn.commit()
        return pos_id


def log_position_close(conn, pos_id, exit_price, exit_reason, pnl_pct, pnl_usd, duration_min):
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
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rl_decisions
            (position_id, timestamp, symbol, action, reward, in_position, unrealized_pnl)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s)
        """, (pos_id, symbol, action, round(reward, 4) if reward else None,
              in_position, round(unrealized_pnl, 4) if unrealized_pnl else None))
        conn.commit()


# ============================================
# PNL CALCULATION
# ============================================

def calc_pnl_pct(direction, entry_price, current_price):
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    if direction == 'long':
        return (current_price - entry_price) / entry_price * 100
    else:
        return (entry_price - current_price) / entry_price * 100


def calc_pnl_pct_with_costs(direction, entry_price, current_price, leverage, entry_time):
    """PnL% mit realen Handelskosten (identisch zu env_v11 Training)."""
    raw_pnl = calc_pnl_pct(direction, entry_price, current_price)
    fees = 0.035 * 2
    hours_held = 0.0
    if entry_time:
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_held = (now - entry_time).total_seconds() / 3600
    funding = 0.01 * hours_held * leverage
    return raw_pnl - fees - funding


# ============================================
# VIRTUAL POSITION HELPERS
# ============================================

def _get_virtual_entry_price(current_price, direction):
    """Virtueller Entry-Preis mit Slippage (0.1% schlechter)."""
    if direction == 'long':
        return current_price * (1 + VIRTUAL_SLIPPAGE)  # Hoeher kaufen
    else:
        return current_price * (1 - VIRTUAL_SLIPPAGE)  # Tiefer shorten


def _count_agent_total_positions(state, db_positions_for_agent):
    """Gesamtzahl Positionen eines Agents (echte DB + virtuelle)."""
    return len(db_positions_for_agent) + len(state.get('virtual_positions', []))


def _get_agent_symbols(state, db_positions_for_agent):
    """Alle Symbole die ein Agent haelt (echt + virtuell)."""
    symbols = {p['symbol'] for p in db_positions_for_agent}
    for vp in state.get('virtual_positions', []):
        symbols.add(vp['symbol'])
    return symbols


# ============================================
# SKIP OUTCOMES
# ============================================

def check_skip_outcomes():
    """Prueft offene Skips ob +1% oder -1% erreicht wurde."""
    if not skip_tracker:
        return

    try:
        s = json.load(open(SETTINGS_PATH))
        db_coins = s['databases']['coins']
        coins_conn = psycopg2.connect(
            dbname=db_coins['name'], user=db_coins['user'],
            password=db_coins['password'], host=db_coins['host'], port=db_coins['port'],
        )
    except Exception:
        return

    resolved = []
    for i, skip in enumerate(skip_tracker):
        entry_price = skip['entry_price']
        symbol = skip['symbol']
        skip_time = skip['time'].replace(tzinfo=None) if skip['time'].tzinfo else skip['time']

        try:
            cur = coins_conn.cursor()
            cur.execute("""
                SELECT MAX(high) as max_high, MIN(low) as min_low
                FROM klines
                WHERE symbol = %s AND open_time >= %s
            """, (symbol, skip_time))
            row = cur.fetchone()
            cur.close()
        except Exception:
            coins_conn.rollback()
            continue

        if not row or row[0] is None:
            continue

        max_high = float(row[0])
        min_low = float(row[1])

        high_pct = (max_high - entry_price) / entry_price * 100
        low_pct = (min_low - entry_price) / entry_price * 100

        if high_pct >= 1.0 and low_pct <= -1.0:
            try:
                cur = coins_conn.cursor()
                cur.execute("""
                    SELECT open_time, high, low FROM klines
                    WHERE symbol = %s AND open_time >= %s
                    ORDER BY open_time
                """, (symbol, skip_time))
                candles = cur.fetchall()
                cur.close()
                correct_direction = None
                for c in candles:
                    c_high_pct = (float(c[1]) - entry_price) / entry_price * 100
                    c_low_pct = (float(c[2]) - entry_price) / entry_price * 100
                    if c_high_pct >= 1.0:
                        correct_direction = 'long'
                        pct_change = c_high_pct
                        break
                    if c_low_pct <= -1.0:
                        correct_direction = 'short'
                        pct_change = c_low_pct
                        break
                if correct_direction is None:
                    continue
            except Exception:
                coins_conn.rollback()
                continue
        elif high_pct >= 1.0:
            correct_direction = 'long'
            pct_change = high_pct
        elif low_pct <= -1.0:
            correct_direction = 'short'
            pct_change = low_pct
        else:
            continue

        v10_correct = skip['dir_v10'] == correct_direction
        v11_correct = skip['dir_v11'] == correct_direction

        duration_min = int((datetime.now(timezone.utc) - skip['time']).total_seconds() / 60)

        try:
            s = json.load(open(SETTINGS_PATH))
            db_cfg = s.get('databases', {}).get('learner', s['databases'].get('app', {}))
            conn = psycopg2.connect(
                dbname=db_cfg.get('name', 'learner'), user=db_cfg['user'],
                password=db_cfg['password'], host=db_cfg['host'], port=db_cfg['port'],
            )
            cur = conn.cursor()

            if skip['dir_v10'] != 'skip':
                cur.execute("""
                    INSERT INTO prediction_feedback
                    (prediction_id, scanner_type, symbol, direction, entry_price,
                     detected_at, resolved_at, status, was_correct, actual_result_pct,
                     duration_minutes, time_result, confidence, take_profit_pct, stop_loss_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """, (0, 'ensemble_v10', skip['symbol'], skip['dir_v10'], entry_price,
                      skip['time'], 'hit_tp' if v10_correct else 'hit_sl',
                      v10_correct, round(pct_change, 4), duration_min,
                      'in_time_tp' if v10_correct else 'in_time_sl',
                      50, 1.0, 1.0))

            if skip['dir_v11'] != 'skip':
                cur.execute("""
                    INSERT INTO prediction_feedback
                    (prediction_id, scanner_type, symbol, direction, entry_price,
                     detected_at, resolved_at, status, was_correct, actual_result_pct,
                     duration_minutes, time_result, confidence, take_profit_pct, stop_loss_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """, (0, 'ensemble_v11', skip['symbol'], skip['dir_v11'], entry_price,
                      skip['time'], 'hit_tp' if v11_correct else 'hit_sl',
                      v11_correct, round(pct_change, 4), duration_min,
                      'in_time_tp' if v11_correct else 'in_time_sl',
                      50, 1.0, 1.0))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"[SKIP-FEEDBACK] {skip['symbol']} | {pct_change:+.2f}% in {duration_min}min | "
                        f"Korrekt: {correct_direction} | "
                        f"V10={skip['dir_v10']}({'OK' if v10_correct else 'FALSCH'}) "
                        f"V11={skip['dir_v11']}({'OK' if v11_correct else 'FALSCH'})")
        except Exception as e:
            logger.error(f"[SKIP-FEEDBACK] DB-Fehler: {e}")

        resolved.append(i)

    try:
        coins_conn.close()
    except:
        pass

    for i in sorted(resolved, reverse=True):
        skip_tracker.pop(i)


# ============================================
# POSITION CLOSE LOGIC — ECHTE POSITIONEN
# ============================================

def _finalize_close_real(app_conn, agent_version, state, pos, current_price, exit_price, pnl_pct, exit_reason):
    """Echte Position schliessen und Punkte verbuchen."""
    symbol = pos['symbol']
    direction = pos['direction']
    leverage = int(pos['leverage'])
    position_size = float(pos['position_size_usd'])
    entry_time = pos['entry_time']
    pos_id = pos['id']
    now = datetime.now(timezone.utc)
    duration_min = int((now - entry_time).total_seconds() / 60) if entry_time else 0

    reward = _reward(pnl_pct, leverage)

    # Wiederholungs-Penalty
    last_trade_times = state.get('last_trade_times', {})
    last_time = last_trade_times.get(symbol)
    if last_time and reward > 0:
        try:
            last_dt = datetime.fromisoformat(last_time)
            if (now - last_dt).total_seconds() < 3600:
                reward *= 0.8
        except:
            pass

    state['total_points'] += reward
    state['week_points'] += reward
    state['week_points_raw'] = state.get('week_points_raw', 0) + reward

    pnl_dollar = position_size * pnl_pct / 100
    fees = position_size * FEE_RATE * 2
    net_pnl = pnl_dollar - fees
    state['total_profit'] += net_pnl
    state['day_pnl'] += net_pnl

    if pnl_pct > 0:
        state['total_wins'] = state.get('total_wins', 0) + 1
    else:
        state['total_losses'] = state.get('total_losses', 0) + 1

    log_position_close(app_conn, pos_id, exit_price or current_price,
                       exit_reason, pnl_pct, net_pnl, duration_min)
    log_decision(app_conn, pos_id, symbol, -1, reward, False, pnl_pct)

    ver = agent_version.upper()
    logger.info(f"[REAL] {ver} SELL {symbol} {direction} {leverage}x | "
                f"PnL: {pnl_pct:+.2f}% ${net_pnl:+.2f} | "
                f"{ver}-Punkte: {state['total_points']:.0f} (reward: {reward:+.2f}) | "
                f"Dauer: {duration_min}min")

    # Experience Buffer (nur echte Trades)
    exp_buffer = v10_experience_buffer if agent_version == 'v10' else v11_experience_buffer
    et = entry_time
    if hasattr(et, 'tzinfo') and et.tzinfo:
        et = et.replace(tzinfo=None)
    exp_buffer.append({
        'symbol': symbol,
        'entry_time': et,
        'exit_time': now.replace(tzinfo=None),
    })


def _finalize_close_virtual(agent_version, state, vpos, current_price):
    """Virtuelle Position schliessen und Punkte verbuchen."""
    symbol = vpos['symbol']
    direction = vpos['direction']
    leverage = int(vpos['leverage'])
    entry_price = float(vpos['entry_price'])
    entry_time_str = vpos['entry_time']
    now = datetime.now(timezone.utc)

    entry_time = datetime.fromisoformat(entry_time_str)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    duration_min = int((now - entry_time).total_seconds() / 60)
    pnl_pct = calc_pnl_pct_with_costs(direction, entry_price, current_price, leverage, entry_time)
    reward = _reward(pnl_pct, leverage)

    # Wiederholungs-Penalty
    last_trade_times = state.get('last_trade_times', {})
    last_time = last_trade_times.get(symbol)
    if last_time and reward > 0:
        try:
            last_dt = datetime.fromisoformat(last_time)
            if (now - last_dt).total_seconds() < 3600:
                reward *= 0.8
        except:
            pass

    state['total_points'] += reward
    state['week_points'] += reward
    state['week_points_raw'] = state.get('week_points_raw', 0) + reward

    # Virtueller PnL — mit identischen Fees+Funding wie echt
    # position_size ist virtuell BASE_TRADE_SIZE
    position_size = BASE_TRADE_SIZE
    pnl_dollar = position_size * pnl_pct / 100
    fees = position_size * FEE_RATE * 2
    net_pnl = pnl_dollar - fees
    state['total_profit'] += net_pnl
    state['day_pnl'] += net_pnl

    if pnl_pct > 0:
        state['total_wins'] = state.get('total_wins', 0) + 1
    else:
        state['total_losses'] = state.get('total_losses', 0) + 1

    ver = agent_version.upper()
    logger.info(f"[VIRTUAL] {ver} SELL {symbol} {direction} {leverage}x | "
                f"PnL: {pnl_pct:+.2f}% | {ver}-Punkte: {state['total_points']:.0f} "
                f"(reward: {reward:+.2f}) | Dauer: {duration_min}min")

    # Virtuelle Trades fliessen NICHT ins Experience-Learning


def _set_pending_close(app_conn, state, pos_id):
    with app_conn.cursor() as cur:
        cur.execute("UPDATE rl_positions SET status = 'pending_close' WHERE id = %s", (pos_id,))
        app_conn.commit()
    state.setdefault('pending_closes', {})[str(pos_id)] = {
        'requested_at': datetime.now(timezone.utc).isoformat(),
        'retried': False,
    }


def _set_back_to_open(app_conn, state, pos_id, symbol):
    with app_conn.cursor() as cur:
        cur.execute("UPDATE rl_positions SET status = 'open' WHERE id = %s", (pos_id,))
        app_conn.commit()
    state.get('pending_closes', {}).pop(str(pos_id), None)
    logger.info(f"{symbol} zurueck auf OPEN (Close fehlgeschlagen)")


def _close_and_log_real(app_conn, agent_version, state, creds, pos, current_price, pnl_pct, exit_reason):
    """Echte Position auf HL schliessen."""
    symbol = pos['symbol']
    pos_id = pos['id']
    coin = symbol.replace('USDC', '').replace('USDT', '')

    # Pruefen ob Position auf HL noch existiert
    try:
        hl_positions = get_hl_open_positions(creds['wallet_address'])
        hl_coins_open = {p['coin'] for p in hl_positions}
        if coin not in hl_coins_open:
            close_logger.warning(f"ALREADY_CLOSED {symbol} | Position nicht mehr auf HL")
            _finalize_close_real(app_conn, agent_version, state, pos, current_price, current_price, pnl_pct, exit_reason)
            return
    except Exception as e:
        close_logger.error(f"HL_CHECK_FAILED {symbol} | {e}")

    close_logger.info(f"CLOSE_ATTEMPT {symbol} [{agent_version}] | reason={exit_reason} | "
                      f"price={current_price:.6f} | pnl={pnl_pct:+.2f}%")
    cancel_all_orders_for_coin_hl(creds, coin)
    result = close_position_hl(creds, coin, creds['wallet_address'])

    if result.get('success'):
        exit_price = result.get('avg_price', 0)
        close_logger.info(f"CLOSE_OK {symbol} [{agent_version}] | exit_price={exit_price:.6f}")
        _finalize_close_real(app_conn, agent_version, state, pos, current_price, exit_price, pnl_pct, exit_reason)
    else:
        close_logger.warning(f"CLOSE_FAILED {symbol} [{agent_version}] -> pending_close")
        _set_pending_close(app_conn, state, pos_id)
        save_agent_state(agent_version, state)


def check_pending_closes(app_conn, v10_state, v11_state, creds):
    """Pending-Close Positionen pruefen (jede 30s Iteration)."""
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT id, symbol, direction, entry_price, entry_time,
                   position_size_usd, leverage, agent_version, is_real
            FROM rl_positions
            WHERE status = 'pending_close'
            ORDER BY entry_time ASC
        """)
        pending = cur.fetchall()

    if not pending:
        return

    try:
        hl_pos_list = get_hl_open_positions(creds['wallet_address'])
        hl_coins_open = {p['coin'] for p in hl_pos_list}
    except:
        hl_coins_open = set()

    now = datetime.now(timezone.utc)

    for pos in pending:
        pos_id = pos['id']
        symbol = pos['symbol']
        coin = symbol.replace('USDC', '').replace('USDT', '')
        direction = pos['direction']
        entry_price = float(pos['entry_price'])
        agent_version = pos.get('agent_version') or 'v11'
        state = v10_state if agent_version == 'v10' else v11_state

        pending_state = state.get('pending_closes', {})
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
        coin_on_hl = coin in hl_coins_open

        if not coin_on_hl:
            try:
                prices_hl = get_current_prices_hl()
                current_price = prices_hl.get(coin, 0)
            except:
                current_price = 0
            if current_price <= 0:
                current_price = entry_price
            pnl_pct = calc_pnl_pct(direction, entry_price, current_price)
            close_logger.info(f"PENDING_CONFIRMED {symbol} [{agent_version}] | Position weg von HL")
            _finalize_close_real(app_conn, agent_version, state, pos, current_price, current_price, pnl_pct, 'agent_exit')
            pending_state.pop(str(pos_id), None)
            continue

        if elapsed >= 60:
            close_logger.warning(f"PENDING_TIMEOUT {symbol} [{agent_version}] -> zurueck auf OPEN")
            try:
                cancel_all_orders_for_coin_hl(creds, coin)
            except:
                pass
            _set_back_to_open(app_conn, state, pos_id, symbol)
            continue

        if elapsed >= 15 and not retried:
            close_logger.info(f"PENDING_RETRY {symbol} [{agent_version}]")
            try:
                close_position_hl(creds, coin, creds['wallet_address'])
            except Exception as e:
                logger.warning(f"Retry sell {symbol} fehlgeschlagen: {e}")
            info['retried'] = True

        state['pending_closes'] = pending_state

    save_agent_state('v10', v10_state)
    save_agent_state('v11', v11_state)


# ============================================
# POSITION MANAGEMENT — ECHTE POSITIONEN (PRO AGENT)
# ============================================

def manage_real_positions_for_agent(model, agent_version, state, app_conn, coins_conn, creds,
                                    prices_hl, btc_1h_cache, eth_1h_cache):
    """Ein Agent managed seine echten DB-Positionen."""
    positions = get_open_positions_db(app_conn, agent_version=agent_version, is_real=True)
    if not positions:
        return

    # DB-HL Sync: Positionen die in DB als echt stehen aber nicht auf HL existieren
    try:
        hl_positions = get_hl_open_positions(creds['wallet_address'])
        hl_coins = {p.get('coin', '') for p in hl_positions}
        for pos in positions:
            coin = pos['symbol'].replace('USDC', '').replace('USDT', '')
            if coin not in hl_coins:
                # Geister-Position: In DB offen, auf HL nicht vorhanden
                current_price = prices_hl.get(coin, float(pos['entry_price']))
                pnl_pct = calc_pnl_pct(pos['direction'], float(pos['entry_price']), current_price)
                logger.warning(f"[SYNC] Geister-Position {pos['symbol']} [{agent_version}] — "
                               f"in DB offen aber nicht auf HL. Schliesse in DB. PnL: {pnl_pct:+.2f}%")
                entry_time = pos['entry_time']
                duration_min = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60) if entry_time else 0
                pnl_usd = float(pos['position_size_usd']) * pnl_pct / 100 if pos.get('position_size_usd') else 0
                log_position_close(app_conn, pos['id'], current_price, 'sync_close',
                                   round(pnl_pct, 4), round(pnl_usd, 2), duration_min)
        # Positionen neu laden nach Sync
        positions = get_open_positions_db(app_conn, agent_version=agent_version, is_real=True)
        if not positions:
            return
    except Exception as e:
        logger.debug(f"[SYNC] HL-Check fehlgeschlagen: {e}")

    now = datetime.now(timezone.utc)
    n_open = _count_agent_total_positions(state, positions)
    closed_coins = set()

    for pos in positions:
        symbol = pos['symbol']
        coin = symbol.replace('USDC', '').replace('USDT', '')
        direction = pos['direction']
        entry_price = float(pos['entry_price'])
        leverage = int(pos['leverage'])
        entry_time = pos['entry_time']
        pos_id = pos['id']

        if coin in closed_coins:
            continue

        current_price = prices_hl.get(coin, 0)
        if current_price <= 0:
            continue

        pnl_pct = calc_pnl_pct(direction, entry_price, current_price)
        pnl_pct_with_costs = calc_pnl_pct_with_costs(direction, entry_price, current_price, leverage, entry_time)
        duration_min = int((now - entry_time).total_seconds() / 60) if entry_time else 0

        pos_state = {
            'in_position': True,
            'direction': 1 if direction == 'long' else -1,
            'unrealized_pnl': pnl_pct_with_costs,
            'duration_min': duration_min,
        }

        try:
            obs = compute_observation_hl(
                coins_conn, symbol,
                position_state=pos_state,
                n_open_positions=n_open,
                btc_1h_cache=btc_1h_cache,
                eth_1h_cache=eth_1h_cache,
            )
        except ConnectionError:
            continue

        mgmt_action, _ = model.predict(obs, deterministic=False)
        mgmt_action = int(mgmt_action)

        # Zwangsverkauf nach 12h (720 Min)
        if duration_min >= 180 and mgmt_action == 0:
            mgmt_action = 1
            close_logger.info(f"FORCE_SELL_3H {symbol} [{agent_version}] {direction} {leverage}x | "
                              f"PnL: {pnl_pct:+.2f}% | Dauer: {duration_min}min")

        if mgmt_action != 0:
            close_logger.info(f"SELL_DECISION {symbol} [{agent_version}] {direction} {leverage}x | "
                              f"PnL: {pnl_pct:+.2f}% | Dauer: {duration_min}min | Action: {mgmt_action}")
            _close_and_log_real(app_conn, agent_version, state, creds, pos, current_price, pnl_pct, 'agent_exit')
            closed_coins.add(coin)
        else:
            log_decision(app_conn, pos_id, symbol, 0, None, True, pnl_pct)

    save_agent_state(agent_version, state)


# ============================================
# POSITION MANAGEMENT — VIRTUELLE POSITIONEN (PRO AGENT)
# ============================================

def manage_virtual_positions_for_agent(model, agent_version, state, coins_conn,
                                       prices_hl, btc_1h_cache, eth_1h_cache):
    """Ein Agent managed seine virtuellen Positionen (im State-File)."""
    virtual_positions = state.get('virtual_positions', [])
    if not virtual_positions:
        return

    now = datetime.now(timezone.utc)
    # Gesamtzahl Positionen dieses Agents
    n_open = len(virtual_positions)  # + echte werden separat gezaehlt, aber fuer Obs reicht virtuell
    remaining = []
    changed = False

    for vpos in virtual_positions:
        symbol = vpos['symbol']
        coin = symbol.replace('USDC', '').replace('USDT', '')
        direction = vpos['direction']
        entry_price = float(vpos['entry_price'])
        leverage = int(vpos['leverage'])
        entry_time_str = vpos['entry_time']

        entry_time = datetime.fromisoformat(entry_time_str)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        current_price = prices_hl.get(coin, 0)
        if current_price <= 0:
            remaining.append(vpos)
            continue

        pnl_pct_with_costs = calc_pnl_pct_with_costs(direction, entry_price, current_price, leverage, entry_time)
        duration_min = int((now - entry_time).total_seconds() / 60)

        pos_state = {
            'in_position': True,
            'direction': 1 if direction == 'long' else -1,
            'unrealized_pnl': pnl_pct_with_costs,
            'duration_min': duration_min,
        }

        try:
            obs = compute_observation_hl(
                coins_conn, symbol,
                position_state=pos_state,
                n_open_positions=n_open,
                btc_1h_cache=btc_1h_cache,
                eth_1h_cache=eth_1h_cache,
            )
        except ConnectionError:
            remaining.append(vpos)
            continue

        mgmt_action, _ = model.predict(obs, deterministic=False)
        mgmt_action = int(mgmt_action)

        # Zwangsverkauf nach 12h (720 Min)
        if duration_min >= 180 and mgmt_action == 0:
            mgmt_action = 1
            logger.info(f"[VIRTUAL] FORCE_SELL_3H {symbol} [{agent_version}] {direction} {leverage}x | "
                        f"PnL-Costs: {pnl_pct_with_costs:+.2f}% | Dauer: {duration_min}min")

        if mgmt_action != 0:
            # Verkaufen (virtuell)
            _finalize_close_virtual(agent_version, state, vpos, current_price)
            changed = True
            # Position wird NICHT in remaining aufgenommen -> entfernt
        else:
            remaining.append(vpos)

    state['virtual_positions'] = remaining

    if changed:
        save_agent_state(agent_version, state)


# ============================================
# INDEPENDENT ENTRY DECISION
# ============================================

def _get_agent_direction(action, leverage_map):
    """Action -> (direction, leverage)."""
    if 1 <= action <= 10:
        return 'long', leverage_map.get(action, 1)
    elif 11 <= action <= 20:
        return 'short', leverage_map.get(action, 1)
    else:
        return 'skip', 0


def independent_entry_decision(model_v10, model_v11, v10_state, v11_state, obs,
                                symbol, current_price, app_conn, creds, config,
                                wallet_balance, cnn_result=None):
    """
    Beide Agenten entscheiden UNABHAENGIG.
    Jeder bekommt seine Position (echt oder virtuell).
    Leader bekommt echte Slots, Follower handelt virtuell.
    Bei Konsens (gleicher Coin, gleiche Richtung) → beide buchen.

    Returns: Anzahl echte Trades die geoeffnet wurden (0 oder 1)
    """
    action_v10, _ = model_v10.predict(obs, deterministic=False)
    action_v11, _ = model_v11.predict(obs, deterministic=False)
    action_v10 = int(action_v10)
    action_v11 = int(action_v11)

    dir_v10, lev_v10 = _get_agent_direction(action_v10, LEVERAGE_MAP_V10)
    dir_v11, lev_v11 = _get_agent_direction(action_v11, LEVERAGE_MAP_V11)

    leader_ver, leader_state, follower_ver, follower_state = get_leader(v10_state, v11_state)
    now = datetime.now(timezone.utc)
    coin = symbol.replace('USDC', '').replace('USDT', '')

    # Leverage-Cap
    max_lev = config['max_leverage']

    # Drawdown-Modus
    if v10_state['week_points'] < 0 and lev_v10 > 3:
        lev_v10 = 3
    if v11_state['week_points'] < 0 and lev_v11 > 3:
        lev_v11 = 3

    if lev_v10 > max_lev:
        lev_v10 = max_lev
    if lev_v11 > max_lev:
        lev_v11 = max_lev

    # Beide skippen?
    if dir_v10 == 'skip' and dir_v11 == 'skip':
        logger.info(f"[DUAL] SKIP {symbol} | V10=skip V11=skip")
        v10_state['total_skips'] = v10_state.get('total_skips', 0) + 1
        v11_state['total_skips'] = v11_state.get('total_skips', 0) + 1
        return 0

    # Echte Positionen auf HL zaehlen
    real_count = get_real_position_count(app_conn)
    real_slot_available = real_count < MAX_REAL_POSITIONS

    # Positionen pro Agent zaehlen (echte DB + virtuelle)
    v10_db_positions = get_open_positions_db(app_conn, agent_version='v10')
    v11_db_positions = get_open_positions_db(app_conn, agent_version='v11')
    v10_total = _count_agent_total_positions(v10_state, v10_db_positions)
    v11_total = _count_agent_total_positions(v11_state, v11_db_positions)
    v10_symbols = _get_agent_symbols(v10_state, v10_db_positions)
    v11_symbols = _get_agent_symbols(v11_state, v11_db_positions)

    trades_opened = 0

    # Konsens pruefen: gleicher Coin, gleiche Richtung
    consensus = (dir_v10 != 'skip' and dir_v11 != 'skip' and dir_v10 == dir_v11)

    # V10 Entscheidung verarbeiten
    if dir_v10 != 'skip' and symbol not in v10_symbols and v10_total < MAX_VIRTUAL_SLOTS:
        is_leader_v10 = (leader_ver == 'v10')
        do_real_v10 = is_leader_v10 and real_slot_available

        if do_real_v10:
            # Echter Trade auf HL
            position_size = _calc_trade_size(wallet_balance, config.get('base_trade_size', BASE_TRADE_SIZE))
            margin = position_size / lev_v10
            if margin <= wallet_balance * 0.5:
                # Frischer Preis
                try:
                    fresh_prices = get_current_prices_hl()
                    fresh = fresh_prices.get(coin, 0)
                    if fresh > 0:
                        current_price_v10 = fresh
                    else:
                        current_price_v10 = current_price
                except:
                    current_price_v10 = current_price

                is_buy = dir_v10 == 'long'
                limit_price = round(current_price_v10 * (0.999 if is_buy else 1.001), 6)
                result = place_limit_order_hl(creds, coin, is_buy, position_size, limit_price, lev_v10)

                if result.get('success'):
                    entry_price = result.get('avg_price', limit_price)
                    pos_id = log_position_open(app_conn, symbol, dir_v10, entry_price,
                                               position_size, lev_v10, 'v10', True)
                    log_decision(app_conn, pos_id, symbol, action_v10, None, True, 0.0)

                    v10_state.setdefault('last_trade_times', {})[symbol] = now.isoformat()
                    v10_state['trades_today'] = v10_state.get('trades_today', 0) + 1
                    v10_state['total_trades'] = v10_state.get('total_trades', 0) + 1
                    trades_opened += 1
                    real_count += 1
                    real_slot_available = real_count < MAX_REAL_POSITIONS

                    logger.info(f"[REAL] V10 BUY {symbol} {dir_v10} {lev_v10}x | "
                                f"${position_size:.2f} @{entry_price:.4f} | V10: {v10_state['total_points']:.0f}pts")
                else:
                    logger.warning(f"HL Order fehlgeschlagen V10 {symbol}: {result.get('error')}")
            else:
                # Margin zu hoch -> virtuell
                do_real_v10 = False

        if not do_real_v10:
            # Virtueller Trade
            virtual_entry = _get_virtual_entry_price(current_price, dir_v10)
            vpos = {
                'symbol': symbol,
                'direction': dir_v10,
                'leverage': lev_v10,
                'entry_price': virtual_entry,
                'entry_time': now.isoformat(),
                'is_real': False,
            }
            v10_state['virtual_positions'].append(vpos)
            v10_state['trades_today'] = v10_state.get('trades_today', 0) + 1
            v10_state['total_trades'] = v10_state.get('total_trades', 0) + 1

            logger.info(f"[VIRTUAL] V10 BUY {symbol} {dir_v10} {lev_v10}x | "
                        f"@{virtual_entry:.4f} | V10: {v10_state['total_points']:.0f}pts")
    elif dir_v10 == 'skip':
        v10_state['total_skips'] = v10_state.get('total_skips', 0) + 1

    # V11 Entscheidung verarbeiten
    if dir_v11 != 'skip' and symbol not in v11_symbols and v11_total < MAX_VIRTUAL_SLOTS:
        is_leader_v11 = (leader_ver == 'v11')
        do_real_v11 = is_leader_v11 and real_slot_available

        if do_real_v11:
            # Echter Trade auf HL
            position_size = _calc_trade_size(wallet_balance, config.get('base_trade_size', BASE_TRADE_SIZE))
            margin = position_size / lev_v11
            if margin <= wallet_balance * 0.5:
                try:
                    fresh_prices = get_current_prices_hl()
                    fresh = fresh_prices.get(coin, 0)
                    if fresh > 0:
                        current_price_v11 = fresh
                    else:
                        current_price_v11 = current_price
                except:
                    current_price_v11 = current_price

                is_buy = dir_v11 == 'long'
                limit_price = round(current_price_v11 * (0.999 if is_buy else 1.001), 6)
                result = place_limit_order_hl(creds, coin, is_buy, position_size, limit_price, lev_v11)

                if result.get('success'):
                    entry_price = result.get('avg_price', limit_price)
                    pos_id = log_position_open(app_conn, symbol, dir_v11, entry_price,
                                               position_size, lev_v11, 'v11', True)
                    log_decision(app_conn, pos_id, symbol, action_v11, None, True, 0.0)

                    v11_state.setdefault('last_trade_times', {})[symbol] = now.isoformat()
                    v11_state['trades_today'] = v11_state.get('trades_today', 0) + 1
                    v11_state['total_trades'] = v11_state.get('total_trades', 0) + 1
                    trades_opened += 1
                    real_count += 1
                    real_slot_available = real_count < MAX_REAL_POSITIONS

                    logger.info(f"[REAL] V11 BUY {symbol} {dir_v11} {lev_v11}x | "
                                f"${position_size:.2f} @{entry_price:.4f} | V11: {v11_state['total_points']:.0f}pts")
                else:
                    logger.warning(f"HL Order fehlgeschlagen V11 {symbol}: {result.get('error')}")
            else:
                do_real_v11 = False

        if not do_real_v11:
            # Virtueller Trade
            virtual_entry = _get_virtual_entry_price(current_price, dir_v11)
            vpos = {
                'symbol': symbol,
                'direction': dir_v11,
                'leverage': lev_v11,
                'entry_price': virtual_entry,
                'entry_time': now.isoformat(),
                'is_real': False,
            }
            v11_state['virtual_positions'].append(vpos)
            v11_state['trades_today'] = v11_state.get('trades_today', 0) + 1
            v11_state['total_trades'] = v11_state.get('total_trades', 0) + 1

            logger.info(f"[VIRTUAL] V11 BUY {symbol} {dir_v11} {lev_v11}x | "
                        f"@{virtual_entry:.4f} | V11: {v11_state['total_points']:.0f}pts")
    elif dir_v11 == 'skip':
        v11_state['total_skips'] = v11_state.get('total_skips', 0) + 1

    # Konsens-Log
    if consensus:
        logger.info(f"[CONSENSUS] V10+V11 einig: {symbol} {dir_v10} | "
                    f"V10 {lev_v10}x, V11 {lev_v11}x")

    # Skip-Tracker (wenn mindestens einer nicht skippt)
    if dir_v10 == 'skip' or dir_v11 == 'skip':
        skip_tracker.append({
            'symbol': symbol,
            'coin': coin,
            'entry_price': current_price,
            'time': now,
            'dir_v10': dir_v10,
            'dir_v11': dir_v11,
            'action_v10': action_v10,
            'action_v11': action_v11,
        })

    # Wiederholungs-Tracking aufraumen
    for st in (v10_state, v11_state):
        st['last_trade_times'] = {
            s: t for s, t in st.get('last_trade_times', {}).items()
            if (now - datetime.fromisoformat(t)).total_seconds() < 7200
        }

    return trades_opened


# ============================================
# CNN SCAN + INDEPENDENT ENTRY
# ============================================

def cnn_scan_and_independent_entry(model_v10, model_v11, app_conn, coins_conn,
                                    v10_state, v11_state, creds, liquid_coins):
    """
    1. CNN scannt alle liquiden Coins (randomisiert)
    2. Bei CNN-Signal: Beide Agenten entscheiden unabhaengig
    """
    config = get_agent_config(app_conn)
    now = datetime.now(timezone.utc)

    try:
        prices_hl = get_current_prices_hl()
    except:
        prices_hl = {}

    wallet_balance = get_hl_balance(creds['wallet_address'])

    # Tages-Trade-Limit
    today = now.strftime('%Y-%m-%d')
    if v10_state.get('trade_day') != today:
        v10_state['trade_day'] = today
        v10_state['trades_today'] = 0
    if v11_state.get('trade_day') != today:
        v11_state['trade_day'] = today
        v11_state['trades_today'] = 0

    # Alle offenen Symbole (echt + virtuell, beide Agents)
    all_open_symbols = set()
    try:
        all_db_pos = get_open_positions_db(app_conn)
        for p in all_db_pos:
            all_open_symbols.add(p['symbol'])
    except:
        pass
    for vp in v10_state.get('virtual_positions', []):
        all_open_symbols.add(vp['symbol'])
    for vp in v11_state.get('virtual_positions', []):
        all_open_symbols.add(vp['symbol'])

    # Echte HL-Positionen als Sicherheitscheck
    try:
        hl_positions = get_hl_open_positions(creds['wallet_address'])
        for p in hl_positions:
            hl_sym = p.get('coin', '') + 'USDC'
            all_open_symbols.add(hl_sym)
    except:
        pass

    trades_opened = 0

    # CNN-Scan: Alle Coins randomisiert
    shuffled_coins = list(liquid_coins)
    random.shuffle(shuffled_coins)

    signals_this_scan = 0

    leader_ver = get_leader(v10_state, v11_state)[0]

    for symbol in shuffled_coins:
        if (v10_state.get('trades_today', 0) + v11_state.get('trades_today', 0)) >= MAX_TRADES_PER_DAY:
            break

        # Beide Agenten voll?
        v10_db = get_open_positions_db(app_conn, agent_version='v10')
        v11_db = get_open_positions_db(app_conn, agent_version='v11')
        v10_total = _count_agent_total_positions(v10_state, v10_db)
        v11_total = _count_agent_total_positions(v11_state, v11_db)
        if v10_total >= MAX_VIRTUAL_SLOTS and v11_total >= MAX_VIRTUAL_SLOTS:
            break

        # Symbol schon offen bei BEIDEN? Dann skip
        # (ein Agent kann den Coin haben, der andere nicht — das ist OK)
        v10_has = symbol in _get_agent_symbols(v10_state, v10_db)
        v11_has = symbol in _get_agent_symbols(v11_state, v11_db)
        if v10_has and v11_has:
            continue

        coin = symbol.replace('USDC', '').replace('USDT', '')
        current_price = prices_hl.get(coin, 0)
        if current_price <= 0:
            continue

        # Observation bauen (DB-Daten fuer Entry)
        n_open_total = v10_total + v11_total
        obs = compute_observation_live(
            coins_conn, symbol, now.replace(tzinfo=None),
            position_state=None,
            n_open_positions=n_open_total,
        )

        # Unabhaengige Entry Decision
        opened = independent_entry_decision(
            model_v10, model_v11, v10_state, v11_state, obs,
            symbol, current_price, app_conn, creds, config,
            wallet_balance,
        )
        trades_opened += opened

    logger.info(f"[DUAL] Scan: {len(shuffled_coins)} Coins, {trades_opened} Trades")

    # States speichern
    save_agent_state('v10', v10_state)
    save_agent_state('v11', v11_state)

    return trades_opened


# ============================================
# LEADER-WECHSEL CHECK
# ============================================

def check_leader_switch(v10_state, v11_state):
    """Nach jedem Verkauf: Leader-Wechsel pruefen."""
    old_leader_v10 = v10_state.get('is_leader', False)
    old_leader_v11 = v11_state.get('is_leader', False)

    new_leader_ver = get_leader(v10_state, v11_state)[0]

    v10_state['is_leader'] = (new_leader_ver == 'v10')
    v11_state['is_leader'] = (new_leader_ver == 'v11')

    # Wechsel erkennen
    if new_leader_ver == 'v10' and not old_leader_v10:
        logger.info(f"[LEADER-SWITCH] V10 ({v10_state['total_points']:.0f}) "
                    f"uebernimmt von V11 ({v11_state['total_points']:.0f})")
    elif new_leader_ver == 'v11' and not old_leader_v11:
        logger.info(f"[LEADER-SWITCH] V11 ({v11_state['total_points']:.0f}) "
                    f"uebernimmt von V10 ({v10_state['total_points']:.0f})")


# ============================================
# EXPERIENCE LEARNING — PRO AGENT (NUR ECHTE TRADES)
# ============================================

def experience_learn_agent(model, agent_version, env_class, model_path, experiences):
    """Ein Agent lernt aus seinen eigenen echten Trade-Erfahrungen."""
    try:
        from rl_agent.learner import preload_market_data

        traded_coins = list(set(e['symbol'] for e in experiences))
        earliest = min(e['entry_time'] for e in experiences)
        latest = max(e['exit_time'] for e in experiences)

        all_coins = traded_coins[:]
        for ref in ['BTCUSDC', 'ETHUSDC']:
            if ref not in all_coins:
                all_coins.append(ref)

        data_start = earliest - timedelta(hours=12)

        coins_conn = get_conn('coins')
        market_data = preload_market_data(coins_conn, all_coins, data_start, latest)
        coins_conn.close()

        span_h = (latest - earliest).total_seconds() / 3600

        env = env_class(
            coins=traded_coins,
            market_data=market_data,
            trade_start=earliest,
            trade_end=latest,
            train_start=earliest,
        )
        model.set_env(env)
        model.learn(total_timesteps=LEARN_STEPS, reset_num_timesteps=False)
        model.save(model_path)
        logger.info(f"[DUAL] {agent_version.upper()} gelernt: {LEARN_STEPS} Steps, "
                    f"{len(traded_coins)} Coins, {span_h:.1f}h | Env-Punkte: {env.total_points:.0f}")

    except Exception as e:
        logger.error(f"[DUAL] {agent_version.upper()} Experience-Lernen fehlgeschlagen: {e}")
        traceback.print_exc()


# ============================================
# MAIN LOOP
# ============================================

def main():
    global running

    print("=" * 70)
    print("  RL Independent Dual-Agent — CNN + V10/V11 Independent Portfolios")
    print(f"  CNN-Threshold: {CNN_CONFIDENCE_THRESHOLD}% | Max: {MAX_REAL_POSITIONS} echte HL-Positionen")
    print(f"  {MAX_VIRTUAL_SLOTS} virtuelle Slots pro Agent")
    print(f"  Entry: {ENTRY_INTERVAL}s, Management: {MGMT_INTERVAL}s")
    print(f"  Lernen: Nach je {LEARN_BATCH_SIZE} echten Trades PRO Agent")
    print(f"  V10: {V10_MODEL_PATH}")
    print(f"  V11: {V11_MODEL_PATH}")
    print("=" * 70)

    # Modelle pruefen
    if not Path(V10_MODEL_PATH).exists():
        logger.error(f"V10 Modell nicht gefunden: {V10_MODEL_PATH}")
        return
    if not Path(V11_MODEL_PATH).exists():
        logger.error(f"V11 Modell nicht gefunden: {V11_MODEL_PATH}")
        return

    # CNN laden
    cnn = get_cnn_model()
    if cnn is None:
        logger.error("CNN-Modell konnte nicht geladen werden!")
        return
    logger.info("CNN-Modell geladen")

    # RL-Modelle laden
    model_v10 = PPO.load(V10_MODEL_PATH)
    logger.info(f"V10 Modell geladen: {V10_MODEL_PATH}")

    model_v11 = PPO.load(V11_MODEL_PATH)
    logger.info(f"V11 Modell geladen: {V11_MODEL_PATH}")

    # States pro Agent laden
    v10_state = load_agent_state('v10')
    v11_state = load_agent_state('v11')

    # Leader initial setzen
    leader_ver = get_leader(v10_state, v11_state)[0]
    v10_state['is_leader'] = (leader_ver == 'v10')
    v11_state['is_leader'] = (leader_ver == 'v11')
    logger.info(f"[DUAL] LEADER: {leader_ver.upper()} "
                f"(V10: {v10_state['total_points']:.0f} vs V11: {v11_state['total_points']:.0f})")

    creds = get_hl_credentials()
    if not creds:
        logger.error("Keine Hyperliquid Credentials!")
        return

    refresh_hl_coin_info()

    # DB-Schema sicherstellen (is_real Spalte)
    try:
        ac = get_conn('app')
        with ac.cursor() as cur:
            cur.execute("ALTER TABLE rl_positions ADD COLUMN IF NOT EXISTS is_real boolean DEFAULT true")
            ac.commit()
        ac.close()
        logger.info("DB-Schema: is_real Spalte sichergestellt")
    except Exception as e:
        logger.warning(f"DB-Schema Anpassung fehlgeschlagen (evtl. schon vorhanden): {e}")

    # Tradeable Coins + Liquiditaets-Filter
    app_conn = get_conn('app')
    tradeable_coins = get_tradeable_coins(app_conn)
    app_conn.close()
    liquid_coins = update_liquid_coins(tradeable_coins)
    last_volume_check = time.time()

    logger.info(f"[V10] Punkte: {v10_state['total_points']:.0f} | Profit: ${v10_state['total_profit']:.2f} | "
                f"Trades: {v10_state.get('total_trades', 0)} | "
                f"Virtuelle Pos: {len(v10_state['virtual_positions'])} | "
                f"Leader: {v10_state['is_leader']}")
    logger.info(f"[V11] Punkte: {v11_state['total_points']:.0f} | Profit: ${v11_state['total_profit']:.2f} | "
                f"Trades: {v11_state.get('total_trades', 0)} | "
                f"Virtuelle Pos: {len(v11_state['virtual_positions'])} | "
                f"Leader: {v11_state['is_leader']}")

    last_scan_time = 0
    last_mgmt_time = 0
    last_leader_check = time.time()
    iteration = 0

    while running:
        try:
            # Day/Week Rollover fuer beide
            _check_day_rollover(v10_state)
            _check_day_rollover(v11_state)
            _check_week_rollover(v10_state)
            _check_week_rollover(v11_state)

            app_conn = get_conn('app')

            config = get_agent_config(app_conn)
            if not config['is_active']:
                app_conn.close()
                time.sleep(POLL_INTERVAL)
                continue

            # 1. Pending-Close Check (echte Positionen)
            check_pending_closes(app_conn, v10_state, v11_state, creds)

            # 2. Skip-Outcomes pruefen
            try:
                check_skip_outcomes()
            except:
                pass

            now_ts = time.time()

            # Liquiditaets-Check 1x taeglich
            if now_ts - last_volume_check >= 86400:
                liquid_coins = update_liquid_coins(tradeable_coins)
                last_volume_check = now_ts

            # 3. Position-Management (alle 30s) — PRO AGENT, echt + virtuell
            if now_ts - last_mgmt_time >= MGMT_INTERVAL:
                coins_conn = get_conn('coins')

                try:
                    prices_hl = get_current_prices_hl()
                except:
                    prices_hl = {}

                try:
                    btc_1h_cache, eth_1h_cache = fetch_hl_ref_candles()
                except ConnectionError as e:
                    logger.error(f"ALARM: HL API nicht erreichbar! ({e})")
                    coins_conn.close()
                    app_conn.close()
                    time.sleep(POLL_INTERVAL)
                    continue

                # V10: Echte Positionen
                manage_real_positions_for_agent(
                    model_v10, 'v10', v10_state, app_conn, coins_conn, creds,
                    prices_hl, btc_1h_cache, eth_1h_cache,
                )
                # V11: Echte Positionen
                manage_real_positions_for_agent(
                    model_v11, 'v11', v11_state, app_conn, coins_conn, creds,
                    prices_hl, btc_1h_cache, eth_1h_cache,
                )

                # V10: Virtuelle Positionen
                manage_virtual_positions_for_agent(
                    model_v10, 'v10', v10_state, coins_conn,
                    prices_hl, btc_1h_cache, eth_1h_cache,
                )
                # V11: Virtuelle Positionen
                manage_virtual_positions_for_agent(
                    model_v11, 'v11', v11_state, coins_conn,
                    prices_hl, btc_1h_cache, eth_1h_cache,
                )

                # Leader-Wechsel alle 2 Stunden pruefen
                if now_ts - last_leader_check >= 7200:
                    check_leader_switch(v10_state, v11_state)
                    last_leader_check = now_ts

                coins_conn.close()
                last_mgmt_time = now_ts

            # 4. CNN-Scan + Independent Entry (alle 5 Min)
            if now_ts - last_scan_time >= ENTRY_INTERVAL:
                coins_conn = get_conn('coins')
                cnn_scan_and_independent_entry(model_v10, model_v11, app_conn, coins_conn,
                                                v10_state, v11_state, creds, liquid_coins)
                coins_conn.close()
                last_scan_time = now_ts

            # 5. Experience-Learning PRO AGENT (alle 100 echte Trades)
            if len(v10_experience_buffer) >= LEARN_BATCH_SIZE:
                logger.info(f"[DUAL] V10 Experience-Lernen: {len(v10_experience_buffer)} Erfahrungen")
                experience_learn_agent(model_v10, 'v10', TradingEnvV10, V10_MODEL_PATH,
                                       list(v10_experience_buffer))
                v10_experience_buffer.clear()

            if len(v11_experience_buffer) >= LEARN_BATCH_SIZE:
                logger.info(f"[DUAL] V11 Experience-Lernen: {len(v11_experience_buffer)} Erfahrungen")
                experience_learn_agent(model_v11, 'v11', TradingEnvV11, V11_MODEL_PATH,
                                       list(v11_experience_buffer))
                v11_experience_buffer.clear()

            # 6. Frontend State updaten
            update_frontend_state(v10_state, v11_state)

            app_conn.close()

            iteration += 1
            if iteration % 60 == 0:  # Alle ~30 Min Status
                try:
                    ac = get_conn('app')
                    n_real = get_real_position_count(ac)
                    n_v10_real = len(get_open_positions_db(ac, agent_version='v10', is_real=True))
                    n_v11_real = len(get_open_positions_db(ac, agent_version='v11', is_real=True))
                    ac.close()
                except:
                    n_real = 0
                    n_v10_real = 0
                    n_v11_real = 0

                n_v10_virt = len(v10_state.get('virtual_positions', []))
                n_v11_virt = len(v11_state.get('virtual_positions', []))
                hl_bal = get_hl_balance(creds['wallet_address'])
                leader_ver = get_leader(v10_state, v11_state)[0]

                v10_wr = v10_state['total_wins'] / (v10_state['total_wins'] + v10_state['total_losses']) * 100 \
                    if (v10_state['total_wins'] + v10_state['total_losses']) > 0 else 0
                v11_wr = v11_state['total_wins'] / (v11_state['total_wins'] + v11_state['total_losses']) * 100 \
                    if (v11_state['total_wins'] + v11_state['total_losses']) > 0 else 0

                logger.info(
                    f"[DUAL] LEADER: {leader_ver.upper()} "
                    f"(V10: {v10_state['total_points']:.0f} vs V11: {v11_state['total_points']:.0f})")
                logger.info(
                    f"[DUAL] Status | HL: ${hl_bal:.2f} | Real: {n_real}/{MAX_REAL_POSITIONS} | "
                    f"V10: {n_v10_real}R+{n_v10_virt}V Pos, {v10_state['total_trades']}T, "
                    f"WR {v10_wr:.0f}%, ${v10_state['total_profit']:.2f} | "
                    f"V11: {n_v11_real}R+{n_v11_virt}V Pos, {v11_state['total_trades']}T, "
                    f"WR {v11_wr:.0f}%, ${v11_state['total_profit']:.2f} | "
                    f"Buf V10:{len(v10_experience_buffer)}/{LEARN_BATCH_SIZE} "
                    f"V11:{len(v11_experience_buffer)}/{LEARN_BATCH_SIZE}")

        except Exception as e:
            logger.error(f"Fehler im Hauptloop: {e}")
            traceback.print_exc()
            try:
                app_conn.close()
            except:
                pass

        time.sleep(POLL_INTERVAL)

    # Shutdown: States speichern
    save_agent_state('v10', v10_state)
    save_agent_state('v11', v11_state)
    update_frontend_state(v10_state, v11_state)
    logger.info("Independent Dual-Agent gestoppt, States gespeichert")


if __name__ == '__main__':
    main()
