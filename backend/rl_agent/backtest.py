#!/usr/bin/env python3
"""
RL-Agent Backtest V2 — PPO Discrete(9) auf CNN-Predictions.

Lädt V2-Modell (Discrete(9) mit Hebel), lernt weiter auf CNN-Predictions.
Agent entscheidet SELBST: Richtung + Hebel. CNN liefert nur Symbol + Zeitpunkt.
Positionen laufen PARALLEL — nicht sequentiell.

Budget: $1000 Start, $20/Trade, ab $2000 → 1/100
Max 50 gleichzeitige Positionen, Hebel aus Agent-Action.

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/backtest.py > /opt/coin/logs/rl_backtest.log 2>&1 &
"""
import sys
import json
import time
import bisect
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v2 import TradingEnvV2, LEVERAGE_MAP
from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SETTINGS_PATH = "/opt/coin/settings.json"
RESULTS_PATH = "/opt/coin/logs/rl_backtest_results.json"
TRADES_LOG_PATH = "/opt/coin/logs/rl_backtest_trades.json"

TEST_START = datetime(2026, 1, 1)
TEST_END = datetime(2026, 3, 9)

SL_PERCENT = 5.0
STEP_MINUTES = 5
FEE_RATE = 0.001  # 0.1% pro Seite

# Budget
START_BALANCE = 1000.0
TRADE_SIZE = 20.0
MAX_FRACTION = 1 / 100
MAX_CONCURRENT = 50

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


# ============================================================
# Market-Daten vorladen
# ============================================================

def preload_market_data(coins_conn, symbols, start, end):
    print("=" * 70)
    print("  Market-Daten vorladen")
    print("=" * 70)

    market_data = {
        'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
    }

    load_start = start - timedelta(days=14)
    load_end = end + timedelta(days=1)

    for table in ['agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']:
        print(f"  {table}...", end='', flush=True)
        total_rows = 0

        for symbol in symbols:
            with coins_conn.cursor() as cur:
                cur.execute(f"""
                    SELECT bucket, open, high, low, close, volume,
                           number_of_trades, taker_buy_base_asset_volume
                    FROM {table}
                    WHERE symbol = %s AND bucket >= %s AND bucket <= %s
                    ORDER BY bucket ASC
                """, (symbol, load_start, load_end))
                rows = cur.fetchall()

            if rows:
                timestamps = []
                for r in rows:
                    ts = r['bucket']
                    if hasattr(ts, 'tzinfo') and ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                    timestamps.append(ts)

                market_data[table][symbol] = {
                    'timestamps': timestamps,
                    'open': np.array([float(r['open']) for r in rows], dtype=np.float32),
                    'high': np.array([float(r['high']) for r in rows], dtype=np.float32),
                    'low': np.array([float(r['low']) for r in rows], dtype=np.float32),
                    'close': np.array([float(r['close']) for r in rows], dtype=np.float32),
                    'volume': np.array([float(r['volume'] or 0) for r in rows], dtype=np.float32),
                    'trades': np.array([float(r['number_of_trades'] or 0) for r in rows], dtype=np.float32),
                    'taker': np.array([float(r['taker_buy_base_asset_volume'] or 0) for r in rows], dtype=np.float32),
                }
                total_rows += len(rows)

        print(f" {total_rows:,} Zeilen, {len(market_data[table])} Coins")

    for ref_symbol, key in [('BTCUSDC', 'btc_1h'), ('ETHUSDC', 'eth_1h')]:
        data = market_data['agg_1h'].get(ref_symbol, {})
        market_data[key] = data
        if data:
            print(f"  {ref_symbol}: {len(data['timestamps'])} 1h-Candles")

    print("  kline_metrics...", end='', flush=True)
    km_data = {}
    km_cols = ['pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
               'pct_300m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m']
    total_km = 0

    for symbol in symbols:
        with coins_conn.cursor() as cur:
            cur.execute("""
                SELECT open_time, pct_30m, pct_60m, pct_90m, pct_120m, pct_180m, pct_240m,
                       pct_300m, pct_360m, pct_420m, pct_480m, pct_540m, pct_600m
                FROM kline_metrics
                WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time ASC
            """, (symbol, load_start, load_end))
            rows = cur.fetchall()

        if rows:
            timestamps = []
            for r in rows:
                ts = r['open_time']
                if hasattr(ts, 'tzinfo') and ts.tzinfo:
                    ts = ts.replace(tzinfo=None)
                timestamps.append(ts)

            entry = {'timestamps': timestamps}
            for col in km_cols:
                entry[col] = np.array(
                    [float(r[col]) if r[col] is not None else 0.0 for r in rows],
                    dtype=np.float32,
                )
            km_data[symbol] = entry
            total_km += len(rows)

    market_data['kline_metrics'] = km_data
    print(f" {total_km:,} Zeilen, {len(km_data)} Coins")

    total_bytes = 0
    for table in ['agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']:
        for sym_data in market_data[table].values():
            for arr in sym_data.values():
                if isinstance(arr, np.ndarray):
                    total_bytes += arr.nbytes
    print(f"\n  RAM: {total_bytes / 1024 / 1024:.0f} MB")
    print()
    return market_data


# ============================================================
# Hilfsfunktionen
# ============================================================

def _get_close(market_data, symbol, current_time):
    data = market_data.get('agg_5m', {}).get(symbol)
    if not data:
        return 0.0
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0 or idx >= len(data['close']):
        return 0.0
    return float(data['close'][idx])


def _check_sl(market_data, symbol, current_time, entry_price, direction):
    data = market_data.get('agg_5m', {}).get(symbol)
    if not data:
        return False
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0 or idx >= len(data['high']):
        return False
    if direction == 1:
        sl_level = entry_price * (1 - SL_PERCENT / 100)
        return float(data['low'][idx]) <= sl_level
    else:
        sl_level = entry_price * (1 + SL_PERCENT / 100)
        return float(data['high'][idx]) >= sl_level


def _slice(market_data, table, symbol, current_time, n_candles):
    data = market_data.get(table, {}).get(symbol)
    if not data or 'timestamps' not in data:
        return empty_candles()
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0:
        return empty_candles()
    start = max(0, idx - n_candles + 1)
    end = idx + 1
    return {k: data[k][start:end] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}


def _slice_ref(market_data, key, current_time, n_candles):
    data = market_data.get(key)
    if not data or 'timestamps' not in data:
        return empty_candles()
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0:
        return empty_candles()
    start = max(0, idx - n_candles + 1)
    end = idx + 1
    return {k: data[k][start:end] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}


def _get_km(market_data, symbol, current_time):
    km_data = market_data.get('kline_metrics', {}).get(symbol)
    if not km_data or 'timestamps' not in km_data:
        return None
    idx = bisect.bisect_right(km_data['timestamps'], current_time) - 1
    if idx < 0:
        return None
    result = {}
    for col in ['pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
                 'pct_300m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m']:
        if col in km_data:
            result[col] = float(km_data[col][idx])
    return result


def _get_obs(market_data, symbol, current_time, position_state=None, n_open=0):
    candles_5m = _slice(market_data, 'agg_5m', symbol, current_time, 144)
    candles_1h = _slice(market_data, 'agg_1h', symbol, current_time, 24)
    candles_4h = _slice(market_data, 'agg_4h', symbol, current_time, 12)
    candles_1d = _slice(market_data, 'agg_1d', symbol, current_time, 14)
    btc_1h = _slice_ref(market_data, 'btc_1h', current_time, 25)
    eth_1h = _slice_ref(market_data, 'eth_1h', current_time, 25)
    km = _get_km(market_data, symbol, current_time)

    return compute_observation(
        candles_5m, candles_1h, candles_4h, candles_1d,
        btc_1h, eth_1h,
        current_time.hour, current_time.weekday(),
        kline_metrics=km,
        position_state=position_state,
        n_open_positions=n_open,
    )


# ============================================================
# Backtest — Zeitbasiert, parallele Positionen
# ============================================================

def run_backtest():
    t_start = time.time()

    print("=" * 70)
    print("  RL-Agent Backtest V2 — Discrete(9) mit Hebel")
    print(f"  Zeitraum: {TEST_START.date()} → {TEST_END.date()}")
    print(f"  Budget: ${START_BALANCE}, Trade: ${TRADE_SIZE}, ab $2000 → 1/{int(1/MAX_FRACTION)}")
    print(f"  Max Positionen: {MAX_CONCURRENT}, Hebel aus Agent-Action")
    print(f"  Agent entscheidet Richtung SELBST (kein CNN-Hint)")
    print("=" * 70)
    print()

    # === CNN-Predictions laden ===
    learner_conn = get_conn('learner')
    with learner_conn.cursor() as cur:
        cur.execute("""
            SELECT prediction_id, symbol, direction, confidence,
                   take_profit_pct, stop_loss_pct, entry_price,
                   detected_at, resolved_at, status
            FROM prediction_feedback
            WHERE detected_at >= %s AND detected_at < %s
            ORDER BY detected_at ASC
        """, (TEST_START, TEST_END))
        predictions = cur.fetchall()
    learner_conn.close()

    for p in predictions:
        if hasattr(p['detected_at'], 'tzinfo') and p['detected_at'].tzinfo:
            p['detected_at'] = p['detected_at'].replace(tzinfo=None)
        if p['resolved_at'] and hasattr(p['resolved_at'], 'tzinfo') and p['resolved_at'].tzinfo:
            p['resolved_at'] = p['resolved_at'].replace(tzinfo=None)

    total = len(predictions)
    tp_count = sum(1 for p in predictions if p['status'] == 'hit_tp')
    sl_count = sum(1 for p in predictions if p['status'] == 'hit_sl')
    exp_count = sum(1 for p in predictions if p['status'] == 'expired')
    baseline_hr = tp_count / (tp_count + sl_count) * 100 if (tp_count + sl_count) > 0 else 0

    print(f"  {total} CNN-Predictions geladen")
    print(f"  {tp_count} TP, {sl_count} SL, {exp_count} Expired")
    print(f"  Baseline HR: {baseline_hr:.1f}%")
    print()

    # Symbole
    symbols = list(set(p['symbol'] for p in predictions))
    if 'BTCUSDC' not in symbols:
        symbols.append('BTCUSDC')
    if 'ETHUSDC' not in symbols:
        symbols.append('ETHUSDC')
    for s in STABLECOINS:
        if s in symbols:
            symbols.remove(s)

    # === Market-Daten ===
    coins_conn = get_conn('coins')
    market_data = preload_market_data(coins_conn, symbols, TEST_START, TEST_END)
    coins_conn.close()

    # === Opportunities für TradingEnv ===
    opportunities = []
    for p in predictions:
        opportunities.append({
            'symbol': p['symbol'],
            'detection_time': p['detected_at'],
            'actual_direction': 'up' if p['direction'] == 'long' else 'down',
        })

    # === V2 Modell laden + Weiterlernen ===
    MODEL_V2_PATH = "/opt/coin/database/data/models/rl_ppo_trading_v2.zip"
    env = TradingEnvV2(opportunities, market_data, train_start=TEST_START)

    try:
        model = PPO.load(MODEL_V2_PATH, env=env)
        print(f"\n  V2 Modell geladen: {MODEL_V2_PATH}")
    except Exception as e:
        print(f"FEHLER: V2 Modell laden fehlgeschlagen: {e}")
        return

    print()
    print("=" * 70)
    print("  Weiterlernen auf CNN-Predictions")
    print("=" * 70)

    learn_steps = 500_000
    print(f"  {learn_steps:,} Timesteps...")
    model.learn(total_timesteps=learn_steps)
    print(f"  Weiterlernen fertig ({time.time()-t_start:.0f}s)")
    print()

    # === Zeitbasierter Backtest ===
    print("=" * 70)
    print("  Backtest (zeitbasiert, parallele Positionen)")
    print("=" * 70)
    print()

    # Predictions nach Zeitpunkt indizieren
    pred_by_time = defaultdict(list)
    for i, p in enumerate(predictions):
        # Auf 5-Min-Raster runden
        dt = p['detected_at']
        minute = (dt.minute // STEP_MINUTES) * STEP_MINUTES
        slot = dt.replace(minute=minute, second=0, microsecond=0)
        pred_by_time[slot].append(i)

    # Zeitachse: alle 5 Min von TEST_START bis TEST_END
    portfolio = START_BALANCE  # $0
    open_positions = []  # {idx, symbol, direction, entry_price, entry_time, margin, leverage, steps}
    trade_log = []       # Alle abgeschlossenen Trades
    results = []         # Alle Entscheidungen
    peak_portfolio = START_BALANCE
    low_portfolio = START_BALANCE

    current_time = TEST_START
    last_print = TEST_START
    total_margin_locked = 0.0

    while current_time < TEST_END:

        # === 1. Offene Positionen managen ===
        still_open = []
        for pos in open_positions:
            pos['steps'] += 1

            price = _get_close(market_data, pos['symbol'], current_time)
            if price <= 0:
                # Keine Daten → schließen mit 0 PnL
                pnl_pct = 0.0
                pnl_dollar = 0.0
                fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
                portfolio += pos['margin'] - fees  # Margin zurück - Fees
                total_margin_locked -= pos['margin']
                trade_log.append({
                    'symbol': pos['symbol'], 'direction': 'long' if pos['direction'] == 1 else 'short',
                    'entry_price': pos['entry_price'], 'exit_price': 0,
                    'entry_time': str(pos['entry_time']), 'exit_time': str(current_time),
                    'pnl_pct': 0.0, 'pnl_dollar': round(-fees, 2),
                    'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
                    'steps': pos['steps'], 'exit_reason': 'no_data',
                    'cnn_direction': pos['cnn_direction'], 'actual_status': pos['actual_status'],
                })
                continue

            # SL prüfen
            if _check_sl(market_data, pos['symbol'], current_time, pos['entry_price'], pos['direction']):
                pnl_pct = -SL_PERCENT
                pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
                fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
                portfolio += pos['margin'] + pnl_dollar - fees
                total_margin_locked -= pos['margin']
                trade_log.append({
                    'symbol': pos['symbol'], 'direction': 'long' if pos['direction'] == 1 else 'short',
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'entry_time': str(pos['entry_time']), 'exit_time': str(current_time),
                    'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(pnl_dollar - fees, 2),
                    'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
                    'steps': pos['steps'], 'exit_reason': 'sl',
                    'cnn_direction': pos['cnn_direction'], 'actual_status': pos['actual_status'],
                })
                continue

            # Unrealisierter PnL
            if pos['direction'] == 1:
                unrealized = (price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                unrealized = (pos['entry_price'] - price) / pos['entry_price'] * 100

            # Agent-Entscheidung: halten oder schließen?
            pos_state = {
                'in_position': True,
                'direction': pos['direction'],
                'unrealized_pnl': unrealized,
                'duration_min': pos['steps'] * STEP_MINUTES,
            }
            obs = _get_obs(market_data, pos['symbol'], current_time,
                          position_state=pos_state, n_open=len(open_positions))
            mgmt_action, _ = model.predict(obs, deterministic=True)
            mgmt_action = int(mgmt_action)

            if mgmt_action != 0:
                # Agent will schließen
                pnl_pct = unrealized
                pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
                fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
                portfolio += pos['margin'] + pnl_dollar - fees
                total_margin_locked -= pos['margin']
                trade_log.append({
                    'symbol': pos['symbol'], 'direction': 'long' if pos['direction'] == 1 else 'short',
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'entry_time': str(pos['entry_time']), 'exit_time': str(current_time),
                    'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(pnl_dollar - fees, 2),
                    'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
                    'steps': pos['steps'], 'exit_reason': 'agent_exit',
                    'cnn_direction': pos['cnn_direction'], 'actual_status': pos['actual_status'],
                })
                continue

            # Timeout: 6h max
            if pos['steps'] >= 6 * 60 // STEP_MINUTES:
                pnl_pct = unrealized
                pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
                fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
                portfolio += pos['margin'] + pnl_dollar - fees
                total_margin_locked -= pos['margin']
                trade_log.append({
                    'symbol': pos['symbol'], 'direction': 'long' if pos['direction'] == 1 else 'short',
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'entry_time': str(pos['entry_time']), 'exit_time': str(current_time),
                    'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(pnl_dollar - fees, 2),
                    'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
                    'steps': pos['steps'], 'exit_reason': 'timeout',
                    'cnn_direction': pos['cnn_direction'], 'actual_status': pos['actual_status'],
                })
                continue

            # Position bleibt offen
            still_open.append(pos)

        open_positions = still_open

        # === 2. Neue Predictions in diesem Zeitslot ===
        slot = current_time
        if slot in pred_by_time:
            for pred_idx in pred_by_time[slot]:
                pred = predictions[pred_idx]
                symbol = pred['symbol']
                cnn_direction = pred['direction']
                actual_status = pred['status']
                n_open = len(open_positions)

                # Agent-Entscheidung
                obs = _get_obs(market_data, symbol, current_time, position_state=None, n_open=n_open)
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                if action == 0 or n_open >= MAX_CONCURRENT:
                    results.append({
                        'symbol': symbol, 'cnn_direction': cnn_direction,
                        'actual_status': actual_status, 'agent_action': 'skip',
                        'detected_at': str(current_time), 'portfolio': round(portfolio - total_margin_locked, 2),
                        'n_open': n_open,
                    })
                    continue

                entry_price = _get_close(market_data, symbol, current_time)
                if entry_price <= 0:
                    results.append({
                        'symbol': symbol, 'cnn_direction': cnn_direction,
                        'actual_status': actual_status, 'agent_action': 'skip',
                        'detected_at': str(current_time), 'portfolio': round(portfolio - total_margin_locked, 2),
                        'n_open': n_open,
                    })
                    continue

                # V2: Action 1-4 = long, 5-8 = short, Hebel aus LEVERAGE_MAP
                agent_direction = 1 if action <= 4 else -1
                agent_dir_str = 'long' if action <= 4 else 'short'
                leverage = LEVERAGE_MAP[action]

                # Positionsgröße
                current_portfolio = portfolio - total_margin_locked
                if current_portfolio > 2000:
                    margin = current_portfolio * MAX_FRACTION
                else:
                    margin = TRADE_SIZE

                # Margin abziehen
                portfolio -= margin
                total_margin_locked += margin

                open_positions.append({
                    'idx': pred_idx, 'symbol': symbol, 'direction': agent_direction,
                    'entry_price': entry_price, 'entry_time': current_time,
                    'margin': margin, 'leverage': leverage, 'steps': 0,
                    'cnn_direction': cnn_direction, 'actual_status': actual_status,
                })

                results.append({
                    'symbol': symbol, 'cnn_direction': cnn_direction,
                    'actual_status': actual_status, 'agent_action': 'trade',
                    'agent_direction': agent_dir_str, 'leverage': leverage,
                    'margin': round(margin, 2),
                    'detected_at': str(current_time), 'portfolio': round(portfolio - total_margin_locked, 2),
                    'n_open': len(open_positions),
                })

        # Portfolio tracken (free balance = portfolio - locked margin)
        current_portfolio = portfolio - total_margin_locked
        if current_portfolio > peak_portfolio:
            peak_portfolio = current_portfolio
        if current_portfolio < low_portfolio:
            low_portfolio = current_portfolio

        # Status print
        if (current_time - last_print).total_seconds() >= 86400:  # Täglich
            print(f"  {current_time.date()} | Portfolio: ${current_portfolio:+.2f} "
                  f"| Open: {len(open_positions)} | Trades: {len(trade_log)} ({time.time()-t_start:.0f}s)")
            last_print = current_time

        current_time += timedelta(minutes=STEP_MINUTES)

    # Letzte Positionen schließen
    for pos in open_positions:
        price = _get_close(market_data, pos['symbol'], TEST_END)
        if price > 0 and pos['entry_price'] > 0:
            if pos['direction'] == 1:
                pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price'] * 100
            if pnl_pct < -SL_PERCENT:
                pnl_pct = -SL_PERCENT
        else:
            pnl_pct = 0.0
        pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
        fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
        portfolio += pos['margin'] + pnl_dollar - fees
        total_margin_locked -= pos['margin']
        trade_log.append({
            'symbol': pos['symbol'], 'direction': 'long' if pos['direction'] == 1 else 'short',
            'entry_price': pos['entry_price'], 'exit_price': price,
            'entry_time': str(pos['entry_time']), 'exit_time': str(TEST_END),
            'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(pnl_dollar - fees, 2),
            'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
            'steps': pos['steps'], 'exit_reason': 'end_of_test',
            'cnn_direction': pos['cnn_direction'], 'actual_status': pos['actual_status'],
        })

    final_portfolio = round(portfolio - total_margin_locked, 2)
    elapsed = time.time() - t_start

    # === REPORT ===
    print()
    print("=" * 70)
    print("  ERGEBNISSE")
    print("=" * 70)
    print()

    trades_taken = [r for r in results if r['agent_action'] == 'trade']
    skips = [r for r in results if r['agent_action'] == 'skip']

    print(f"--- Übersicht ---")
    print(f"  Predictions gesamt:  {total}")
    print(f"  Agent tradet:        {len(trades_taken)} ({len(trades_taken)/total*100:.1f}%)")
    print(f"  Agent skippt:        {len(skips)} ({len(skips)/total*100:.1f}%)")
    print(f"  Abgeschlossene:      {len(trade_log)}")
    print()

    # Richtung
    if trades_taken:
        agent_long = [t for t in trades_taken if t.get('agent_direction') == 'long']
        agent_short = [t for t in trades_taken if t.get('agent_direction') == 'short']
        cnn_agree = [t for t in trades_taken if t.get('agent_direction') == t['cnn_direction']]

        print(f"--- Richtungswahl ---")
        print(f"  Agent Long:          {len(agent_long)}")
        print(f"  Agent Short:         {len(agent_short)}")
        print(f"  Stimmt mit CNN:      {len(cnn_agree)} ({len(cnn_agree)/len(trades_taken)*100:.1f}%)")
        print()

    # Konto
    print(f"--- Konto ---")
    print(f"  Start:               ${START_BALANCE:.2f}")
    print(f"  Ende:                ${final_portfolio:.2f}")
    print(f"  Peak:                ${peak_portfolio:.2f}")
    print(f"  Tief:                ${low_portfolio:.2f}")
    if peak_portfolio > 0:
        max_dd = (peak_portfolio - low_portfolio) / peak_portfolio * 100
        print(f"  Max Drawdown:        {max_dd:.1f}%")
    print()

    # Performance aus Trade-Log
    if trade_log:
        wins = [t for t in trade_log if t['pnl_dollar'] > 0]
        losses = [t for t in trade_log if t['pnl_dollar'] < 0]
        even = [t for t in trade_log if t['pnl_dollar'] == 0]

        win_rate = len(wins) / len(trade_log) * 100
        total_pnl = sum(t['pnl_dollar'] for t in trade_log)
        avg_pnl = np.mean([t['pnl_pct'] for t in trade_log])
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        avg_lev = np.mean([t['leverage'] for t in trade_log])

        print(f"--- Performance ---")
        print(f"  Win-Rate:            {win_rate:.1f}% ({len(wins)} W / {len(losses)} L / {len(even)} 0)")
        print(f"  Avg PnL/Trade:       {avg_pnl:+.3f}%")
        print(f"  Avg Win:             {avg_win:+.3f}%")
        print(f"  Avg Loss:            {avg_loss:+.3f}%")
        print(f"  Total PnL ($):       ${total_pnl:+.2f}")
        print(f"  Ø Hebel:             {avg_lev:.1f}x")
        print()

    # Hebel-Verteilung
    if trade_log:
        lev_dist = defaultdict(int)
        for t in trade_log:
            lev_dist[t['leverage']] += 1
        print(f"--- Hebel-Verteilung ---")
        for lev in sorted(lev_dist):
            lev_trades = [t for t in trade_log if t['leverage'] == lev]
            lev_pnl = sum(t['pnl_dollar'] for t in lev_trades)
            lev_wr = sum(1 for t in lev_trades if t['pnl_dollar'] > 0) / len(lev_trades) * 100
            print(f"  {lev:>2}x: {lev_dist[lev]:>5} Trades | WR: {lev_wr:.0f}% | PnL: ${lev_pnl:+.2f}")
        print()

    # Exit-Gründe
    if trade_log:
        exit_reasons = defaultdict(int)
        for t in trade_log:
            exit_reasons[t['exit_reason']] += 1
        print(f"--- Exit-Gründe ---")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trade_log) * 100
            avg = np.mean([t['pnl_pct'] for t in trade_log if t['exit_reason'] == reason])
            dollar = sum(t['pnl_dollar'] for t in trade_log if t['exit_reason'] == reason)
            print(f"  {reason:<16} {count:>6} ({pct:.1f}%)  Ø PnL: {avg:+.3f}%  ${dollar:+.2f}")
        print()

    # Skips
    if skips:
        skip_tp = sum(1 for s in skips if s['actual_status'] == 'hit_tp')
        skip_sl = sum(1 for s in skips if s['actual_status'] == 'hit_sl')
        skip_exp = sum(1 for s in skips if s['actual_status'] == 'expired')
        skip_hr = skip_tp / (skip_tp + skip_sl) * 100 if (skip_tp + skip_sl) > 0 else 0

        print(f"--- Geskippte Predictions ---")
        print(f"  Gesamt:              {len(skips)}")
        print(f"  Davon wären TP:      {skip_tp} ({skip_hr:.1f}%)")
        print(f"  Davon wären SL:      {skip_sl}")
        print(f"  Davon Expired:       {skip_exp}")
        print()

    # Wochen
    if trade_log:
        print(f"--- Wochen ---")
        weekly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0})
        for t in trade_log:
            dt = datetime.fromisoformat(t['entry_time'])
            w = dt.strftime('%Y-KW%W')
            weekly[w]['trades'] += 1
            weekly[w]['pnl'] += t['pnl_dollar']
            if t['pnl_dollar'] > 0:
                weekly[w]['wins'] += 1
            elif t['pnl_dollar'] < 0:
                weekly[w]['losses'] += 1

        print(f"  {'Woche':<12} {'Trades':>7} {'W/L':>8} {'HR':>6} {'PnL $':>12}")
        for w in sorted(weekly):
            d = weekly[w]
            hr = f"{d['wins']/d['trades']*100:.0f}%" if d['trades'] else "-"
            wl = f"{d['wins']}/{d['losses']}" if d['trades'] else "-"
            print(f"  {w:<12} {d['trades']:>7} {wl:>8} {hr:>6} ${d['pnl']:>+11.2f}")
        print()

    print(f"Laufzeit: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # === Trade-Log speichern ===
    try:
        with open(TRADES_LOG_PATH, 'w') as f:
            json.dump(trade_log, f, indent=2, default=str)
        print(f"Trade-Log gespeichert: {TRADES_LOG_PATH} ({len(trade_log)} Trades)")
    except Exception as e:
        print(f"Trade-Log speichern fehlgeschlagen: {e}")

    # === Ergebnisse speichern ===
    try:
        with open(RESULTS_PATH, 'w') as f:
            json.dump({
                'test_period': f"{TEST_START.date()} → {TEST_END.date()}",
                'start_balance': START_BALANCE,
                'final_portfolio': final_portfolio,
                'peak': round(peak_portfolio, 2),
                'low': round(low_portfolio, 2),
                'total_predictions': total,
                'baseline_hr': round(baseline_hr, 2),
                'trades_taken': len(trades_taken),
                'trades_completed': len(trade_log),
                'skips': len(skips),
                'results': results,
            }, f, indent=2, default=str)
        print(f"Ergebnisse gespeichert: {RESULTS_PATH}")
    except Exception as e:
        print(f"Ergebnisse speichern fehlgeschlagen: {e}")


if __name__ == '__main__':
    run_backtest()
