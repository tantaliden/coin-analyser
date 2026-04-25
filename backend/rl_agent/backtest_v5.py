#!/usr/bin/env python3
"""
RL-Agent Backtest V5 — Autonomous Agent (ohne CNN-Predictions).

Agent sieht alle HL-Coins und entscheidet selbst wann und was er handelt.
Kein CNN, keine Predictions, kein Skip-Penalty.
60min max Haltezeit, SL -5%.

Zeitraum: 01.01.2026 → 13.03.2026

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/backtest_v5.py > /opt/coin/logs/backtest_v5.log 2>&1 &
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

from rl_agent.env_v5 import TradingEnvV5, LEVERAGE_MAP, _reward, MAX_HOLD_MINUTES
from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SETTINGS_PATH = "/opt/coin/settings.json"
RESULTS_PATH = "/opt/coin/logs/rl_backtest_v5_results.json"
TRADES_LOG_PATH = "/opt/coin/logs/rl_backtest_v5_trades.json"
MODEL_PATH = "/opt/coin/database/data/models/rl_ppo_trading_v5.zip"

TEST_START = datetime(2026, 1, 1)
TEST_END = datetime(2026, 3, 14)

SL_PERCENT = 5.0
STEP_MINUTES = 5
FEE_RATE = 0.001

START_BALANCE = 1000.0
BASE_TRADE_SIZE = 15.0
MAX_CONCURRENT = 50
START_POINTS = 500.0

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']


def _calc_trade_size(portfolio):
    if portfolio < 1500:
        return BASE_TRADE_SIZE
    if portfolio < 5000:
        return portfolio / 100
    step = int((portfolio - 5000) / 1000)
    divisor = 110 + step * 10
    if divisor > 200:
        divisor = 200
    return portfolio / divisor


def _check_day_rollover(state, current_time):
    today = current_time.strftime('%Y-%m-%d')
    if state['current_day'] == today:
        return
    if state['current_day']:
        if state['day_pnl'] < 0:
            state['losing_streak_days'] += 1
            if state['losing_streak_days'] >= 2:
                penalty_pct = min(2 ** (state['losing_streak_days'] - 2), 20) / 100
                penalty = state['total_points'] * penalty_pct
                state['total_points'] -= penalty
                print(f"    [VERLUST-SERIE] Tag {state['losing_streak_days']}: "
                      f"-{penalty_pct*100:.0f}% = -{penalty:.0f} Punkte")
        else:
            if state['losing_streak_days'] > 0:
                print(f"    [VERLUST-SERIE] Beendet nach {state['losing_streak_days']} Tagen")
            state['losing_streak_days'] = 0
    state['current_day'] = today
    state['day_pnl'] = 0.0


def _check_week_rollover(state, current_time):
    iso = current_time.isocalendar()
    current_week = f"{iso[0]}-W{iso[1]:02d}"
    if state['current_week'] == current_week:
        return
    if state['current_week']:
        raw_points = state['week_points_raw']
        prev_raw = state.get('prev_week_raw_points', 0)
        threshold = prev_raw * 1.10 if prev_raw > 0 and state['winning_streak_weeks'] > 0 else 0
        if raw_points > 0 and raw_points >= threshold:
            state['winning_streak_weeks'] += 1
            multiplier = 1.20 + (state['winning_streak_weeks'] - 1) * 0.05
            bonus_points = raw_points * (multiplier - 1.0)
            state['total_points'] += bonus_points
            print(f"    [WOCHEN-BONUS] Woche {state['winning_streak_weeks']}: "
                  f"x{multiplier:.2f} auf {raw_points:.0f} Roh = "
                  f"+{bonus_points:.0f} Bonus (Vorwoche: {prev_raw:.0f}, Schwelle: {threshold:.0f})")
        else:
            if state['winning_streak_weeks'] > 0:
                reason = "negativ" if raw_points <= 0 else f"unter Schwelle ({raw_points:.0f} < {threshold:.0f})"
                print(f"    [WOCHEN-SERIE] Beendet nach {state['winning_streak_weeks']} Wochen ({reason})")
            state['winning_streak_weeks'] = 0
        state['prev_week_raw_points'] = raw_points
    state['current_week'] = current_week
    state['week_points'] = 0.0
    state['week_points_raw'] = 0.0


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def preload_market_data(coins_conn, symbols, start, end):
    print("=" * 70)
    print("  Market-Daten vorladen")
    print("=" * 70)

    market_data = {
        'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
    }
    load_start = start - timedelta(days=14)
    load_end = end + timedelta(days=1)

    print(f"  klines_1m...", end='', flush=True)
    total_1m = 0
    for symbol in symbols:
        with coins_conn.cursor() as cur:
            cur.execute("""
                SELECT open_time, open, high, low, close, volume, trades, taker_buy_base
                FROM klines WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time ASC
            """, (symbol, start - timedelta(days=1), load_end))
            rows = cur.fetchall()
        if rows:
            timestamps = [r['open_time'].replace(tzinfo=None) if hasattr(r['open_time'], 'tzinfo') and r['open_time'].tzinfo else r['open_time'] for r in rows]
            market_data['klines_1m'][symbol] = {
                'timestamps': timestamps,
                'open': np.array([float(r['open']) for r in rows], dtype=np.float32),
                'high': np.array([float(r['high']) for r in rows], dtype=np.float32),
                'low': np.array([float(r['low']) for r in rows], dtype=np.float32),
                'close': np.array([float(r['close']) for r in rows], dtype=np.float32),
                'volume': np.array([float(r['volume'] or 0) for r in rows], dtype=np.float32),
                'trades': np.array([float(r['trades'] or 0) for r in rows], dtype=np.float32),
                'taker': np.array([float(r['taker_buy_base'] or 0) for r in rows], dtype=np.float32),
            }
            total_1m += len(rows)
    print(f" {total_1m:,} Zeilen, {len(market_data['klines_1m'])} Coins")

    for table in ['agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']:
        print(f"  {table}...", end='', flush=True)
        total_rows = 0
        for symbol in symbols:
            with coins_conn.cursor() as cur:
                cur.execute(f"""
                    SELECT bucket, open, high, low, close, volume,
                           number_of_trades, taker_buy_base_asset_volume
                    FROM {table} WHERE symbol = %s AND bucket >= %s AND bucket <= %s
                    ORDER BY bucket ASC
                """, (symbol, load_start, load_end))
                rows = cur.fetchall()
            if rows:
                timestamps = [r['bucket'].replace(tzinfo=None) if hasattr(r['bucket'], 'tzinfo') and r['bucket'].tzinfo else r['bucket'] for r in rows]
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
                FROM kline_metrics WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time ASC
            """, (symbol, load_start, load_end))
            rows = cur.fetchall()
        if rows:
            timestamps = [r['open_time'].replace(tzinfo=None) if hasattr(r['open_time'], 'tzinfo') and r['open_time'].tzinfo else r['open_time'] for r in rows]
            entry = {'timestamps': timestamps}
            for col in km_cols:
                entry[col] = np.array([float(r[col]) if r[col] is not None else 0.0 for r in rows], dtype=np.float32)
            km_data[symbol] = entry
            total_km += len(rows)
    market_data['kline_metrics'] = km_data
    print(f" {total_km:,} Zeilen, {len(km_data)} Coins")

    total_bytes = sum(arr.nbytes for table in ['klines_1m', 'agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']
                      for sym_data in market_data[table].values() for arr in sym_data.values() if isinstance(arr, np.ndarray))
    print(f"\n  RAM: {total_bytes / 1024 / 1024:.0f} MB\n")
    return market_data


def _get_close(market_data, symbol, current_time):
    data = market_data.get('klines_1m', {}).get(symbol)
    if not data:
        data = market_data.get('agg_5m', {}).get(symbol)
    if not data:
        return 0.0
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0 or idx >= len(data['close']):
        return 0.0
    return float(data['close'][idx])


def _check_sl(market_data, symbol, current_time, entry_price, direction):
    data = market_data.get('klines_1m', {}).get(symbol)
    if not data:
        data = market_data.get('agg_5m', {}).get(symbol)
    if not data:
        return False
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0 or idx >= len(data['high']):
        return False
    if direction == 1:
        return float(data['low'][idx]) <= entry_price * (1 - SL_PERCENT / 100)
    else:
        return float(data['high'][idx]) >= entry_price * (1 + SL_PERCENT / 100)


def _get_obs(market_data, symbol, current_time, position_state=None, n_open=0):
    def _slice(table, sym, n):
        d = market_data.get(table, {}).get(sym)
        if not d or 'timestamps' not in d:
            return empty_candles()
        idx = bisect.bisect_right(d['timestamps'], current_time) - 1
        if idx < 0:
            return empty_candles()
        s = max(0, idx - n + 1)
        return {k: d[k][s:idx+1] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}

    def _slice_ref(key, n):
        d = market_data.get(key)
        if not d or 'timestamps' not in d:
            return empty_candles()
        idx = bisect.bisect_right(d['timestamps'], current_time) - 1
        if idx < 0:
            return empty_candles()
        s = max(0, idx - n + 1)
        return {k: d[k][s:idx+1] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}

    km_data = market_data.get('kline_metrics', {}).get(symbol)
    km = None
    if km_data and 'timestamps' in km_data:
        idx = bisect.bisect_right(km_data['timestamps'], current_time) - 1
        if idx >= 0:
            km = {}
            for col in ['pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
                         'pct_300m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m']:
                if col in km_data:
                    km[col] = float(km_data[col][idx])

    return compute_observation(
        _slice('agg_5m', symbol, 144), _slice('agg_1h', symbol, 24),
        _slice('agg_4h', symbol, 12), _slice('agg_1d', symbol, 14),
        _slice_ref('btc_1h', 25), _slice_ref('eth_1h', 25),
        current_time.hour, current_time.weekday(),
        kline_metrics=km, position_state=position_state, n_open_positions=n_open,
    )


def run_backtest():
    t_start = time.time()

    print("=" * 70)
    print("  RL-Agent Backtest V5 — Autonomous (kein CNN)")
    print(f"  Zeitraum: {TEST_START.date()} → {TEST_END.date()}")
    print(f"  Budget: ${START_BALANCE}, Trade: ${BASE_TRADE_SIZE}, progressiv")
    print(f"  Max Positionen: {MAX_CONCURRENT}, Hebel 1x-10x")
    print(f"  Max Haltezeit: {MAX_HOLD_MINUTES}min, SL: -{SL_PERCENT}%")
    print(f"  Kein Skip-Penalty — Agent entscheidet frei")
    print("=" * 70)
    print()

    # HL-Coins laden
    app_conn = get_conn('app')
    with app_conn.cursor() as cur:
        cur.execute("SELECT symbol FROM coin_info WHERE 'hyperliquid' = ANY(exchanges)")
        coins = [r['symbol'] for r in cur.fetchall()]
    app_conn.close()
    coins = [c for c in coins if c not in STABLECOINS]
    if 'BTCUSDC' not in coins:
        coins.append('BTCUSDC')
    if 'ETHUSDC' not in coins:
        coins.append('ETHUSDC')
    print(f"  {len(coins)} HL-Coins")

    # Market-Daten laden
    coins_conn = get_conn('coins')
    market_data = preload_market_data(coins_conn, coins, TEST_START, TEST_END)
    coins_conn.close()

    # V5 Modell laden
    env = TradingEnvV5(coins, market_data, TEST_START, TEST_END, train_start=TEST_START)
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print(f"  V5 Modell geladen: {MODEL_PATH}")
    except Exception as e:
        print(f"FEHLER: V5 Modell laden fehlgeschlagen: {e}")
        return

    print()
    print("=" * 70)
    print("  Backtest (zeitbasiert, alle Coins, parallele Positionen)")
    print("=" * 70)
    print()

    portfolio = START_BALANCE
    open_positions = []
    trade_log = []
    peak_portfolio = START_BALANCE
    low_portfolio = START_BALANCE
    total_margin_locked = 0.0
    total_skips = 0
    total_entries = 0

    state = {
        'total_points': START_POINTS,
        'current_day': '', 'day_pnl': 0.0, 'losing_streak_days': 0,
        'current_week': '', 'week_points': 0.0, 'week_points_raw': 0.0,
        'prev_week_raw_points': 0, 'winning_streak_weeks': 0,
        'max_losing_streak': 0, 'max_winning_streak': 0,
    }

    current_time = TEST_START
    last_print = TEST_START
    last_trade_times = {}

    while current_time < TEST_END:
        _check_day_rollover(state, current_time)
        _check_week_rollover(state, current_time)

        if state['losing_streak_days'] > state['max_losing_streak']:
            state['max_losing_streak'] = state['losing_streak_days']
        if state['winning_streak_weeks'] > state['max_winning_streak']:
            state['max_winning_streak'] = state['winning_streak_weeks']

        drawdown_mode = state['week_points'] < 0

        # === 1. Offene Positionen managen ===
        still_open = []
        closed_coins = set()
        for pos in open_positions:
            pos['steps'] += 1
            duration_min = pos['steps'] * STEP_MINUTES
            coin = pos['symbol'].replace('USDC', '').replace('USDT', '')

            if coin in closed_coins:
                continue

            price = _get_close(market_data, pos['symbol'], current_time)
            if price <= 0:
                still_open.append(pos)
                continue

            if pos['direction'] == 1:
                unrealized = (price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                unrealized = (pos['entry_price'] - price) / pos['entry_price'] * 100

            exit_reason = None

            # SL
            if _check_sl(market_data, pos['symbol'], current_time, pos['entry_price'], pos['direction']):
                unrealized = -SL_PERCENT
                exit_reason = 'sl'
            # Timeout 60min
            elif duration_min >= MAX_HOLD_MINUTES:
                exit_reason = 'timeout'
            else:
                # Agent-Entscheidung
                pos_state = {
                    'in_position': True,
                    'direction': pos['direction'],
                    'unrealized_pnl': unrealized,
                    'duration_min': duration_min,
                }
                obs = _get_obs(market_data, pos['symbol'], current_time,
                              position_state=pos_state, n_open=len(open_positions))
                mgmt_action, _ = model.predict(obs, deterministic=True)
                if int(mgmt_action) != 0:
                    exit_reason = 'agent_exit'

            if exit_reason:
                pnl_pct = unrealized
                reward = _reward(pnl_pct, pos['leverage'])
                # Wiederholungs-Penalty
                lt = last_trade_times.get(pos['symbol'])
                if lt and reward > 0 and (current_time - lt).total_seconds() < 3600:
                    reward *= 0.8
                state['total_points'] += reward
                state['week_points'] += reward
                state['week_points_raw'] += reward
                pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
                fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
                net_pnl = pnl_dollar - fees
                portfolio += pos['margin'] + pnl_dollar - fees
                total_margin_locked -= pos['margin']
                state['day_pnl'] += net_pnl
                last_trade_times[pos['symbol']] = current_time
                closed_coins.add(coin)
                trade_log.append({
                    'symbol': pos['symbol'],
                    'direction': 'long' if pos['direction'] == 1 else 'short',
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'entry_time': str(pos['entry_time']), 'exit_time': str(current_time),
                    'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(net_pnl, 2),
                    'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
                    'duration_min': duration_min, 'exit_reason': exit_reason,
                    'reward': round(reward, 2), 'total_points': round(state['total_points'], 1),
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # === 2. Neue Trades — alle Coins durchgehen ===
        n_open = len(open_positions)
        open_symbols = {p['symbol'] for p in open_positions}

        for symbol in coins:
            if n_open >= MAX_CONCURRENT:
                break
            if symbol in open_symbols:
                continue

            price = _get_close(market_data, symbol, current_time)
            if price <= 0:
                continue

            obs = _get_obs(market_data, symbol, current_time, position_state=None, n_open=n_open)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            if action == 0:
                total_skips += 1
                continue

            agent_direction = 1 if action <= 10 else -1
            leverage = LEVERAGE_MAP[action]

            if drawdown_mode and leverage > 3:
                leverage = 3

            current_portfolio = portfolio - total_margin_locked
            margin = _calc_trade_size(current_portfolio)
            if margin > current_portfolio * 0.5:
                continue

            portfolio -= margin
            total_margin_locked += margin
            n_open += 1
            total_entries += 1
            open_symbols.add(symbol)

            open_positions.append({
                'symbol': symbol, 'direction': agent_direction,
                'entry_price': price, 'entry_time': current_time,
                'margin': margin, 'leverage': leverage, 'steps': 0,
            })

        # Portfolio tracken
        current_portfolio = portfolio - total_margin_locked
        if current_portfolio > peak_portfolio:
            peak_portfolio = current_portfolio
        if current_portfolio < low_portfolio:
            low_portfolio = current_portfolio

        # Tages-Print
        if (current_time - last_print).total_seconds() >= 86400:
            print(f"  {current_time.date()} | Portfolio: ${current_portfolio:+.2f} "
                  f"| Punkte: {state['total_points']:+.0f} "
                  f"| Open: {len(open_positions)} | Trades: {len(trade_log)} "
                  f"| Verlust: {state['losing_streak_days']}d | Wochen: {state['winning_streak_weeks']}w")
            last_print = current_time

        current_time += timedelta(minutes=STEP_MINUTES)

    # Restliche Positionen schließen
    for pos in open_positions:
        price = _get_close(market_data, pos['symbol'], TEST_END)
        if price > 0 and pos['entry_price'] > 0:
            if pos['direction'] == 1:
                pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price'] * 100
        else:
            pnl_pct = 0.0
        reward = _reward(pnl_pct, pos['leverage'])
        state['total_points'] += reward
        state['week_points'] += reward
        state['week_points_raw'] += reward
        pnl_dollar = pos['margin'] * pos['leverage'] * pnl_pct / 100
        fees = pos['margin'] * pos['leverage'] * FEE_RATE * 2
        portfolio += pos['margin'] + pnl_dollar - fees
        total_margin_locked -= pos['margin']
        trade_log.append({
            'symbol': pos['symbol'],
            'direction': 'long' if pos['direction'] == 1 else 'short',
            'entry_price': pos['entry_price'], 'exit_price': price,
            'entry_time': str(pos['entry_time']), 'exit_time': str(TEST_END),
            'pnl_pct': round(pnl_pct, 4), 'pnl_dollar': round(pnl_dollar - fees, 2),
            'margin': round(pos['margin'], 2), 'leverage': pos['leverage'],
            'duration_min': pos['steps'] * STEP_MINUTES, 'exit_reason': 'end_of_test',
            'reward': round(reward, 2), 'total_points': round(state['total_points'], 1),
        })

    final_portfolio = round(portfolio - total_margin_locked, 2)
    elapsed = time.time() - t_start

    # === REPORT ===
    print()
    print("=" * 70)
    print("  ERGEBNISSE")
    print("=" * 70)
    print()

    print(f"--- Übersicht ---")
    print(f"  Coin-Präsentationen: {total_skips + total_entries}")
    print(f"  Agent tradet:        {total_entries}")
    print(f"  Agent skippt:        {total_skips}")
    print(f"  Abgeschlossene:      {len(trade_log)}")
    print()

    print(f"--- Konto ---")
    print(f"  Start:               ${START_BALANCE:.2f}")
    print(f"  Ende:                ${final_portfolio:.2f}")
    print(f"  Peak:                ${peak_portfolio:.2f}")
    print(f"  Tief:                ${low_portfolio:.2f}")
    if peak_portfolio > 0:
        max_dd = (peak_portfolio - low_portfolio) / peak_portfolio * 100
        print(f"  Max Drawdown:        {max_dd:.1f}%")
    print()

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

        rewards = [t['reward'] for t in trade_log]
        pos_rewards = [r for r in rewards if r > 0]
        neg_rewards = [r for r in rewards if r < 0]
        print(f"--- Punktesystem ---")
        print(f"  Startpunkte:         {START_POINTS:.0f}")
        print(f"  Endpunkte:           {state['total_points']:+.0f}")
        print(f"  Ø Reward/Trade:      {np.mean(rewards):+.2f}")
        print(f"  Positive Rewards:    {len(pos_rewards)} (Ø {np.mean(pos_rewards):+.2f})" if pos_rewards else "  Positive Rewards:    0")
        print(f"  Negative Rewards:    {len(neg_rewards)} (Ø {np.mean(neg_rewards):+.2f})" if neg_rewards else "  Negative Rewards:    0")
        print()

        print(f"--- Serien ---")
        print(f"  Max Verlust-Serie:   {state['max_losing_streak']} Tage")
        print(f"  Max Wochen-Serie:    {state['max_winning_streak']} Wochen")
        print()

        print(f"--- Hebel-Verteilung ---")
        lev_dist = defaultdict(int)
        for t in trade_log:
            lev_dist[t['leverage']] += 1
        for lev in sorted(lev_dist):
            lev_trades = [t for t in trade_log if t['leverage'] == lev]
            lev_pnl = sum(t['pnl_dollar'] for t in lev_trades)
            lev_wr = sum(1 for t in lev_trades if t['pnl_dollar'] > 0) / len(lev_trades) * 100
            lev_reward = sum(t['reward'] for t in lev_trades)
            print(f"  {lev:>2}x: {lev_dist[lev]:>5} Trades | WR: {lev_wr:.0f}% | PnL: ${lev_pnl:+.2f} | Reward: {lev_reward:+.1f}")
        print()

        print(f"--- Exit-Gründe ---")
        exit_reasons = defaultdict(int)
        for t in trade_log:
            exit_reasons[t['exit_reason']] += 1
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trade_log) * 100
            avg = np.mean([t['pnl_pct'] for t in trade_log if t['exit_reason'] == reason])
            dollar = sum(t['pnl_dollar'] for t in trade_log if t['exit_reason'] == reason)
            print(f"  {reason:<16} {count:>6} ({pct:.1f}%)  Ø PnL: {avg:+.3f}%  ${dollar:+.2f}")
        print()

        print(f"--- Wochen ---")
        weekly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'points': 0.0})
        for t in trade_log:
            dt = datetime.fromisoformat(t['entry_time'])
            w = dt.strftime('%Y-KW%W')
            weekly[w]['trades'] += 1
            weekly[w]['pnl'] += t['pnl_dollar']
            weekly[w]['points'] += t.get('reward', 0)
            if t['pnl_dollar'] > 0:
                weekly[w]['wins'] += 1
            elif t['pnl_dollar'] < 0:
                weekly[w]['losses'] += 1

        print(f"  {'Woche':<12} {'Trades':>7} {'W/L':>8} {'HR':>6} {'PnL $':>12} {'Punkte':>10}")
        for w in sorted(weekly):
            d = weekly[w]
            hr = f"{d['wins']/d['trades']*100:.0f}%" if d['trades'] else "-"
            wl = f"{d['wins']}/{d['losses']}" if d['trades'] else "-"
            print(f"  {w:<12} {d['trades']:>7} {wl:>8} {hr:>6} ${d['pnl']:>+11.2f} {d['points']:>+10.0f}")
        print()

    print(f"Laufzeit: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    try:
        with open(TRADES_LOG_PATH, 'w') as f:
            json.dump(trade_log, f, indent=2, default=str)
        print(f"Trade-Log gespeichert: {TRADES_LOG_PATH} ({len(trade_log)} Trades)")
    except Exception as e:
        print(f"Trade-Log speichern fehlgeschlagen: {e}")

    try:
        with open(RESULTS_PATH, 'w') as f:
            json.dump({
                'test_period': f"{TEST_START.date()} → {TEST_END.date()}",
                'start_balance': START_BALANCE, 'final_portfolio': final_portfolio,
                'peak': round(peak_portfolio, 2), 'low': round(low_portfolio, 2),
                'total_entries': total_entries, 'total_skips': total_skips,
                'trades_completed': len(trade_log),
                'total_points': round(state['total_points'], 1),
                'max_losing_streak': state['max_losing_streak'],
                'max_winning_streak': state['max_winning_streak'],
            }, f, indent=2, default=str)
        print(f"Ergebnisse gespeichert: {RESULTS_PATH}")
    except Exception as e:
        print(f"Ergebnisse speichern fehlgeschlagen: {e}")


if __name__ == '__main__':
    run_backtest()
