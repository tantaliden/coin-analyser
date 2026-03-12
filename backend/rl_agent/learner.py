#!/usr/bin/env python3
"""
RL-Agent Learner — Periodisches Nachtraining (alle 12h).

Unabhängig vom Agent-Service. Läuft als eigener systemd Timer.

Ablauf:
1. Synthetische Opportunities aus letzten 6 Monaten generieren
2. Kopie des aktuellen Modells laden
3. Weitertrainieren (500k Steps)
4. Beide Modelle auf letzten 14 Tagen backtesten (gleicher Zeitraum!)
5. Besseres Modell behalten

Usage:
    systemctl start rl-learner  (oder Timer)
    cd /opt/coin/backend
    nohup python3 -u rl_agent/learner.py > /opt/coin/logs/rl_learner.log 2>&1 &
"""
import sys
import json
import time
import shutil
import bisect
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v3 import TradingEnvV3, LEVERAGE_MAP, _reward
from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = Path("/opt/coin/database/data/models/rl_ppo_trading_v3.zip")
MODEL_BACKUP = Path("/opt/coin/database/data/models/rl_ppo_trading_v3_backup.zip")
LOG_PATH = "/opt/coin/logs/rl_learner.log"

TRAIN_MONTHS = 6          # Letzte 6 Monate für Training
EVAL_DAYS = 14            # Letzte 14 Tage für Vergleich
LEARN_STEPS = 500_000     # Nachtraining Steps
SL_PERCENT = 5.0
STEP_MINUTES = 5

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
# Opportunities generieren (identisch zu trainer_v3.py)
# ============================================================

def generate_opportunities(coins_conn, start, end):
    """Synthetische Opportunities: ±5% Moves in 6h-Fenstern."""
    print(f"  Opportunities generieren: {start.date()} → {end.date()}")

    with coins_conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, bucket, open, high, low, close
            FROM agg_1h
            WHERE bucket >= %s AND bucket < %s
            ORDER BY symbol, bucket
        """, (start, end))
        all_rows = cur.fetchall()
    print(f"  {len(all_rows):,} agg_1h Zeilen")

    coins = {}
    current_sym = None
    current_rows = []
    for row in all_rows:
        if row['symbol'] != current_sym:
            if current_sym and current_rows:
                coins[current_sym] = current_rows
            current_sym = row['symbol']
            current_rows = []
        current_rows.append(row)
    if current_sym and current_rows:
        coins[current_sym] = current_rows

    for s in STABLECOINS:
        coins.pop(s, None)

    opportunities = []
    MOVE = 5.0
    WINDOW = 6
    CHECK = 4

    for symbol, rows in coins.items():
        if len(rows) < WINDOW + 1:
            continue
        last_idx = -CHECK
        for i in range(0, len(rows) - WINDOW, CHECK):
            if i - last_idx < CHECK:
                continue
            base = float(rows[i]['close'])
            if base <= 0:
                continue

            tp_thresh = base * (1 + MOVE / 100)
            sl_thresh = base * (1 - MOVE / 100)
            tp_time = sl_time = None

            for j in range(i + 1, min(i + WINDOW + 1, len(rows))):
                h, l = float(rows[j]['high']), float(rows[j]['low'])
                if not tp_time and h >= tp_thresh:
                    tp_time = rows[j]['bucket']
                if not sl_time and l <= sl_thresh:
                    sl_time = rows[j]['bucket']
                if tp_time or sl_time:
                    break

            if not tp_time and not sl_time:
                continue

            if tp_time and sl_time:
                actual_dir = 'up' if tp_time <= sl_time else 'down'
            elif tp_time:
                actual_dir = 'up'
            else:
                actual_dir = 'down'

            dt = rows[i]['bucket']
            if hasattr(dt, 'tzinfo') and dt.tzinfo:
                dt = dt.replace(tzinfo=None)

            opportunities.append({
                'symbol': symbol,
                'detection_time': dt,
                'actual_direction': actual_dir,
            })
            last_idx = i

    opportunities.sort(key=lambda o: o['detection_time'])

    up = sum(1 for o in opportunities if o['actual_direction'] == 'up')
    down = len(opportunities) - up
    print(f"  {len(opportunities):,} Opportunities ({up} Up, {down} Down)")
    return opportunities


# ============================================================
# Market-Daten vorladen (identisch zu trainer_v3.py)
# ============================================================

def preload_market_data(coins_conn, symbols, start, end):
    market_data = {'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {}}
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
        print(f" {total_rows:,}")

    for ref_symbol, key in [('BTCUSDC', 'btc_1h'), ('ETHUSDC', 'eth_1h')]:
        market_data[key] = market_data['agg_1h'].get(ref_symbol, {})

    # kline_metrics
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
    print(f" {total_km:,}")
    return market_data


# ============================================================
# Mini-Backtest für Modell-Vergleich
# ============================================================

def evaluate_model(model, market_data, opportunities, eval_start, eval_end):
    """
    Schneller Backtest auf einem Zeitabschnitt.
    Gibt total_reward zurück (Summe aller Rewards).
    Identische Logik wie env_v3.py.
    """
    eval_opps = [o for o in opportunities
                 if eval_start <= o['detection_time'] < eval_end]

    if not eval_opps:
        return 0.0, 0, 0, 0

    total_reward = 0.0
    trades = 0
    wins = 0
    losses = 0

    for opp in eval_opps:
        symbol = opp['symbol']
        current_time = opp['detection_time']

        # Entry-Entscheidung
        obs = _get_obs(market_data, symbol, current_time, n_open=0)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        if action == 0:
            continue

        entry_price = _get_close(market_data, symbol, current_time)
        if entry_price <= 0:
            continue

        direction = 1 if action <= 10 else -1
        leverage = LEVERAGE_MAP[action]
        trades += 1

        # Position simulieren (max 6h)
        for step in range(1, 6 * 60 // STEP_MINUTES + 1):
            current_time += timedelta(minutes=STEP_MINUTES)
            price = _get_close(market_data, symbol, current_time)

            if price <= 0:
                break

            # SL
            if direction == 1:
                sl_level = entry_price * (1 - SL_PERCENT / 100)
                hit_sl = price <= sl_level
            else:
                sl_level = entry_price * (1 + SL_PERCENT / 100)
                hit_sl = price >= sl_level

            if hit_sl:
                reward = _reward(-SL_PERCENT, leverage)
                total_reward += reward
                losses += 1
                break

            # PnL
            if direction == 1:
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - price) / entry_price * 100

            # Agent-Entscheidung
            pos_state = {
                'in_position': True,
                'direction': direction,
                'unrealized_pnl': pnl_pct,
                'duration_min': step * STEP_MINUTES,
            }
            obs = _get_obs(market_data, symbol, current_time,
                          position_state=pos_state, n_open=1)
            mgmt_action, _ = model.predict(obs, deterministic=True)

            if int(mgmt_action) != 0:
                reward = _reward(pnl_pct, leverage)
                total_reward += reward
                if pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1
                break
        else:
            # Timeout
            if direction == 1:
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - price) / entry_price * 100
            reward = _reward(pnl_pct, leverage)
            total_reward += reward
            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1

    return total_reward, trades, wins, losses


# ============================================================
# Hilfsfunktionen (aus backtest_v3.py)
# ============================================================

def _get_close(market_data, symbol, current_time):
    data = market_data.get('agg_5m', {}).get(symbol)
    if not data:
        return 0.0
    idx = bisect.bisect_right(data['timestamps'], current_time) - 1
    if idx < 0 or idx >= len(data['close']):
        return 0.0
    return float(data['close'][idx])


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
# Hauptfunktion
# ============================================================

def run_learning_cycle():
    t_start = time.time()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    train_end = now
    train_start = now - timedelta(days=TRAIN_MONTHS * 30)
    eval_start = now - timedelta(days=EVAL_DAYS)
    eval_end = now

    print("=" * 70)
    print("  RL-Agent Learner — Periodisches Nachtraining")
    print(f"  Training:   {train_start.date()} → {train_end.date()} ({TRAIN_MONTHS} Monate)")
    print(f"  Vergleich:  {eval_start.date()} → {eval_end.date()} ({EVAL_DAYS} Tage)")
    print(f"  Steps:      {LEARN_STEPS:,}")
    print("=" * 70)
    print()

    if not MODEL_PATH.exists():
        print("[LEARNER] FEHLER: Kein Modell vorhanden!")
        return

    # 1. Daten laden
    coins_conn = get_conn('coins')
    opps = generate_opportunities(coins_conn, train_start, train_end)

    if len(opps) < 100:
        print(f"[LEARNER] Zu wenig Opportunities ({len(opps)}), Abbruch.")
        coins_conn.close()
        return

    symbols = list(set(o['symbol'] for o in opps))
    if 'BTCUSDC' not in symbols:
        symbols.append('BTCUSDC')
    if 'ETHUSDC' not in symbols:
        symbols.append('ETHUSDC')

    print()
    market_data = preload_market_data(coins_conn, symbols, train_start, train_end)
    coins_conn.close()

    # 2. Altes Modell evaluieren (auf letzten 14 Tagen)
    print()
    print("=" * 70)
    print("  Phase 1: Altes Modell evaluieren")
    print("=" * 70)

    env = TradingEnvV3(opps, market_data, train_start=train_start)
    old_model = PPO.load(str(MODEL_PATH), env=env)

    old_reward, old_trades, old_wins, old_losses = evaluate_model(
        old_model, market_data, opps, eval_start, eval_end
    )
    old_wr = old_wins / old_trades * 100 if old_trades > 0 else 0

    print(f"  Altes Modell: {old_trades} Trades, {old_wins}W/{old_losses}L "
          f"({old_wr:.0f}% WR), Reward: {old_reward:+.1f}")

    # 3. Backup + Nachtrainieren
    print()
    print("=" * 70)
    print("  Phase 2: Nachtrainieren")
    print("=" * 70)

    shutil.copy2(str(MODEL_PATH), str(MODEL_BACKUP))
    print(f"  Backup: {MODEL_BACKUP}")

    new_model = PPO.load(str(MODEL_PATH), env=env)
    new_model.learning_rate = 0.0001
    new_model.lr_schedule = lambda _: 0.0001

    print(f"  Training: {LEARN_STEPS:,} Steps...")
    new_model.learn(total_timesteps=LEARN_STEPS)

    # 4. Neues Modell evaluieren (gleicher Zeitraum!)
    print()
    print("=" * 70)
    print("  Phase 3: Neues Modell evaluieren")
    print("=" * 70)

    new_reward, new_trades, new_wins, new_losses = evaluate_model(
        new_model, market_data, opps, eval_start, eval_end
    )
    new_wr = new_wins / new_trades * 100 if new_trades > 0 else 0

    print(f"  Neues Modell: {new_trades} Trades, {new_wins}W/{new_losses}L "
          f"({new_wr:.0f}% WR), Reward: {new_reward:+.1f}")

    # 5. Vergleich + Entscheidung
    print()
    print("=" * 70)
    print("  Vergleich (gleiche 14 Tage)")
    print("=" * 70)
    print(f"  Alt:  Reward {old_reward:+.1f} | {old_trades} Trades | {old_wr:.0f}% WR")
    print(f"  Neu:  Reward {new_reward:+.1f} | {new_trades} Trades | {new_wr:.0f}% WR")

    if new_reward > old_reward:
        new_model.save(str(MODEL_PATH))
        print(f"\n  ✓ NEUES Modell übernommen! (Reward {new_reward:+.1f} > {old_reward:+.1f})")
        print(f"    Backup bleibt: {MODEL_BACKUP}")
        swapped = True
    else:
        print(f"\n  ✗ Altes Modell bleibt. (Reward {old_reward:+.1f} >= {new_reward:+.1f})")
        # Backup entfernen — altes Modell ist unangetastet
        MODEL_BACKUP.unlink(missing_ok=True)
        swapped = False

    elapsed = time.time() - t_start
    print(f"\n  Laufzeit: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Log-Eintrag für Nachverfolgung
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'train_period': f"{train_start.date()} → {train_end.date()}",
        'eval_period': f"{eval_start.date()} → {eval_end.date()}",
        'old_reward': round(old_reward, 1),
        'old_trades': old_trades,
        'old_wr': round(old_wr, 1),
        'new_reward': round(new_reward, 1),
        'new_trades': new_trades,
        'new_wr': round(new_wr, 1),
        'swapped': swapped,
        'elapsed_s': round(elapsed, 0),
    }

    log_path = Path("/opt/coin/logs/rl_learner_history.json")
    history = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                history = json.load(f)
        except:
            pass
    history.append(log_entry)
    # Nur letzte 100 Einträge behalten
    history = history[-100:]
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  Log: {log_path}")


if __name__ == '__main__':
    run_learning_cycle()
