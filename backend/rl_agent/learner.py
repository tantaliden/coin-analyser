#!/usr/bin/env python3
"""
RL-Agent Continuous Learner V6 — Weitertrainieren, nicht ersetzen.

Der "Geist" bleibt erhalten. Kein Vergleich alt/neu, kein Swap.
Das Modell wird geladen, auf den letzten 6 Monaten weitertrainiert,
und gespeichert. Immer dasselbe Gehirn, immer dazulernend.

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

from rl_agent.env_v6 import TradingEnvV6, LEVERAGE_MAP, _reward, MAX_HOLD_MINUTES
from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = Path("/opt/coin/database/data/models/rl_ppo_trading_v6.zip")
MODEL_BACKUP = Path("/opt/coin/database/data/models/rl_ppo_trading_v6_backup.zip")

TRAIN_MONTHS = 6          # Letzte 6 Monate
LEARN_STEPS = 500_000     # Pro Lernzyklus
SL_PERCENT = 5.0
ENTRY_STEP_MINUTES = 5
MGMT_STEP_MINUTES = 1

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def preload_market_data(coins_conn, symbols, start, end):
    market_data = {'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {}}
    load_start = start - timedelta(days=14)
    load_end = end + timedelta(days=1)

    # klines_1m
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


def run_learning_cycle():
    t_start = time.time()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    train_start = now - timedelta(days=TRAIN_MONTHS * 30)
    train_end = now
    data_start = train_start - timedelta(days=90)

    print("=" * 70)
    print("  RL-Agent Continuous Learner V6")
    print(f"  Gleicher Geist — weitertrainieren, nicht ersetzen")
    print(f"  Zeitraum:   {train_start.date()} -> {train_end.date()} ({TRAIN_MONTHS} Monate)")
    print(f"  Steps:      {LEARN_STEPS:,}")
    print("=" * 70)
    print()

    if not MODEL_PATH.exists():
        print("[LEARNER] FEHLER: Kein V6 Modell vorhanden!")
        return

    # 1. Coins + Daten laden
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

    coins_conn = get_conn('coins')
    print()
    market_data = preload_market_data(coins_conn, coins, data_start, train_end)
    coins_conn.close()

    # 2. Backup (Sicherheit, nicht fuer Vergleich)
    print()
    shutil.copy2(str(MODEL_PATH), str(MODEL_BACKUP))
    print(f"  Backup: {MODEL_BACKUP}")

    # 3. Modell laden und WEITERTRAINIEREN
    env = TradingEnvV6(coins, market_data, train_start, train_end, train_start=train_start)
    model = PPO.load(str(MODEL_PATH), env=env)
    model.learning_rate = 0.00005   # Niedrige LR — verfeinern, nicht umlernen
    model.lr_schedule = lambda _: 0.00005

    print(f"  Modell geladen, weitertrainieren mit LR=0.00005")
    print(f"  Training: {LEARN_STEPS:,} Steps...")
    print()

    model.learn(total_timesteps=LEARN_STEPS)

    # 4. Speichern — gleicher Pfad, gleicher Geist
    model.save(str(MODEL_PATH))
    print(f"\n[LEARNER] Modell gespeichert: {MODEL_PATH}")

    # Stats
    print(f"\n  Total Punkte: {env.total_points:.0f}")
    print(f"  Verlust-Serie: {env.losing_streak_days}d")
    print(f"  Wochen-Serie: {env.winning_streak_weeks}w")

    coin_counts = getattr(env, 'coin_trade_counts', {})
    print(f"  Verschiedene Coins: {len(coin_counts)}")

    for lev in range(1, 11):
        s = env.leverage_stats[lev]
        if s['trades'] > 0:
            avg = s['total_points'] / s['trades']
            print(f"  {lev:2d}x: {s['trades']:6d} Trades, Avg {avg:+.3f}")

    elapsed = time.time() - t_start
    print(f"\nLernen fertig in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Log
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'train_period': f"{train_start.date()} -> {train_end.date()}",
        'learn_steps': LEARN_STEPS,
        'total_points': round(env.total_points, 1),
        'unique_coins': len(coin_counts),
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
    history = history[-100:]
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    run_learning_cycle()
