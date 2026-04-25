#!/usr/bin/env python3
"""
RL-Agent V7 Training — Vom Beobachter zum Trader.

Phase 1: Jan - Maerz 2025 — Markt erforschen (5m, einfaches +/- PnL)
Phase 2: Apr 2025 - Feb 2026 — Traden lernen (5m, volles Reward-System)
Phase 3: Maerz 2026 - heute — Feinschliff (1m, niedrigere LR)

256-256 MLP, from scratch.
Speichert als rl_ppo_trading_v7.zip (ueberschreibt NICHT v6).

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/train_v7.py > /opt/coin/logs/rl_train_v7.log 2>&1 &
"""
import sys
import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v6 import TradingEnvV6

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_DIR = Path("/opt/coin/database/data/models")
MODEL_PATH = MODEL_DIR / "rl_ppo_trading_v7.zip"

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']

PHASES = [
    {
        'name': 'Phase 1: Markt erforschen',
        'data_start': datetime(2025, 1, 1),
        'trade_start': datetime(2025, 1, 1),
        'trade_end': datetime(2025, 4, 1),
        'mgmt_step': 5,
        'steps': 10_000_000,
        'lr': 0.0001,
    },
    {
        'name': 'Phase 2: Traden lernen',
        'data_start': datetime(2025, 1, 1),
        'trade_start': datetime(2025, 4, 1),
        'trade_end': datetime(2026, 3, 1),
        'mgmt_step': 5,
        'steps': 30_000_000,
        'lr': 0.0001,
    },
    {
        'name': 'Phase 3: Feinschliff 1m',
        'data_start': datetime(2025, 12, 1),
        'trade_start': datetime(2026, 3, 1),
        'trade_end': datetime(2026, 3, 19),
        'mgmt_step': 1,
        'steps': 10_000_000,
        'lr': 0.00005,
    },
]


class PhaseCallback(BaseCallback):
    """Loggt Stats alle 500k Steps."""

    def __init__(self, env, phase_name, model_path, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.phase_name = phase_name
        self.model_path = model_path
        self.last_log = 0
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_log >= 500_000:
            self.last_log = self.num_timesteps
            self._log()
        # Checkpoint alle 5M Steps
        if self.num_timesteps - self.last_save >= 5_000_000:
            self.last_save = self.num_timesteps
            self.model.save(str(self.model_path))
            print(f"  [CHECKPOINT] Gespeichert bei {self.num_timesteps:,} Steps")
        return True

    def _on_training_end(self):
        self._log()

    def _log(self):
        coin_counts = getattr(self.env, 'coin_trade_counts', {})
        total_trades = sum(coin_counts.values())
        print(f"\n[{self.phase_name}] {self.num_timesteps:,} steps | "
              f"Punkte: {self.env.total_points:.0f} | "
              f"Trades: {total_trades} | "
              f"Zeit: {self.env.global_time} | "
              f"Coins: {len(coin_counts)}")
        for lev in range(1, 11):
            s = self.env.leverage_stats[lev]
            if s['trades'] > 0:
                avg = s['total_points'] / s['trades']
                print(f"  {lev:2d}x: {s['trades']:6d} Trades, Avg {avg:+.3f}")
        print(flush=True)


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def preload_market_data(coins_conn, symbols, data_start, trade_end, load_1m=True):
    """Market-Daten laden."""
    market_data = {'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {}}
    load_start = data_start - timedelta(days=14)
    load_end = trade_end + timedelta(days=1)

    if load_1m:
        market_data['klines_1m'] = {}
        print("  klines_1m...", end='', flush=True)
        total_1m = 0
        for symbol in symbols:
            with coins_conn.cursor() as cur:
                cur.execute("""
                    SELECT open_time, open, high, low, close, volume, trades, taker_buy_base
                    FROM klines WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                    ORDER BY open_time ASC
                """, (symbol, data_start, load_end))
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
        print(f" {total_1m:,} Zeilen")

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
            timestamps = [r['open_time'].replace(tzinfo=None) if hasattr(r['open_time'], 'tzinfo') and r['open_time'].tzinfo else r['open_time'] for r in rows]
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


def run():
    t_total = time.time()

    print("=" * 70)
    print("  RL-Agent V7 Training — Vom Beobachter zum Trader")
    print("  Phase 1: Markt erforschen (Jan-Maerz 2025, 5m)")
    print("  Phase 2: Traden lernen (Apr 2025-Feb 2026, 5m)")
    print("  Phase 3: Feinschliff (Maerz 2026, 1m)")
    print("  256-256 MLP, from scratch")
    print(f"  Modell: {MODEL_PATH}")
    print("=" * 70)
    print()

    # Coins laden
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
    print(f"  {len(coins)} HL-Coins\n")

    # Neues Modell from scratch
    dummy_env = TradingEnvV6(coins, {'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {}, 'kline_metrics': {}},
                              datetime(2025, 1, 1), datetime(2025, 1, 2))
    model = PPO(
        "MlpPolicy", dummy_env,
        learning_rate=0.0001, n_steps=4096, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, target_kl=0.03,
        verbose=1,
        policy_kwargs={'net_arch': [256, 256]},
    )
    print("  Neues PPO-Modell erstellt: 256-256 MLP\n")

    for i, phase in enumerate(PHASES):
        t_phase = time.time()
        print("=" * 70)
        print(f"  {phase['name']}")
        print(f"  Zeitraum: {phase['trade_start'].date()} -> {phase['trade_end'].date()}")
        print(f"  Management: {phase['mgmt_step']}m, LR: {phase['lr']}, Steps: {phase['steps']:,}")
        print("=" * 70)

        # klines_1m nur in Phase 3 (1m Management)
        load_1m = phase['mgmt_step'] <= 2
        coins_conn = get_conn('coins')
        market_data = preload_market_data(coins_conn, coins, phase['data_start'], phase['trade_end'], load_1m=load_1m)
        coins_conn.close()

        env = TradingEnvV6(
            coins=coins,
            market_data=market_data,
            trade_start=phase['trade_start'],
            trade_end=phase['trade_end'],
            train_start=phase['trade_start'],
            mgmt_step_minutes=phase['mgmt_step'],
        )

        model.set_env(env)
        model.learning_rate = phase['lr']
        model.lr_schedule = lambda _, lr=phase['lr']: lr

        callback = PhaseCallback(env, phase['name'], MODEL_PATH)
        model.learn(total_timesteps=phase['steps'], callback=callback, reset_num_timesteps=False)

        # Phase speichern
        phase_path = MODEL_DIR / f"rl_ppo_trading_v7_phase{i+1}.zip"
        model.save(str(phase_path))
        model.save(str(MODEL_PATH))
        print(f"\n  Phase {i+1} gespeichert: {phase_path}")

        # Stats
        coin_counts = getattr(env, 'coin_trade_counts', {})
        total_trades = sum(coin_counts.values())
        elapsed_phase = time.time() - t_phase
        print(f"  Punkte: {env.total_points:.0f}")
        print(f"  Trades: {total_trades}")
        print(f"  Coins: {len(coin_counts)}")
        for lev in range(1, 11):
            s = env.leverage_stats[lev]
            if s['trades'] > 0:
                avg = s['total_points'] / s['trades']
                print(f"  {lev:2d}x: {s['trades']:6d} Trades, Avg {avg:+.3f}")
        print(f"  Laufzeit: {elapsed_phase:.0f}s ({elapsed_phase/3600:.1f}h)\n")

    elapsed = time.time() - t_total
    print(f"\n[V7] Finales Modell gespeichert: {MODEL_PATH}")
    print(f"Gesamtlaufzeit: {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == '__main__':
    run()
