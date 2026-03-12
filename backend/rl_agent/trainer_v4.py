#!/usr/bin/env python3
"""
RL-Agent Training V4 — Serien-Mechanik + zeitbasierte Hebel-Freischaltung.

Erweiterungen gegenüber V3:
  1. Verlust-Serien-Strafe (1%/2%/4%/...20% ab Tag 2)
  2. Wochen-Bonus (x1.20, +0.05 pro Folgewoche)
  3. Hebel-Freischaltung: Monat 1-2 max 3x, 3-4 max 5x, 6+ max 10x
  4. Drawdown-Modus: Woche im Minus → max 3x
  5. Wiederholungs-Penalty: Gleicher Coin <1h → Gewinn x0.8

Zeitraum: 01.07.2024 → 31.12.2025 (18 Monate)
Logfile: /opt/coin/logs/rl_v4_leverage_stats.json

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/trainer_v4.py > /opt/coin/logs/rl_training_v4.log 2>&1 &
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

from rl_agent.env_v4 import TradingEnvV4

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_DIR = Path("/opt/coin/database/data/models")
MODEL_PATH = MODEL_DIR / "rl_ppo_trading_v4.zip"
STATS_LOG = Path("/opt/coin/logs/rl_v4_leverage_stats.json")

TRAIN_START = datetime(2024, 7, 1)
TRAIN_END = datetime(2026, 1, 1)

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']


class LeverageStatsCallback(BaseCallback):
    """Loggt Hebel-Statistiken alle 500k Steps."""

    def __init__(self, env, log_path, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_path = log_path
        self.last_log = 0

    def _on_step(self):
        if self.num_timesteps - self.last_log >= 500_000:
            self.last_log = self.num_timesteps
            self._log_stats()
        return True

    def _on_training_end(self):
        self._log_stats()

    def _log_stats(self):
        stats = self.env.leverage_stats
        log_entry = {
            'timesteps': self.num_timesteps,
            'total_points': round(self.env.total_points, 1),
            'losing_streak_days': self.env.losing_streak_days,
            'winning_streak_weeks': self.env.winning_streak_weeks,
            'leverage': {}
        }
        for lev in range(1, 11):
            s = stats[lev]
            avg = s['total_points'] / max(s['trades'], 1)
            log_entry['leverage'][str(lev)] = {
                'trades': s['trades'],
                'total_points': round(s['total_points'], 1),
                'avg_points': round(avg, 3),
            }

        # Append to log
        existing = []
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    existing = json.load(f)
            except:
                pass
        existing.append(log_entry)
        with open(self.log_path, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"\n[V4-STATS] {self.num_timesteps:,} steps | "
              f"Punkte: {self.env.total_points:.0f} | "
              f"Verlust: {self.env.losing_streak_days}d | "
              f"Wochen: {self.env.winning_streak_weeks}w")
        for lev in range(1, 11):
            s = stats[lev]
            if s['trades'] > 0:
                avg = s['total_points'] / s['trades']
                print(f"  {lev:2d}x: {s['trades']:6d} Trades, "
                      f"Ø {avg:+.3f} Pkt/Trade, "
                      f"Gesamt: {s['total_points']:+.1f}")
        print()


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def generate_opportunities(coins_conn):
    """Synthetische Opportunities: ±5% Moves in 6h-Fenstern."""
    print("=" * 70)
    print("  Synthetische Opportunities generieren")
    print(f"  Zeitraum: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print("=" * 70)

    print("Lade agg_1h...")
    with coins_conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, bucket, open, high, low, close
            FROM agg_1h
            WHERE bucket >= %s AND bucket < %s
            ORDER BY symbol, bucket
        """, (TRAIN_START, TRAIN_END))
        all_rows = cur.fetchall()
    print(f"  {len(all_rows)} Zeilen")

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
    print(f"  {len(coins)} Coins")

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
    print(f"  {len(opportunities)} Opportunities")
    print(f"  {up} Up ({up/len(opportunities)*100:.1f}%), {down} Down ({down/len(opportunities)*100:.1f}%)")
    print()
    return opportunities


def preload_market_data(coins_conn, symbols, start, end):
    """Market-Daten komplett in RAM laden."""
    print("=" * 70)
    print("  Market-Daten vorladen")
    print("=" * 70)

    market_data = {
        'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
    }

    load_start = start - timedelta(days=14)
    load_end = end + timedelta(days=15)

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


def train():
    t_start = time.time()
    coins_conn = get_conn('coins')

    opps = generate_opportunities(coins_conn)

    symbols = list(set(o['symbol'] for o in opps))
    if 'BTCUSDC' not in symbols:
        symbols.append('BTCUSDC')
    if 'ETHUSDC' not in symbols:
        symbols.append('ETHUSDC')

    market_data = preload_market_data(coins_conn, symbols, TRAIN_START, TRAIN_END)
    coins_conn.close()

    print("=" * 70)
    print("  PPO Training V4 — Serien-Mechanik + Hebel-Freischaltung")
    print(f"  {len(opps):,} Opportunities, {len(symbols)} Coins")
    print(f"  Zeitraum: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Hebel: Monat 1-2 max 3x, 3-4 max 5x, ab 6 max 10x")
    print(f"  Verlust-Serie: Ab Tag 2 → 1%/2%/4%/.../20% Punkte-Abzug")
    print(f"  Wochen-Bonus: x1.20, +0.05 pro Folgewoche")
    print(f"  Drawdown: Woche im Minus → max 3x")
    print(f"  Wiederholung: Gleicher Coin <1h → Gewinn x0.8")
    print("=" * 70)
    print()

    env = TradingEnvV4(opps, market_data, train_start=TRAIN_START)

    # Logfile leeren
    if STATS_LOG.exists():
        STATS_LOG.unlink()

    callback = LeverageStatsCallback(env, STATS_LOG)

    target_kl = 0.03
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=target_kl,
        verbose=1,
        policy_kwargs={
            'net_arch': [256, 256],
        },
    )
    print(f"[RL-AGENT] PPO V4 Modell erstellt: 256-256 MLP, Discrete(21)")

    total_timesteps = 10_000_000
    print(f"Training: {total_timesteps:,} Timesteps")
    print()

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(MODEL_PATH))
    print(f"\n[RL-AGENT] V4 Modell gespeichert: {MODEL_PATH}")

    # Finale Stats ausgeben
    print("\n" + "=" * 70)
    print("  FINALE HEBEL-STATISTIKEN")
    print("=" * 70)
    for lev in range(1, 11):
        s = env.leverage_stats[lev]
        if s['trades'] > 0:
            avg = s['total_points'] / s['trades']
            print(f"  {lev:2d}x: {s['trades']:6d} Trades, "
                  f"Ø {avg:+.3f} Pkt/Trade, "
                  f"Gesamt: {s['total_points']:+.1f}")
    print(f"\n  Total Punkte: {env.total_points:.0f}")
    print(f"  Verlust-Serie max: {env.losing_streak_days}d")
    print(f"  Wochen-Serie max: {env.winning_streak_weeks}w")

    elapsed = time.time() - t_start
    print(f"\nTraining fertig in {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == '__main__':
    train()
