#!/usr/bin/env python3
"""
RL-Agent Training V6 — Autonomous Agent, 1-Min Management.

Wie V5 aber: Positions-Management im 1-Minuten-Takt.
Agent lernt schneller auf Preisbewegungen zu reagieren.

- klines_1m fuer praezise Preise in Position
- Entry-Scan weiterhin alle 5 Min
- 20M Steps (mehr Steps pro Trade wegen 1-Min-Takt)
- Gleicher Zeitraum: 01.07.2025 -> 31.12.2025

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/trainer_v6.py > /opt/coin/logs/rl_training_v6.log 2>&1 &
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
MODEL_PATH = MODEL_DIR / "rl_ppo_trading_v6.zip"
STATS_LOG = Path("/opt/coin/logs/rl_v6_leverage_stats.json")

# Marktdaten laden ab hier (3 Monate Vorlauf)
DATA_START = datetime(2025, 7, 1)
# Trading-Zeitraum (3 Monate: Okt-Dez)
TRADE_START = datetime(2025, 10, 1)
TRADE_END = datetime(2026, 1, 1)

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
        coin_counts = getattr(self.env, 'coin_trade_counts', {})
        log_entry = {
            'timesteps': self.num_timesteps,
            'total_points': round(self.env.total_points, 1),
            'losing_streak_days': self.env.losing_streak_days,
            'winning_streak_weeks': self.env.winning_streak_weeks,
            'global_time': str(self.env.global_time),
            'unique_coins': len(coin_counts),
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

        print(f"\n[V6-STATS] {self.num_timesteps:,} steps | "
              f"Punkte: {self.env.total_points:.0f} | "
              f"Zeit: {self.env.global_time} | "
              f"Verlust: {self.env.losing_streak_days}d | "
              f"Wochen: {self.env.winning_streak_weeks}w | "
              f"Coins: {len(coin_counts)}")
        for lev in range(1, 11):
            s = stats[lev]
            if s['trades'] > 0:
                avg = s['total_points'] / s['trades']
                print(f"  {lev:2d}x: {s['trades']:6d} Trades, "
                      f"Avg {avg:+.3f} Pkt/Trade, "
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


def get_hl_coins(app_conn):
    """HL-tradeable Coins laden."""
    with app_conn.cursor() as cur:
        cur.execute("SELECT symbol FROM coin_info WHERE 'hyperliquid' = ANY(exchanges)")
        symbols = [r['symbol'] for r in cur.fetchall()]
    symbols = [s for s in symbols if s not in STABLECOINS]
    print(f"  {len(symbols)} HL-tradeable Coins geladen")
    return symbols


def preload_market_data(coins_conn, symbols, data_start, trade_end):
    """Market-Daten komplett in RAM laden (inkl. klines_1m)."""
    print("=" * 70)
    print("  Market-Daten vorladen (V6: mit klines_1m)")
    print(f"  Daten ab: {data_start.date()} (Vorlauf fuer Features)")
    print(f"  Trading bis: {trade_end.date()}")
    print("=" * 70)

    market_data = {
        'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
    }

    load_start = data_start - timedelta(days=14)
    load_end = trade_end + timedelta(days=15)

    # klines_1m — nur fuer Trading-Zeitraum (nicht Vorlauf)
    klines_1m_start = trade_end - timedelta(days=190)  # Trade-Start mit Puffer
    print(f"  klines_1m ({klines_1m_start.date()} -> {load_end.date()})...", end='', flush=True)
    total_1m = 0
    for symbol in symbols:
        with coins_conn.cursor() as cur:
            cur.execute("""
                SELECT open_time, open, high, low, close, volume, trades, taker_buy_base
                FROM klines WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time ASC
            """, (symbol, klines_1m_start, load_end))
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
    for table in ['klines_1m', 'agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']:
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
    app_conn = get_conn('app')

    coins = get_hl_coins(app_conn)
    app_conn.close()

    if 'BTCUSDC' not in coins:
        coins.append('BTCUSDC')
    if 'ETHUSDC' not in coins:
        coins.append('ETHUSDC')

    market_data = preload_market_data(coins_conn, coins, DATA_START, TRADE_END)
    coins_conn.close()

    print("=" * 70)
    print("  PPO Training V6 — Autonomous Agent, 1-Min Management")
    print(f"  {len(coins)} HL-Coins, kein CNN")
    print(f"  Daten-Vorlauf: {DATA_START.date()} (3 Monate)")
    print(f"  Trading: {TRADE_START.date()} -> {TRADE_END.date()}")
    print(f"  Entry: alle 5 Min, Management: jede 1 Min")
    print(f"  Max Haltezeit: 60min, SL: -5%")
    print(f"  Hebel: Monat 1-2 max 3x, 3-4 max 5x, ab 5 max 10x")
    print("=" * 70)
    print()

    env = TradingEnvV6(
        coins=coins,
        market_data=market_data,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
        train_start=TRADE_START,
    )

    if STATS_LOG.exists():
        STATS_LOG.unlink()

    callback = LeverageStatsCallback(env, STATS_LOG)

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
        target_kl=0.03,
        verbose=1,
        policy_kwargs={
            'net_arch': [256, 256],
        },
    )
    print(f"[RL-AGENT] PPO V6 Modell erstellt: 256-256 MLP, Discrete(21)")

    total_timesteps = 50_000_000
    print(f"Training: {total_timesteps:,} Timesteps")
    print()

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(MODEL_PATH))
    print(f"\n[RL-AGENT] V6 Modell gespeichert: {MODEL_PATH}")

    # Finale Stats
    print("\n" + "=" * 70)
    print("  FINALE HEBEL-STATISTIKEN (gesamt)")
    print("=" * 70)
    for lev in range(1, 11):
        s = env.leverage_stats[lev]
        if s['trades'] > 0:
            avg = s['total_points'] / s['trades']
            print(f"  {lev:2d}x: {s['trades']:6d} Trades, "
                  f"Avg {avg:+.3f} Pkt/Trade, "
                  f"Gesamt: {s['total_points']:+.1f}")
    print(f"\n  Total Punkte: {env.total_points:.0f}")
    print(f"  Verlust-Serie max: {env.losing_streak_days}d")
    print(f"  Wochen-Serie max: {env.winning_streak_weeks}w")

    # Coin-Verteilung
    coin_counts = getattr(env, 'coin_trade_counts', {})
    print(f"\n  Verschiedene Coins getradet: {len(coin_counts)} von {len(coins)}")
    if coin_counts:
        sorted_coins = sorted(coin_counts.items(), key=lambda x: -x[1])
        print(f"  Top 20 Coins:")
        for sym, count in sorted_coins[:20]:
            print(f"    {sym:<16} {count:>8} Trades")
        if len(sorted_coins) > 20:
            rest = sum(c for _, c in sorted_coins[20:])
            print(f"    {'... Rest':<16} {rest:>8} Trades ({len(sorted_coins)-20} Coins)")

    # Phasen-Analyse (ab 15M und ab 25M Steps)
    log_path = STATS_LOG
    if log_path.exists():
        try:
            with open(log_path) as f:
                all_stats = json.load(f)

            for phase_start, phase_name in [(15_000_000, "ab 15M"), (25_000_000, "ab 25M")]:
                # Checkpoint kurz vor phase_start finden
                before = [s for s in all_stats if s['timesteps'] <= phase_start]
                after = [s for s in all_stats if s['timesteps'] > phase_start]
                if not before or not after:
                    continue

                base = before[-1]
                final = all_stats[-1]

                print(f"\n" + "=" * 70)
                print(f"  PHASEN-ANALYSE: {phase_name} Steps ({phase_start:,} -> {final['timesteps']:,})")
                print("=" * 70)

                phase_points = final['total_points'] - base['total_points']
                print(f"  Punkte in Phase: {phase_points:+.0f}")

                # Unique Coins Entwicklung
                base_coins = base.get('unique_coins', '?')
                final_coins = final.get('unique_coins', '?')
                print(f"  Coins: {base_coins} -> {final_coins}")

                for lev in range(1, 11):
                    b = base['leverage'].get(str(lev), {'trades': 0, 'total_points': 0})
                    f_ = final['leverage'].get(str(lev), {'trades': 0, 'total_points': 0})
                    d_trades = f_['trades'] - b['trades']
                    d_points = f_['total_points'] - b['total_points']
                    if d_trades > 0:
                        avg = d_points / d_trades
                        print(f"  {lev:2d}x: {d_trades:>8} Trades, "
                              f"Avg {avg:+.3f} Pkt/Trade, "
                              f"Gesamt: {d_points:+.1f}")
        except Exception as e:
            print(f"  Phasen-Analyse fehlgeschlagen: {e}")

    elapsed = time.time() - t_start
    print(f"\nTraining fertig in {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == '__main__':
    train()
