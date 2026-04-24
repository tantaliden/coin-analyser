#!/usr/bin/env python3
"""
RL-Agent Fine-Tuning V6 — Ein Durchlauf auf aktuellen Daten.

Laedt das bestehende V6-Modell und trainiert es auf Jan-Maerz 2026 weiter.
Ein Durchlauf, kein Loop — der Agent erlebt die Daten wie live.
Der "Geist" bleibt erhalten, nur neues Wissen kommt dazu.

Usage:
    cd /opt/coin/backend
    nohup python3 -u rl_agent/finetune_v6.py > /opt/coin/logs/rl_finetune_v6.log 2>&1 &
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_v6 import TradingEnvV6

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_DIR = Path("/opt/coin/database/data/models")
MODEL_PATH = MODEL_DIR / "rl_ppo_trading_v6_sl25.zip"
MODEL_BACKUP = MODEL_DIR / "rl_ppo_trading_v6_pre_finetune.zip"
SOURCE_MODEL = MODEL_DIR / "rl_ppo_trading_v6_pre_finetune.zip"  # Basis-Modell laden

# Fine-Tuning: Hybrid 5m→1m, gleiche Einstellungen wie live
DATA_START = datetime(2025, 10, 1)  # Vorlauf fuer Features
TRADE_START = datetime(2026, 1, 1)
TRADE_END = datetime(2026, 3, 18)

# Hybrid: 5m bis 10.03, dann 1m bis 18.03
FINETUNE_STEPS = 15_000_000

STABLECOINS = ['USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC']


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def preload_market_data(coins_conn, symbols, data_start, trade_end):
    """Market-Daten komplett in RAM laden (inkl. klines_1m)."""
    print("=" * 70)
    print("  Market-Daten vorladen")
    print("=" * 70)

    market_data = {
        'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
    }
    load_start = data_start - timedelta(days=14)
    load_end = trade_end + timedelta(days=1)

    print(f"  klines_1m...", end='', flush=True)
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


def run():
    t_start = time.time()

    print("=" * 70)
    print("  RL-Agent Fine-Tuning V6")
    print(f"  Bestehendes Modell weitertrainieren auf {TRADE_START.date()} -> {TRADE_END.date()}")
    print(f"  Ein Durchlauf, {FINETUNE_STEPS:,} Steps")
    print(f"  Der Geist bleibt — nur neues Wissen kommt dazu")
    print("=" * 70)
    print()

    if not SOURCE_MODEL.exists():
        print(f"[FINETUNE] FEHLER: Basis-Modell nicht gefunden: {SOURCE_MODEL}")
        return

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
    print(f"  {len(coins)} HL-Coins")

    # Daten laden
    coins_conn = get_conn('coins')
    market_data = preload_market_data(coins_conn, coins, DATA_START, TRADE_END)
    coins_conn.close()

    # Env erstellen
    env = TradingEnvV6(
        coins=coins,
        market_data=market_data,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
        train_start=TRADE_START,
    )

    # Bestehendes Modell laden — NICHT neu erstellen!
    import shutil
    print(f"  Lade Basis-Modell: {SOURCE_MODEL}")

    model = PPO.load(str(SOURCE_MODEL), env=env)
    model.learning_rate = 0.00005  # Niedrigere LR fuer Fine-Tuning
    model.lr_schedule = lambda _: 0.00005

    print(f"  Modell geladen, Fine-Tuning mit LR=0.00005")
    print(f"  Training: {FINETUNE_STEPS:,} Steps...")
    print()

    model.learn(total_timesteps=FINETUNE_STEPS)
    model.save(str(MODEL_PATH))

    print(f"\n[FINETUNE] Modell gespeichert: {MODEL_PATH}")

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
    print(f"\nFine-Tuning fertig in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    run()
