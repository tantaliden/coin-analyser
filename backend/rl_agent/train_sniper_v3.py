#!/usr/bin/env python3
"""
V3.0 Sniper Training — Richtige Observations, hartes Reward-System.

Alles unter +5% nach Hebel = Bestrafung. Große Moves belohnt.
Basierend auf env_wallet.py Observations (bewährt).

P1: Jan-Mär 2025 (10M Steps)
P2: Apr-Dez 2025 (3M Steps pro Monat)
P3: Okt-Dez 2025 nochmal (5M Steps)
P4: Fine-Tuning Dez 2025 (500k Steps)
Livetest: 01.01.2026 - 05.04.2026 (Single-Pass, 1-Min)

Usage:
    cd /opt/coin/backend
    nohup /opt/coin/venv/bin/python3 -u /opt/training/scripts/train_sniper_v3.py > /opt/coin/logs/rl_train_sniper_v3.log 2>&1 &
"""
import sys
import json
import time
import gc
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

sys.path.insert(0, '/opt/coin/backend')
sys.path.insert(0, '/opt/training')

from rl_agent.env_sniper_v3 import TradingEnvSniperV3, N_FEATURES, LEVERAGE_MAP

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_DIR = Path("/opt/training/models")
MODEL_PATH = MODEL_DIR / "rl_sniper_v3.zip"

TOP50_COINS = [
    'BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC', 'DOGEUSDC', 'BNBUSDC',
    'SUIUSDC', 'TRUMPUSDC', 'ADAUSDC', 'TRXUSDC', 'ENAUSDC', 'ZECUSDC',
    'LINKUSDC', 'LTCUSDC', 'AVAXUSDC', 'XPLUSDC', 'WIFUSDC', 'HBARUSDC',
    'PENGUUSDC', 'WLDUSDC', 'UNIUSDC', 'TAOUSDC', 'AAVEUSDC', 'ASTERUSDC',
    'NEARUSDC', 'PNUTUSDC', 'ARBUSDC', 'XLMUSDC', 'APTUSDC', 'SEIUSDC',
    'DOTUSDC', 'SUSDC', 'BCHUSDC', 'CRVUSDC', 'RUNEUSDC', 'FETUSDC',
    'CAKEUSDC', 'FILUSDC', 'OMUSDC', 'WLFIUSDC', 'VIRTUALUSDC', 'TONUSDC',
    'AVNTUSDC', 'ETHFIUSDC', 'OPUSDC', 'PUMPUSDC', 'LDOUSDC', 'PAXGUSDC',
    'ICPUSDC', 'AIXBTUSDC',
]

P1_STEPS = 10_000_000
P2_STEPS_PER_MONTH = 3_000_000
P3_STEPS = 5_000_000
P4_STEPS = 500_000

LEARN_BATCH_SIZE = 150
LEARN_STEPS = 2000

LIVETEST_START = datetime(2026, 1, 1)
LIVETEST_END = datetime(2026, 4, 5, 12, 0)


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def preload_market_data(coins_conn, symbols, data_start, trade_end):
    """Lädt ALLE Daten inkl. klines_1m (fehlte im V2 Sniper)."""
    market_data = {'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {}}
    load_start = data_start - timedelta(days=14)
    load_end = trade_end + timedelta(days=1)

    # klines_1m (für 1-Min Management-Preise)
    print(f"    klines_1m...", end='', flush=True)
    total_1m = 0
    for symbol in symbols:
        with coins_conn.cursor() as cur:
            cur.execute("""
                SELECT open_time, open, high, low, close, volume, trades, taker_buy_base
                FROM klines WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time ASC
            """, (symbol, data_start - timedelta(days=1), load_end))
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
    print(f" {total_1m:,}")

    for table in ['agg_5m', 'agg_1h', 'agg_4h', 'agg_1d']:
        print(f"    {table}...", end='', flush=True)
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

    print("    kline_metrics...", end='', flush=True)
    km_cols = ['pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
               'pct_300m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m']
    km_data = {}
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


def run_phase(model, phase_name, month_start, month_end, coins, steps):
    """Training-Phase mit PPO.learn()."""
    print(f"\n  --- {phase_name}: {month_start.strftime('%Y-%m')} bis {month_end.strftime('%Y-%m')} ({steps:,} Steps) ---")

    coins_conn = get_conn('coins')
    market_data = preload_market_data(coins_conn, coins, month_start, month_end)
    coins_conn.close()

    env = TradingEnvSniperV3(
        coins=coins, market_data=market_data,
        trade_start=month_start, trade_end=month_end,
        train_start=month_start, mgmt_step_minutes=1,
    )

    model.set_env(env)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)

    total = env.total_wins + env.total_losses
    wr = env.total_wins / max(total, 1) * 100
    short_pct = env.total_shorts / max(env.total_longs + env.total_shorts, 1) * 100
    avg_hold = env.total_hold_minutes / max(total, 1)
    print(f"    Punkte={env.total_points:.0f} | Trades={total} | WR={wr:.1f}% | "
          f"Hold={avg_hold:.0f}m | Short={short_pct:.0f}%")

    del market_data, env
    gc.collect()
    return model


def run_livetest(model, coins):
    """Single-Pass durch 01.01.2026 - 05.04.2026, jede Minute einmal."""
    print(f"\n{'='*70}")
    print(f"  LIVETEST: {LIVETEST_START} -> {LIVETEST_END}")
    print(f"  Single-Pass, 1-Min, manueller Loop")
    print(f"{'='*70}")

    total_points = 2000.0
    weekly_target_points = 0
    week_start_points = 2000.0
    current_week = ''
    total_wins = 0
    total_losses = 0
    total_longs = 0
    total_shorts = 0
    total_hold_minutes = 0.0
    learn_count = 0
    trades_since_learn = 0

    months = []
    current = LIVETEST_START
    while current < LIVETEST_END:
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_end = min(next_month, LIVETEST_END)
        months.append((current, month_end))
        current = next_month

    print(f"  {len(months)} Monate\n")

    for m_idx, (m_start, m_end) in enumerate(months):
        month_name = m_start.strftime('%Y-%m')
        ckpt = MODEL_DIR / f"rl_sniper_v3_livetest_{month_name}.zip"

        if ckpt.exists():
            print(f"  {month_name}: Checkpoint vorhanden, lade...")
            model = PPO.load(str(ckpt))
            continue

        print(f"\n  --- {month_name} ({m_idx+1}/{len(months)}) ---")

        data_start = m_start - timedelta(days=14)
        coins_conn = get_conn('coins')
        market_data = preload_market_data(coins_conn, coins, data_start, m_end)
        coins_conn.close()

        import rl_agent.env_sniper_v3 as _env_mod
        _env_mod.ENTRY_STEP_MINUTES = 1

        env = TradingEnvSniperV3(
            coins=coins, market_data=market_data,
            trade_start=m_start, trade_end=m_end,
            train_start=m_start, mgmt_step_minutes=1,
        )

        # State wiederherstellen
        env.total_points = total_points
        env.weekly_target_points = weekly_target_points
        env.week_start_points = week_start_points
        env.current_week = current_week
        env.total_wins = total_wins
        env.total_losses = total_losses
        env.total_longs = total_longs
        env.total_shorts = total_shorts
        env.total_hold_minutes = total_hold_minutes

        # Manueller Loop
        obs, _ = env.reset()
        month_trades = 0

        while env.global_time < m_end:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                cc = env.coin_trade_counts
                current_total = sum(cc.values())
                if current_total > month_trades:
                    trades_since_learn += (current_total - month_trades)
                    month_trades = current_total

                if trades_since_learn >= LEARN_BATCH_SIZE:
                    model.set_env(env)
                    model.learning_rate = 0.00003
                    model.lr_schedule = lambda _, lr=0.00003: lr
                    model.learn(total_timesteps=LEARN_STEPS, reset_num_timesteps=False)
                    learn_count += 1
                    trades_since_learn = 0

                if env.global_time < m_end:
                    obs, _ = env.reset()
                else:
                    break

        # State sichern
        total_points = env.total_points
        weekly_target_points = env.weekly_target_points
        week_start_points = env.week_start_points
        current_week = env.current_week
        total_wins = env.total_wins
        total_losses = env.total_losses
        total_longs = env.total_longs
        total_shorts = env.total_shorts
        total_hold_minutes = env.total_hold_minutes
        total_trades = total_wins + total_losses

        wr = total_wins / max(total_trades, 1) * 100
        avg_hold = total_hold_minutes / max(total_trades, 1)
        short_pct = total_shorts / max(total_longs + total_shorts, 1) * 100

        print(f"  {month_name}: Punkte={total_points:.0f} | Trades={total_trades} | "
              f"WR={wr:.1f}% | Hold={avg_hold:.0f}m | Short={short_pct:.0f}% | Learn={learn_count}")

        model.save(str(ckpt))
        del market_data, env
        gc.collect()

    return model, total_points, total_wins + total_losses, total_wins, total_losses, learn_count


def run():
    t_total = time.time()

    print("=" * 70)
    print("  V3.0 Sniper Training — Richtige Observations, hartes Reward")
    print(f"  Coins: {len(TOP50_COINS)} (Top 50 HL nach Volumen)")
    print(f"  Reward: <5% nach Hebel = Bestrafung, >5% = Belohnung")
    print(f"  Hebel: 5x-10x | Max 5 Positionen | SL -10% | Timeout 24h")
    print(f"  Livetest: 01.01.2026 - 05.04.2026 (Single-Pass)")
    print("=" * 70)

    coins = TOP50_COINS

    # Dummy-Env für PPO-Init
    dummy_env = TradingEnvSniperV3(
        coins=coins,
        market_data={'klines_1m': {}, 'agg_5m': {}, 'agg_1h': {}, 'agg_4h': {}, 'agg_1d': {},
                     'btc_1h': {}, 'eth_1h': {}, 'kline_metrics': {}},
        trade_start=datetime(2025, 1, 1), trade_end=datetime(2025, 1, 2),
    )

    model = PPO(
        'MlpPolicy', dummy_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 512]),
        verbose=0,
    )

    # === P1: Jan-Mär 2025 ===
    model = run_phase(model, "P1", datetime(2025, 1, 1), datetime(2025, 4, 1), coins, P1_STEPS)
    model.save(str(MODEL_DIR / "rl_sniper_v3_p1.zip"))
    print(f"  P1 gespeichert")

    # === P2: Apr-Dez 2025 monatsweise ===
    for month in range(4, 13):
        m_start = datetime(2025, month, 1)
        m_end = datetime(2025, month + 1, 1) if month < 12 else datetime(2026, 1, 1)
        model = run_phase(model, f"P2-{month:02d}", m_start, m_end, coins, P2_STEPS_PER_MONTH)
        model.save(str(MODEL_DIR / f"rl_sniper_v3_p2_{month:02d}.zip"))

    model.save(str(MODEL_DIR / "rl_sniper_v3_p2.zip"))
    print(f"\n  P2 komplett gespeichert")

    # === P3: Okt-Dez 2025 nochmal ===
    model = run_phase(model, "P3", datetime(2025, 10, 1), datetime(2026, 1, 1), coins, P3_STEPS)
    model.save(str(MODEL_DIR / "rl_sniper_v3_p3.zip"))
    print(f"  P3 gespeichert")

    # === P4: Fine-Tuning Dez 2025 ===
    model = run_phase(model, "P4", datetime(2025, 12, 1), datetime(2026, 1, 1), coins, P4_STEPS)
    model.save(str(MODEL_DIR / "rl_sniper_v3_p4.zip"))
    print(f"  P4 gespeichert")

    # === Livetest ===
    model, points, total, wins, losses, learns = run_livetest(model, coins)

    # Finales Modell
    import shutil
    final_path = MODEL_DIR / "rl_sniper_v3_final.zip"
    model.save(str(final_path))
    model.save(str(MODEL_PATH))

    backup_dir = MODEL_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(str(final_path), str(backup_dir / f"rl_sniper_v3_FRESH_{ts}.zip"))

    elapsed = time.time() - t_total
    wr = wins / max(total, 1) * 100
    print(f"\n{'='*70}")
    print(f"  V3.0 Sniper FERTIG!")
    print(f"  Punkte: {points:.0f} | Trades: {total} | WR: {wr:.1f}%")
    print(f"  Learn-Zyklen: {learns}")
    print(f"  Finales Modell: {final_path}")
    print(f"  Laufzeit: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
