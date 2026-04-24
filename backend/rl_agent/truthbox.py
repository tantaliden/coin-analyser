#!/usr/bin/env python3
"""
Truthbox — Replay-Simulator für RL-Agent.

Spielt historische Daten Minute für Minute ab, exakt wie der Live-Service.
Observations kommen über compute_observation_live() mit striktem Zeitfilter.
KEIN Preloading in Arrays — alles über DB-Queries wie im Live-Betrieb.

Keine echten Trades. Alles simuliert, alles geloggt.

Usage (auf dem Live-Server):
    nohup python3 -u /opt/coin/backend/rl_agent/truthbox.py > /opt/coin/logs/truthbox.log 2>&1 &
"""
import json
import csv
import sys
import time
import random
import bisect
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.features import compute_observation_live, get_kline_metrics_from_db, N_FEATURES
from rl_agent.env_wallet import LEVERAGE_MAP, N_FEATURES as WALLET_N_FEATURES, TAKER_FEE, FUNDING_RATE_PER_HOUR

# === KONFIGURATION ===
SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = "/opt/coin/database/data/models/rl_wallet_v1.zip"
OUTPUT_CSV = "/opt/coin/logs/truthbox_march2026_v2.csv"
OUTPUT_LOG = "/opt/coin/logs/truthbox_v2.log"

# Zeitraum
SIM_START = datetime(2026, 3, 1, 0, 0)
SIM_END = datetime(2026, 4, 1, 0, 0)

# Live-Regeln
MAX_CONCURRENT = 15
BASE_TRADE_SIZE = 20.0
SCAN_MINUTES = {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56}
MGMT_INTERVAL_MIN = 5  # Position-Management alle 5 Min
SELL_BLOCK_LOW = -0.5
SELL_BLOCK_HIGH = 0.5
TIMEOUT_MINUTES = 1440  # 24h Timeout
TIMEOUT_LOSS_MULTIPLIER = 4  # Verlust-Bestrafung bei Timeout

STABLECOINS = {'USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC'}


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_tradeable_coins(app_conn):
    with app_conn.cursor() as cur:
        cur.execute("SELECT symbol FROM coin_info WHERE 'hyperliquid' = ANY(exchanges)")
        coins = [r['symbol'] for r in cur.fetchall()]
    return [c for c in coins if c not in STABLECOINS]


def get_price_at(coins_conn, symbol, at_time):
    """Letzter bekannter Close-Preis aus agg_5m VOR at_time."""
    with coins_conn.cursor() as cur:
        cur.execute("""
            SELECT close FROM agg_5m
            WHERE symbol = %s AND bucket <= %s
            ORDER BY bucket DESC LIMIT 1
        """, (symbol, at_time))
        row = cur.fetchone()
    if row:
        return float(row['close'])
    return 0.0


def get_price_1m_at(coins_conn, symbol, at_time):
    """Letzter bekannter Close-Preis aus klines (1m) VOR at_time."""
    with coins_conn.cursor() as cur:
        cur.execute("""
            SELECT close FROM klines
            WHERE symbol = %s AND open_time <= %s
            ORDER BY open_time DESC LIMIT 1
        """, (symbol, at_time))
        row = cur.fetchone()
    if row:
        return float(row['close'])
    return 0.0


def calc_pnl_pct(direction, entry_price, current_price):
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    if direction == 'long':
        return (current_price - entry_price) / entry_price * 100
    else:
        return (entry_price - current_price) / entry_price * 100


def calc_pnl_with_costs(direction, entry_price, current_price, leverage, entry_time, current_time):
    raw_pnl = calc_pnl_pct(direction, entry_price, current_price)
    fees = TAKER_FEE * 2
    hours_held = (current_time - entry_time).total_seconds() / 3600
    funding = FUNDING_RATE_PER_HOUR * hours_held * leverage
    return raw_pnl - fees - funding


def get_observation(coins_conn, symbol, current_time, state, position_state=None, n_open=0):
    """
    Observation über den LIVE-Codepfad: compute_observation_live().
    Strikter Zeitfilter: Nur Daten <= current_time.
    + 2 Wallet-Extra-Features (59 total).
    """
    base_obs = compute_observation_live(
        coins_conn, symbol, current_time,
        position_state=position_state,
        n_open_positions=n_open,
    )

    obs = np.zeros(WALLET_N_FEATURES, dtype=np.float32)
    obs[:N_FEATURES] = base_obs[:N_FEATURES]
    obs[57] = state.get('weekly_target_points', 0) / 50.0
    week_start = state.get('week_start_points', state.get('total_points', 2000))
    obs[58] = (state.get('total_points', 2000) - week_start) / max(week_start, 1)

    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


class SimPosition:
    """Eine simulierte Position."""
    __slots__ = ('symbol', 'direction', 'leverage', 'entry_price', 'entry_time',
                 'size_usd', 'pos_id')
    _next_id = 1

    def __init__(self, symbol, direction, leverage, entry_price, entry_time, size_usd):
        self.symbol = symbol
        self.direction = direction
        self.leverage = leverage
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size_usd = size_usd
        self.pos_id = SimPosition._next_id
        SimPosition._next_id += 1


def main():
    print("=" * 70)
    print("  TRUTHBOX — Replay-Simulator")
    print(f"  Zeitraum: {SIM_START} -> {SIM_END}")
    print(f"  Modell: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Regeln: {MAX_CONCURRENT} Positionen, ${BASE_TRADE_SIZE}, kein SL, {TIMEOUT_MINUTES}min Timeout (Verlust x{TIMEOUT_LOSS_MULTIPLIER})")
    print(f"  Observations: compute_observation_live() (DB-Queries, strikter Zeitfilter)")
    print("=" * 70)

    if not Path(MODEL_PATH).exists():
        print(f"FEHLER: Modell nicht gefunden: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH)
    print(f"Modell geladen: {MODEL_PATH}")

    # Coins laden
    app_conn = get_conn('app')
    tradeable_coins = get_tradeable_coins(app_conn)
    app_conn.close()
    print(f"{len(tradeable_coins)} tradeable Coins")

    # State (wie im Live-Service)
    state = {
        'total_points': 2000.0,
        'total_profit': 0.0,
        'current_day': '',
        'current_week': '',
        'day_pnl': 0.0,
        'week_points': 0.0,
        'week_points_raw': 0.0,
        'weekly_target_points': 0,
        'week_start_points': 2000.0,
    }

    # Offene Positionen
    open_positions = []  # Liste von SimPosition

    # Stats
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl_usd = 0.0

    # CSV Setup
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'timestamp', 'event', 'symbol', 'direction', 'leverage',
        'action', 'entry_price', 'exit_price', 'pnl_pct', 'pnl_usd',
        'duration_min', 'n_open', 'total_points', 'total_profit',
        'obs_sample_0', 'obs_sample_5', 'obs_sample_14', 'obs_sample_30',
        'obs_sample_40', 'obs_sample_52',
    ])

    # Hauptloop: Minute für Minute
    current_time = SIM_START
    last_scan_min = -1
    last_mgmt_time = SIM_START - timedelta(minutes=MGMT_INTERVAL_MIN)
    coins_conn = get_conn('coins')
    iteration = 0
    last_status_time = SIM_START

    print(f"\nStarte Replay ab {SIM_START}...")
    t_real_start = time.time()

    while current_time < SIM_END:
        try:
            # Tages/Wochen-Rollover
            today = current_time.strftime('%Y-%m-%d')
            if state['current_day'] != today:
                state['current_day'] = today
                state['day_pnl'] = 0.0

            iso = current_time.isocalendar()
            week_key = f"{iso[0]}-W{iso[1]:02d}"
            if state['current_week'] != week_key:
                if state['current_week']:
                    pct_change = (state['total_points'] - state['week_start_points']) / max(state['week_start_points'], 1) * 100
                    if pct_change >= 15.0:
                        state['weekly_target_points'] += 1
                state['current_week'] = week_key
                state['week_points'] = 0.0
                state['week_points_raw'] = 0.0
                state['week_start_points'] = state['total_points']

            now_min = current_time.minute
            n_open = len(open_positions)

            # === POSITION MANAGEMENT (alle 5 Min) ===
            if (current_time - last_mgmt_time).total_seconds() >= MGMT_INTERVAL_MIN * 60:
                last_mgmt_time = current_time
                closed_in_mgmt = []

                for pos in open_positions:
                    # Preis über 1m-Klines (wie im Live: HL-API, hier DB-1m)
                    current_price = get_price_1m_at(coins_conn, pos.symbol, current_time)
                    if current_price <= 0:
                        current_price = get_price_at(coins_conn, pos.symbol, current_time)
                    if current_price <= 0:
                        continue

                    pnl_pct_raw = calc_pnl_pct(pos.direction, pos.entry_price, current_price)
                    pnl_pct_costs = calc_pnl_with_costs(
                        pos.direction, pos.entry_price, current_price,
                        pos.leverage, pos.entry_time, current_time,
                    )
                    duration_min = int((current_time - pos.entry_time).total_seconds() / 60)

                    # === TIMEOUT CHECK (24h) ===
                    if duration_min >= TIMEOUT_MINUTES:
                        pnl_for_stats = pnl_pct_raw
                        # Verlust-Bestrafung: PnL x4 bei negativem PnL
                        if pnl_pct_raw < 0:
                            pnl_for_stats = pnl_pct_raw * TIMEOUT_LOSS_MULTIPLIER

                        pnl_usd = pos.size_usd * pnl_pct_raw / 100
                        fees = pos.size_usd * 0.00035 * 2
                        net_pnl = pnl_usd - fees

                        total_trades += 1
                        if pnl_pct_raw > 0:
                            total_wins += 1
                        else:
                            total_losses += 1
                        total_pnl_usd += net_pnl

                        # Punkte mit Bestrafung
                        state['total_points'] += pnl_for_stats

                        writer.writerow([
                            current_time.isoformat(), 'TIMEOUT', pos.symbol,
                            pos.direction, pos.leverage, -1,
                            f'{pos.entry_price:.6f}', f'{current_price:.6f}',
                            f'{pnl_pct_raw:.4f}', f'{net_pnl:.2f}',
                            duration_min, n_open, f'{state["total_points"]:.1f}',
                            f'{total_pnl_usd:.2f}', '', '', '', '', '', '',
                        ])

                        closed_in_mgmt.append(pos)
                        penalty_info = f' (x{TIMEOUT_LOSS_MULTIPLIER} Penalty)' if pnl_pct_raw < 0 else ''
                        print(f'  [{current_time}] TIMEOUT {pos.symbol} {pos.direction} {pos.leverage}x | '
                              f'PnL: {pnl_pct_raw:+.2f}% ${net_pnl:+.2f} | {duration_min}min{penalty_info}')
                        continue

                    pos_state = {
                        'in_position': True,
                        'direction': 1 if pos.direction == 'long' else -1,
                        'unrealized_pnl': pnl_pct_costs,
                        'duration_min': duration_min,
                    }

                    obs = get_observation(
                        coins_conn, pos.symbol, current_time, state,
                        position_state=pos_state,
                        n_open=n_open,
                    )

                    mgmt_action, _ = model.predict(obs, deterministic=False)
                    mgmt_action = int(mgmt_action)

                    # Verkaufs-Block
                    if mgmt_action != 0 and SELL_BLOCK_LOW <= pnl_pct_raw <= SELL_BLOCK_HIGH:
                        mgmt_action = 0

                    if mgmt_action != 0:
                        # CLOSE
                        pnl_usd = pos.size_usd * pnl_pct_raw / 100
                        fees = pos.size_usd * 0.00035 * 2
                        net_pnl = pnl_usd - fees

                        total_trades += 1
                        if pnl_pct_raw > 0:
                            total_wins += 1
                        else:
                            total_losses += 1
                        total_pnl_usd += net_pnl

                        writer.writerow([
                            current_time.isoformat(), 'CLOSE', pos.symbol,
                            pos.direction, pos.leverage, mgmt_action,
                            f'{pos.entry_price:.6f}', f'{current_price:.6f}',
                            f'{pnl_pct_raw:.4f}', f'{net_pnl:.2f}',
                            duration_min, n_open, f'{state["total_points"]:.1f}',
                            f'{total_pnl_usd:.2f}',
                            f'{obs[0]:.4f}', f'{obs[5]:.4f}', f'{obs[14]:.4f}',
                            f'{obs[30]:.4f}', f'{obs[40]:.4f}', f'{obs[52]:.4f}',
                        ])

                        closed_in_mgmt.append(pos)
                        print(f"  [{current_time}] CLOSE {pos.symbol} {pos.direction} {pos.leverage}x | "
                              f"PnL: {pnl_pct_raw:+.2f}% ${net_pnl:+.2f} | {duration_min}min")

                for pos in closed_in_mgmt:
                    open_positions.remove(pos)
                n_open = len(open_positions)

            # === ENTRY SCAN (bei :01, :06, :11, ...) ===
            if now_min in SCAN_MINUTES and last_scan_min != now_min:
                last_scan_min = now_min
                open_symbols = {p.symbol for p in open_positions}

                shuffled = list(tradeable_coins)
                random.shuffle(shuffled)

                for symbol in shuffled:
                    if n_open >= MAX_CONCURRENT:
                        break
                    if symbol in open_symbols:
                        continue

                    current_price = get_price_at(coins_conn, symbol, current_time)
                    if current_price <= 0:
                        continue

                    obs = get_observation(
                        coins_conn, symbol, current_time, state,
                        position_state=None,
                        n_open=n_open,
                    )

                    action, _ = model.predict(obs, deterministic=False)
                    action = int(action)

                    if action == 0:
                        continue

                    direction = 'long' if action <= 10 else 'short'
                    leverage = LEVERAGE_MAP[action]

                    # Drawdown-Modus
                    if state['week_points'] < 0 and leverage > 3:
                        leverage = 3

                    pos = SimPosition(symbol, direction, leverage, current_price, current_time, BASE_TRADE_SIZE)
                    open_positions.append(pos)
                    n_open += 1
                    open_symbols.add(symbol)

                    writer.writerow([
                        current_time.isoformat(), 'OPEN', symbol,
                        direction, leverage, action,
                        f'{current_price:.6f}', '', '', '',
                        '', n_open, f'{state["total_points"]:.1f}',
                        f'{total_pnl_usd:.2f}',
                        f'{obs[0]:.4f}', f'{obs[5]:.4f}', f'{obs[14]:.4f}',
                        f'{obs[30]:.4f}', f'{obs[40]:.4f}', f'{obs[52]:.4f}',
                    ])

                    print(f"  [{current_time}] OPEN {symbol} {direction} {leverage}x @{current_price:.4f}")

            # Status alle 24 Stunden
            if (current_time - last_status_time).total_seconds() >= 86400:
                wr = total_wins / total_trades * 100 if total_trades > 0 else 0
                elapsed_real = time.time() - t_real_start
                sim_days = (current_time - SIM_START).total_seconds() / 86400
                print(f"\n=== {current_time.strftime('%Y-%m-%d')} | "
                      f"Trades: {total_trades} (WR: {wr:.1f}%) | "
                      f"PnL: ${total_pnl_usd:+.2f} | "
                      f"Offen: {n_open} | "
                      f"Real-Zeit: {elapsed_real/60:.1f}min für {sim_days:.1f} Sim-Tage ===\n")
                last_status_time = current_time
                csv_file.flush()

            # Nächste Minute
            current_time += timedelta(minutes=1)
            iteration += 1

            # DB-Connection alle 1000 Iterationen erneuern
            if iteration % 1000 == 0:
                try:
                    coins_conn.close()
                except:
                    pass
                coins_conn = get_conn('coins')

        except Exception as e:
            print(f"FEHLER bei {current_time}: {e}")
            traceback.print_exc()
            # DB-Connection erneuern und weitermachen
            try:
                coins_conn.close()
            except:
                pass
            coins_conn = get_conn('coins')
            current_time += timedelta(minutes=1)

    # === ABSCHLUSS ===
    # Alle noch offenen Positionen zum letzten Preis schließen
    for pos in open_positions:
        current_price = get_price_at(coins_conn, pos.symbol, current_time)
        if current_price > 0:
            pnl_pct_raw = calc_pnl_pct(pos.direction, pos.entry_price, current_price)
            pnl_usd = pos.size_usd * pnl_pct_raw / 100
            fees = pos.size_usd * 0.00035 * 2
            net_pnl = pnl_usd - fees
            total_trades += 1
            if pnl_pct_raw > 0:
                total_wins += 1
            else:
                total_losses += 1
            total_pnl_usd += net_pnl
            duration_min = int((current_time - pos.entry_time).total_seconds() / 60)
            writer.writerow([
                current_time.isoformat(), 'FORCE_CLOSE', pos.symbol,
                pos.direction, pos.leverage, -1,
                f'{pos.entry_price:.6f}', f'{current_price:.6f}',
                f'{pnl_pct_raw:.4f}', f'{net_pnl:.2f}',
                duration_min, 0, f'{state["total_points"]:.1f}',
                f'{total_pnl_usd:.2f}', '', '', '', '', '', '',
            ])

    csv_file.close()
    coins_conn.close()

    elapsed = time.time() - t_real_start
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    print("\n" + "=" * 70)
    print("  TRUTHBOX ERGEBNIS")
    print("=" * 70)
    print(f"  Zeitraum:    {SIM_START.date()} -> {SIM_END.date()}")
    print(f"  Trades:      {total_trades}")
    print(f"  Win-Rate:    {wr:.1f}% ({total_wins}W / {total_losses}L)")
    print(f"  PnL:         ${total_pnl_usd:+.2f}")
    print(f"  Avg PnL:     ${total_pnl_usd/total_trades:+.2f}/Trade" if total_trades > 0 else "")
    print(f"  Real-Zeit:   {elapsed/60:.1f} Minuten")
    print(f"  CSV:         {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()
