#!/usr/bin/env python3
"""
Initial-Learning für V1.1 Agent — Clean Backup + alle bisherigen Trades.

Ablauf:
1. Agent-Service stoppen
2. Aktuelles Modell sichern
3. Clean-Backup vom Live-Server holen
4. Alle geschlossenen Trades aus DB laden
5. In Batches von 150 Experience-Learning durchführen (mit fixem Replay-Reward)
6. Agent-Service starten

Usage:
    /opt/coin/venv/bin/python3 -u /opt/coin/backend/rl_agent/initial_learn_v11.py
"""
import json
import os
import sys
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.env_wallet import LEVERAGE_MAP, _reward, TradingEnvWallet as TradingEnvLive
from rl_agent.learner import preload_market_data

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = "/opt/coin/database/data/models/rl_wallet_v11.zip"
CLEAN_BACKUP = "/opt/coin/database/data/models/rl_wallet_v11_clean.zip"
CURRENT_BACKUP = "/opt/coin/database/data/models/rl_wallet_v11_pre_initial_learn.zip"
LIVE_SERVER_BACKUP = "root@82.165.236.163:/opt/training/models/rl_wallet_v11_final.zip"

BATCH_SIZE = 150
LEARN_STEPS = 2000


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_closed_trades():
    """Alle geschlossenen Trades aus DB laden, chronologisch sortiert."""
    conn = get_conn('app')
    with conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, direction, leverage, entry_price, entry_time, exit_time,
                   pnl_percent, pnl_usd, exit_reason
            FROM rl_positions
            WHERE status = 'closed'
            ORDER BY exit_time ASC
        """)
        rows = cur.fetchall()
    conn.close()
    return rows


def build_experiences(trades):
    """DB-Trades in Experience-Format umwandeln."""
    experiences = []
    for t in trades:
        pnl_pct = float(t['pnl_percent'] or 0)
        leverage = int(t['leverage'] or 1)
        direction = t['direction']

        # Reward berechnen wie im Live-Service
        reward = _reward(pnl_pct, leverage)

        entry_time = t['entry_time']
        exit_time = t['exit_time']
        if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
            entry_time = entry_time.replace(tzinfo=None)
        if hasattr(exit_time, 'tzinfo') and exit_time.tzinfo:
            exit_time = exit_time.replace(tzinfo=None)

        experiences.append({
            'symbol': t['symbol'],
            'direction': direction,
            'leverage': leverage,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': float(t['entry_price']),
            'pnl_pct': pnl_pct,
            'reward': reward,
            'action': leverage if direction == 'long' else leverage + 10,
        })
    return experiences


def experience_learn_batch(model, experiences, batch_num, total_batches):
    """Ein Batch Experience-Learning mit fixem Replay-Reward."""
    traded_coins = list(set(e['symbol'] for e in experiences))
    earliest = min(e['entry_time'] for e in experiences)
    latest = max(e['exit_time'] for e in experiences)

    all_coins = traded_coins[:]
    for ref in ['BTCUSDC', 'ETHUSDC']:
        if ref not in all_coins:
            all_coins.append(ref)

    data_start = earliest - timedelta(hours=12)

    print(f"\n  [Batch {batch_num}/{total_batches}] Lade Marktdaten: "
          f"{len(all_coins)} Coins, {earliest} -> {latest}")

    coins_conn = get_conn('coins')
    market_data = preload_market_data(coins_conn, all_coins, data_start, latest)
    coins_conn.close()

    event_times = [e['entry_time'] for e in experiences]

    env = TradingEnvLive(
        coins=traded_coins,
        market_data=market_data,
        trade_start=earliest,
        trade_end=latest,
        train_start=earliest,
        event_times=event_times,
    )

    # Echte Rewards injizieren (gefixtes Format)
    env._replay_rewards = {}
    for e in experiences:
        key = (e['symbol'], e['entry_time'].isoformat()[:16])
        env._replay_rewards[key] = {
            'reward': e['reward'],
            'direction': e['direction'],
            'pnl_pct': e['pnl_pct'],
        }

    model.set_env(env)
    model.learn(total_timesteps=LEARN_STEPS, reset_num_timesteps=False)

    wins = sum(1 for e in experiences if e['pnl_pct'] > 0)
    losses = len(experiences) - wins
    avg_reward = sum(e['reward'] for e in experiences) / max(len(experiences), 1)
    span_h = (latest - earliest).total_seconds() / 3600

    print(f"  [Batch {batch_num}/{total_batches}] Fertig: {LEARN_STEPS} Steps | "
          f"{len(experiences)} Trades ({wins}W/{losses}L) | "
          f"{len(traded_coins)} Coins | {span_h:.1f}h | "
          f"Avg Reward: {avg_reward:+.2f}")


def main():
    print("=" * 70)
    print("  V1.1 INITIAL-LEARNING — Clean Backup + alle Trades")
    print("=" * 70)

    # 1. Agent stoppen
    print("\n[1/6] Agent-Service stoppen...")
    subprocess.run(["systemctl", "stop", "rl-agent"], check=True)
    time.sleep(2)
    result = subprocess.run(["systemctl", "is-active", "rl-agent"], capture_output=True, text=True)
    if result.stdout.strip() == "active":
        print("  FEHLER: Agent läuft noch!")
        return
    print("  Agent gestoppt.")

    # 2. Aktuelles Modell sichern
    print("\n[2/6] Aktuelles Modell sichern...")
    if Path(MODEL_PATH).exists():
        subprocess.run(["cp", MODEL_PATH, CURRENT_BACKUP], check=True)
        print(f"  Backup: {CURRENT_BACKUP}")

    # 3. Clean-Backup vom Live-Server holen
    print("\n[3/6] Clean V1.1 vom Live-Server holen...")
    result = subprocess.run(
        ["scp", LIVE_SERVER_BACKUP, CLEAN_BACKUP],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FEHLER: scp fehlgeschlagen: {result.stderr}")
        # Fallback: Agent wieder starten
        subprocess.run(["systemctl", "start", "rl-agent"])
        return
    # Clean-Backup als aktives Modell setzen
    subprocess.run(["cp", CLEAN_BACKUP, MODEL_PATH], check=True)
    print(f"  Clean-Modell geladen: {CLEAN_BACKUP}")

    # 4. Trades laden
    print("\n[4/6] Geschlossene Trades aus DB laden...")
    trades = get_closed_trades()
    print(f"  {len(trades)} geschlossene Trades gefunden")

    if not trades:
        print("  Keine Trades — starte Agent ohne Learning.")
        subprocess.run(["systemctl", "start", "rl-agent"], check=True)
        return

    experiences = build_experiences(trades)
    wins = sum(1 for e in experiences if e['pnl_pct'] > 0)
    losses = len(experiences) - wins
    print(f"  {wins}W / {losses}L | "
          f"Zeitraum: {experiences[0]['entry_time']} -> {experiences[-1]['exit_time']}")

    # 5. Batched Experience-Learning
    print("\n[5/6] Experience-Learning in Batches...")
    model = PPO.load(MODEL_PATH)
    print(f"  Modell geladen: {MODEL_PATH}")

    total_batches = (len(experiences) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(experiences), BATCH_SIZE):
        batch = experiences[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            experience_learn_batch(model, batch, batch_num, total_batches)
            model.save(str(MODEL_PATH))
            print(f"  Modell gespeichert nach Batch {batch_num}")
        except Exception as e:
            print(f"  FEHLER bei Batch {batch_num}: {e}")
            traceback.print_exc()
            # Weitermachen mit nächstem Batch

    print(f"\n  Learning abgeschlossen: {total_batches} Batches, "
          f"{len(experiences)} Trades verarbeitet")

    # 6. Agent starten
    print("\n[6/6] Agent-Service starten...")
    subprocess.run(["systemctl", "start", "rl-agent"], check=True)
    time.sleep(2)
    result = subprocess.run(["systemctl", "is-active", "rl-agent"], capture_output=True, text=True)
    status = result.stdout.strip()
    print(f"  Agent-Status: {status}")

    if status == "active":
        print("\n" + "=" * 70)
        print("  FERTIG — V1.1 Agent läuft mit frisch gelerntem Modell")
        print(f"  Clean-Backup: {CLEAN_BACKUP}")
        print(f"  Pre-Learning-Backup: {CURRENT_BACKUP}")
        print("=" * 70)
    else:
        print("\n  WARNUNG: Agent ist nicht aktiv! Logs prüfen:")
        print("  journalctl -u rl-agent -n 30 --no-pager")


if __name__ == '__main__':
    main()
