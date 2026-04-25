#!/bin/bash
TRAIN_PID=$1
LOG="/opt/coin/logs/rl_training_v4.log"

echo "$(date): Warte auf Training-PID $TRAIN_PID..."

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done

echo "$(date): Training beendet! Starte Backtest..."
tail -5 "$LOG"

cd /opt/coin/backend
/opt/coin/venv/bin/python3 -u rl_agent/backtest_v4.py > /opt/coin/logs/backtest_v4.log 2>&1

echo "$(date): Backtest abgeschlossen!"
tail -30 /opt/coin/logs/backtest_v4.log
