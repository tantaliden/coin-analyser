#!/bin/bash
# Gap-Closer: Wartet bis Default-Sim (PID 132194) fertig ist, startet dann 01.03.-06.03.
TARGET_PID=132194
LOG="/opt/coin/logs/sim_gap_close.log"
SCANNER="/opt/coin/backend/momentum/scanner.py"
VENV="/opt/coin/venv/bin/python"

echo "[$(date)] Warte auf PID $TARGET_PID (Default-Sim)..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $TARGET_PID beendet. Starte Lücken-Sim 01.03. - 06.03.2026..."
sleep 10

cd /opt/coin/backend/momentum
nohup $VENV $SCANNER --sim 2026-03-01T00:00:00+01:00 2026-03-06T23:59:00+01:00 5 > $LOG 2>&1 &
NEW_PID=$!

echo "[$(date)] Lücken-Sim gestartet mit PID $NEW_PID"
echo "Log: $LOG"
