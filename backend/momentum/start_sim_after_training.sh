#!/bin/bash
# Warte bis Base-Model-Training fertig ist (PID 105791)
while kill -0 105791 2>/dev/null; do
    sleep 10
done

echo "[$(date)] Base-Model Training fertig"

# Cache leeren
find /opt/coin/backend/momentum -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Simulation starten
echo "[$(date)] Starte Simulation..."
nohup /opt/coin/venv/bin/python /opt/coin/backend/momentum/scanner.py \
  --sim "2026-01-01T00:00:00+00:00" "2026-03-01T00:00:00+00:00" 5 \
  > /opt/coin/logs/sim_default.log 2>&1 &

echo "[$(date)] Simulation gestartet, PID: $!"
