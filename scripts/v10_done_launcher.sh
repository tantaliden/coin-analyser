#!/bin/bash
# Wartet auf V10-Training-Ende auf dem Analyse-Server.
# Danach: V10 -> Live-Server, V11-Training starten, Agent starten.
#
# Usage: nohup /opt/coin/scripts/v10_done_launcher.sh > /opt/coin/logs/v10_launcher.log 2>&1 &

ANALYSE="root@82.165.236.163"
LIVE_MODEL_DIR="/opt/coin/database/data/models"
LOG="/opt/coin/logs/rl_train_v10.log"

echo "$(date) | Warte auf V10-Training-Ende..."

while true; do
    # Prüfe ob V10 fertig ist (Finales Modell gespeichert)
    DONE=$(ssh $ANALYSE "grep -c '\[V10\] Finales Modell gespeichert' $LOG 2>/dev/null")
    if [ "$DONE" -ge 1 ] 2>/dev/null; then
        echo "$(date) | V10 Training FERTIG!"
        break
    fi

    # Alternativ: Prüfe ob Prozess noch läuft
    RUNNING=$(ssh $ANALYSE "pgrep -f 'train_v10.py' | wc -l")
    if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
        # Prozess weg — prüfe ob erfolgreich oder abgestürzt
        if ssh $ANALYSE "grep -q 'Phase 3' $LOG"; then
            echo "$(date) | V10 Prozess beendet (Phase 3 erreicht)"
            break
        else
            echo "$(date) | WARNUNG: V10 Prozess weg aber Phase 3 nicht erreicht!"
            echo "$(date) | Manuell prüfen! Breche ab."
            exit 1
        fi
    fi

    sleep 300  # Alle 5 Min prüfen
done

# === 1. V10-Modell auf Live-Server kopieren ===
echo "$(date) | Kopiere V10-Modell auf Live-Server..."
scp $ANALYSE:$LIVE_MODEL_DIR/rl_ppo_trading_v10.zip $LIVE_MODEL_DIR/rl_ppo_trading_v10.zip
if [ $? -ne 0 ]; then
    echo "$(date) | FEHLER: SCP fehlgeschlagen!"
    exit 1
fi
echo "$(date) | V10-Modell kopiert."

# === 2. V11-Training auf Analyse-Server starten ===
echo "$(date) | Starte V11-Training auf Analyse-Server..."
ssh $ANALYSE "cd /opt/coin/backend && nohup /opt/coin/venv/bin/python3 -u rl_agent/train_v11.py > /opt/coin/logs/rl_train_v11.log 2>&1 &"
sleep 5
V11_RUNNING=$(ssh $ANALYSE "pgrep -f 'train_v11.py' | wc -l")
if [ "$V11_RUNNING" -ge 1 ] 2>/dev/null; then
    echo "$(date) | V11-Training gestartet."
else
    echo "$(date) | WARNUNG: V11-Training scheint nicht zu laufen!"
fi

# === 3. RL-Agent (V10) auf Live-Server starten ===
echo "$(date) | Starte RL-Agent V10 auf Live-Server..."
systemctl enable rl-agent
systemctl start rl-agent
sleep 5
if systemctl is-active --quiet rl-agent; then
    echo "$(date) | RL-Agent V10 LIVE! Service läuft."
else
    echo "$(date) | WARNUNG: RL-Agent Service nicht aktiv!"
fi

# === 4. Heartbeat-Watchdog wieder aktivieren ===
systemctl enable heartbeat-watchdog
systemctl start heartbeat-watchdog
echo "$(date) | Heartbeat-Watchdog aktiviert."

echo "$(date) | === FERTIG ==="
echo "$(date) | V10 -> Live (Agent läuft)"
echo "$(date) | V11 -> Training auf Analyse-Server"
