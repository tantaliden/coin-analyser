#!/bin/bash
# W-V1.0 Auto-Launcher
# Wartet auf Training-Ende, macht Backup, startet Agent + Health
#
# Usage: nohup /opt/coin/scripts/w1_auto_launch.sh > /opt/coin/logs/w1_launcher.log 2>&1 &

LOG="/opt/coin/logs/rl_train_w_v1.log"
MODEL="/opt/training/models/rl_wallet_v1.zip"
LIVE_MODEL="/opt/coin/database/data/models/rl_wallet_v1.zip"
BACKUP_DIR="/opt/training/models/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "$(date) | W-V1.0 Auto-Launcher gestartet"
echo "$(date) | Warte auf Training-Ende..."

while true; do
    # Prüfe ob Training fertig ist
    if grep -q "Finales Modell gespeichert\|Training abgeschlossen\|Phase 3.*gespeichert" "$LOG" 2>/dev/null; then
        echo "$(date) | Training FERTIG!"
        break
    fi

    # Prüfe ob Prozess noch läuft
    if ! pgrep -f "train_w_v1.py" > /dev/null; then
        # Prozess weg — prüfe ob erfolgreich
        if grep -q "Phase 3" "$LOG" 2>/dev/null; then
            LAST_STEP=$(grep '\[Phase 3' "$LOG" | tail -1 | grep -oP '\d+(?= steps)')
            if [ "$LAST_STEP" -ge 49000000 ] 2>/dev/null; then
                echo "$(date) | Training Prozess beendet (Phase 3 bei ${LAST_STEP} Steps)"
                break
            fi
        fi
        echo "$(date) | WARNUNG: Training Prozess weg aber nicht fertig!"
        echo "$(date) | Letzter Log-Eintrag:"
        tail -3 "$LOG"
        echo "$(date) | Warte 5 Min und prüfe erneut..."
        sleep 300
        if ! pgrep -f "train_w_v1.py" > /dev/null; then
            echo "$(date) | Training wirklich beendet. Starte trotzdem..."
            break
        fi
    fi

    sleep 120  # Alle 2 Min prüfen
done

# === 1. BACKUP ===
echo "$(date) | Erstelle Backup..."
mkdir -p "$BACKUP_DIR"

cp "$MODEL" "${BACKUP_DIR}/rl_wallet_v1_FINAL_${TIMESTAMP}.zip"
cp /opt/training/models/rl_wallet_v1_phase1.zip "${BACKUP_DIR}/rl_wallet_v1_phase1_${TIMESTAMP}.zip"
cp /opt/training/models/rl_wallet_v1_phase2.zip "${BACKUP_DIR}/rl_wallet_v1_phase2_${TIMESTAMP}.zip" 2>/dev/null
cp /opt/training/envs/env_wallet.py "${BACKUP_DIR}/env_wallet_${TIMESTAMP}.py"
cp /opt/training/scripts/train_w_v1.py "${BACKUP_DIR}/train_w_v1_${TIMESTAMP}.py"

# Original-Backup (NICHT ANFASSEN)
cp "$MODEL" "/opt/training/models/rl_wallet_v1_ORIGINAL.zip"

echo "$(date) | Backup erstellt: ${BACKUP_DIR}/"
ls -la "${BACKUP_DIR}/"*${TIMESTAMP}*

# === 2. MODELL KOPIEREN ===
echo "$(date) | Kopiere Modell fuer Live..."
cp "$MODEL" "$LIVE_MODEL"
echo "$(date) | Modell kopiert: $LIVE_MODEL"

# === 3. STATE RESET ===
echo "$(date) | Setze Agent-State zurueck..."
cat > /opt/coin/database/data/models/rl_agent_state.json << 'EOF'
{"total_points":2000.0,"total_profit":0.0,"total_trades":0,"total_wins":0,"total_losses":0,"losing_streak_days":0,"winning_streak_weeks":0,"current_day":"","current_week":"","day_pnl":0.0,"week_points":0.0,"week_points_raw":0.0,"prev_week_raw_points":0,"last_trade_times":{},"trades_today":0,"trade_day":"","weekly_target_points":0,"week_start_points":2000.0}
EOF

# === 4. AGENT STARTEN ===
echo "$(date) | Starte RL-Agent (W-V1.0)..."
systemctl enable rl-agent
systemctl start rl-agent
sleep 10

if systemctl is-active --quiet rl-agent; then
    echo "$(date) | RL-Agent W-V1.0 LIVE!"
    journalctl -u rl-agent --no-pager -n 5
else
    echo "$(date) | FEHLER: RL-Agent nicht gestartet!"
    journalctl -u rl-agent --no-pager -n 10
fi

# === 5. HEALTH + WATCHDOG ===
echo "$(date) | Aktiviere Watchdog..."
systemctl enable heartbeat-watchdog
systemctl start heartbeat-watchdog

# Health-Check: rl-agent statt rl-ensemble
sed -i "s/rl-ensemble/rl-agent/g" /opt/coin/backend/services/telegram_bot.py 2>/dev/null
sed -i "s/rl-ensemble/rl-agent/g" /opt/coin/backend/meta/routes.py 2>/dev/null
systemctl restart analyser-telegram-bot coin-analyser-api 2>/dev/null

echo "$(date) | === FERTIG ==="
echo "$(date) | W-V1.0 Agent LIVE"
echo "$(date) | Backup: ${BACKUP_DIR}/"
echo "$(date) | Original: /opt/training/models/rl_wallet_v1_ORIGINAL.zip"
