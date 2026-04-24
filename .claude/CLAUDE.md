# Coin-Analyser (Tresor) - Projektregeln

## Projektstruktur
- **Backend**: `/opt/coin/backend/` - Python/FastAPI, Port 8002
- **Frontend**: `/opt/coin/frontend/` - React/Vite, Port 3002
- **Config**: `/opt/coin/settings.json` - Single Source of Truth (DB-Credentials, Binance Keys, etc.)
- **Training**: `/opt/training/` - Modelle, Scripts, Envs (auf Live-Server)
- **Services**: `coin-analyser-api`, `coin-analyser-frontend`, `rl-agent`, `agg-refresher`
- **DB**: `analyser_app` (PostgreSQL) für App-Daten, `coins` (TimescaleDB) für Kurs-Daten, `learner` für Feedback
- **Domain**: tradebot.tantaliden.com (Live-Server 82.165.236.163)
- **venv**: `/opt/coin/venv/`

## 4-File-Prinzip
`naming.js`, `router.js`, `settings.json`, `index.js` sind die Kern-Config-Dateien. Keine Hardcodes außerhalb dieser Struktur.

## RL-Agent (W-V1.0 — aktiv seit 31.03.2026)
- **Service**: `rl-agent.service` → `service.py` (ACTIVE_VERSION='w1')
- **Env**: `env_wallet.py` (59 Features, Short-Incentives, Wochenziel 15%)
- **Modell**: `rl_wallet_v1.zip` (unter /opt/coin/database/data/models/)
- **Entry**: DB-Daten (compute_observation_live), Coins shuffled
- **Management**: HL-Live-Daten (compute_observation_hl), 5 Min Takt
- **Experience-Learning**: Nach 150 Trades, 2000 Steps
- **Max 15 Positionen, $20 Trade-Size, kein SL, 3h Timeout**
- **Verkaufs-Block**: -0.5% bis +0.5%
- **Zentrale Funktion**: `get_observation()` — Agent sieht KEINEN Unterschied Training/Live

## Wichtige Dateien
- `backend/rl_agent/service.py` — W-V1.0 Live-Service
- `backend/rl_agent/env_wallet.py` — Training-Environment (59 Features)
- `backend/rl_agent/features.py` — compute_observation (Training + Live)
- `backend/rl_agent/trader.py` — Hyperliquid Trading API
- `backend/rl_agent/routes.py` — RL-Agent Frontend API
- `backend/wallet/routes.py` — Wallet API (Binance deaktiviert, nur HL)
- `database/services/agg_refresher.py` — Agg + Metrics Refresh (38s nach Bucket-Ende)
- `database/services/ingestor.py` — Binance klines Ingestor

## Workflow vor Änderungen
1. Dateien komplett lesen bevor Änderungen
2. Backup machen (Modelle, Config)
3. Skalpell, nicht Machete — minimale Änderungen
4. Syntax-Check vor Deploy
5. Nach Frontend-Änderung: `cd /opt/coin/frontend && npm run build`
6. Services neustarten: `systemctl restart coin-analyser-api coin-analyser-frontend`
7. Kein sed auf große Dateien

## Training (unter /opt/training/)
- **Models**: /opt/training/models/ (Checkpoints, ORIGINAL Backups)
- **Scripts**: /opt/training/scripts/ (train_w_v1.py, train_w_v1_p345.py, etc.)
- **Envs**: /opt/training/envs/ (env_wallet.py, env_long.py, etc.)
- **ORIGINAL Backups NICHT anfassen**

## Auto-Memory
- **Pfad**: `/root/.claude/projects/-opt-coin/memory/MEMORY.md`
- Bei jeder Session: MEMORY.md lesen und am Ende aktualisieren
- MUSS gepflegt werden!

## Kommunikation
- Deutsch, per Du
- Nicht raten, fragen wenn unklar
- Keine endlosen Infos — machen, dann Zusammenfassung
- Wenn mehr Ressourcen/Zeit nötig: SOFORT informieren
- Filigran, nicht dumpfbackenmäßig
