# Coin-Analyser (Tresor) - Projektregeln

## Projektstruktur
- **Backend**: `/opt/coin/backend/` - Python/FastAPI, Port 8002
- **Frontend**: `/opt/coin/frontend/` - React/Vite, Port 3002
- **Config**: `/opt/coin/settings.json` - Single Source of Truth (DB-Credentials, Binance Keys, etc.)
- **Services**: `coin-analyser-api`, `coin-analyser-frontend`, `trade-tracker`
- **DB**: `analyser_app` (PostgreSQL) für App-Daten, `coins` (TimescaleDB) für Kurs-Daten
- **Domain**: tresor.tantaliden.com
- **venv**: `/opt/coin/venv/`

## 4-File-Prinzip
`naming.js`, `router.js`, `settings.json`, `index.js` sind die Kern-Config-Dateien. Keine Hardcodes außerhalb dieser Struktur.

## Backend-Module
auth, meta, search, groups, coins, indicators, user, wallet, bot, momentum, services

## Frontend-Module
MomentumModule.jsx, SearchModule.jsx, SearchResultsModule.jsx, ChartModule.jsx, wallet/, bot/, groups/, indicators/, chart/

## Wichtige Dateien
- `backend/momentum/scanner.py` - Scanner-Loop, check_active_predictions(), update_stats()
- `backend/momentum/routes.py` - Predictions API, Trade Execution (Rocket-Button)
- `backend/services/trade_tracker.py` - Erkennt OCO Sells auf Binance, trägt sie in trade_history ein, resolved Predictions
- `backend/wallet/routes.py` - Balance, Positions, Orders, History, realized-pnl
- `frontend/src/modules/MomentumModule.jsx` - Scanner UI, Predictions-Tabelle, Stats Drill-Down, Trade-Dialog

## Workflow vor Änderungen
1. **Git commit** vor jeder Änderung (Backup-Stand)
2. Dateien komplett lesen bevor Änderungen
3. Skalpell, nicht Machete - minimale Änderungen
4. Nach Änderung: `cd /opt/coin/frontend && npm run build`, dann Services neustarten
5. Services neustarten: `systemctl restart coin-analyser-api coin-analyser-frontend trade-tracker`

## Datenfluss Trade-Execution (Rocket-Button)
1. User klickt Rocket → `POST /api/v1/momentum/trade/{prediction_id}`
2. Market Buy auf Binance → OCO Order (TP + SL) wird gesetzt
3. Buy wird in `trade_history` geloggt mit `prediction_id`
4. `trade_tracker` Service erkennt Sell auf Binance → loggt Sell + resolved Prediction + aktualisiert Stats

## Bekannte Themen
- Predictions die gelöscht werden brechen die Zuordnung zu trade_history (prediction_id wird NULL via FK SET NULL)
- Scanner resolved Predictions über Kursvergleich - wenn der Kurs nach OCO-Fill zurückläuft, verpasst der Scanner den Hit
- trade_tracker (seit 2026-02-25) fängt das jetzt ab: resolved Predictions beim Sell-Log

## Auto-Memory
- **Pfad**: `/root/.claude/projects/-opt-coin/memory/MEMORY.md`
- Bei jeder Session: MEMORY.md lesen und am Ende aktualisieren
- Projektstand, offene Baustellen, Erkenntnisse dort festhalten
- MEMORY.md ist das Gedächtnis zwischen Sessions - MUSS gepflegt werden!

## Kommunikation
- Deutsch, per Du
- Nicht raten, fragen wenn unklar
- Keine endlosen Infos - machen, dann Zusammenfassung
- /home/claude/ enthält weitere Projekt-Docs und Regeln
