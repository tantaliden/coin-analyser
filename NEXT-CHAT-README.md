# COIN-ANALYSER Status - 02.02.2026

## AKTUELLER STAND

Backend ist FERTIG und läuft auf Port 8002.

### Getestet und funktioniert:
- `curl http://localhost:8002/` → OK
- `curl http://localhost:8002/api/v1/meta/health` → Beide DBs connected
- `curl http://localhost:8002/api/v1/meta/config` → Config OK

### Service:
- Systemd: coin-analyser-api.service (enabled, running)
- Port: 8002
- Pfad: /opt/coin/backend/

### GitHub:
- https://github.com/tantaliden/coin-analyser
- Letzter Commit: "Backend modules complete"

### Module (alle fertig):
auth, meta, search, groups, coins, indicators, user, wallet, bot

### Datenbanken:
- coins (klines) - volker_admin
- analyser_app (users, indicators) - volker_admin

## NÄCHSTE SCHRITTE

1. Nginx für coin.tantaliden.de → localhost:8002
2. Frontend aufsetzen (Port 3002)
3. Alte analyser-api Services stoppen (Data-Ingestors laufen lassen!)

## WICHTIGE PFADE

- Neues Projekt: /opt/coin/
- Altes Projekt: /opt/analyser/ (READ ONLY)
- Settings: /opt/coin/settings.json
- Venv: /opt/coin/venv/

## 4-FILE PRINZIP EINHALTEN!
