# Coin-Analyser

Modulare Crypto-Analyse-Plattform für coin.tantaliden.de

## 4-File-Prinzip

| Datei | Zweck |
|-------|-------|
| `naming.js` | Single Source of Truth - alle Namen/Konstanten |
| `router.js` | Module-Loader - verbindet alles |
| `settings.json` | Konfiguration - keine Hardcodes |
| `index.js` | Entry Point |

## Struktur

```
/opt/coin/
├── naming.js
├── router.js
├── settings.json
├── index.js
├── backend/
│   ├── shared/      # Database, Utils
│   ├── auth/        # Login, JWT
│   ├── search/      # Event-Suche
│   ├── candles/     # Kerzen-API
│   ├── indicators/  # Indikatoren
│   └── meta/        # Health, Symbols
└── frontend/
```

## Ports

- API: 8002 (analyser nutzt 8001)
- Frontend: 3002 (analyser nutzt 3001)

## Datenbanken

Nutzt die bestehenden DBs:
- `coins` (klines, aggregates)
- `analyser_app` (users, indicators)
