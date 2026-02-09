"""META ROUTES - Health, Symbols, Stats, Config"""
import json
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_coins_db, get_app_db
from auth.auth import get_current_user

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

router = APIRouter(prefix="/api/v1/meta", tags=["meta"])

@router.get("/health")
async def health():
    coins_ok = app_ok = False
    try:
        with get_coins_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                coins_ok = True
    except: pass
    try:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                app_ok = True
    except: pass
    return {
        "status": "ok" if (coins_ok and app_ok) else "degraded", 
        "version": SETTINGS['app']['version'],
        "databases": {"coins": {"connected": coins_ok}, "app": {"connected": app_ok}}, 
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/symbols")
async def get_symbols(current_user: dict = Depends(get_current_user)):
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval = '1m' ORDER BY symbol")
            symbols = [row['symbol'] for row in cur.fetchall()]
    return {"symbols": symbols, "count": len(symbols)}

@router.get("/intervals")
async def get_intervals(current_user: dict = Depends(get_current_user)):
    return {
        "intervals": [
            "1m", "2m", "5m", "10m", "15m", "30m", 
            "1h", "2h", "4h", "6h", "8h", "12h", 
            "1d", "2d", "3d", "7d", "15d", "30d", 
            "1M", "2M", "3M", "6M", 
            "1y", "2y", "3y"
        ]
    }

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT symbol) as symbols FROM klines WHERE interval='1m'")
            symbols = cur.fetchone()['symbols']
            cur.execute("SELECT MIN(open_time) as earliest, MAX(open_time) as latest FROM klines WHERE interval='1m'")
            times = cur.fetchone()
    return {
        "symbols": symbols, 
        "earliest": times['earliest'].isoformat() if times['earliest'] else None, 
        "latest": times['latest'].isoformat() if times['latest'] else None
    }

@router.get("/config")
async def get_frontend_config():
    """Frontend-Konfiguration - Single Source of Truth für UI"""
    return {
        "app": {
            "name": SETTINGS['app']['name'], 
            "version": SETTINGS['app']['version']
        },
        "search": {
            "defaultTargetPercent": 5,
            "defaultDurationMinutes": 120,
            "defaultDirection": "up"
        },
        # kline_metrics Durations - die verfügbaren Zeitfenster für Event-Suche
        "klineMetricsDurations": [30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600],
        # Timeframes für Charts
        "timeframes": {
            "chartOptions": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "aggregates": ["2m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        },
        # Indicator Fields
        "indicatorFields": [
            {"key": "close", "label": "Close", "color": "#3b82f6", "type": "price"},
            {"key": "volume", "label": "Volume", "color": "#f59e0b", "type": "volume"},
            {"key": "trades", "label": "Trades", "color": "#22c55e", "type": "count"},
            {"key": "high", "label": "High", "color": "#ef4444", "type": "price"},
            {"key": "low", "label": "Low", "color": "#06b6d4", "type": "price"},
        ],
        "indicatorOperations": [">", "<", ">=", "<=", "between"],
        # Event Colors für Chart-Overlay
        "eventColors": [
            "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ef4444",
            "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#14b8a6",
            "#f43f5e", "#6366f1", "#78716c", "#0ea5e9", "#d946ef"
        ],
        # UI Theme
        "ui": {
            "theme": {
                "colors": {
                    "bg": "#0f1419",
                    "surface": "#1a1f2e",
                    "border": "#2d3748",
                    "hover": "#374151",
                    "text": "#e5e7eb",
                    "muted": "#9ca3af",
                    "accent-blue": "#3b82f6",
                    "accent-green": "#22c55e",
                    "accent-red": "#ef4444",
                    "accent-yellow": "#eab308"
                }
            }
        }
    }
