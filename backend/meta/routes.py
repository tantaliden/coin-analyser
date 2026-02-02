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
    return {"status": "ok" if (coins_ok and app_ok) else "degraded", "version": SETTINGS['app']['version'],
            "databases": {"coins": {"connected": coins_ok}, "app": {"connected": app_ok}}, "timestamp": datetime.utcnow().isoformat()}

@router.get("/symbols")
async def get_symbols(current_user: dict = Depends(get_current_user)):
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval = '1m' ORDER BY symbol")
            symbols = [row['symbol'] for row in cur.fetchall()]
    return {"symbols": symbols, "count": len(symbols)}

@router.get("/intervals")
async def get_intervals(current_user: dict = Depends(get_current_user)):
    return {"intervals": ["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "2d", "3d", "7d", "15d", "30d", "1M", "2M", "3M", "6M", "1y", "2y", "3y"]}

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT symbol) as symbols FROM klines WHERE interval='1m'")
            symbols = cur.fetchone()['symbols']
            cur.execute("SELECT MIN(open_time) as earliest, MAX(open_time) as latest FROM klines WHERE interval='1m'")
            times = cur.fetchone()
    return {"symbols": symbols, "earliest": times['earliest'].isoformat() if times['earliest'] else None, "latest": times['latest'].isoformat() if times['latest'] else None}

@router.get("/config")
async def get_frontend_config():
    return {
        "app": {"name": SETTINGS['app']['name'], "version": SETTINGS['app']['version']},
        "search": SETTINGS.get('search', {"defaultTargetPercent": 5, "defaultDurationMinutes": 120, "defaultDirection": "up"}),
        "indicatorFields": ["close", "volume", "trades", "high", "low"],
        "indicatorOperations": [">", "<", ">=", "<=", "between"],
    }
