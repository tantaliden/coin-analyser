"""
META ROUTES - Health, Config, Symbols, Stats
Lädt alle Werte aus naming.js und settings.json
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_coins_db, get_app_db
from shared.naming_loader import NAMING
from auth.auth import get_current_user

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

router = APIRouter(prefix="/api/v1/meta", tags=["meta"])

@router.get("/health")
async def health():
    """Health Check für beide Datenbanken"""
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
    
    # Momentum Scanner Health (Default + 2h)
    def _check_scanner_heartbeat(hb_path):
        info = {}
        ok = False
        try:
            if os.path.exists(hb_path):
                age_seconds = time.time() - os.path.getmtime(hb_path)
                if age_seconds < 300:
                    ok = True
                    with open(hb_path) as f:
                        info = json.load(f)
                info['age_seconds'] = round(age_seconds)
        except Exception:
            pass
        return ok, info

    scanner_ok, scanner_info = _check_scanner_heartbeat('/opt/coin/logs/.scanner_heartbeat')
    scanner_2h_ok, scanner_2h_info = _check_scanner_heartbeat('/opt/coin/logs/.scanner_2h_heartbeat')

    all_ok = coins_ok and app_ok and scanner_ok and scanner_2h_ok
    return {
        "status": "ok" if all_ok else ("degraded" if (coins_ok and app_ok) else "critical"),
        "version": SETTINGS['app']['version'],
        "databases": {
            "coins": {"connected": coins_ok},
            "app": {"connected": app_ok}
        },
        "momentum_scanner": {
            "healthy": scanner_ok,
            **scanner_info
        },
        "momentum_scanner_2h": {
            "healthy": scanner_2h_ok,
            **scanner_2h_info
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/symbols")
async def get_symbols(current_user: dict = Depends(get_current_user)):
    """Alle verfügbaren Symbole aus klines"""
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM klines WHERE interval = '1m' ORDER BY symbol")
            symbols = [row['symbol'] for row in cur.fetchall()]
    return {"symbols": symbols, "count": len(symbols)}

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Datenbank-Statistiken"""
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
    """Frontend-Konfiguration aus naming.js und settings.json"""
    return {
        "app": {
            "name": SETTINGS['app']['name'], 
            "version": SETTINGS['app']['version']
        },
        "search": SETTINGS.get('search', {}),
        "trading": SETTINGS.get('trading', {}),
        # Aus naming.js
        "klineMetricsDurations": NAMING.get('klineMetricsDurations', [30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600]),
        "timeframes": NAMING.get('timeframes', {}),
        "candleTimeframes": NAMING.get('candleTimeframes', []),
        "avgPeriods": NAMING.get('avgPeriods', []),
        "indicatorFields": NAMING.get('indicatorFields', []),
        "indicatorOperations": NAMING.get('indicatorOperations', []),
        "indicatorAggregators": NAMING.get('indicatorAggregators', []),
        "eventColors": NAMING.get('eventColors', []),
        "overlapEventColors": NAMING.get('overlapEventColors', []),
        "setColorOptions": NAMING.get('setColorOptions', []),
        "searchResultColumns": NAMING.get('searchResultColumns', []),
        "labels": NAMING.get('labels', {}),
        # UI Theme aus settings.json
        "ui": SETTINGS.get('ui', {})
    }
