"""Endpoint: liefert Anomalie-Vorschlaege fuer Chart-Daten."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import pytz

from shared.database import get_coins_db
from auth.auth import get_current_user
from .candle_aggregator import load_candles_for_symbol
from .anomaly_detector import detect_anomalies

router = APIRouter(prefix="/api/v1/search", tags=["anomaly"])
BERLIN_TZ = pytz.timezone('Europe/Berlin')


class AnomalyRequest(BaseModel):
    symbol: str
    start_time: str          # ISO
    end_time: str            # ISO (= event_start fuer Prehistory-Scan)
    candle_timeframe: int    # Minuten


@router.post("/anomalies")
async def find_anomalies(request: AnomalyRequest, current_user: dict = Depends(get_current_user)):
    if request.candle_timeframe <= 0:
        raise HTTPException(400, "candle_timeframe muss > 0 sein")
    try:
        start = datetime.fromisoformat(request.start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(request.end_time.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(400, f"Ungueltiges Datum: {e}")

    with get_coins_db() as conn:
        with conn.cursor() as cur:
            candles = load_candles_for_symbol(
                request.symbol, start, end, request.candle_timeframe, cur
            )
    if not candles:
        raise HTTPException(404, f"Keine Candles fuer {request.symbol} im Zeitraum")

    suggestions = detect_anomalies(candles, event_start_time=end)
    # open_time als ISO zurueckgeben
    for s in suggestions:
        s['open_time'] = s['open_time'].isoformat() if hasattr(s['open_time'], 'isoformat') else s['open_time']

    return {
        "symbol": request.symbol,
        "candle_timeframe": request.candle_timeframe,
        "candles_scanned": len(candles),
        "suggestions": suggestions,
    }
