"""Batch-Anomalie-Endpoint: Scannt Prehistory aller Suchergebnisse, aggregiert Anomalien
nach Haeufigkeit (metric + Offset-Bucket vor Event). Pro Event zaehlt jede Gruppe nur 1x.
Keine Fallbacks — fehlt Feld, ValueError."""

from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_coins_db
from auth.auth import get_current_user
from .candle_aggregator import load_candles_for_symbol
from .anomaly_detector import detect_anomalies, detect_patterns, _normalize_dt
from .predictor_settings import BATCH_ANOMALY_SETTINGS

router = APIRouter(prefix="/api/v1/search", tags=["anomaly"])


class BatchEvent(BaseModel):
    symbol: str
    event_start: str  # ISO


class BatchRequest(BaseModel):
    events: List[BatchEvent]
    prehistory_minutes: int
    candle_timeframe: int
    bucket_minutes: int = BATCH_ANOMALY_SETTINGS["bucket_minutes"]


def _parse_event_start(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace('Z', '+00:00').replace(' ', 'T'))
    return _normalize_dt(dt)


@router.post("/anomalies/batch")
async def anomalies_batch(req: BatchRequest, current_user: dict = Depends(get_current_user)):
    if req.candle_timeframe <= 0:
        raise HTTPException(400, "candle_timeframe muss > 0 sein")
    if req.prehistory_minutes <= 0:
        raise HTTPException(400, "prehistory_minutes muss > 0 sein")
    if req.bucket_minutes <= 0:
        raise HTTPException(400, "bucket_minutes muss > 0 sein")
    if not req.events:
        raise HTTPException(400, "events darf nicht leer sein")

    # Aggregation: key = (metric, bucket_start_min) -> set of event_keys
    groups: dict = {}
    # Summen fuer Mittelwerte: key -> (sum_abs_z, count_hits)
    zs: dict = {}

    scanned = 0
    with_anomalies = 0

    with get_coins_db() as conn:
        with conn.cursor() as cur:
            for idx, ev in enumerate(req.events):
                try:
                    ev_start = _parse_event_start(ev.event_start)
                except ValueError as e:
                    raise HTTPException(400, f"Event {idx}: ungueltiges event_start ({e})")

                start = ev_start - timedelta(minutes=req.prehistory_minutes)
                end = ev_start
                candles = load_candles_for_symbol(
                    ev.symbol, start, end, req.candle_timeframe, cur
                )
                if not candles:
                    continue
                scanned += 1

                suggestions = detect_anomalies(candles, event_start_time=end)
                pattern_hits = detect_patterns(candles, event_start_time=end)

                seen_in_event = set()
                event_key = f"{ev.symbol}|{ev.event_start}|{idx}"

                # 1) Z-Score-Anomalien (metrics)
                for sug in suggestions:
                    open_time = _normalize_dt(sug['open_time'])
                    if open_time is None:
                        continue
                    offset_min = (ev_start - open_time).total_seconds() / 60.0
                    bucket_start = int(offset_min // req.bucket_minutes) * req.bucket_minutes
                    for an in sug['anomalies']:
                        metric = an['metric']
                        key = (metric, bucket_start)
                        zs.setdefault(key, [0.0, 0])
                        zs[key][0] += abs(float(an['z_score']))
                        zs[key][1] += 1
                        if key in seen_in_event:
                            continue
                        seen_in_event.add(key)
                        groups.setdefault(key, set()).add(event_key)

                # 2) Candle-Muster (pattern:<pid>)
                for ph in pattern_hits:
                    open_time = _normalize_dt(ph['open_time'])
                    if open_time is None:
                        continue
                    offset_min = (ev_start - open_time).total_seconds() / 60.0
                    bucket_start = int(offset_min // req.bucket_minutes) * req.bucket_minutes
                    metric = f"pattern:{ph['pattern_id']}"
                    key = (metric, bucket_start)
                    zs.setdefault(key, [0.0, 0])
                    zs[key][0] += float(ph['score'])
                    zs[key][1] += 1
                    if key in seen_in_event:
                        continue
                    seen_in_event.add(key)
                    groups.setdefault(key, set()).add(event_key)

                if seen_in_event:
                    with_anomalies += 1

    # In Liste umwandeln + sortieren. event_refs = {symbol, event_start} fuer Cascade-Filter.
    result = []
    for (metric, bucket_start), event_set in groups.items():
        z_sum, z_count = zs.get((metric, bucket_start), (0.0, 0))
        avg_z = round(z_sum / z_count, 2) if z_count > 0 else 0.0
        view_count = len(event_set)
        freq = round(view_count / scanned * 100, 1) if scanned > 0 else 0.0
        event_refs = []
        for k in sorted(event_set):
            # Format "symbol|event_start|idx" -> zurueck in dict
            parts = k.split('|')
            if len(parts) >= 2:
                event_refs.append({"symbol": parts[0], "event_start": parts[1]})
        result.append({
            "metric": metric,
            "offset_bucket_start_min": bucket_start,
            "offset_bucket_end_min": bucket_start + req.bucket_minutes,
            "view_count": view_count,
            "total_scanned": scanned,
            "frequency_pct": freq,
            "avg_abs_z_score": avg_z,
            "hit_count_total": z_count,
            "event_refs": event_refs,
        })

    # Sortiert: Haeufigkeit desc, dann naeher am Event zuerst (kleiner bucket_start)
    result.sort(key=lambda g: (-g["view_count"], g["offset_bucket_start_min"]))

    return {
        "scanned_events": scanned,
        "requested_events": len(req.events),
        "events_with_anomalies": with_anomalies,
        "bucket_minutes": req.bucket_minutes,
        "prehistory_minutes": req.prehistory_minutes,
        "candle_timeframe": req.candle_timeframe,
        "groups": result,
    }
