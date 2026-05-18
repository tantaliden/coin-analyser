"""Speichert Anomalie-Itemset als Indicator-Set. Items werden als indicator_items
mit indicator_type='anomaly:<metric>' abgelegt. primary_context ist Pflicht."""

import json
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-save"])


class AnomalyItem(BaseModel):
    metric: str                    # z.B. 'volume', 'rsi_14', 'pattern:hammer'
    bucket_start_min: int          # Minuten vor Event (positiv)
    bucket_end_min: int


class PrimaryContext(BaseModel):
    search_date_from: str
    search_date_to: str
    search_percent_min: float
    search_percent_max: Optional[float] = None
    search_direction: str
    search_duration_minutes: int
    events_at_creation: int
    coin_group_id: Optional[int] = None


class SaveAnomalySetRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str
    duration_minutes: int
    direction: str
    target_percent: float
    prehistory_minutes: int
    candle_timeframe: int
    items: List[AnomalyItem]
    primary_context: PrimaryContext
    view_count: int          # Wie viele Events dieses Itemset erfuellen
    frequency_pct: float


@router.post("/sets/from-anomaly-itemset")
async def save_set_from_anomaly_itemset(req: SaveAnomalySetRequest, current_user: dict = Depends(get_current_user)):
    if not req.items:
        raise HTTPException(400, "items darf nicht leer sein")
    ctx = req.primary_context

    meta = {
        "source": "anomaly_itemset",
        "view_count": req.view_count,
        "frequency_pct": req.frequency_pct,
    }

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO indicator_sets (
                    owner_id, name, description, duration_minutes, direction,
                    target_percent, prehistory_minutes, candle_timeframe,
                    search_pattern, coin_group_id,
                    search_date_from, search_date_to,
                    search_percent_min, search_percent_max,
                    search_direction, search_duration_minutes, events_at_creation
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING set_id
            """, (
                current_user['user_id'], req.name, req.description,
                req.duration_minutes, req.direction, req.target_percent,
                req.prehistory_minutes, req.candle_timeframe,
                json.dumps(meta), ctx.coin_group_id,
                ctx.search_date_from, ctx.search_date_to,
                ctx.search_percent_min, ctx.search_percent_max,
                ctx.search_direction, ctx.search_duration_minutes, ctx.events_at_creation,
            ))
            set_id = cur.fetchone()['set_id']

            for i, item in enumerate(req.items):
                indicator_type = f"anomaly:{item.metric}"
                # time_start_minutes / time_end_minutes: Konvention wie save_from_drawings
                # Negativ = vor Event. bucket_start_min=10 bedeutet -10..-bucket_end
                time_start = -item.bucket_end_min
                time_end = -item.bucket_start_min
                cur.execute("""
                    INSERT INTO indicator_items (
                        set_id, time_start_minutes, time_end_minutes, indicator_type,
                        sort_order
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (set_id, time_start, time_end, indicator_type, i))

            conn.commit()

    return {"set_id": set_id, "name": req.name, "item_count": len(req.items)}
