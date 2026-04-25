"""Speichert Indicator-Set aus ChartView-Kriterien. Ohne Fallbacks."""
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_app_db
from auth.auth import get_current_user
from search.predictor_settings import SET_DEFAULTS, INITIAL_POINT_SETTINGS
from .criteria_mapper import map_kind_to_db, resolve_time_window, resolve_aggregator, serialize_fuzzy

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-save"])


class FuzzyIn(BaseModel):
    # Alle Felder sind Pflicht — keine Fallbacks
    valueTolerance: float
    timeTolerance: float
    slopeTolerance: float
    ratioTolerance: float
    useRange: bool = False
    rangeMin: Optional[float] = None
    rangeMax: Optional[float] = None


class CriterionIn(BaseModel):
    kind: str
    field: str
    value: Optional[float] = None
    value2: Optional[float] = None
    field2: Optional[str] = None
    time_offset: Optional[int] = None
    time_offset2: Optional[int] = None
    time_offset_from: Optional[int] = None
    time_offset_to: Optional[int] = None
    pattern_id: Optional[str] = None
    sequence: Optional[str] = None
    fuzzy: FuzzyIn  # Pflicht
    is_initial_point: bool = False
    initial_fixed_offset: Optional[int] = None


class InitialPointConfigIn(BaseModel):
    match_mode: str
    match_threshold: int
    enforce_sequence: bool
    window_minutes: int


class PrimarySearchContext(BaseModel):
    """Zeitraum + Filter der Primary-Suche, die das Set produzierte — fuer Reproduzierbarkeit."""
    search_date_from: str  # ISO
    search_date_to: str    # ISO
    search_percent_min: float
    search_percent_max: Optional[float] = None
    search_direction: str
    search_duration_minutes: int
    events_at_creation: int
    coin_group_id: Optional[int] = None


class SaveSetRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str = ''
    duration_minutes: int
    direction: str
    target_percent: float
    prehistory_minutes: int
    candle_timeframe: int
    criteria: List[CriterionIn] = []
    initial_points: List[CriterionIn] = []
    initial_point_config: InitialPointConfigIn
    global_fuzzy: FuzzyIn
    primary_context: PrimarySearchContext


@router.post("/sets/from-drawings")
async def save_set_from_drawings(request: SaveSetRequest, current_user: dict = Depends(get_current_user)):
    """Speichert neues Indicator-Set inkl. Primary-Search-Kontext fuer Reproduzierbarkeit."""
    if not request.criteria and not request.initial_points:
        raise HTTPException(400, "Weder Kriterien noch Initialpunkte angegeben")

    ctx = request.primary_context

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO indicator_sets (
                    owner_id, name, description, duration_minutes, direction,
                    target_percent, prehistory_minutes, candle_timeframe,
                    initial_point_config, search_pattern, coin_group_id,
                    search_date_from, search_date_to,
                    search_percent_min, search_percent_max,
                    search_direction, search_duration_minutes, events_at_creation
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING set_id
            """, (
                current_user['user_id'], request.name, request.description,
                request.duration_minutes, request.direction, request.target_percent,
                request.prehistory_minutes, request.candle_timeframe,
                json.dumps(request.initial_point_config.model_dump()),
                json.dumps({'global_fuzzy': request.global_fuzzy.model_dump()}),
                ctx.coin_group_id,
                ctx.search_date_from, ctx.search_date_to,
                ctx.search_percent_min, ctx.search_percent_max,
                ctx.search_direction, ctx.search_duration_minutes, ctx.events_at_creation,
            ))
            set_id = cur.fetchone()['set_id']

            for i, ip in enumerate(request.initial_points):
                _insert_item(cur, set_id, ip, i, is_initial=True,
                             timeframe=request.candle_timeframe)

            for i, c in enumerate(request.criteria):
                _insert_item(cur, set_id, c, len(request.initial_points) + i,
                             is_initial=False, timeframe=request.candle_timeframe)

            conn.commit()
            return {"set_id": set_id, "name": request.name}


def _insert_item(cur, set_id, crit, sort_order, is_initial, timeframe):
    fuzzy_json = serialize_fuzzy(crit.fuzzy)
    indicator_type, op, pattern_data = map_kind_to_db(crit)
    time_start, time_end = resolve_time_window(crit)
    agg_str = resolve_aggregator(timeframe)

    cur.execute("""
        INSERT INTO indicator_items (
            set_id, time_start_minutes, time_end_minutes, indicator_type,
            condition_operator, condition_value, condition_value2,
            aggregator, pattern_data, sort_order,
            fuzzy_config, is_initial_point, initial_fixed_offset
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        set_id, time_start, time_end, indicator_type,
        op, crit.value, crit.value2,
        agg_str, json.dumps(pattern_data) if pattern_data else None, sort_order,
        json.dumps(fuzzy_json), is_initial, crit.initial_fixed_offset,
    ))
