"""Laedt ein Set fuer den Load-Flow: Mainsearch-Params + Drawings-Rekonstruktion."""
import json
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-load"])


def _json_field(raw):
    if raw is None:
        return None
    return raw if isinstance(raw, dict) else json.loads(raw)


def _item_to_drawing(item):
    """Rekonstruiert eine Drawing-Konfiguration aus einem indicator_item.
    Wird im Frontend verwendet um die DrawingCanvas neu zu fuellen."""
    fuzzy = _json_field(item.get('fuzzy_config'))
    pattern_data = _json_field(item.get('pattern_data'))

    base = {
        'item_id': item['item_id'],
        'is_initial_point': bool(item.get('is_initial_point')),
        'indicator_type': item['indicator_type'],
        'condition_operator': item['condition_operator'],
        'condition_value': float(item['condition_value']) if item.get('condition_value') is not None else None,
        'condition_value2': float(item['condition_value2']) if item.get('condition_value2') is not None else None,
        'time_start_minutes': item['time_start_minutes'],
        'time_end_minutes': item['time_end_minutes'],
        'aggregator': item['aggregator'],
        'pattern_data': pattern_data,
        'fuzzy_config': fuzzy,
        'initial_fixed_offset': item.get('initial_fixed_offset'),
    }
    return base


@router.get("/sets/{set_id}/load")
async def load_set(set_id: int, current_user: dict = Depends(get_current_user)):
    """Liefert das komplette Set + Items + Mainsearch-Params fuer den Load-Flow im ChartView."""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT * FROM indicator_sets
                           WHERE set_id = %s AND (is_public = TRUE OR owner_id = %s)""",
                        (set_id, current_user['user_id']))
            s = cur.fetchone()
            if not s:
                raise HTTPException(404, "Set nicht gefunden")

            cur.execute("SELECT * FROM indicator_items WHERE set_id = %s ORDER BY sort_order, item_id",
                        (set_id,))
            items = [dict(r) for r in cur.fetchall()]

    drawings = [_item_to_drawing(it) for it in items]
    search_pattern = _json_field(s.get('search_pattern')) or {}
    ip_config = _json_field(s.get('initial_point_config')) or {}

    return {
        "set_id": s['set_id'],
        "name": s['name'],
        "description": s.get('description'),
        "mainsearch_params": {
            "start_date": s.get('search_date_from').isoformat() if s.get('search_date_from') else None,
            "end_date": s.get('search_date_to').isoformat() if s.get('search_date_to') else None,
            "target_percent": float(s['search_percent_min']) if s.get('search_percent_min') else None,
            "max_percent": float(s['search_percent_max']) if s.get('search_percent_max') else None,
            "direction": s.get('search_direction'),
            "duration_minutes": s.get('search_duration_minutes'),
            "coin_group_id": s.get('coin_group_id'),
            "events_at_creation": s.get('events_at_creation'),
        },
        "set_params": {
            "duration_minutes": s['duration_minutes'],
            "direction": s['direction'],
            "target_percent": float(s['target_percent']) if s.get('target_percent') else None,
            "prehistory_minutes": s.get('prehistory_minutes'),
            "candle_timeframe": s.get('candle_timeframe'),
        },
        "global_fuzzy": search_pattern.get('global_fuzzy'),
        "initial_point_config": ip_config,
        "drawings": drawings,
    }
