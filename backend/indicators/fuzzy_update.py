"""Fuzzy-Only Update. Zeit, Grundwerte, Bereich, Operator sind GESPERRT — nur Unschaerfe editierbar."""
import json
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-fuzzy"])


ALLOWED_FUZZY_KEYS = {'valueTolerance', 'timeTolerance', 'slopeTolerance', 'ratioTolerance',
                      'useRange', 'rangeMin', 'rangeMax'}

ALLOWED_IP_CONFIG_KEYS = {'match_mode', 'match_threshold', 'enforce_sequence', 'window_minutes'}


class FuzzyUpdate(BaseModel):
    valueTolerance: Optional[float] = None
    timeTolerance: Optional[float] = None
    slopeTolerance: Optional[float] = None
    ratioTolerance: Optional[float] = None
    useRange: Optional[bool] = None
    rangeMin: Optional[float] = None
    rangeMax: Optional[float] = None


@router.put("/items/{item_id}/fuzzy")
async def update_item_fuzzy(item_id: int, update: FuzzyUpdate, current_user: dict = Depends(get_current_user)):
    patch = update.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(400, "Keine Fuzzy-Werte angegeben")

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT i.fuzzy_config, s.owner_id, s.is_locked
                FROM indicator_items i
                JOIN indicator_sets s ON i.set_id = s.set_id
                WHERE i.item_id = %s
            """, (item_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Item not found")
            if row['owner_id'] != current_user['user_id']:
                raise HTTPException(403, "Not your set")
            if row['is_locked']:
                raise HTTPException(423, "Set is locked")

            if row['fuzzy_config'] is None:
                raise HTTPException(500, "Item hat keine fuzzy_config — kaputter Datensatz")

            current = row['fuzzy_config'] if isinstance(row['fuzzy_config'], dict) else json.loads(row['fuzzy_config'])
            current.update(patch)

            cur.execute("UPDATE indicator_items SET fuzzy_config = %s WHERE item_id = %s",
                        (json.dumps(current), item_id))
            conn.commit()
            return {"item_id": item_id, "fuzzy_config": current}


@router.put("/sets/{set_id}/initial-point-config")
async def update_initial_point_config(set_id: int, config: dict, current_user: dict = Depends(get_current_user)):
    if not isinstance(config, dict):
        raise HTTPException(400, "Body muss dict sein")

    unknown = set(config.keys()) - ALLOWED_IP_CONFIG_KEYS
    if unknown:
        raise HTTPException(400, f"Unbekannte Schluessel: {unknown}")

    if not config:
        raise HTTPException(400, "Keine aenderbaren Werte angegeben")

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id, is_locked, initial_point_config FROM indicator_sets WHERE set_id = %s", (set_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Set not found")
            if row['owner_id'] != current_user['user_id']:
                raise HTTPException(403, "Not your set")
            if row['is_locked']:
                raise HTTPException(423, "Set is locked")

            if row['initial_point_config'] is None:
                raise HTTPException(500, "Set hat keine initial_point_config — kaputter Datensatz")

            current = row['initial_point_config'] if isinstance(row['initial_point_config'], dict) else json.loads(row['initial_point_config'])
            current.update(config)

            cur.execute("UPDATE indicator_sets SET initial_point_config = %s, updated_at = NOW() WHERE set_id = %s",
                        (json.dumps(current), set_id))
            conn.commit()
            return {"set_id": set_id, "initial_point_config": current}
