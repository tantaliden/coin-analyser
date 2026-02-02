"""USER STATE ROUTES - Persistence"""
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/user", tags=["user"])

class UserStateUpdate(BaseModel):
    current_set_id: Optional[int] = None
    current_filters: Optional[dict] = None
    prehistory_minutes: Optional[int] = None
    selected_event_ids: Optional[List[int]] = None
    chart_settings: Optional[dict] = None
    module_layouts: Optional[List[dict]] = None

@router.get("/state")
async def get_user_state(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT current_set_id, current_filters, prehistory_minutes, selected_event_ids, chart_settings, module_layouts, updated_at
                FROM user_state WHERE user_id = %s""", (user_id,))
            row = cur.fetchone()
            if not row:
                cur.execute("INSERT INTO user_state (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))
                conn.commit()
                cur.execute("""SELECT current_set_id, current_filters, prehistory_minutes, selected_event_ids, chart_settings, module_layouts, updated_at
                    FROM user_state WHERE user_id = %s""", (user_id,))
                row = cur.fetchone()
            if row:
                return {"current_set_id": row['current_set_id'], "current_filters": row['current_filters'],
                    "prehistory_minutes": row['prehistory_minutes'], "selected_event_ids": row['selected_event_ids'],
                    "chart_settings": row['chart_settings'], "module_layouts": row['module_layouts'] or [],
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None}
            return {"current_set_id": None, "current_filters": {"durationMinutes": 120, "minPercent": 5, "maxPercent": 100, "direction": "up"},
                "prehistory_minutes": 720, "selected_event_ids": [], "chart_settings": {"mode": "overlap", "chartType": "line"},
                "module_layouts": [], "updated_at": None}

@router.put("/state")
async def update_user_state(state: UserStateUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    updates, params = [], []
    if state.current_set_id is not None:
        updates.append("current_set_id = %s")
        params.append(state.current_set_id if state.current_set_id > 0 else None)
    if state.current_filters is not None:
        updates.append("current_filters = %s")
        params.append(json.dumps(state.current_filters))
    if state.prehistory_minutes is not None:
        updates.append("prehistory_minutes = %s")
        params.append(state.prehistory_minutes)
    if state.selected_event_ids is not None:
        updates.append("selected_event_ids = %s")
        params.append(json.dumps(state.selected_event_ids))
    if state.chart_settings is not None:
        updates.append("chart_settings = %s")
        params.append(json.dumps(state.chart_settings))
    if not updates:
        return {"status": "no_changes"}
    params.append(user_id)
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_state (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))
            cur.execute(f"UPDATE user_state SET {', '.join(updates)} WHERE user_id = %s", params)
            conn.commit()
    return {"status": "updated"}
