"""Zentraler Session-State Endpoint: UI-Zustand aggregiert in einem JSONB-Dict.
Liest/schreibt user_state.ui_extras + drawings — device-uebergreifend persistent."""

import json
from typing import Optional, Dict
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/user", tags=["user-session"])


class SessionStatePatch(BaseModel):
    """Partial-Update. Nur gesetzte Keys werden gemerged."""
    search: Optional[Dict] = None      # searchStore-Slice
    module: Optional[Dict] = None      # moduleStore-Slice
    drawings: Optional[Dict] = None    # drawingsStore map
    extras: Optional[Dict] = None      # beliebige weitere UI-States


def _ensure_row(cur, user_id):
    cur.execute("SELECT user_id FROM user_state WHERE user_id = %s", (user_id,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO user_state (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))


def _decode(raw):
    if raw is None:
        return {}
    return raw if isinstance(raw, dict) else json.loads(raw)


@router.get("/session-state")
async def get_session_state(current_user: dict = Depends(get_current_user)):
    """Liefert den aggregierten UI-State: drawings + ui_extras.split(search/module/extras)."""
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            _ensure_row(cur, user_id)
            cur.execute("SELECT drawings, ui_extras FROM user_state WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            conn.commit()

    extras = _decode(row['ui_extras'])
    return {
        "drawings": _decode(row['drawings']),
        "search": extras.get('search', {}),
        "module": extras.get('module', {}),
        "extras": {k: v for k, v in extras.items() if k not in ('search', 'module')},
    }


@router.put("/session-state")
async def put_session_state(patch: SessionStatePatch, current_user: dict = Depends(get_current_user)):
    """Merge-Update: nur angegebene Bereiche ueberschrieben. Kein Fallback — ungueltiger Body = 400."""
    user_id = current_user['user_id']

    with get_app_db() as conn:
        with conn.cursor() as cur:
            _ensure_row(cur, user_id)
            cur.execute("SELECT drawings, ui_extras FROM user_state WHERE user_id = %s", (user_id,))
            row = cur.fetchone()

            new_drawings = _decode(row['drawings'])
            new_extras = _decode(row['ui_extras'])

            if patch.drawings is not None:
                new_drawings = patch.drawings

            if patch.search is not None:
                new_extras['search'] = patch.search
            if patch.module is not None:
                new_extras['module'] = patch.module
            if patch.extras is not None:
                # merge weitere extras
                for k, v in patch.extras.items():
                    new_extras[k] = v

            cur.execute("""UPDATE user_state
                           SET drawings = %s, ui_extras = %s, updated_at = NOW()
                           WHERE user_id = %s""",
                        (json.dumps(new_drawings), json.dumps(new_extras), user_id))
            conn.commit()

    return {"ok": True}
