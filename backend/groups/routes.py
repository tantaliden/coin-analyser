"""GROUPS ROUTES - Coin Groups CRUD"""
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/groups", tags=["groups"])

class GroupCreate(BaseModel):
    name: str
    color: str = "#3B82F6"

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None

@router.get("")
async def get_groups(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT g.group_id as id, g.name, g.color, g.created_at,
                       COALESCE(array_agg(m.symbol) FILTER (WHERE m.symbol IS NOT NULL), '{}') as coins
                FROM coin_groups g
                LEFT JOIN coin_group_members m ON g.group_id = m.group_id
                WHERE g.user_id = %s
                GROUP BY g.group_id
                ORDER BY g.name
            """, (current_user['user_id'],))
            groups = cur.fetchall()
    return {"groups": [dict(g) for g in groups]}

@router.post("")
async def create_group(request: GroupCreate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO coin_groups (user_id, name, color) VALUES (%s, %s, %s) RETURNING group_id",
                       (current_user['user_id'], request.name, request.color))
            group_id = cur.fetchone()['group_id']
            conn.commit()
    return {"id": group_id, "name": request.name, "color": request.color}

@router.put("/{group_id}")
async def update_group(group_id: int, request: GroupUpdate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM coin_groups WHERE group_id = %s", (group_id,))
            group = cur.fetchone()
            if not group or group['user_id'] != current_user['user_id']:
                raise HTTPException(status_code=404, detail="Group not found")
            updates, values = [], []
            if request.name:
                updates.append("name = %s")
                values.append(request.name)
            if request.color:
                updates.append("color = %s")
                values.append(request.color)
            if updates:
                values.append(group_id)
                cur.execute(f"UPDATE coin_groups SET {', '.join(updates)} WHERE group_id = %s", values)
                conn.commit()
    return {"message": "Group updated"}

@router.delete("/{group_id}")
async def delete_group(group_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM coin_groups WHERE group_id = %s AND user_id = %s RETURNING group_id",
                       (group_id, current_user['user_id']))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Group not found")
            conn.commit()
    return {"message": "Group deleted"}

@router.post("/{group_id}/coins")
async def add_coins_to_group(group_id: int, symbols: List[str], current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM coin_groups WHERE group_id = %s", (group_id,))
            group = cur.fetchone()
            if not group or group['user_id'] != current_user['user_id']:
                raise HTTPException(status_code=404, detail="Group not found")
            for symbol in symbols:
                cur.execute("INSERT INTO coin_group_members (group_id, symbol) VALUES (%s, %s) ON CONFLICT DO NOTHING", (group_id, symbol))
            conn.commit()
    return {"message": f"Added {len(symbols)} coins"}

@router.delete("/{group_id}/coins/{symbol}")
async def remove_coin_from_group(group_id: int, symbol: str, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM coin_groups WHERE group_id = %s", (group_id,))
            group = cur.fetchone()
            if not group or group['user_id'] != current_user['user_id']:
                raise HTTPException(status_code=404, detail="Group not found")
            cur.execute("DELETE FROM coin_group_members WHERE group_id = %s AND symbol = %s", (group_id, symbol))
            conn.commit()
    return {"message": "Coin removed"}
