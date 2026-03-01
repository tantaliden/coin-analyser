"""USER ROUTES - State + Layout Persistence + Settings"""
import json
import requests
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user, encrypt_value, decrypt_value

router = APIRouter(prefix="/api/v1/user", tags=["user"])

# === STATE ===

class UserStateUpdate(BaseModel):
    current_set_id: Optional[int] = None
    current_filters: Optional[dict] = None
    prehistory_minutes: Optional[int] = None
    selected_event_ids: Optional[List[int]] = None
    chart_settings: Optional[dict] = None
    current_layout_id: Optional[int] = None

@router.get("/state")
async def get_user_state(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT current_set_id, current_filters, prehistory_minutes,
                selected_event_ids, chart_settings, module_layouts, updated_at
                FROM user_state WHERE user_id = %s""", (user_id,))
            row = cur.fetchone()
            if not row:
                cur.execute("INSERT INTO user_state (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
                conn.commit()
                cur.execute("""SELECT current_set_id, current_filters, prehistory_minutes,
                    selected_event_ids, chart_settings, module_layouts, updated_at
                    FROM user_state WHERE user_id = %s""", (user_id,))
                row = cur.fetchone()
            if row:
                return {
                    "current_set_id": row['current_set_id'],
                    "current_filters": row['current_filters'],
                    "prehistory_minutes": row['prehistory_minutes'],
                    "selected_event_ids": row['selected_event_ids'],
                    "chart_settings": row['chart_settings'],
                    "module_layouts": row['module_layouts'] or [],
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                }
            return {"current_set_id": None, "current_filters": {}, "prehistory_minutes": 720,
                    "selected_event_ids": [], "chart_settings": {}, "module_layouts": [], "updated_at": None}

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
            cur.execute("INSERT INTO user_state (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
            cur.execute(f"UPDATE user_state SET {', '.join(updates)} WHERE user_id = %s", params)
            conn.commit()
    return {"status": "updated"}

# === LAYOUTS (user_layouts Tabelle) ===

class LayoutSave(BaseModel):
    name: str
    layout_data: List[dict]
    is_default: Optional[bool] = False

class LayoutUpdate(BaseModel):
    layout_data: List[dict]

@router.get("/layouts")
async def get_layouts(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, name, layout_data, is_default, updated_at
                FROM user_layouts WHERE user_id = %s ORDER BY is_default DESC, name""",
                (current_user['user_id'],))
            rows = cur.fetchall()
    return {"layouts": [{
        "id": r['id'], "name": r['name'], "layout_data": r['layout_data'],
        "is_default": r['is_default'],
        "updated_at": r['updated_at'].isoformat() if r['updated_at'] else None
    } for r in rows]}

@router.post("/layouts")
async def create_layout(data: LayoutSave, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            if data.is_default:
                cur.execute("UPDATE user_layouts SET is_default = false WHERE user_id = %s", (user_id,))
            cur.execute("""INSERT INTO user_layouts (user_id, name, layout_data, is_default)
                VALUES (%s, %s, %s, %s) RETURNING id""",
                (user_id, data.name, json.dumps(data.layout_data), data.is_default))
            new_id = cur.fetchone()['id']
            conn.commit()
    return {"id": new_id, "name": data.name}

@router.put("/layouts/{layout_id}")
async def update_layout(layout_id: int, data: LayoutUpdate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""UPDATE user_layouts SET layout_data = %s, updated_at = NOW()
                WHERE id = %s AND user_id = %s""",
                (json.dumps(data.layout_data), layout_id, current_user['user_id']))
            conn.commit()
    return {"status": "updated"}

@router.put("/layouts/{layout_id}/default")
async def set_default_layout(layout_id: int, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE user_layouts SET is_default = false WHERE user_id = %s", (user_id,))
            cur.execute("UPDATE user_layouts SET is_default = true WHERE id = %s AND user_id = %s", (layout_id, user_id))
            conn.commit()
    return {"status": "updated"}

@router.delete("/layouts/{layout_id}")
async def delete_layout(layout_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_layouts WHERE id = %s AND user_id = %s", (layout_id, current_user['user_id']))
            conn.commit()
    return {"status": "deleted"}

# === SETTINGS (API Keys) ===

class ApiKeyUpdate(BaseModel):
    api_key: str
    api_secret: str

@router.get("/settings")
async def get_settings(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT
                binance_api_key_encrypted, binance_api_valid,
                hyperliquid_api_key_encrypted, hyperliquid_api_valid
                FROM users WHERE user_id = %s""", (user_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "binance": {
            "configured": bool(row['binance_api_key_encrypted']),
            "valid": bool(row['binance_api_valid']),
            "key_hint": decrypt_value(row['binance_api_key_encrypted'])[-4:] if row['binance_api_key_encrypted'] else None
        },
        "hyperliquid": {
            "configured": bool(row['hyperliquid_api_key_encrypted']),
            "valid": bool(row['hyperliquid_api_valid']),
            "key_hint": decrypt_value(row['hyperliquid_api_key_encrypted'])[-4:] if row['hyperliquid_api_key_encrypted'] else None
        }
    }

@router.put("/settings/binance")
async def update_binance_keys(data: ApiKeyUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    # Test: Binance API Ping
    try:
        from binance.client import Client as BinanceClient
        client = BinanceClient(data.api_key, data.api_secret)
        client.get_account()
        valid = True
    except Exception as e:
        return {"status": "error", "message": f"Binance API Test fehlgeschlagen: {str(e)}"}
    # Speichern
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""UPDATE users SET
                binance_api_key_encrypted = %s, binance_api_secret_encrypted = %s, binance_api_valid = %s
                WHERE user_id = %s""",
                (encrypt_value(data.api_key), encrypt_value(data.api_secret), valid, user_id))
            conn.commit()
    return {"status": "ok", "valid": valid}

@router.put("/settings/hyperliquid")
async def update_hyperliquid_keys(data: ApiKeyUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    # Test: Hyperliquid API — hole User State mit der Wallet-Adresse (api_key = wallet address)
    try:
        resp = requests.post("https://api.hyperliquid.xyz/info",
            json={"type": "clearinghouseState", "user": data.api_key},
            headers={"Content-Type": "application/json"}, timeout=10)
        if resp.status_code != 200:
            return {"status": "error", "message": f"Hyperliquid API Fehler: HTTP {resp.status_code}"}
        result = resp.json()
        # Wenn die Antwort ein Dict mit marginSummary ist, ist der Key gültig
        if not isinstance(result, dict) or 'marginSummary' not in result:
            return {"status": "error", "message": "Wallet-Adresse nicht erkannt oder kein Account"}
        valid = True
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Hyperliquid API nicht erreichbar: {str(e)}"}
    # Speichern
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""UPDATE users SET
                hyperliquid_api_key_encrypted = %s, hyperliquid_api_secret_encrypted = %s, hyperliquid_api_valid = %s
                WHERE user_id = %s""",
                (encrypt_value(data.api_key), encrypt_value(data.api_secret), valid, user_id))
            conn.commit()
    return {"status": "ok", "valid": valid}
