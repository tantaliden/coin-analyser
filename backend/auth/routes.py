"""
AUTH ROUTES - Login, Refresh, Password, API-Key
"""

import json
import hmac
import hashlib
import time
import base64
from pathlib import Path
from datetime import datetime
from cryptography.fernet import Fernet
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException, Depends, status
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_user_by_email, get_current_user
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# === MODELS ===

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class ApiKeyRequest(BaseModel):
    api_key: str
    api_secret: str

# === ENCRYPTION ===

def get_fernet():
    key = SETTINGS.get('encryptionKey', 'default_key_change_me!!!!!')
    key_bytes = hashlib.sha256(key.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)

def encrypt_value(value: str) -> str:
    return get_fernet().encrypt(value.encode()).decode()

def decrypt_value(encrypted: str) -> str:
    return get_fernet().decrypt(encrypted.encode()).decode()

# === ROUTES ===

@router.post("/login")
async def login(request: LoginRequest):
    user = get_user_by_email(request.email)
    
    if not user or not verify_password(request.password, user['password_hash']):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
    if not user['is_active']:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is deactivated")
    
    access_token = create_access_token(user['user_id'], user['email'], user['role'])
    refresh_token = create_refresh_token(user['user_id'])
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {"user_id": user['user_id'], "email": user['email'], "role": user['role']}
    }

@router.post("/refresh")
async def refresh(request: RefreshRequest):
    payload = decode_token(request.refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    
    user_id = int(payload.get("sub"))
    
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, email, role, is_active FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
    
    if not user or not user['is_active']:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated")
    
    access_token = create_access_token(user['user_id'], user['email'], user['role'])
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/password")
async def change_password(request: PasswordChangeRequest, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE user_id = %s", (current_user['user_id'],))
            user = cur.fetchone()
            
            if not verify_password(request.current_password, user['password_hash']):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
            
            new_hash = hash_password(request.new_password)
            cur.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (new_hash, current_user['user_id']))
            conn.commit()
    
    return {"message": "Password changed"}

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@router.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, email, role, is_active, binance_api_valid, created_at, updated_at
                FROM users WHERE user_id = %s
            """, (current_user['user_id'],))
            user = cur.fetchone()
    
    return {
        "user_id": user['user_id'],
        "email": user['email'],
        "role": user['role'],
        "is_active": user['is_active'],
        "api_key_configured": user['binance_api_valid'],
        "created_at": user['created_at'].isoformat() if user['created_at'] else None,
        "updated_at": user['updated_at'].isoformat() if user['updated_at'] else None
    }

@router.get("/api-key/status")
async def get_api_key_status(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT binance_api_key_encrypted, binance_api_valid, updated_at
                FROM users WHERE user_id = %s
            """, (current_user['user_id'],))
            user = cur.fetchone()
    
    if not user or not user['binance_api_key_encrypted']:
        return {"configured": False, "valid": False, "last_validated": None, "key_suffix": None}
    
    try:
        decrypted = decrypt_value(user['binance_api_key_encrypted'])
        key_suffix = decrypted[-4:] if len(decrypted) >= 4 else "****"
    except:
        key_suffix = "****"
    
    return {
        "configured": True,
        "valid": user['binance_api_valid'],
        "last_validated": user['updated_at'].isoformat() if user['updated_at'] else None,
        "key_suffix": key_suffix
    }

@router.post("/api-key")
async def save_api_key(request: ApiKeyRequest, current_user: dict = Depends(get_current_user)):
    try:
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(request.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        
        headers = {'X-MBX-APIKEY': request.api_key}
        url = f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            error_msg = response.json().get('msg', 'Unknown error')
            return {"valid": False, "message": f"Binance API error: {error_msg}"}
        
    except requests.exceptions.Timeout:
        return {"valid": False, "message": "Connection timeout"}
    except Exception as e:
        return {"valid": False, "message": f"Connection error: {str(e)}"}
    
    encrypted_key = encrypt_value(request.api_key)
    encrypted_secret = encrypt_value(request.api_secret)
    
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET binance_api_key_encrypted = %s, binance_api_secret_encrypted = %s,
                       binance_api_valid = TRUE, updated_at = NOW()
                WHERE user_id = %s
            """, (encrypted_key, encrypted_secret, current_user['user_id']))
            conn.commit()
    
    return {"valid": True, "message": "API key saved successfully"}

@router.delete("/api-key")
async def delete_api_key(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET binance_api_key_encrypted = NULL, binance_api_secret_encrypted = NULL,
                       binance_api_valid = FALSE, updated_at = NOW()
                WHERE user_id = %s
            """, (current_user['user_id'],))
            conn.commit()
    
    return {"message": "API key deleted"}
