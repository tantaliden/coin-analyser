"""
AUTH ROUTES - Login, Refresh, Password, Me
Verwendet bestehende users und user_sessions Tabellen
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException, status, Depends

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import (
    verify_password, hash_password,
    create_access_token, create_refresh_token, decode_token,
    get_user_by_email, get_current_user
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

def hash_token(token: str) -> str:
    """Hash refresh token für DB-Speicherung"""
    return hashlib.sha256(token.encode()).hexdigest()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/login")
async def login(request: LoginRequest):
    """Login mit Email und Passwort"""
    user = get_user_by_email(request.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not verify_password(request.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is deactivated"
        )
    
    access_token = create_access_token(user['user_id'], user['email'], user['role'])
    refresh_token = create_refresh_token(user['user_id'])
    
    # Session speichern (mit Hash)
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO user_sessions (user_id, refresh_token_hash, expires_at)
                   VALUES (%s, %s, NOW() + INTERVAL '7 days')""",
                (user['user_id'], hash_token(refresh_token))
            )
            conn.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "user_id": user['user_id'],
            "email": user['email'],
            "role": user['role']
        }
    }

@router.post("/refresh")
async def refresh_token_endpoint(refresh_token: str):
    """Refresh Access Token"""
    payload = decode_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user_id = int(payload.get("sub"))
    token_hash = hash_token(refresh_token)
    
    # Prüfe ob Session existiert
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT user_id FROM user_sessions 
                   WHERE user_id = %s AND refresh_token_hash = %s AND expires_at > NOW()""",
                (user_id, token_hash)
            )
            session = cur.fetchone()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )
    
    # User holen
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, role, is_active FROM users WHERE user_id = %s",
                (user_id,)
            )
            user = cur.fetchone()
    
    if not user or not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    new_access_token = create_access_token(user['user_id'], user['email'], user['role'])
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Aktueller User"""
    return {
        "user_id": current_user['user_id'],
        "email": current_user['email'],
        "role": current_user['role']
    }

@router.post("/password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Passwort ändern"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE user_id = %s",
                (current_user['user_id'],)
            )
            user = cur.fetchone()
    
    if not verify_password(request.current_password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    new_hash = hash_password(request.new_password)
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE user_id = %s",
                (new_hash, current_user['user_id'])
            )
            conn.commit()
    
    return {"message": "Password changed successfully"}

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout - Sessions löschen"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_sessions WHERE user_id = %s",
                (current_user['user_id'],)
            )
            conn.commit()
    
    return {"message": "Logged out successfully"}
