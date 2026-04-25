"""
AUTH - JWT Token Handling, Password Verification
Verwendet bestehende users Tabelle aus analyser_app
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

JWT_SECRET = SETTINGS['auth']['jwtSecret']
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = SETTINGS['auth']['jwtExpiryMinutes']
REFRESH_EXPIRY_DAYS = SETTINGS['auth']['refreshExpiryDays']

security = HTTPBearer()

def verify_password(password: str, hashed: str) -> bool:
    """Prüft Passwort gegen bcrypt Hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def hash_password(password: str) -> str:
    """Erstellt bcrypt Hash"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(user_id: int, email: str, role: str) -> str:
    """Erstellt JWT Access Token"""
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: int) -> str:
    """Erstellt JWT Refresh Token"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_EXPIRY_DAYS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    """Dekodiert JWT Token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

def get_user_by_email(email: str) -> Optional[dict]:
    """Holt User aus DB anhand Email"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, password_hash, role, is_active FROM users WHERE email = %s",
                (email,)
            )
            return cur.fetchone()

def get_user_by_id(user_id: int) -> Optional[dict]:
    """Holt User aus DB anhand ID"""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, role, is_active FROM users WHERE user_id = %s",
                (user_id,)
            )
            return cur.fetchone()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI Dependency für authentifizierten User"""
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user_id = int(payload.get("sub"))
    user = get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user['is_active']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is deactivated"
        )
    
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI Dependency für Admin User"""
    if current_user['role'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# === ENCRYPTION ===
from cryptography.fernet import Fernet
import base64
import hashlib

def get_encryption_key() -> bytes:
    """Erstellt Fernet Key aus encryptionKey in settings.json"""
    key = SETTINGS['auth']['encryptionKey'].encode()
    # Pad oder hash zu 32 bytes für Fernet
    key_hash = hashlib.sha256(key).digest()
    return base64.urlsafe_b64encode(key_hash)

def encrypt_value(value: str) -> str:
    """Verschlüsselt einen Wert"""
    if not value:
        return None
    f = Fernet(get_encryption_key())
    return f.encrypt(value.encode()).decode()

def decrypt_value(encrypted: str) -> str:
    """Entschlüsselt einen Wert"""
    if not encrypted:
        return None
    f = Fernet(get_encryption_key())
    return f.decrypt(encrypted.encode()).decode()
