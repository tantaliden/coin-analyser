"""AUTH MODULE - JWT Token Handling, Password Hashing"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.fernet import Fernet
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

JWT_SECRET = SETTINGS['auth']['jwt_secret']
JWT_ALGORITHM = SETTINGS['auth'].get('jwt_algorithm', 'HS256')
JWT_EXPIRY_MINUTES = SETTINGS['auth'].get('access_token_expire_minutes', 1440)

# Encryption for API keys
ENCRYPTION_KEY = SETTINGS.get('encryption', {}).get('key', 'default-key-32-bytes-long!!')
FERNET_KEY = hashlib.sha256(ENCRYPTION_KEY.encode()).digest()
FERNET_KEY_B64 = __import__('base64').urlsafe_b64encode(FERNET_KEY)
fernet = Fernet(FERNET_KEY_B64)

security = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRY_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": str(user_id), "exp": expire, "type": "refresh"}, JWT_SECRET, algorithm=JWT_ALGORITHM)

def encrypt_value(value: str) -> str:
    return fernet.encrypt(value.encode()).decode()

def decrypt_value(encrypted: str) -> str:
    return fernet.decrypt(encrypted.encode()).decode()

def get_user_by_email(email: str) -> Optional[dict]:
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cur.fetchone()

def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            return cur.fetchone()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return dict(user)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    if not current_user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin required")
    return current_user

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SETTINGS['auth']['jwt_secret'], algorithms=[SETTINGS['auth']['jwt_algorithm']])
        return payload
    except JWTError:
        return None
