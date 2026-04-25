"""
DATABASE - Connection Management
Lädt Konfiguration aus settings.json
"""

import json
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

COINS_DB = SETTINGS['databases']['coins']
APP_DB = SETTINGS['databases']['app']

@contextmanager
def get_coins_db():
    conn = psycopg2.connect(
        host=COINS_DB['host'],
        port=COINS_DB['port'],
        dbname=COINS_DB['name'],
        user=COINS_DB['user'],
        password=COINS_DB['password'],
        cursor_factory=RealDictCursor
    )
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_app_db():
    conn = psycopg2.connect(
        host=APP_DB['host'],
        port=APP_DB['port'],
        dbname=APP_DB['name'],
        user=APP_DB['user'],
        password=APP_DB['password'],
        cursor_factory=RealDictCursor
    )
    try:
        yield conn
    finally:
        conn.close()
