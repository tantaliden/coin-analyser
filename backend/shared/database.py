"""
DATABASE - Zentrale DB-Verbindungen
Lädt Config aus settings.json (4-File-Prinzip)
"""

import json
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

# Settings laden
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / 'settings.json') as f:
    SETTINGS = json.load(f)

# Connection Pools
_pools = {}


def _get_pool(db_name: str) -> ThreadedConnectionPool:
    """Holt oder erstellt Connection Pool für eine DB."""
    if db_name not in _pools:
        cfg = SETTINGS['databases'][db_name]
        _pools[db_name] = ThreadedConnectionPool(
            minconn=cfg['pool']['min'],
            maxconn=cfg['pool']['max'],
            host=cfg['host'],
            port=cfg['port'],
            dbname=cfg['name'],
            user=cfg['user'],
            password=cfg['password']
        )
    return _pools[db_name]


@contextmanager
def get_coins_db():
    """Connection zur coins DB (klines, aggregates)."""
    pool = _get_pool('coins')
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def get_app_db():
    """Connection zur analyser_app DB (users, indicators)."""
    pool = _get_pool('app')
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def query_coins(sql: str, params: tuple = None) -> list:
    """Query auf coins DB, gibt Liste von Dicts zurück."""
    with get_coins_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def query_app(sql: str, params: tuple = None) -> list:
    """Query auf app DB, gibt Liste von Dicts zurück."""
    with get_app_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def execute_app(sql: str, params: tuple = None) -> int:
    """Execute auf app DB, gibt affected rows zurück."""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def close_pools():
    """Alle Pools schließen (für Shutdown)."""
    for pool in _pools.values():
        pool.closeall()
    _pools.clear()
