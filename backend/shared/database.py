"""DATABASE - Zentrale DB-Verbindungen"""
import json
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / 'settings.json') as f:
    SETTINGS = json.load(f)

_pools = {}

def _get_pool(db_name: str) -> ThreadedConnectionPool:
    if db_name not in _pools:
        cfg = SETTINGS['database'][db_name]
        _pools[db_name] = ThreadedConnectionPool(
            minconn=2, maxconn=10,
            host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
            user=cfg['user'], password=cfg['password']
        )
    return _pools[db_name]

@contextmanager
def get_coins_db():
    pool = _get_pool('coins')
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

@contextmanager
def get_app_db():
    pool = _get_pool('app')
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

def query_coins(sql, params=None):
    with get_coins_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

def query_app(sql, params=None):
    with get_app_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

def execute_app(sql, params=None):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount

def close_pools():
    for pool in _pools.values():
        pool.closeall()
    _pools.clear()
