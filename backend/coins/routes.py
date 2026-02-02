"""COINS ROUTES - Coin Info, Networks, Categories"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/coins", tags=["coins"])

@router.get("")
async def get_coins(network: Optional[str] = None, category: Optional[str] = None, search: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            query = "SELECT symbol, base_asset, name, network, categories, price_precision, qty_precision, min_notional, min_qty FROM coin_info WHERE 1=1"
            params = []
            if network:
                query += " AND network = %s"
                params.append(network)
            if category:
                query += " AND %s = ANY(categories)"
                params.append(category)
            if search:
                query += " AND (symbol ILIKE %s OR base_asset ILIKE %s OR name ILIKE %s)"
                params.extend([f"%{search}%"] * 3)
            query += " ORDER BY symbol"
            cur.execute(query, params)
            coins = cur.fetchall()
    return {"coins": [dict(c) for c in coins]}

@router.get("/networks")
async def get_networks(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT network FROM coin_info WHERE network IS NOT NULL ORDER BY network")
            networks = [row['network'] for row in cur.fetchall()]
    return {"networks": networks}

@router.get("/categories")
async def get_categories(current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT category, COUNT(*) as coin_count FROM (SELECT unnest(categories) as category FROM coin_info WHERE categories IS NOT NULL) sub
                GROUP BY category HAVING COUNT(*) >= 4 ORDER BY category
            """)
            categories = [row['category'] for row in cur.fetchall()]
    return {"categories": categories}

@router.put("/{symbol}/network")
async def update_coin_network(symbol: str, network: str, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE coin_info SET network = %s, updated_at = NOW() WHERE symbol = %s RETURNING symbol", (network, symbol))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Coin not found")
            conn.commit()
    return {"message": f"Network updated to {network}"}
