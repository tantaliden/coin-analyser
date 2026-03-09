"""RL-Agent API Routes — Config und Status über Bot-Modul."""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user
from rl_agent.trader import get_hl_balance, get_hl_open_positions

router = APIRouter(prefix="/api/v1/rl-agent", tags=["rl-agent"])


class RLConfigUpdate(BaseModel):
    is_active: Optional[bool] = None
    min_trade_size: Optional[float] = None
    max_capital_fraction: Optional[float] = None
    max_leverage: Optional[float] = None
    max_concurrent_positions: Optional[int] = None


@router.get("/config")
async def get_rl_config(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM rl_agent_config WHERE user_id = %s", (user_id,))
            config = cur.fetchone()
            if not config:
                cur.execute(
                    "INSERT INTO rl_agent_config (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (user_id,),
                )
                conn.commit()
                return {
                    "is_active": False,
                    "min_trade_size": 25.0,
                    "max_capital_fraction": 0.05,
                    "max_leverage": 5.0,
                    "max_concurrent_positions": 5,
                }
    return {
        "is_active": config["is_active"],
        "min_trade_size": float(config["min_trade_size"]),
        "max_capital_fraction": float(config["max_capital_fraction"]),
        "max_leverage": float(config["max_leverage"]),
        "max_concurrent_positions": config["max_concurrent_positions"],
    }


@router.put("/config")
async def update_rl_config(request: RLConfigUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    updates, values = [], []

    if request.is_active is not None:
        updates.append("is_active = %s")
        values.append(request.is_active)
    if request.min_trade_size is not None:
        if request.min_trade_size < 5:
            return {"error": "Mindestbetrag 5 USD"}
        updates.append("min_trade_size = %s")
        values.append(request.min_trade_size)
    if request.max_capital_fraction is not None:
        if not 0.01 <= request.max_capital_fraction <= 0.5:
            return {"error": "Kapitalanteil muss zwischen 1% und 50% liegen"}
        updates.append("max_capital_fraction = %s")
        values.append(request.max_capital_fraction)
    if request.max_leverage is not None:
        if not 1 <= request.max_leverage <= 50:
            return {"error": "Hebel muss zwischen 1x und 50x liegen"}
        updates.append("max_leverage = %s")
        values.append(request.max_leverage)
    if request.max_concurrent_positions is not None:
        updates.append("max_concurrent_positions = %s")
        values.append(request.max_concurrent_positions)

    if not updates:
        return {"error": "Keine Änderungen"}

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rl_agent_config (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )
            cur.execute(
                f"UPDATE rl_agent_config SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = %s",
                values + [user_id],
            )
            conn.commit()
    return {"status": "updated"}


@router.get("/status")
async def get_rl_status(current_user: dict = Depends(get_current_user)):
    """Aktueller Status: offene Trades, Performance, Agent-Stats."""
    user_id = current_user["user_id"]

    with get_app_db() as conn:
        with conn.cursor() as cur:
            # Config
            cur.execute("SELECT is_active FROM rl_agent_config WHERE user_id = %s", (user_id,))
            config = cur.fetchone()
            is_active = config["is_active"] if config else False

            # Offene Trades
            cur.execute("SELECT COUNT(*) as cnt FROM rl_trades WHERE status = 'open'")
            open_trades = cur.fetchone()["cnt"]

            # Gesamt-Performance
            cur.execute(
                """
                SELECT COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl_usd > 0) as winners,
                       COUNT(*) FILTER (WHERE pnl_usd <= 0) as losers,
                       COALESCE(SUM(pnl_usd), 0) as total_pnl,
                       COALESCE(AVG(pnl_usd), 0) as avg_pnl,
                       COALESCE(MAX(pnl_usd), 0) as best_trade,
                       COALESCE(MIN(pnl_usd), 0) as worst_trade
                FROM rl_trades WHERE status = 'closed'
                """
            )
            perf = cur.fetchone()

            # Observations
            cur.execute(
                """
                SELECT COUNT(*) as total,
                       COUNT(*) FILTER (WHERE agent_action = 'taken') as taken,
                       COUNT(*) FILTER (WHERE agent_action = 'skipped') as skipped
                FROM rl_observations
                """
            )
            obs = cur.fetchone()

    # Hyperliquid Balance
    try:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT hyperliquid_wallet_address FROM users WHERE user_id = %s", (user_id,))
                addr = cur.fetchone()
        hl_balance = get_hl_balance(addr["hyperliquid_wallet_address"]) if addr and addr["hyperliquid_wallet_address"] else 0
    except:
        hl_balance = 0

    return {
        "is_active": is_active,
        "hl_balance": round(hl_balance, 2),
        "open_trades": open_trades,
        "total_trades": perf["total"],
        "winners": perf["winners"],
        "losers": perf["losers"],
        "total_pnl": round(float(perf["total_pnl"]), 2),
        "avg_pnl": round(float(perf["avg_pnl"]), 2),
        "best_trade": round(float(perf["best_trade"]), 2),
        "worst_trade": round(float(perf["worst_trade"]), 2),
        "observations_total": obs["total"],
        "observations_taken": obs["taken"],
        "observations_skipped": obs["skipped"],
    }


@router.get("/trades")
async def get_rl_trades(
    status: str = "all",
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """RL-Agent Trades abfragen."""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            if status == "all":
                cur.execute(
                    """
                    SELECT id, symbol, direction, leverage, entry_price, entry_time,
                           exit_price, exit_time, exit_reason, position_size_usd,
                           pnl_usd, pnl_percent, fees_usd, duration_minutes, status
                    FROM rl_trades ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, symbol, direction, leverage, entry_price, entry_time,
                           exit_price, exit_time, exit_reason, position_size_usd,
                           pnl_usd, pnl_percent, fees_usd, duration_minutes, status
                    FROM rl_trades WHERE status = %s ORDER BY created_at DESC LIMIT %s
                    """,
                    (status, limit),
                )
            trades = cur.fetchall()

    return {
        "trades": [
            {
                "id": t["id"],
                "symbol": t["symbol"],
                "direction": t["direction"],
                "leverage": float(t["leverage"]),
                "entry_price": float(t["entry_price"]) if t["entry_price"] else None,
                "entry_time": t["entry_time"].isoformat() if t["entry_time"] else None,
                "exit_price": float(t["exit_price"]) if t["exit_price"] else None,
                "exit_time": t["exit_time"].isoformat() if t["exit_time"] else None,
                "exit_reason": t["exit_reason"],
                "size_usd": float(t["position_size_usd"]),
                "pnl_usd": float(t["pnl_usd"]) if t["pnl_usd"] else None,
                "pnl_percent": float(t["pnl_percent"]) if t["pnl_percent"] else None,
                "fees_usd": float(t["fees_usd"]) if t["fees_usd"] else None,
                "duration_min": t["duration_minutes"],
                "status": t["status"],
            }
            for t in trades
        ]
    }


@router.get("/monthly")
async def get_rl_monthly(current_user: dict = Depends(get_current_user)):
    """Monats-Archiv."""
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM rl_monthly_summary ORDER BY month DESC LIMIT 12")
            rows = cur.fetchall()

    return {
        "months": [
            {
                "month": r["month"],
                "total_trades": r["total_trades"],
                "winners": r["winners"],
                "losers": r["losers"],
                "total_pnl": float(r["total_pnl_usd"]),
                "avg_pnl": float(r["avg_trade_pnl"]),
                "best_trade": float(r["best_trade_pnl"]),
                "worst_trade": float(r["worst_trade_pnl"]),
                "avg_leverage": float(r["avg_leverage"]),
                "start_capital": float(r["start_capital"]) if r["start_capital"] else None,
                "end_capital": float(r["end_capital"]) if r["end_capital"] else None,
            }
            for r in rows
        ]
    }
