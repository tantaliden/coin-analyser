"""RL-Agent API Routes V3 — Config, Status, Trades, Backtest."""
import json
import subprocess
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db
from auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/rl-agent", tags=["rl-agent"])

MODEL_PATH = Path("/opt/coin/database/data/models/rl_ppo_trading_v3.zip")
STATE_PATH = Path("/opt/coin/database/data/models/rl_agent_state.json")
RESULTS_PATH = Path("/opt/coin/logs/rl_backtest_v3_results.json")
TRADES_LOG_PATH = Path("/opt/coin/logs/rl_backtest_v3_trades.json")


class RLConfigUpdate(BaseModel):
    is_active: Optional[bool] = None
    max_leverage: Optional[int] = None
    max_concurrent_positions: Optional[int] = None
    base_trade_size: Optional[float] = None


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
                    "is_active": False, "max_leverage": 10,
                    "max_concurrent_positions": 50,
                    "base_trade_size": 15.0,
                }
    return {
        "is_active": config["is_active"],
        "max_leverage": config["max_leverage"],
        "max_concurrent_positions": config["max_concurrent_positions"],
        "base_trade_size": float(config["base_trade_size"]) if config.get("base_trade_size") else 15.0,
    }


@router.put("/config")
async def update_rl_config(request: RLConfigUpdate, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    updates, values = [], []

    if request.is_active is not None:
        updates.append("is_active = %s")
        values.append(request.is_active)
    if request.max_leverage is not None:
        if not 1 <= request.max_leverage <= 10:
            return {"error": "Hebel muss zwischen 1x und 10x liegen"}
        updates.append("max_leverage = %s")
        values.append(request.max_leverage)
    if request.max_concurrent_positions is not None:
        if not 1 <= request.max_concurrent_positions <= 50:
            return {"error": "Max Positionen zwischen 1 und 50"}
        updates.append("max_concurrent_positions = %s")
        values.append(request.max_concurrent_positions)
    if request.base_trade_size is not None:
        if request.base_trade_size < 15:
            return {"error": "Mindestens $15"}
        updates.append("base_trade_size = %s")
        values.append(request.base_trade_size)

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


@router.post("/service/{action}")
async def control_service(action: str, current_user: dict = Depends(get_current_user)):
    """Service starten/stoppen + DB-Flag setzen."""
    if action not in ('start', 'stop'):
        return {"error": "Nur 'start' oder 'stop'"}

    user_id = current_user["user_id"]

    if action == 'start':
        subprocess.run(['/usr/bin/systemctl', 'start', 'rl-agent'], capture_output=True, timeout=10)
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE rl_agent_config SET is_active = true, updated_at = NOW() WHERE user_id = %s", (user_id,))
                conn.commit()
    else:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE rl_agent_config SET is_active = false, updated_at = NOW() WHERE user_id = %s", (user_id,))
                conn.commit()
        subprocess.run(['/usr/bin/systemctl', 'stop', 'rl-agent'], capture_output=True, timeout=10)

    result = subprocess.run(['/usr/bin/systemctl', 'is-active', 'rl-agent'], capture_output=True, text=True, timeout=5)
    is_running = result.stdout.strip() == 'active'

    return {"status": "ok", "service_running": is_running, "is_active": action == 'start'}


@router.get("/status")
async def get_rl_status(current_user: dict = Depends(get_current_user)):
    """Agent-Status: Modell, Performance, Punkte, letzte Entscheidungen."""

    # Agent-State aus Datei (Punkte, Portfolio)
    agent_state = {}
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                agent_state = json.load(f)
        except:
            pass

    with get_app_db() as conn:
        with conn.cursor() as cur:
            # Config
            cur.execute("SELECT is_active FROM rl_agent_config WHERE user_id = %s",
                        (current_user["user_id"],))
            config = cur.fetchone()
            is_active = config["is_active"] if config else False

            # Offene Positionen
            cur.execute("SELECT COUNT(*) as cnt FROM rl_positions WHERE status = 'open'")
            open_count = cur.fetchone()["cnt"]

            # Performance (geschlossene Positionen)
            cur.execute("""
                SELECT COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl_usd > 0) as winners,
                       COUNT(*) FILTER (WHERE pnl_usd <= 0) as losers,
                       COALESCE(SUM(pnl_usd), 0) as total_pnl,
                       COALESCE(AVG(pnl_usd), 0) as avg_pnl,
                       COALESCE(MAX(pnl_usd), 0) as best_trade,
                       COALESCE(MIN(pnl_usd), 0) as worst_trade,
                       COALESCE(AVG(leverage), 1) as avg_leverage
                FROM rl_positions WHERE status = 'closed'
            """)
            perf = cur.fetchone()

            # Letzte 10 Entscheidungen
            cur.execute("""
                SELECT d.symbol, d.action, d.reward, d.in_position, d.unrealized_pnl,
                       d.timestamp,
                       p.direction, p.leverage, p.position_size_usd, p.status as pos_status,
                       p.pnl_usd, p.exit_reason
                FROM rl_decisions d
                LEFT JOIN rl_positions p ON d.position_id = p.id
                ORDER BY d.id DESC LIMIT 10
            """)
            recent = cur.fetchall()

    # Punkt-Bonus berechnen
    total_points = agent_state.get('total_points', 0)
    if total_points >= 5000:
        bonus = 2.0
    elif total_points >= 2000:
        bonus = 1.5
    elif total_points >= 500:
        bonus = 1.2
    else:
        bonus = 1.0

    # Service Status
    try:
        svc_result = subprocess.run(['/usr/bin/systemctl', 'is-active', 'rl-agent'],
                                     capture_output=True, text=True, timeout=5)
        service_running = svc_result.stdout.strip() == 'active'
    except:
        service_running = False

    return {
        "is_active": is_active,
        "service_running": service_running,
        "model_exists": MODEL_PATH.exists(),
        "model_type": "PPO-V3 Discrete(21)",
        "open_positions": open_count,
        "total_trades": perf["total"],
        "winners": perf["winners"],
        "losers": perf["losers"],
        "total_pnl": round(float(perf["total_pnl"]), 2),
        "avg_pnl": round(float(perf["avg_pnl"]), 2),
        "best_trade": round(float(perf["best_trade"]), 2),
        "worst_trade": round(float(perf["worst_trade"]), 2),
        "avg_leverage": round(float(perf["avg_leverage"]), 1),
        "total_points": round(total_points, 0),
        "point_bonus": bonus,
        "total_profit": round(agent_state.get('total_profit', 0), 2),
        "day_pnl": round(agent_state.get('day_pnl', 0), 2),
        "week_points": round(agent_state.get('week_points', 0), 0),
        "losing_streak_days": agent_state.get('losing_streak_days', 0),
        "winning_streak_weeks": agent_state.get('winning_streak_weeks', 0),
        "week_bonus_multiplier": round(1.20 + max(agent_state.get('winning_streak_weeks', 0) - 1, 0) * 0.05, 2) if agent_state.get('winning_streak_weeks', 0) > 0 else 0,
        "recent_decisions": [
            {
                "symbol": r["symbol"],
                "action": r["action"],
                "direction": r["direction"],
                "leverage": int(r["leverage"]) if r["leverage"] else None,
                "reward": round(float(r["reward"]), 2) if r["reward"] else None,
                "unrealized_pnl": round(float(r["unrealized_pnl"]), 2) if r["unrealized_pnl"] else None,
                "in_position": r["in_position"],
                "pos_status": r["pos_status"],
                "exit_reason": r["exit_reason"],
                "timestamp": r["timestamp"].isoformat() if r["timestamp"] else None,
            }
            for r in recent
        ],
    }


@router.get("/trades")
async def get_rl_trades(status: str = "all", limit: int = 50,
                        current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            if status == "all":
                cur.execute("""
                    SELECT id, symbol, direction, leverage, entry_price, entry_time,
                           exit_price, exit_time, exit_reason, position_size_usd,
                           pnl_usd, pnl_percent, duration_minutes, status, exchange
                    FROM rl_positions ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            else:
                cur.execute("""
                    SELECT id, symbol, direction, leverage, entry_price, entry_time,
                           exit_price, exit_time, exit_reason, position_size_usd,
                           pnl_usd, pnl_percent, duration_minutes, status, exchange
                    FROM rl_positions WHERE status = %s ORDER BY created_at DESC LIMIT %s
                """, (status, limit))
            trades = cur.fetchall()

    return {
        "trades": [
            {
                "id": t["id"], "symbol": t["symbol"], "direction": t["direction"],
                "leverage": int(t["leverage"]),
                "entry_price": float(t["entry_price"]) if t["entry_price"] else None,
                "entry_time": t["entry_time"].isoformat() if t["entry_time"] else None,
                "exit_price": float(t["exit_price"]) if t["exit_price"] else None,
                "exit_time": t["exit_time"].isoformat() if t["exit_time"] else None,
                "exit_reason": t["exit_reason"],
                "size_usd": float(t["position_size_usd"]) if t["position_size_usd"] else None,
                "pnl_usd": float(t["pnl_usd"]) if t["pnl_usd"] else None,
                "pnl_percent": float(t["pnl_percent"]) if t["pnl_percent"] else None,
                "duration_min": t["duration_minutes"],
                "status": t["status"],
                "exchange": t["exchange"],
            }
            for t in trades
        ]
    }


@router.get("/backtest")
async def get_backtest_results(current_user: dict = Depends(get_current_user)):
    """V3 Backtest-Ergebnisse."""
    if not RESULTS_PATH.exists():
        return {"error": "Kein Backtest durchgeführt"}

    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)

        result = {
            "test_period": data.get('test_period'),
            "start_balance": data.get('start_balance'),
            "final_portfolio": data.get('final_portfolio'),
            "peak": data.get('peak'),
            "low": data.get('low'),
            "total_predictions": data.get('total_predictions'),
            "baseline_hr": data.get('baseline_hr'),
            "trades_taken": data.get('trades_taken'),
            "trades_completed": data.get('trades_completed'),
            "skips": data.get('skips'),
        }

        # Trade-Log laden (detaillierte Trades)
        if TRADES_LOG_PATH.exists():
            with open(TRADES_LOG_PATH) as f:
                trade_log = json.load(f)

            wins = [t for t in trade_log if t['pnl_dollar'] > 0]
            losses = [t for t in trade_log if t['pnl_dollar'] < 0]

            result["performance"] = {
                "total_trades": len(trade_log),
                "winners": len(wins),
                "losers": len(losses),
                "win_rate": round(len(wins) / len(trade_log) * 100, 1) if trade_log else 0,
                "total_pnl": round(sum(t['pnl_dollar'] for t in trade_log), 2),
            }

        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/monthly")
async def get_rl_monthly(current_user: dict = Depends(get_current_user)):
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
