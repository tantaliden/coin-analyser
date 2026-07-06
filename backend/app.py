"""COIN-ANALYSER - FastAPI Main Application"""
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load settings
SETTINGS_PATH = Path(__file__).resolve().parent.parent / 'settings.json'
with open(SETTINGS_PATH) as f:
    SETTINGS = json.load(f)

app = FastAPI(
    title="Coin-Analyser API",
    version=SETTINGS.get('app', {}).get('version', '1.0.0'),
    description="Crypto Trading Analysis Platform"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.get('server', {}).get('cors', {}).get('origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from auth import router as auth_router
from meta import router as meta_router, predictor_settings_router
from search import router as search_router
from search.counter_search import router as counter_search_router
from search.anomaly_endpoint import router as anomaly_router
from search.batch_anomaly_endpoint import router as batch_anomaly_router
from indicators.save_anomaly_set import router as save_anomaly_router
from search.anomaly_itemsets_endpoint import router as anomaly_itemsets_router
from groups import router as groups_router
from coins import router as coins_router
from indicators import router as indicators_router, save_router as indicators_save_router, fuzzy_router as indicators_fuzzy_router, scan_router as indicators_scan_router, load_router as indicators_load_router, fields_router as indicators_fields_router
from user import router as user_router, session_router as user_session_router
from wallet import router as wallet_router
from bot import router as bot_router
from momentum import router as momentum_router
from rl_agent.routes import router as rl_agent_router
from predictor import router as predictor_router
from seq import router as seq_router
from reach import router as reach_router
from nnf import router as nnf_router
from claudewallet import router as claudewallet_router

# Include routers
app.include_router(auth_router)
app.include_router(meta_router)
app.include_router(predictor_settings_router)
app.include_router(search_router)
app.include_router(counter_search_router)
app.include_router(anomaly_router)
app.include_router(batch_anomaly_router)
app.include_router(save_anomaly_router)
app.include_router(anomaly_itemsets_router)
app.include_router(groups_router)
app.include_router(coins_router)
app.include_router(indicators_router)
app.include_router(indicators_save_router)
app.include_router(indicators_fuzzy_router)
app.include_router(indicators_scan_router)
app.include_router(indicators_load_router)
app.include_router(indicators_fields_router)
app.include_router(user_router)
app.include_router(user_session_router)
app.include_router(wallet_router)
app.include_router(bot_router)
app.include_router(momentum_router)
app.include_router(rl_agent_router)
app.include_router(predictor_router)
app.include_router(seq_router)
app.include_router(reach_router)
app.include_router(nnf_router)
app.include_router(claudewallet_router)

@app.on_event("startup")
async def _start_hl_ws():
    """Startet HL-Websocket-Subscriber als Background-Thread.
    Liefert mids/positions/orders push-basiert an Wallet- und Predictor-Routes."""
    try:
        from wallet.hl_ws_state import ensure_started
        from shared.database import get_app_db
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT hyperliquid_wallet_address FROM users WHERE user_id=1")
                row = cur.fetchone()
        addr = row.get('hyperliquid_wallet_address') if row else None
        if addr:
            print(f"[APP-STARTUP] HL-WS-Subscriber starting for {addr}", flush=True)
            ensure_started(addr)
            print(f"[APP-STARTUP] HL-WS-Subscriber started", flush=True)
        else:
            print(f"[APP-STARTUP] No HL address — WS skipped", flush=True)
    except Exception as e:
        import traceback
        print(f"[APP-STARTUP] HL-WS error: {e}", flush=True)
        traceback.print_exc()


@app.get("/")
async def root():
    return {"app": "coin-analyser", "version": SETTINGS.get('app', {}).get('version'), "status": "running"}

if __name__ == "__main__":
    port = SETTINGS.get('server', {}).get('port', 8002)
    uvicorn.run(app, host="0.0.0.0", port=port)
