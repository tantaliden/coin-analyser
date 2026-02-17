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
from meta import router as meta_router
from search import router as search_router
from groups import router as groups_router
from coins import router as coins_router
from indicators import router as indicators_router
from user import router as user_router
from wallet import router as wallet_router
from bot import router as bot_router
from momentum import router as momentum_router

# Include routers
app.include_router(auth_router)
app.include_router(meta_router)
app.include_router(search_router)
app.include_router(groups_router)
app.include_router(coins_router)
app.include_router(indicators_router)
app.include_router(user_router)
app.include_router(wallet_router)
app.include_router(bot_router)
app.include_router(momentum_router)

@app.get("/")
async def root():
    return {"app": "coin-analyser", "version": SETTINGS.get('app', {}).get('version'), "status": "running"}

if __name__ == "__main__":
    port = SETTINGS.get('server', {}).get('port', 8002)
    uvicorn.run(app, host="0.0.0.0", port=port)
