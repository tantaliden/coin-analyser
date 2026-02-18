#!/usr/bin/env python3
"""
CoinGecko Coin Info Updater
Holt name, categories und network (asset_platform_id) für alle Coins
Läuft als Service alle 12h oder einmalig mit --once

Liest settings.json aus /opt/coin/settings.json
Schreibt in analyser_app.coin_info
"""

import json
import time
import sys
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import requests
from pathlib import Path
from datetime import datetime

# Settings laden
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # /opt/coin
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

DB_APP = SETTINGS['databases']['app']
BINANCE_API = SETTINGS['binance']['liveApiUrl']
QUOTE_ASSET = 'USDC'

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
RATE_LIMIT_DELAY = 4  # CoinGecko free tier: 10-30 req/min

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

def get_db():
    return psycopg2.connect(
        host=DB_APP['host'],
        port=DB_APP['port'],
        dbname=DB_APP['name'],
        user=DB_APP['user'],
        password=DB_APP['password']
    )

# =============================================
# PHASE 1: Binance Exchange Info (Precision etc.)
# =============================================

def fetch_and_update_exchange_info():
    log(f"Fetching Binance exchange info for {QUOTE_ASSET} pairs...")
    
    r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    
    symbols = []
    for s in data.get('symbols', []):
        if s.get('quoteAsset') != QUOTE_ASSET:
            continue
        if s.get('status') != 'TRADING':
            continue
        
        price_precision = 8
        qty_precision = 8
        min_notional = 10.0
        min_qty = 0.00001
        
        for f in s.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick = f.get('tickSize', '0.00000001')
                price_precision = len(tick.rstrip('0').split('.')[-1]) if '.' in tick else 0
            elif f['filterType'] == 'LOT_SIZE':
                step = f.get('stepSize', '0.00000001')
                qty_precision = len(step.rstrip('0').split('.')[-1]) if '.' in step else 0
                min_qty = float(f.get('minQty', 0.00001))
            elif f['filterType'] == 'NOTIONAL':
                min_notional = float(f.get('minNotional', 10.0))
        
        symbols.append((
            s['symbol'], s['baseAsset'],
            price_precision, qty_precision, min_notional, min_qty,
            datetime.utcnow()
        ))
    
    log(f"Found {len(symbols)} {QUOTE_ASSET} trading pairs")
    
    conn = get_db()
    cur = conn.cursor()
    execute_values(cur, """
        INSERT INTO coin_info (symbol, base_asset, price_precision, qty_precision, min_notional, min_qty, updated_at)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            base_asset = EXCLUDED.base_asset,
            price_precision = EXCLUDED.price_precision,
            qty_precision = EXCLUDED.qty_precision,
            min_notional = EXCLUDED.min_notional,
            min_qty = EXCLUDED.min_qty,
            updated_at = NOW()
    """, symbols)
    conn.commit()
    cur.close()
    conn.close()
    
    log(f"✓ Updated {len(symbols)} coins (exchange info)")
    return len(symbols)

# =============================================
# PHASE 2: CoinGecko Enrichment (Name, Categories, Network)
# =============================================

def coingecko_search(symbol):
    try:
        url = f"{COINGECKO_BASE}/search?query={symbol}"
        r = requests.get(url, timeout=10)
        
        if r.status_code == 429:
            log(f"  RATE LIMIT - waiting 60s...")
            time.sleep(60)
            return coingecko_search(symbol)
        
        r.raise_for_status()
        coins = r.json().get('coins', [])
        
        for coin in coins:
            if coin.get('symbol', '').upper() == symbol.upper():
                return coin.get('id')
        return None
    except Exception as e:
        log(f"  Search error for {symbol}: {e}")
        return None

def coingecko_details(coin_id):
    try:
        url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=false&developer_data=false"
        r = requests.get(url, timeout=10)
        
        if r.status_code == 429:
            log(f"  RATE LIMIT - waiting 60s...")
            time.sleep(60)
            return coingecko_details(coin_id)
        
        if r.status_code == 404:
            return None
        
        r.raise_for_status()
        data = r.json()
        
        categories = data.get('categories', [])
        categories = [c for c in categories if c and not any(
            x in c.lower() for x in ['portfolio', 'index', 'alleged', 'ftx', 'sec']
        )]
        
        return {
            'name': data.get('name'),
            'categories': categories,
            'network': data.get('asset_platform_id')
        }
    except Exception as e:
        log(f"  Details error for {coin_id}: {e}")
        return None

def enrich_with_coingecko(force_all=False):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    if force_all:
        cur.execute("SELECT symbol, base_asset FROM coin_info ORDER BY symbol")
    else:
        cur.execute("SELECT symbol, base_asset FROM coin_info WHERE name IS NULL ORDER BY symbol")
    
    coins = cur.fetchall()
    total = len(coins)
    
    if total == 0:
        log("All coins already enriched. Use --all to force refresh.")
        conn.close()
        return
    
    log(f"Enriching {total} coins via CoinGecko (~{total * RATE_LIMIT_DELAY * 2 // 60} min estimated)")
    
    updated = 0
    failed = 0
    
    for i, coin in enumerate(coins):
        symbol = coin['symbol']
        base = coin['base_asset']
        
        log(f"[{i+1}/{total}] {symbol} ({base})...")
        
        time.sleep(RATE_LIMIT_DELAY)
        coin_id = coingecko_search(base)
        
        if not coin_id:
            log(f"  SKIP - no CoinGecko ID")
            failed += 1
            continue
        
        time.sleep(RATE_LIMIT_DELAY)
        details = coingecko_details(coin_id)
        
        if not details:
            log(f"  SKIP - no details")
            failed += 1
            continue
        
        cur.execute("""
            UPDATE coin_info 
            SET name = %s, categories = %s, network = %s, updated_at = NOW()
            WHERE symbol = %s
        """, (details['name'], details['categories'], details['network'], symbol))
        conn.commit()
        
        log(f"  ✓ {details['name']} | {details['network'] or 'native'} | {len(details['categories'])} categories")
        updated += 1
    
    cur.close()
    conn.close()
    log(f"DONE: {updated} enriched, {failed} failed")

# =============================================
# MAIN
# =============================================

def run_once(force_all=False):
    log("="*50)
    log("COIN INFO UPDATE START")
    log("="*50)
    
    fetch_and_update_exchange_info()
    enrich_with_coingecko(force_all)
    
    log("✓ Complete!")

def run_loop():
    log("Starting coin info updater loop (every 12h)")
    while True:
        try:
            run_once()
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        log("Next update in 12 hours...")
        time.sleep(12 * 60 * 60)

if __name__ == '__main__':
    if '--once' in sys.argv:
        force = '--all' in sys.argv
        run_once(force_all=force)
    else:
        run_loop()
