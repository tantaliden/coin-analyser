"""
TRADE TRACKER SERVICE
Prueft regelmaessig offene Buys ohne zugehoerigen Sell.
Holt Sell-Trades von Binance und traegt sie in trade_history ein.
"""

import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

COIN_DIR = Path('/opt/coin')
ROOT_DIR = Path(__file__).resolve().parent.parent  # /opt/coin/backend

sys.path.insert(0, str(ROOT_DIR))

import psycopg2
import psycopg2.extras
from binance.client import Client as BinanceClient
from cryptography.fernet import Fernet
import base64, hashlib

with open(COIN_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

DB_APP = SETTINGS['databases']['app']

def get_db():
    conn = psycopg2.connect(
        host=DB_APP['host'], port=DB_APP['port'],
        dbname=DB_APP['name'], user=DB_APP['user'], password=DB_APP['password']
    )
    conn.autocommit = False
    return conn

def get_encryption_key():
    key = SETTINGS['auth']['encryptionKey'].encode()
    key_hash = hashlib.sha256(key).digest()
    return base64.urlsafe_b64encode(key_hash)

def decrypt_value(encrypted):
    if not encrypted:
        return None
    f = Fernet(get_encryption_key())
    return f.decrypt(encrypted.encode()).decode()

def get_binance_client(user_row):
    api_key = decrypt_value(user_row['binance_api_key_encrypted'])
    api_secret = decrypt_value(user_row['binance_api_secret_encrypted'])
    return BinanceClient(api_key, api_secret)

def check_open_trades():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute("""
            SELECT t.id, t.user_id, t.symbol, t.price as buy_price,
                   t.quantity as buy_qty, t.quote_amount as buy_quote,
                   t.order_id, t.prediction_id, t.executed_at
            FROM trade_history t
            WHERE t.side = 'buy'
              AND NOT EXISTS (
                  SELECT 1 FROM trade_history t2
                  WHERE t2.prediction_id = t.prediction_id
                    AND t2.side = 'sell'
              )
              AND t.executed_at >= NOW() - INTERVAL '7 days'
            ORDER BY t.executed_at ASC
        """)
        open_buys = cur.fetchall()

        if not open_buys:
            return 0

        print(f"[TRACKER] {len(open_buys)} offene Buys gefunden")

        user_clients = {}
        filled_count = 0

        for buy in open_buys:
            user_id = buy['user_id']
            symbol = buy['symbol']
            buy_time = buy['executed_at']

            if user_id not in user_clients:
                cur.execute("""
                    SELECT binance_api_key_encrypted, binance_api_secret_encrypted, binance_api_valid
                    FROM users WHERE user_id = %s AND binance_api_valid = true
                """, (user_id,))
                user = cur.fetchone()
                if not user:
                    continue
                try:
                    user_clients[user_id] = get_binance_client(user)
                except Exception as e:
                    print(f"[TRACKER] User {user_id}: Client Fehler: {e}")
                    continue

            client = user_clients[user_id]

            try:
                buy_ts = int(buy_time.timestamp() * 1000)
                trades = client.get_my_trades(symbol=symbol, startTime=buy_ts, limit=50)

                sell_trades = [t for t in trades if not t['isBuyer'] and int(t['time']) > buy_ts]

                if not sell_trades:
                    continue

                total_sell_qty = sum(float(t['qty']) for t in sell_trades)
                total_sell_quote = sum(float(t['quoteQty']) for t in sell_trades)
                avg_sell_price = total_sell_quote / total_sell_qty if total_sell_qty > 0 else 0

                buy_qty = float(buy['buy_qty'])
                if total_sell_qty < buy_qty * 0.5:
                    continue

                sell_order_id = str(sell_trades[0].get('orderId', ''))
                sell_time = datetime.fromtimestamp(int(sell_trades[-1]['time']) / 1000, tz=timezone.utc)

                buy_quote = float(buy['buy_quote'])
                pnl = total_sell_quote - buy_quote
                pnl_pct = (pnl / buy_quote * 100) if buy_quote > 0 else 0

                cur.execute("""
                    INSERT INTO trade_history (user_id, prediction_id, symbol, side, price, quantity, quote_amount, order_id, is_bot_trade, executed_at)
                    VALUES (%s, %s, %s, 'sell', %s, %s, %s, %s, TRUE, %s)
                """, (user_id, buy['prediction_id'], symbol, avg_sell_price, total_sell_qty, total_sell_quote, sell_order_id, sell_time))

                conn.commit()
                filled_count += 1

                sign = "+" if pnl >= 0 else ""
                print(f"[TRACKER] {symbol}: Sell | {sign}{pnl:.2f} USDC ({sign}{pnl_pct:.1f}%)")

            except Exception as e:
                print(f"[TRACKER] {symbol}: Fehler: {e}")
                conn.rollback()
                continue

        return filled_count

    except Exception as e:
        print(f"[TRACKER] Fehler: {e}")
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def run_loop():
    print(f"[TRACKER] Trade Tracker gestartet - prueft alle 30s")
    while True:
        try:
            filled = check_open_trades()
            if filled > 0:
                print(f"[TRACKER] {filled} Sells nachgetragen")
        except Exception as e:
            print(f"[TRACKER] Loop-Fehler: {e}")
            traceback.print_exc()
        time.sleep(30)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        filled = check_open_trades()
        print(f"[TRACKER] Ergebnis: {filled} Sells nachgetragen")
    else:
        run_loop()
