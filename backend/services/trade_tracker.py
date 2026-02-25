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

def resolve_stats(cur, user_id):
    """Stats aktualisieren nach Prediction-Resolution"""
    time_periods = {
        '24h': "detected_at >= NOW() - INTERVAL '24 hours'",
        '7d': "detected_at >= NOW() - INTERVAL '7 days'",
        '30d': "detected_at >= NOW() - INTERVAL '30 days'",
        'all': "TRUE"
    }
    combos = []
    for tp, tw in time_periods.items():
        combos.append((tp, tw, None))
        combos.append((f'long_{tp}', tw, 'long'))
        combos.append((f'short_{tp}', tw, 'short'))

    for period_key, time_where, direction in combos:
        dir_where = f" AND direction = '{direction}'" if direction else ""
        cur.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE was_correct = true) as correct,
                COUNT(*) FILTER (WHERE was_correct = false) as incorrect,
                COUNT(*) FILTER (WHERE status = 'expired') as expired,
                AVG(confidence) as avg_conf,
                AVG(actual_result_pct) as avg_result,
                MAX(actual_result_pct) as best,
                MIN(actual_result_pct) as worst,
                AVG(duration_minutes) as avg_dur
            FROM momentum_predictions
            WHERE user_id = %s AND status != 'active' AND {time_where}{dir_where}
        """, (user_id,))
        row = cur.fetchone()
        total = row['total'] or 0
        correct = row['correct'] or 0
        hit_rate = (correct / total * 100) if total > 0 else 0
        cur.execute("""
            INSERT INTO momentum_stats (user_id, period, total_predictions, correct_predictions,
                incorrect_predictions, expired_predictions, avg_confidence, avg_result_pct,
                best_result_pct, worst_result_pct, hit_rate_pct, avg_duration_minutes, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, period) DO UPDATE SET
                total_predictions = EXCLUDED.total_predictions,
                correct_predictions = EXCLUDED.correct_predictions,
                incorrect_predictions = EXCLUDED.incorrect_predictions,
                expired_predictions = EXCLUDED.expired_predictions,
                avg_confidence = EXCLUDED.avg_confidence,
                avg_result_pct = EXCLUDED.avg_result_pct,
                best_result_pct = EXCLUDED.best_result_pct,
                worst_result_pct = EXCLUDED.worst_result_pct,
                hit_rate_pct = EXCLUDED.hit_rate_pct,
                avg_duration_minutes = EXCLUDED.avg_duration_minutes,
                updated_at = NOW()
        """, (user_id, period_key, total, correct, row['incorrect'] or 0,
              row['expired'] or 0, row['avg_conf'], row['avg_result'],
              row['best'], row['worst'], round(hit_rate, 2),
              int(row['avg_dur']) if row['avg_dur'] else None))


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
                  WHERE t2.side = 'sell'
                    AND t2.symbol = t.symbol
                    AND t2.user_id = t.user_id
                    AND (
                      (t.prediction_id IS NOT NULL AND t2.prediction_id = t.prediction_id)
                      OR
                      (t.prediction_id IS NULL AND t2.prediction_id IS NULL
                       AND t2.executed_at >= t.executed_at
                       AND t2.executed_at <= t.executed_at + INTERVAL '24 hours')
                    )
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

                # Prediction resolven wenn vorhanden
                if buy['prediction_id']:
                    buy_price = float(buy['buy_price'])
                    if buy_price > 0:
                        result_pct = ((avg_sell_price - buy_price) / buy_price) * 100
                    else:
                        result_pct = 0
                    new_status = 'hit_tp' if pnl >= 0 else 'hit_sl'
                    duration = int((sell_time - buy_time).total_seconds() / 60)
                    cur.execute("""
                        UPDATE momentum_predictions
                        SET status = %s, was_correct = %s, actual_result_pct = %s,
                            duration_minutes = %s, resolved_at = %s
                        WHERE prediction_id = %s AND status = 'active'
                    """, (new_status, pnl >= 0, round(result_pct, 4), duration, sell_time, buy['prediction_id']))
                    print(f"[TRACKER] Prediction #{buy['prediction_id']} → {new_status} ({result_pct:+.2f}%)")
                    # Stats aktualisieren
                    resolve_stats(cur, user_id)

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
