#!/usr/bin/env python3
"""
Signal-Service — Supervised ML Entry + regelbasierte Exits.

Entry: PyTorch Signal-Modell (skip/long/short) mit Confidence-Threshold
Exit: Trailing Stop, SL, Timeout — feste Regeln, kein Agent

Scan: :01/:06/:11/:16/:21/:26/:31/:36/:41/:46/:51/:56 (sync mit Agg-Refresh)
Management: Alle 5 Min

Usage:
    systemctl start signal-service
"""
import json
import time
import signal
import sys
import logging
import random
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_agent.features import compute_observation_live, N_FEATURES
from rl_agent.trader import (
    get_hl_credentials,
    get_hl_balance,
    get_hl_open_positions,
    get_available_coins_hl,
    get_current_prices_hl,
    place_limit_order_hl,
    close_position_hl,
    cancel_all_orders_for_coin_hl,
    refresh_hl_coin_info,
)

SETTINGS_PATH = "/opt/coin/settings.json"
MODEL_PATH = "/opt/coin/database/data/models/signal_model_best.pth"
STATE_PATH = "/opt/coin/database/data/models/signal_service_state.json"

# Timing
POLL_INTERVAL = 30
SCAN_MINUTES = {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56}
MGMT_INTERVAL = 300  # 5 Min — identisch zur Truthbox

# === DEFAULTS (werden von DB-Config überschrieben) ===
CONFIDENCE_THRESHOLD = 0.60
MAX_CONCURRENT = 10
BASE_TRADE_SIZE = 20.0
LEVERAGE = 5
TRAILING_STOP_PCT = 2.0
SL_PCT = 3.0
MIN_HOLD_MINUTES = 15
TIMEOUT_MINUTES = 240

FEE_RATE = 0.00035
MAX_TRADES_PER_DAY = 100
MIN_24H_VOLUME = 20_000
TP_PCT = 0  # 0 = aus
TS_TIGHTEN_ENABLED = False
TS_TIGHTEN_THRESHOLD = 5.0
COOLDOWN_HOURS = 2  # Sperre pro Coin nach Trade

# Cooldown-Tracker: {symbol: datetime} — wann zuletzt gehandelt
coin_cooldowns = {}


def load_config_from_db():
    """Lädt konfigurierbare Parameter aus rl_agent_config."""
    global CONFIDENCE_THRESHOLD, MAX_CONCURRENT, BASE_TRADE_SIZE, LEVERAGE
    global TRAILING_STOP_PCT, SL_PCT, TP_PCT, TIMEOUT_MINUTES, TS_TIGHTEN_ENABLED, TS_TIGHTEN_THRESHOLD
    try:
        conn = get_conn('app')
        with conn.cursor() as cur:
            cur.execute("""
                SELECT max_leverage, max_concurrent_positions, base_trade_size,
                       confidence_threshold, trailing_stop_pct, sl_pct, tp_pct,
                       timeout_minutes, ts_tighten_enabled, ts_tighten_threshold
                FROM rl_agent_config WHERE user_id = 1
            """)
            row = cur.fetchone()
        conn.close()
        if row:
            LEVERAGE = int(row['max_leverage'] or 5)
            MAX_CONCURRENT = int(row['max_concurrent_positions'] or 10)
            BASE_TRADE_SIZE = float(row['base_trade_size'] or 20.0)
            CONFIDENCE_THRESHOLD = float(row['confidence_threshold'] or 0.60)
            TRAILING_STOP_PCT = float(row['trailing_stop_pct'] or 2.0)
            SL_PCT = float(row['sl_pct'] or 3.0)
            TP_PCT = float(row.get('tp_pct') or 0)
            TIMEOUT_MINUTES = int(row.get('timeout_minutes') or 240)
            TS_TIGHTEN_ENABLED = bool(row.get('ts_tighten_enabled', False))
            TS_TIGHTEN_THRESHOLD = float(row.get('ts_tighten_threshold') or 5.0)
            print(f"[SIGNAL-SERVICE] Config geladen: Hebel={LEVERAGE}x, Max={MAX_CONCURRENT}, "
                  f"Size=${BASE_TRADE_SIZE}, Conf={CONFIDENCE_THRESHOLD}, "
                  f"TS={TRAILING_STOP_PCT}% (nach Hebel, nur positiv), SL={SL_PCT}%"
                  f"{f', Verschärfung ab {TS_TIGHTEN_THRESHOLD}%' if TS_TIGHTEN_ENABLED else ''}")
    except Exception as e:
        print(f"[SIGNAL-SERVICE] Config-Laden fehlgeschlagen (nutze Defaults): {e}")

STABLECOINS = {'USDCUSDC', 'USDTUSDC', 'BUSDUSDC', 'DAIUSDC', 'TUSDUSDC', 'FDUSDUSDC'}

HIDDEN_SIZES = [512, 256, 128]

# Close-Logger
close_logger = logging.getLogger('signal_closes')
close_logger.setLevel(logging.DEBUG)
_cl_handler = logging.FileHandler('/opt/coin/logs/signal_closes.log')
_cl_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
close_logger.addHandler(_cl_handler)
close_logger.propagate = False

running = True


def signal_handler(sig, frame):
    global running
    print(f"[SIGNAL-SERVICE] Signal {sig} empfangen, stoppe...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class SignalModel(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_sizes=None, n_classes=3, dropout=0.3):
        super().__init__()
        hidden_sizes = hidden_sizes or HIDDEN_SIZES
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_size = h
        layers.append(nn.Linear(prev_size, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def get_conn(db_key):
    s = json.load(open(SETTINGS_PATH))
    db = s['databases'][db_key]
    return psycopg2.connect(
        dbname=db['name'], user=db['user'], password=db['password'],
        host=db['host'], port=db['port'],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def load_state():
    if Path(STATE_PATH).exists():
        return json.load(open(STATE_PATH))
    return {
        'total_trades': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_profit': 0.0,
        'day_trades': 0,
        'current_day': '',
        'signals_correct': 0,
        'signals_total': 0,
    }


def save_state(state):
    json.dump(state, open(STATE_PATH, 'w'), indent=2)


def log_position_open(app_conn, symbol, direction, leverage, entry_price, position_size):
    with app_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rl_positions (symbol, direction, entry_price, entry_time, leverage,
                                      position_size_usd, status, mode, exchange)
            VALUES (%s, %s, %s, NOW(), %s, %s, 'open', 'live', 'hyperliquid')
            RETURNING id
        """, (symbol, direction, entry_price, leverage, position_size))
        pos_id = cur.fetchone()['id']
        app_conn.commit()
    return pos_id


def log_position_close(app_conn, pos_id, exit_price, exit_reason, pnl_pct, net_pnl, duration_min):
    with app_conn.cursor() as cur:
        cur.execute("""
            UPDATE rl_positions
            SET exit_price = %s, exit_time = NOW(), exit_reason = %s,
                pnl_percent = %s, pnl_usd = %s, duration_minutes = %s, status = 'closed'
            WHERE id = %s
        """, (exit_price, exit_reason, pnl_pct, net_pnl, duration_min, pos_id))
        app_conn.commit()


def get_open_positions_from_db(app_conn):
    with app_conn.cursor() as cur:
        cur.execute("""
            SELECT id, symbol, direction, leverage, entry_price, entry_time, position_size_usd
            FROM rl_positions WHERE status = 'open'
            ORDER BY entry_time ASC
        """)
        return cur.fetchall()


def calc_pnl(direction, entry_price, current_price):
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    if direction == 'long':
        return (current_price - entry_price) / entry_price * 100
    return (entry_price - current_price) / entry_price * 100


# ============================================================
# Position Management — Trailing Stop, SL, Timeout
# ============================================================

def manage_positions(app_conn, creds, wallet_address, state, peak_tracker):
    """Prüft alle offenen Positionen gegen die Regeln."""
    positions = get_open_positions_from_db(app_conn)
    hl_positions = get_hl_open_positions(wallet_address)
    hl_by_coin = {}
    for hp in hl_positions:
        coin = hp['coin']
        hl_by_coin[coin] = hp

    now = datetime.now(timezone.utc)
    closed_count = 0

    for pos in positions:
        symbol = pos['symbol']
        coin = symbol.replace('USDC', '')
        pos_id = pos['id']
        direction = pos['direction']
        entry_price = float(pos['entry_price'])
        entry_time = pos['entry_time']
        position_size = float(pos['position_size_usd'] or BASE_TRADE_SIZE)

        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        # Aktueller Preis von HL
        hl_pos = hl_by_coin.get(coin)
        if hl_pos:
            current_price = float(hl_pos['entry_price'])
            # Verwende unrealized PnL für genaueren Preis
            unrealized = float(hl_pos.get('unrealized_pnl', 0))
            size_val = abs(float(hl_pos.get('size', 0)))
            if size_val > 0 and entry_price > 0:
                if direction == 'long':
                    current_price = entry_price * (1 + unrealized / (size_val * entry_price))
                else:
                    current_price = entry_price * (1 - unrealized / (size_val * entry_price))
        else:
            # Position nicht auf HL — vermutlich schon geschlossen
            log_position_close(app_conn, pos_id, entry_price, 'not_on_exchange', 0, 0, 0)
            print(f"[SIGNAL-SERVICE] Position #{pos_id} {symbol} nicht auf HL gefunden — closed")
            closed_count += 1
            continue

        # pnl_pct ist schon nach Hebel (Rückrechnung aus HL unrealized_pnl)
        pnl_pct = calc_pnl(direction, entry_price, current_price)
        duration_min = int((now - entry_time).total_seconds() / 60)

        # Peak-Tracking
        peak_key = str(pos_id)
        if peak_key not in peak_tracker:
            peak_tracker[peak_key] = 0.0
        if pnl_pct > peak_tracker[peak_key]:
            peak_tracker[peak_key] = pnl_pct
        peak_pnl = peak_tracker[peak_key]

        exit_reason = None

        # 1. Hard SL (0 = aus)
        if SL_PCT > 0 and pnl_pct <= -SL_PCT:
            exit_reason = 'sl'

        # 2. Take-Profit
        elif TP_PCT > 0 and pnl_pct >= TP_PCT:
            exit_reason = 'tp'

        # 3. Timeout (0 = aus)
        elif TIMEOUT_MINUTES > 0 and duration_min >= TIMEOUT_MINUTES:
            exit_reason = 'timeout'

        # 4. Trailing Stop (nur wenn PnL positiv)
        elif duration_min >= MIN_HOLD_MINUTES and pnl_pct > 0 and peak_pnl > 0:
            ts = TRAILING_STOP_PCT
            # Optionale Verschärfung
            if TS_TIGHTEN_ENABLED and peak_pnl >= TS_TIGHTEN_THRESHOLD:
                ts = ts / 2
            drawdown_from_peak = peak_pnl - pnl_pct
            if drawdown_from_peak >= ts:
                exit_reason = 'trailing_stop'

        if exit_reason:
            # Position schließen
            try:
                result = close_position_hl(creds, coin, wallet_address)
                success = result.get('success', False)
                exit_price = result.get('avg_price', current_price)
            except Exception as e:
                print(f"[SIGNAL-SERVICE] Close fehlgeschlagen {symbol}: {e}")
                continue

            if not success:
                print(f"[SIGNAL-SERVICE] Close nicht bestätigt {symbol}")
                continue

            final_pnl = calc_pnl(direction, entry_price, exit_price)
            # position_size ist Notional ($20), nicht Margin — Hebel ist schon drin
            fees = position_size * FEE_RATE * 2
            net_pnl = position_size * final_pnl / 100 - fees

            log_position_close(app_conn, pos_id, exit_price, exit_reason, final_pnl, net_pnl, duration_min)

            # Cooldown setzen
            coin_cooldowns[symbol] = datetime.now(timezone.utc)

            state['total_trades'] += 1
            if net_pnl > 0:
                state['total_wins'] += 1
            else:
                state['total_losses'] += 1
            state['total_profit'] += net_pnl

            # Cleanup peak tracker
            if peak_key in peak_tracker:
                del peak_tracker[peak_key]

            close_logger.info(
                f"{exit_reason.upper():>13} {symbol} {direction} | "
                f"PnL: {final_pnl:+.2f}% ${net_pnl:+.2f} | "
                f"Peak: {peak_pnl:.2f}% | {duration_min}min"
            )

            print(f"[SIGNAL-SERVICE] {exit_reason.upper()} {symbol} {direction} {LEVERAGE}x | "
                  f"PnL: {final_pnl:+.2f}% ${net_pnl:+.2f} | Peak: {peak_pnl:.2f}% | {duration_min}min")

            closed_count += 1

    return closed_count


# ============================================================
# Entry Scan — Signal-Modell
# ============================================================

def scan_entries(model, app_conn, creds, wallet_address, state, coins_conn):
    """Scannt alle Coins, fragt Signal-Modell, öffnet Positionen."""
    positions = get_open_positions_from_db(app_conn)
    n_open = len(positions)

    if n_open >= MAX_CONCURRENT:
        return 0

    # Tages-Limit
    today = datetime.now().strftime('%Y-%m-%d')
    if state['current_day'] != today:
        state['current_day'] = today
        state['day_trades'] = 0
    if state['day_trades'] >= MAX_TRADES_PER_DAY:
        return 0

    open_symbols = {p['symbol'] for p in positions}

    # Blacklist aus DB
    try:
        with app_conn.cursor() as cur:
            cur.execute("SELECT symbol FROM coin_info WHERE blacklisted = true")
            blacklisted = {r['symbol'] if hasattr(r, 'get') else r[0] for r in cur.fetchall()}
        app_conn.commit()
    except Exception:
        try: app_conn.rollback()
        except Exception: pass
        blacklisted = set()

    # Alle HL-Coins holen
    available_coins = get_available_coins_hl()
    tradeable = [c + 'USDC' for c in available_coins if c + 'USDC' not in STABLECOINS and c + 'USDC' not in blacklisted]
    random.shuffle(tradeable)

    # Aktuelle Mid-Preise
    try:
        prices = get_current_prices_hl()  # {coin: float(mid_price)}
    except:
        prices = {}

    trades_opened = 0
    scanned = 0
    signals_found = 0

    for symbol in tradeable:
        if n_open + trades_opened >= MAX_CONCURRENT:
            break
        if symbol in open_symbols:
            continue

        # Cooldown prüfen
        last_trade = coin_cooldowns.get(symbol)
        if last_trade and (datetime.now(timezone.utc) - last_trade).total_seconds() < COOLDOWN_HOURS * 3600:
            continue

        coin = symbol.replace('USDC', '')

        mid_price = prices.get(coin, 0)
        if mid_price <= 0:
            continue

        scanned += 1

        # Features via Live-Codepfad (DB-Queries)
        try:
            base_obs = compute_observation_live(
                coins_conn, symbol, datetime.now(),
                position_state=None,
                n_open_positions=n_open + trades_opened,
            )
            features = np.nan_to_num(base_obs[:N_FEATURES], nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            continue

        # Modell-Prediction
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            output = model(x)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        if pred_class == 0:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        signals_found += 1
        direction = 'long' if pred_class == 1 else 'short'
        is_buy = direction == 'long'
        print(f"[SIGNAL-SERVICE] Signal: {symbol} {direction} | Conf: {confidence:.3f} | Preis: {mid_price:.6f}")

        # Trade ausführen
        try:
            result = place_limit_order_hl(
                creds, coin, is_buy, BASE_TRADE_SIZE,
                mid_price, LEVERAGE,
            )
            if not result.get('success'):
                print(f"[SIGNAL-SERVICE] Order nicht erfolgreich {symbol}: {result}")
                continue

            entry_price = result.get('avg_price', mid_price)
            pos_id = log_position_open(app_conn, symbol, direction, LEVERAGE,
                                        entry_price, BASE_TRADE_SIZE)

            trades_opened += 1
            state['day_trades'] += 1

            print(f"[SIGNAL-SERVICE] OPEN {symbol} {direction} {LEVERAGE}x | "
                  f"${BASE_TRADE_SIZE} @{entry_price:.4f} | "
                  f"Conf: {confidence:.2f} | Offen: {n_open + trades_opened}")

        except Exception as e:
            print(f"[SIGNAL-SERVICE] Entry fehlgeschlagen {symbol}: {e}")

    print(f"[SIGNAL-SERVICE] Scan: {scanned} Coins geprüft, {signals_found} Signale, {trades_opened} eröffnet")
    return trades_opened


# ============================================================
# Main Loop
# ============================================================

def main():
    global running

    print("=" * 70)
    print(f"  Signal-Service — ML Entry + Regelbasierte Exits")
    print(f"  Trailing Stop: {TRAILING_STOP_PCT}% | SL: {SL_PCT}% | Timeout: {TIMEOUT_MINUTES}min")
    print(f"  Hebel: {LEVERAGE}x | Max: {MAX_CONCURRENT} | Confidence: >= {CONFIDENCE_THRESHOLD}")
    print(f"  Scan: :01/:06/:11/... | Management: {MGMT_INTERVAL}s")
    print(f"  Modell: {MODEL_PATH}")
    print("=" * 70)

    # Modell laden
    if not Path(MODEL_PATH).exists():
        print(f"[SIGNAL-SERVICE] FEHLER: Modell nicht gefunden: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = SignalModel(
        input_size=checkpoint.get('input_size', N_FEATURES),
        hidden_sizes=checkpoint.get('hidden_sizes', HIDDEN_SIZES),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[SIGNAL-SERVICE] Modell geladen: {MODEL_PATH}")

    # Config aus DB laden
    load_config_from_db()

    # Credentials
    creds = get_hl_credentials()
    wallet_address = creds['wallet_address']
    print(f"[SIGNAL-SERVICE] Wallet: {wallet_address}")

    balance = get_hl_balance(wallet_address)
    print(f"[SIGNAL-SERVICE] Balance: ${balance:.2f}")

    # State laden
    state = load_state()
    peak_tracker = {}  # pos_id -> peak_pnl

    # DB-Connections
    app_conn = get_conn('app')
    coins_conn = get_conn('coins')

    # HL Coin-Info refreshen
    try:
        refresh_hl_coin_info()
    except:
        pass

    last_scan_min = -1
    last_mgmt_time = datetime.now(timezone.utc) - timedelta(seconds=MGMT_INTERVAL)
    status_interval = 300  # Status alle 5 Min
    last_status_time = time.time()

    print(f"[SIGNAL-SERVICE] Gestartet. Warte auf Scan-Minute...")

    while running:
        try:
            now = datetime.now(timezone.utc)
            now_min = now.minute

            # === POSITION MANAGEMENT ===
            if (now - last_mgmt_time).total_seconds() >= MGMT_INTERVAL:
                last_mgmt_time = now
                try:
                    manage_positions(app_conn, creds, wallet_address, state, peak_tracker)
                except Exception as e:
                    print(f"[SIGNAL-SERVICE] Management-Fehler: {e}")
                    traceback.print_exc()

            # === ENTRY SCAN ===
            if now_min in SCAN_MINUTES and last_scan_min != now_min:
                last_scan_min = now_min
                print(f"[SIGNAL-SERVICE] Scan gestartet (Minute :{now_min:02d})...")
                t_scan = time.time()
                try:
                    # DB-Connections erneuern
                    try:
                        coins_conn.close()
                    except:
                        pass
                    coins_conn = get_conn('coins')

                    opened = scan_entries(model, app_conn, creds, wallet_address, state, coins_conn)
                    print(f"[SIGNAL-SERVICE] Scan fertig: {opened} Trades eröffnet in {time.time()-t_scan:.1f}s")
                except Exception as e:
                    print(f"[SIGNAL-SERVICE] Scan-Fehler: {e}")
                    traceback.print_exc()

            # === STATUS + CONFIG RELOAD ===
            if time.time() - last_status_time >= status_interval:
                last_status_time = time.time()
                load_config_from_db()
                try:
                    balance = get_hl_balance(wallet_address)
                except:
                    balance = 0
                positions = get_open_positions_from_db(app_conn)
                wr = state['total_wins'] / max(state['total_trades'], 1) * 100

                print(f"[SIGNAL-SERVICE] Status | HL: ${balance:.2f} | "
                      f"Profit: ${state['total_profit']:.2f} | "
                      f"Trades: {state['total_trades']} (WR: {wr:.1f}%) | "
                      f"Offen: {len(positions)} | "
                      f"Heute: {state['day_trades']}/{MAX_TRADES_PER_DAY}")

                save_state(state)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[SIGNAL-SERVICE] Unerwarteter Fehler: {e}")
            traceback.print_exc()
            # DB-Connections erneuern
            try:
                app_conn.close()
            except:
                pass
            try:
                coins_conn.close()
            except:
                pass
            app_conn = get_conn('app')
            coins_conn = get_conn('coins')
            time.sleep(10)

    # Cleanup
    save_state(state)
    try:
        app_conn.close()
        coins_conn.close()
    except:
        pass
    print("[SIGNAL-SERVICE] Gestoppt.")


if __name__ == '__main__':
    main()
