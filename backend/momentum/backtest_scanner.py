#!/usr/bin/env python3
"""
BACKTEST SCANNER
Lässt den Scanner über historische Daten laufen und prüft ob TP oder SL zuerst getroffen wird.
Zeitraum: Konfigurierbarer Monat, default letzter Monat.
"""

import sys, os, json, time, logging
from datetime import datetime, timedelta, timezone
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# === LOGGING ===
LOG_FILE = '/opt/coin/logs/backtest_scanner.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backtest')

# === DB ===
def coins_db():
    return psycopg2.connect(host='localhost', dbname='coins', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)

def app_db():
    return psycopg2.connect(host='localhost', dbname='analyser_app', user='volker_admin',
                            password='VoltiStrongPass2025', cursor_factory=RealDictCursor)

# === SCANNER FUNKTIONEN (kopiert aus scanner.py) ===
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_ema(values, period):
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = (v - ema) * multiplier + ema
    return ema

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])

# Import Scanner v2 analyze_symbol + Hilfsfunktionen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner import analyze_symbol, calc_body_ratio, calc_consecutive


def load_scan_config():
    """Lädt aktuelle Scanner-Config aus DB (gleiche Thresholds wie Live-Scanner)"""
    with app_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM momentum_scan_config WHERE is_active = true LIMIT 1")
        return cur.fetchone() or {}


# === BACKTEST LOGIK ===

def get_symbols(conn):
    """Alle USDC Symbole die genug Daten haben"""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT symbol FROM klines 
        WHERE interval = '1m' AND open_time >= NOW() - INTERVAL '60 days'
        GROUP BY symbol HAVING COUNT(*) > 10000
        ORDER BY symbol
    """)
    return [r['symbol'] for r in cur.fetchall()]

def get_candles(conn, symbol, timeframe_view, before_time, limit):
    """Candles VOR einem bestimmten Zeitpunkt laden (simuliert 'jetzt' im Backtest)"""
    cur = conn.cursor()
    if timeframe_view == '1h':
        table = 'agg_1h'
    elif timeframe_view == '4h':
        table = 'agg_4h'
    elif timeframe_view == '1d':
        table = 'agg_1d'
    else:
        raise ValueError(f"Unknown timeframe: {timeframe_view}")
    
    cur.execute(f"""
        SELECT bucket as open_time, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s AND bucket < %s
        ORDER BY bucket DESC
        LIMIT %s
    """, (symbol, before_time, limit))
    rows = cur.fetchall()
    rows.reverse()  # Chronologisch sortieren
    return rows

def get_1m_candles_forward(conn, symbol, from_time, hours=24):
    """1m Candles NACH dem Signal holen (für TP/SL Check)"""
    cur = conn.cursor()
    end_time = from_time + timedelta(hours=hours)
    cur.execute("""
        SELECT open_time, open, high, low, close 
        FROM klines 
        WHERE symbol = %s AND interval = '1m' AND open_time >= %s AND open_time < %s
        ORDER BY open_time
    """, (symbol, from_time, end_time))
    return cur.fetchall()

def compute_market_context(conn, scan_time, symbols):
    """Berechnet Market Context zu einem historischen Zeitpunkt"""
    cur = conn.cursor()
    
    # 4h Veränderung für alle Symbole
    changes = []
    # Vereinfacht: Top 20 Symbole nach Volumen für Breadth
    cur.execute("""
        SELECT symbol, close
        FROM agg_1h
        WHERE bucket = (SELECT MAX(bucket) FROM agg_1h WHERE bucket <= %s)
          AND symbol = ANY(%s)
    """, (scan_time, symbols[:50]))
    current_prices = {r['symbol']: r['close'] for r in cur.fetchall()}
    
    t_4h = scan_time - timedelta(hours=4)
    cur.execute("""
        SELECT symbol, close
        FROM agg_1h
        WHERE bucket = (SELECT MAX(bucket) FROM agg_1h WHERE bucket <= %s)
          AND symbol = ANY(%s)
    """, (t_4h, list(current_prices.keys())))
    old_prices = {r['symbol']: r['close'] for r in cur.fetchall()}
    
    t_1h = scan_time - timedelta(hours=1)
    cur.execute("""
        SELECT symbol, close
        FROM agg_1h
        WHERE bucket = (SELECT MAX(bucket) FROM agg_1h WHERE bucket <= %s)
          AND symbol = ANY(%s)
    """, (t_1h, list(current_prices.keys())))
    prices_1h = {r['symbol']: r['close'] for r in cur.fetchall()}
    
    changes_4h = []
    positive_1h = 0
    total_1h = 0
    for sym in current_prices:
        if sym in old_prices and old_prices[sym] > 0:
            pct = ((current_prices[sym] - old_prices[sym]) / old_prices[sym]) * 100
            changes_4h.append(pct)
        if sym in prices_1h and prices_1h[sym] > 0:
            total_1h += 1
            if current_prices[sym] > prices_1h[sym]:
                positive_1h += 1
    
    if not changes_4h:
        return None
    
    avg_4h = np.mean(changes_4h)
    breadth_4h = len([c for c in changes_4h if c > 0]) / len(changes_4h)
    breadth_1h = positive_1h / total_1h if total_1h > 0 else 0.5
    
    # Trend Score
    trend = 0
    if avg_4h > 1.0: trend = 0.8
    elif avg_4h > 0.5: trend = 0.5
    elif avg_4h > 0: trend = 0.2
    elif avg_4h > -0.5: trend = -0.2
    elif avg_4h > -1.0: trend = -0.5
    else: trend = -0.8
    
    return {
        'market_trend': trend,
        'avg_4h': avg_4h,
        'breadth_4h': breadth_4h,
        'breadth_1h': breadth_1h,
    }

def check_tp_sl(forward_candles, entry_price, direction, tp_pct=2.0, sl_pct=2.0):
    """Prüft ob TP oder SL zuerst getroffen wird. Minute für Minute."""
    if not forward_candles:
        return 'no_data', 0, 0, 0
    
    max_favorable = 0
    max_adverse = 0
    
    for i, c in enumerate(forward_candles):
        if direction == 'long':
            # Long: Preis steigt = gut
            high_pct = ((c['high'] - entry_price) / entry_price) * 100
            low_pct = ((c['low'] - entry_price) / entry_price) * 100
            max_favorable = max(max_favorable, high_pct)
            max_adverse = min(max_adverse, low_pct)
            
            # SL zuerst prüfen (wenn low SL durchbricht)
            if low_pct <= -sl_pct:
                return 'hit_sl', i + 1, max_favorable, max_adverse
            # Dann TP
            if high_pct >= tp_pct:
                return 'hit_tp', i + 1, max_favorable, max_adverse
        else:
            # Short: Preis fällt = gut
            # Favorable = wie weit Preis gefallen ist
            low_pct = ((entry_price - c['low']) / entry_price) * 100  # positiv wenn Preis fällt
            high_pct = ((entry_price - c['high']) / entry_price) * 100  # negativ wenn Preis steigt
            max_favorable = max(max_favorable, low_pct)
            max_adverse = min(max_adverse, high_pct)
            
            # SL: Preis steigt um sl_pct
            if -high_pct >= sl_pct:  # high_pct ist negativ wenn Preis steigt
                return 'hit_sl', i + 1, max_favorable, max_adverse
            # TP: Preis fällt um tp_pct  
            if low_pct >= tp_pct:
                return 'hit_tp', i + 1, max_favorable, max_adverse
    
    return 'expired', len(forward_candles), max_favorable, max_adverse


def run_backtest(start_date, end_date, tp_pct=2.0, sl_pct=2.0, min_confidence=65,
                 scan_interval_hours=1, max_forward_hours=24,
                 long_pct_30m_min=2.5, short_pct_30m_min=1.0, short_pct_60m_max=-3.0):
    """
    Hauptfunktion: Scanner über historische Daten laufen lassen.
    Scannt alle X Stunden, für jedes Symbol, prüft ob TP/SL erreicht wird.
    """
    scan_config = load_scan_config()
    logger.info(f"Loaded scan_config: {dict(scan_config)}")

    logger.info(f"{'='*60}")
    logger.info(f"BACKTEST START")
    logger.info(f"Period: {start_date} → {end_date}")
    logger.info(f"TP={tp_pct}% SL={sl_pct}% MinConf={min_confidence}")
    logger.info(f"Long pct_30m>={long_pct_30m_min}% Short pct_30m<=-{short_pct_30m_min}%")
    logger.info(f"Short pct_60m blocker: <{short_pct_60m_max}%")
    logger.info(f"Scan every {scan_interval_hours}h, forward window {max_forward_hours}h")
    logger.info(f"{'='*60}")

    cconn = coins_db()
    
    # Symbole laden
    symbols = get_symbols(cconn)
    logger.info(f"Symbols: {len(symbols)}")
    
    # Ergebnis-Tracker
    results = []
    total_scans = 0
    total_signals = 0
    
    # Zeitschleife: alle X Stunden von start bis end
    scan_time = start_date
    scan_count = 0
    
    while scan_time <= end_date:
        scan_count += 1
        signals_this_scan = 0
        
        # Market Context berechnen
        mkt_ctx = compute_market_context(cconn, scan_time, symbols)
        
        for symbol in symbols:
            # Candles laden (wie der Live-Scanner es tut)
            candles_1h = get_candles(cconn, symbol, '1h', scan_time, 50)
            candles_4h = get_candles(cconn, symbol, '4h', scan_time, 20)
            candles_1d = get_candles(cconn, symbol, '1d', scan_time, 15)
            
            if not candles_1h:
                continue
            
            current_price = candles_1h[-1]['close']
            if not current_price or current_price <= 0:
                continue
            
            # Scanner analysieren
            signal = analyze_symbol(candles_1h, candles_4h, candles_1d, current_price, market_context=mkt_ctx, scan_config=scan_config)
            
            if signal is None:
                continue
            if signal['confidence'] < min_confidence:
                continue
            
            # pct_30m / pct_60m Filter (wie im Live-Scanner)
            ccur = cconn.cursor()
            ccur.execute("""
                SELECT pct_30m, pct_60m FROM kline_metrics
                WHERE symbol = %s AND open_time <= %s
                ORDER BY open_time DESC LIMIT 1
            """, (symbol, scan_time))
            metrics = ccur.fetchone()
            
            if metrics and metrics['pct_30m'] is not None:
                pct_30m = float(metrics['pct_30m'])
                if signal['direction'] == 'long' and pct_30m < long_pct_30m_min:
                    continue
                if signal['direction'] == 'short' and pct_30m > -short_pct_30m_min:
                    continue
                
                # Short-Blocker pct_60m
                if signal['direction'] == 'short' and metrics.get('pct_60m') is not None:
                    pct_60m = float(metrics['pct_60m'])
                    if pct_60m < short_pct_60m_max:
                        continue
            
            # Signal gefunden → Forward-Candles laden und TP/SL prüfen
            forward = get_1m_candles_forward(cconn, symbol, scan_time, max_forward_hours)
            
            outcome, minutes, max_fav, max_adv = check_tp_sl(
                forward, signal['entry_price'], signal['direction'], tp_pct, sl_pct
            )
            
            pct_30m_val = float(metrics['pct_30m']) if metrics and metrics['pct_30m'] else None
            
            results.append({
                'scan_time': scan_time,
                'symbol': symbol,
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'entry_price': signal['entry_price'],
                'outcome': outcome,
                'minutes_to_outcome': minutes,
                'max_favorable': round(max_fav, 2),
                'max_adverse': round(max_adv, 2),
                'pct_30m': pct_30m_val,
                'rsi': signal['indicators']['rsi_1h'],
                'vol_ratio': signal['indicators']['vol_ratio'],
                'trend_4h': signal['indicators']['trend_4h'],
                'hh_hl': signal['indicators']['hh_hl'],
            })
            
            signals_this_scan += 1
            total_signals += 1
        
        total_scans += 1
        
        if scan_count % 24 == 0:
            logger.info(f"[PROGRESS] {scan_time:%Y-%m-%d %H:00} | Scans: {total_scans} | Signals: {total_signals}")
        
        scan_time += timedelta(hours=scan_interval_hours)
    
    cconn.close()
    
    # === AUSWERTUNG ===
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Period: {start_date} → {end_date}")
    logger.info(f"Total scans: {total_scans}")
    logger.info(f"Total signals: {total_signals}")
    
    for direction in ['long', 'short']:
        d_results = [r for r in results if r['direction'] == direction]
        if not d_results:
            continue
        
        tp = len([r for r in d_results if r['outcome'] == 'hit_tp'])
        sl = len([r for r in d_results if r['outcome'] == 'hit_sl'])
        expired = len([r for r in d_results if r['outcome'] == 'expired'])
        no_data = len([r for r in d_results if r['outcome'] == 'no_data'])
        total = len(d_results)
        resolved = tp + sl
        hit_rate = (tp / resolved * 100) if resolved > 0 else 0
        
        logger.info(f"\n  {direction.upper()}: {total} signals")
        logger.info(f"    TP: {tp} | SL: {sl} | Expired: {expired} | No-Data: {no_data}")
        logger.info(f"    Hit-Rate: {hit_rate:.1f}% ({tp}/{resolved} resolved)")
        
        # Durchschnittliche Zeit bis Outcome
        tp_times = [r['minutes_to_outcome'] for r in d_results if r['outcome'] == 'hit_tp']
        sl_times = [r['minutes_to_outcome'] for r in d_results if r['outcome'] == 'hit_sl']
        if tp_times:
            logger.info(f"    Avg time to TP: {np.mean(tp_times):.0f} min ({np.mean(tp_times)/60:.1f}h)")
        if sl_times:
            logger.info(f"    Avg time to SL: {np.mean(sl_times):.0f} min ({np.mean(sl_times)/60:.1f}h)")
        
        # Max favorable bei SLs (zeigt ob Signal "fast" richtig war)
        sl_favs = [r['max_favorable'] for r in d_results if r['outcome'] == 'hit_sl']
        if sl_favs:
            logger.info(f"    SL predictions max favorable: avg={np.mean(sl_favs):.2f}% med={np.median(sl_favs):.2f}%")
            never_moved = len([f for f in sl_favs if f < 0.5])
            logger.info(f"    SL with max_fav < 0.5% (never moved right): {never_moved}/{len(sl_favs)} ({never_moved/len(sl_favs)*100:.0f}%)")
        
        # Confidence Breakdown
        for conf_min, conf_max in [(65, 69), (70, 79), (80, 89), (90, 100)]:
            c_results = [r for r in d_results if conf_min <= r['confidence'] <= conf_max]
            if not c_results:
                continue
            c_tp = len([r for r in c_results if r['outcome'] == 'hit_tp'])
            c_sl = len([r for r in c_results if r['outcome'] == 'hit_sl'])
            c_res = c_tp + c_sl
            c_hr = (c_tp / c_res * 100) if c_res > 0 else 0
            logger.info(f"    Conf {conf_min}-{conf_max}: {c_hr:.1f}% ({c_tp}/{c_res})")
        
        # pct_30m Breakdown
        pct_vals = [(r['pct_30m'], r['outcome']) for r in d_results if r['pct_30m'] is not None]
        if pct_vals:
            logger.info(f"\n    pct_30m Breakdown:")
            if direction == 'long':
                for lo, hi in [(1.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 99)]:
                    bucket = [(p, o) for p, o in pct_vals if lo <= p < hi]
                    if not bucket:
                        continue
                    b_tp = len([o for _, o in bucket if o == 'hit_tp'])
                    b_sl = len([o for _, o in bucket if o == 'hit_sl'])
                    b_res = b_tp + b_sl
                    b_hr = (b_tp / b_res * 100) if b_res > 0 else 0
                    logger.info(f"      pct_30m {lo:.1f}-{hi:.1f}%: {b_hr:.1f}% ({b_tp}/{b_res})")
            else:
                for lo, hi in [(-2.0, -1.0), (-3.0, -2.0), (-4.0, -3.0), (-99, -4.0)]:
                    bucket = [(p, o) for p, o in pct_vals if lo <= p < hi]
                    if not bucket:
                        continue
                    b_tp = len([o for _, o in bucket if o == 'hit_tp'])
                    b_sl = len([o for _, o in bucket if o == 'hit_sl'])
                    b_res = b_tp + b_sl
                    b_hr = (b_tp / b_res * 100) if b_res > 0 else 0
                    logger.info(f"      pct_30m {lo:.1f} to {hi:.1f}%: {b_hr:.1f}% ({b_tp}/{b_res})")
    
    # Worst offenders: Symbole die immer daneben liegen
    logger.info(f"\n{'='*60}")
    logger.info(f"SYMBOL BREAKDOWN (Top Gewinner + Verlierer)")
    logger.info(f"{'='*60}")
    
    from collections import defaultdict
    sym_stats = defaultdict(lambda: {'tp': 0, 'sl': 0})
    for r in results:
        if r['outcome'] in ('hit_tp', 'hit_sl'):
            sym_stats[r['symbol']][r['outcome'].split('_')[1]] += 1
    
    sym_list = []
    for sym, st in sym_stats.items():
        total = st['tp'] + st['sl']
        if total >= 3:
            sym_list.append((sym, st['tp'], st['sl'], total, st['tp'] / total * 100))
    
    sym_list.sort(key=lambda x: x[4])
    
    logger.info("\n  WORST (lowest hit-rate):")
    for sym, tp, sl, total, hr in sym_list[:10]:
        logger.info(f"    {sym:15s}: {hr:5.1f}% ({tp} TP / {sl} SL / {total} total)")
    
    logger.info("\n  BEST (highest hit-rate):")
    for sym, tp, sl, total, hr in sym_list[-10:]:
        logger.info(f"    {sym:15s}: {hr:5.1f}% ({tp} TP / {sl} SL / {total} total)")
    
    # Signal-Analyse: Welche Signale korrelieren mit TP vs SL?
    logger.info(f"\n{'='*60}")
    logger.info(f"INDICATOR BREAKDOWN")
    logger.info(f"{'='*60}")
    
    for direction in ['long', 'short']:
        d_results = [r for r in results if r['direction'] == direction and r['outcome'] in ('hit_tp', 'hit_sl')]
        if not d_results:
            continue
        
        logger.info(f"\n  {direction.upper()}:")
        
        # RSI Buckets
        for lo, hi in [(0, 30), (30, 45), (45, 55), (55, 70), (70, 100)]:
            bucket = [r for r in d_results if lo <= r['rsi'] < hi]
            if not bucket:
                continue
            b_tp = len([r for r in bucket if r['outcome'] == 'hit_tp'])
            b_res = len(bucket)
            logger.info(f"    RSI {lo}-{hi}: {b_tp/b_res*100:.1f}% ({b_tp}/{b_res})")
        
        # Trend 4h
        for lo, hi in [(-1.0, -0.3), (-0.3, 0.3), (0.3, 1.0)]:
            bucket = [r for r in d_results if lo <= r['trend_4h'] <= hi]
            if not bucket:
                continue
            b_tp = len([r for r in bucket if r['outcome'] == 'hit_tp'])
            b_res = len(bucket)
            logger.info(f"    Trend4h {lo:+.1f} to {hi:+.1f}: {b_tp/b_res*100:.1f}% ({b_tp}/{b_res})")
        
        # HH/HL
        for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]:
            bucket = [r for r in d_results if lo <= r['hh_hl'] <= hi]
            if not bucket:
                continue
            b_tp = len([r for r in bucket if r['outcome'] == 'hit_tp'])
            b_res = len(bucket)
            logger.info(f"    HH/HL {lo:.1f}-{hi:.1f}: {b_tp/b_res*100:.1f}% ({b_tp}/{b_res})")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    # Default: Letzter Monat
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    
    # Override per Argument
    if len(sys.argv) >= 3:
        start = datetime.strptime(sys.argv[1], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(sys.argv[2], '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    run_backtest(start, end)
