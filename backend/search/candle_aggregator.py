"""Laedt und aggregiert Candle-Daten aus klines/agg-Tabellen. Arbeitet in-memory."""

from datetime import datetime, timedelta
from .predictor_settings import AGG_TABLE_MAP


def _naive(dt):
    if dt is None or not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt
    return dt.replace(tzinfo=None)


def load_candles_for_symbol(symbol, start_time, end_time, timeframe_minutes, cursor):
    """Laedt Candles fuer ein Symbol im Zeitraum. DB hat naive timestamps."""
    # Facade-Views klines_Xm (open_time, trades als native Spalte)
    table = AGG_TABLE_MAP.get(timeframe_minutes)
    if table is None:
        table = 'klines_1m'
    start_time = _naive(start_time)
    end_time = _naive(end_time)
    cursor.execute(f"""
        SELECT open_time, open, high, low, close, volume, trades
        FROM {table}
        WHERE symbol = %s AND open_time >= %s AND open_time < %s
        ORDER BY open_time
    """, (symbol, start_time, end_time))
    
    rows = cursor.fetchall()
    return [dict(r) for r in rows]


def aggregate_candles(candles_1m, timeframe_minutes):
    """Aggregiert 1m-Candles zu groesserem Timeframe. Nur noetig wenn keine passende agg-Tabelle existiert."""
    if timeframe_minutes <= 1 or not candles_1m:
        return candles_1m
    
    result = []
    for i in range(0, len(candles_1m), timeframe_minutes):
        chunk = candles_1m[i:i + timeframe_minutes]
        if not chunk:
            continue
        result.append({
            'open_time': chunk[0]['open_time'],
            'open': chunk[0]['open'],
            'high': max(float(c['high']) for c in chunk),
            'low': min(float(c['low']) for c in chunk),
            'close': chunk[-1]['close'],
            'volume': sum(float(c['volume']) for c in chunk),
            'trades': sum(int(c.get('trades', 0)) for c in chunk),
        })
    return result


def get_candle_at_offset(candles, base_time, offset_minutes, tolerance_minutes=0):
    """Findet die Candle bei base_time + offset_minutes (mit optionaler Toleranz).
    Arbeitet auf in-memory Array statt DB-Query."""
    target = base_time + timedelta(minutes=offset_minutes)
    tol = timedelta(minutes=tolerance_minutes)
    
    best = None
    best_diff = None
    for c in candles:
        t = c['open_time']
        if tolerance_minutes > 0:
            if t < target - tol or t > target + tol:
                continue
        diff = abs((t - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best = c
            best_diff = diff
    
    return best


def find_candles_in_window(candles, base_time, from_offset_min, to_offset_min):
    """Alle Candles im Zeitfenster [base_time + from_offset, base_time + to_offset]."""
    t_from = base_time + timedelta(minutes=from_offset_min)
    t_to = base_time + timedelta(minutes=to_offset_min)
    return [c for c in candles if t_from <= c['open_time'] <= t_to]


def get_day_open_from_candles(candles, event_time, tz):
    """Holt den Open-Preis der ersten Candle des Tages (00:00 Berlin) aus in-memory Daten."""
    if event_time.tzinfo:
        berlin_time = event_time.astimezone(tz)
    else:
        berlin_time = tz.localize(event_time)
    
    day_start = tz.localize(datetime(berlin_time.year, berlin_time.month, berlin_time.day))
    
    for c in candles:
        t = c['open_time']
        if hasattr(t, 'astimezone'):
            t = t.astimezone(tz)
        elif t.tzinfo is None:
            t = tz.localize(t)
        if t >= day_start:
            return float(c['open'])
    
    return None
