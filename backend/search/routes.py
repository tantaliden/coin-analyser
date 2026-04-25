"""
SEARCH ROUTES - Event Search, Candles
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import pytz
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, Query

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_coins_db, get_app_db
from auth.auth import get_current_user

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / 'settings.json') as f:
    SETTINGS = json.load(f)

BERLIN_TZ = pytz.timezone('Europe/Berlin')
ALLOWED_DURATIONS = [30, 60, 90, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600]

router = APIRouter(prefix="/api/v1/search", tags=["search"])

# === MODELS ===

class EventSearchRequest(BaseModel):
    symbols: Optional[List[str]] = None
    start_date: str
    end_date: str
    target_percent: float = Field(default=5.0, ge=0.1, le=100.0)
    duration_minutes: int = Field(default=120, ge=10, le=1440)
    search_interval: str = Field(default="1m")
    direction: str = Field(default="up")
    limit: Optional[int] = Field(default=None, ge=1, le=100000)
    offset: int = Field(default=0, ge=0)

# === HELPERS ===

def get_table_for_interval(interval: str) -> str:
    mapping = {
        '1m': 'klines_1m', '5m': 'klines_5m', '15m': 'klines_15m', '30m': 'klines_30m',
        '1h': 'klines_1h', '4h': 'klines_4h', '1d': 'klines_1d',
    }
    return mapping.get(interval, 'klines_1m')

# === ROUTES ===

@router.get("/events")
async def search_events_get(
    duration_minutes: int = 120,
    min_percent: float = 5.0,
    max_percent: float = 999999.0,
    direction: str = "up",
    groups: str = "",
    start_date: str = "",
    end_date: str = "",
    weekdays: str = "",
    hour_start: int = -1,
    hour_end: int = -1,
    hl_only: bool = True,
    limit: int = 100000,
    current_user: dict = Depends(get_current_user)
):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        if 'T' in start_date:
            start_date = start_date.split('T')[0]
        if 'T' in end_date:
            end_date = end_date.split('T')[0]
        start_dt = BERLIN_TZ.localize(datetime.strptime(start_date, "%Y-%m-%d"))
        end_dt = BERLIN_TZ.localize(datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    
    if duration_minutes not in ALLOWED_DURATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid duration_minutes. Allowed: {ALLOWED_DURATIONS}")
    
    if direction not in ['up', 'down', 'both']:
        raise HTTPException(status_code=400, detail="direction must be 'up', 'down', or 'both'")
    
    # Parse weekday filter: "0,1,2" = Mo,Di,Mi (0=Montag, 6=Sonntag)
    weekday_filter = None
    if weekdays:
        weekday_filter = set(int(w) for w in weekdays.split(',') if w.strip().isdigit())
    
    # Hour filter (Berlin time)
    use_hour_filter = hour_start >= 0 and hour_end >= 0
    
    symbols = []
    if groups:
        group_ids = [int(g) for g in groups.split(',') if g.strip().isdigit()]
        if group_ids:
            with get_app_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT symbol FROM coin_group_members WHERE group_id = ANY(%s)", (group_ids,))
                    symbols = [row['symbol'] for row in cur.fetchall()]
    
    # HL-Only Filter
    if hl_only:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM coin_info WHERE hl_sz_decimals IS NOT NULL")
                hl_symbols = set(row['symbol'] for row in cur.fetchall())
        if symbols:
            symbols = [s for s in symbols if s in hl_symbols]
        else:
            symbols = list(hl_symbols)
    
    pct_column = f"pct_{duration_minutes}m"
    events = []
    
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            if not symbols:
                cur.execute("SELECT DISTINCT symbol FROM kline_metrics ORDER BY symbol")
                symbols = [row['symbol'] for row in cur.fetchall()]
            
            for symbol in symbols:
                if len(events) >= limit:
                    break
                
                if direction == 'up':
                    query = f"""
                        SELECT open_time, {pct_column} as change_pct FROM kline_metrics
                        WHERE symbol = %s AND open_time >= %s AND open_time < %s
                          AND {pct_column} >= %s AND {pct_column} <= %s AND {pct_column} IS NOT NULL
                        ORDER BY open_time
                    """
                    cur.execute(query, (symbol, start_dt, end_dt, min_percent, max_percent))
                elif direction == 'down':
                    query = f"""
                        SELECT open_time, {pct_column} as change_pct FROM kline_metrics
                        WHERE symbol = %s AND open_time >= %s AND open_time < %s
                          AND {pct_column} <= %s AND {pct_column} >= %s AND {pct_column} IS NOT NULL
                        ORDER BY open_time
                    """
                    cur.execute(query, (symbol, start_dt, end_dt, -min_percent, -max_percent))
                else:
                    query = f"""
                        SELECT open_time, {pct_column} as change_pct FROM kline_metrics
                        WHERE symbol = %s AND open_time >= %s AND open_time < %s
                          AND (({pct_column} >= %s AND {pct_column} <= %s) OR ({pct_column} <= %s AND {pct_column} >= %s))
                          AND {pct_column} IS NOT NULL
                        ORDER BY open_time
                    """
                    cur.execute(query, (symbol, start_dt, end_dt, min_percent, max_percent, -min_percent, -max_percent))
                
                hits = cur.fetchall()
                if not hits:
                    continue
                
                last_event_end = None
                for hit in hits:
                    if len(events) >= limit:
                        break
                    
                    event_end_time = hit['open_time']
                    event_start_time = event_end_time - timedelta(minutes=duration_minutes)
                    
                    event_start_berlin = event_start_time.astimezone(BERLIN_TZ) if event_start_time.tzinfo else BERLIN_TZ.localize(event_start_time)
                    if event_start_berlin < start_dt or event_start_berlin > end_dt:
                        continue
                    
                    # Weekday filter (Berlin time, 0=Monday)
                    if weekday_filter is not None:
                        if event_start_berlin.weekday() not in weekday_filter:
                            continue
                    
                    # Hour filter (Berlin time)
                    if use_hour_filter:
                        h = event_start_berlin.hour
                        if hour_start <= hour_end:
                            if h < hour_start or h > hour_end:
                                continue
                        else:
                            # Wrap around midnight, e.g. 22-06
                            if h < hour_start and h > hour_end:
                                continue
                    
                    if last_event_end and event_start_time < last_event_end:
                        continue
                    
                    events.append({
                        "id": f"{symbol}_{event_start_time.strftime('%Y%m%d%H%M')}",
                        "symbol": symbol,
                        "event_start": event_start_time.isoformat(),
                        "event_end": event_end_time.isoformat(),
                        "change_percent": round(float(hit['change_pct']), 2),
                        "duration_minutes": duration_minutes,
                        "direction": 'up' if float(hit['change_pct']) >= 0 else 'down'
                    })
                    last_event_end = event_end_time
    
    return {
        "results": events,
        "total": len(events),
        "filters": {"duration_minutes": duration_minutes, "min_percent": min_percent, "max_percent": max_percent,
                    "direction": direction, "start_date": start_date, "end_date": end_date,
                    "weekdays": weekdays, "hour_start": hour_start, "hour_end": hour_end}
    }

@router.post("/events")
async def search_events(request: EventSearchRequest, current_user: dict = Depends(get_current_user)):
    if request.duration_minutes not in ALLOWED_DURATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid duration_minutes. Allowed: {ALLOWED_DURATIONS}")
    
    if request.direction not in ['up', 'down']:
        raise HTTPException(status_code=400, detail="direction must be 'up' or 'down'")
    
    try:
        start_dt = BERLIN_TZ.localize(datetime.strptime(request.start_date, "%Y-%m-%d"))
        end_dt = BERLIN_TZ.localize(datetime.strptime(request.end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    pct_column = f"pct_{request.duration_minutes}m"
    events = []
    
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            if request.symbols:
                symbols = request.symbols
            else:
                cur.execute("SELECT DISTINCT symbol FROM kline_metrics ORDER BY symbol")
                symbols = [row['symbol'] for row in cur.fetchall()]
            
            for symbol in symbols:
                if request.limit and len(events) >= request.limit + request.offset:
                    break
                
                if request.direction == 'up':
                    query = f"""
                        SELECT open_time, {pct_column} as change_pct FROM kline_metrics
                        WHERE symbol = %s AND open_time >= %s AND open_time < %s
                          AND {pct_column} >= %s AND {pct_column} IS NOT NULL
                        ORDER BY open_time
                    """
                    cur.execute(query, (symbol, start_dt, end_dt, request.target_percent))
                else:
                    query = f"""
                        SELECT open_time, {pct_column} as change_pct FROM kline_metrics
                        WHERE symbol = %s AND open_time >= %s AND open_time < %s
                          AND {pct_column} <= %s AND {pct_column} IS NOT NULL
                        ORDER BY open_time
                    """
                    cur.execute(query, (symbol, start_dt, end_dt, -request.target_percent))
                
                hits = cur.fetchall()
                if not hits:
                    continue
                
                last_event_end = None
                for hit in hits:
                    if request.limit and len(events) >= request.limit + request.offset:
                        break
                    
                    event_end_time = hit['open_time']
                    event_start_time = event_end_time - timedelta(minutes=request.duration_minutes)
                    
                    event_start_berlin = event_start_time.astimezone(BERLIN_TZ) if event_start_time.tzinfo else BERLIN_TZ.localize(event_start_time)
                    if event_start_berlin < start_dt or event_start_berlin > end_dt:
                        continue
                    
                    if last_event_end and event_start_time < last_event_end:
                        continue
                    
                    cur.execute("""
                        SELECT open_time, open, high, low, close, volume, trades
                        FROM klines_1m WHERE symbol = %s
                          AND open_time >= %s AND open_time <= %s
                        ORDER BY open_time
                    """, (symbol, event_start_time, event_end_time))
                    
                    candles = cur.fetchall()
                    if len(candles) < 2:
                        continue
                    
                    start_price = float(candles[0]['open'])
                    end_price = float(candles[-1]['close'])
                    if start_price == 0:
                        continue
                    
                    total_volume = sum(float(c['volume']) for c in candles)
                    trades_count = sum(int(c['trades'] or 0) for c in candles)
                    
                    max_price = max(float(c['high']) for c in candles)
                    min_price = min(float(c['low']) for c in candles)
                    volatility_pct = ((max_price - min_price) / start_price) * 100 if start_price > 0 else 0
                    
                    running_max = float(candles[0]['high'])
                    max_drawdown = 0.0
                    for c in candles:
                        if float(c['high']) > running_max:
                            running_max = float(c['high'])
                        drawdown = ((running_max - float(c['low'])) / running_max) * 100 if running_max > 0 else 0
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                    
                    events.append({
                        "id": f"{symbol}_{candles[0]['open_time'].strftime('%Y%m%d%H%M')}",
                        "symbol": symbol,
                        "event_start": candles[0]['open_time'].astimezone(BERLIN_TZ).strftime('%Y-%m-%d %H:%M'),
                        "event_end": candles[-1]['open_time'].astimezone(BERLIN_TZ).strftime('%Y-%m-%d %H:%M'),
                        "start_price": start_price,
                        "end_price": end_price,
                        "change_percent": round(float(hit['change_pct']), 2),
                        "duration_minutes": request.duration_minutes,
                        "volume": round(total_volume, 2),
                        "trades_count": trades_count,
                        "volatility_pct": round(volatility_pct, 2),
                        "max_drawdown_pct": round(max_drawdown, 2),
                        "candles_count": len(candles),
                        "avg_volume_per_candle": round(total_volume / len(candles), 2)
                    })
                    last_event_end = event_end_time
    
    if request.direction == 'up':
        events.sort(key=lambda x: x['change_percent'], reverse=True)
    else:
        events.sort(key=lambda x: x['change_percent'], reverse=False)
    
    total = len(events)
    if request.limit:
        events = events[request.offset:request.offset + request.limit]
    elif request.offset:
        events = events[request.offset:]
    
    return {
        "events": events, "total": total, "limit": request.limit, "offset": request.offset,
        "parameters": {"start_date": request.start_date, "end_date": request.end_date,
                       "target_percent": request.target_percent, "duration_minutes": request.duration_minutes,
                       "direction": request.direction}
    }

@router.get("/candles")
async def get_candles(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1m",
    current_user: dict = Depends(get_current_user)
):
    try:
        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")
    
    table = get_table_for_interval(interval)
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT open_time as time, open, high, low, close, volume, trades,
                       quote_asset_volume, taker_buy_base, taker_buy_quote,
                       funding, open_interest, premium, oracle_px, mark_px, mid_px,
                       bbo_bid_px, bbo_ask_px, bbo_bid_sz, bbo_ask_sz,
                       spread_bps, book_imbalance_5, book_depth_5
                FROM {table}
                WHERE symbol = %s AND open_time >= %s AND open_time <= %s
                ORDER BY open_time
            """
            cur.execute(query, (symbol, start_dt, end_dt))
            rows = cur.fetchall()

    def _fnum(v):
        return float(v) if v is not None else None

    candles = []
    for row in rows:
        candles.append({
            "time": int(row['time'].timestamp()),
            "open": _fnum(row['open']),
            "high": _fnum(row['high']),
            "low": _fnum(row['low']),
            "close": _fnum(row['close']),
            "volume": _fnum(row['volume']),
            "trades": int(row['trades'] or 0),
            "quote_asset_volume": _fnum(row['quote_asset_volume']),
            "taker_buy_base": _fnum(row['taker_buy_base']),
            "taker_buy_quote": _fnum(row['taker_buy_quote']),
            "funding": _fnum(row.get('funding')),
            "open_interest": _fnum(row.get('open_interest')),
            "premium": _fnum(row.get('premium')),
            "oracle_px": _fnum(row.get('oracle_px')),
            "mark_px": _fnum(row.get('mark_px')),
            "mid_px": _fnum(row.get('mid_px')),
            "bbo_bid_px": _fnum(row.get('bbo_bid_px')),
            "bbo_ask_px": _fnum(row.get('bbo_ask_px')),
            "bbo_bid_sz": _fnum(row.get('bbo_bid_sz')),
            "bbo_ask_sz": _fnum(row.get('bbo_ask_sz')),
            "spread_bps": _fnum(row.get('spread_bps')),
            "book_imbalance_5": _fnum(row.get('book_imbalance_5')),
            "book_depth_5": _fnum(row.get('book_depth_5')),
        })
    
    return {"symbol": symbol, "interval": interval, "candles": candles, "count": len(candles)}


# === PRE-CANDLES ===

class PreCandleEvent(BaseModel):
    id: str
    symbol: str
    event_start: str

class PreCandlesRequest(BaseModel):
    events: List[PreCandleEvent]
    timeframe: str = "15m"
    candle_count: int = 10
    prehistory_minutes: Optional[int] = None
    time_end_minutes: Optional[int] = None

@router.post("/pre-candles")
async def get_pre_event_candles(
    request: PreCandlesRequest,
    current_user: dict = Depends(get_current_user)
):
    """Holt für jedes Event die X Candles VOR event_start. Gibt Farben zurück."""
    timeframe_map = {
        '2m': 'agg_2m', '5m': 'agg_5m', '10m': 'agg_10m', '15m': 'agg_15m', '30m': 'agg_30m',
        '1h': 'agg_1h', '2h': 'agg_2h', '4h': 'agg_4h', '6h': 'agg_6h', '12h': 'agg_12h',
        '1d': 'agg_1d', '3d': 'agg_3d', '7d': 'agg_7d'
    }
    
    agg_table = timeframe_map.get(request.timeframe)
    if not agg_table:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {request.timeframe}")
    
    candle_count = request.candle_count
    results = {}
    
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '60s'")
            
            for event in request.events:
                symbol = event.symbol
                event_start = event.event_start
                event_id = event.id
                
                if not symbol or not event_start:
                    continue
                
                try:
                    end_time = event_start
                    if request.prehistory_minutes and request.time_end_minutes:
                        offset_minutes = request.prehistory_minutes - request.time_end_minutes
                        if isinstance(event_start, str):
                            end_time = datetime.fromisoformat(event_start.replace('Z', '+00:00')) - timedelta(minutes=offset_minutes)
                        else:
                            end_time = event_start - timedelta(minutes=offset_minutes)
                    
                    cur.execute(f"""
                        SELECT bucket, open, close
                        FROM {agg_table}
                        WHERE symbol = %s AND bucket < %s
                        ORDER BY bucket DESC
                        LIMIT %s
                    """, (symbol, end_time, candle_count))
                    
                    rows = cur.fetchall()
                    
                    if len(rows) < 1:
                        results[event_id] = []
                        continue
                    
                    candles = []
                    rows_reversed = list(reversed(rows))
                    
                    for row in rows_reversed:
                        if float(row['close']) >= float(row['open']):
                            candles.append('green')
                        else:
                            candles.append('red')
                    
                    results[event_id] = candles
                    
                except Exception as e:
                    results[event_id] = []
    
    return {"candles": results, "timeframe": request.timeframe, "count": candle_count}


# === CASCADE (alias for /indicators/validate) ===

class CascadeEvent(BaseModel):
    id: str
    symbol: str
    event_start: str
    event_end: Optional[str] = None
    change_percent: Optional[float] = None
    duration_minutes: Optional[int] = None
    direction: Optional[str] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    volume: Optional[float] = None
    trades_count: Optional[int] = None
    volatility_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    candles_count: Optional[int] = None
    avg_volume_per_candle: Optional[float] = None

class CascadeIndicator(BaseModel):
    item_id: Optional[int] = None
    indicator_type: Optional[str] = None
    field: Optional[str] = None
    condition_operator: Optional[str] = None
    condition_value: Optional[float] = None
    time_start_minutes: Optional[int] = None
    time_end_minutes: Optional[int] = None
    aggregator: Optional[str] = None
    color: Optional[str] = None
    is_active: Optional[bool] = True
    pattern_data: Optional[dict] = None
    pattern_type: Optional[str] = None

class CascadeRequest(BaseModel):
    events: List[dict]
    indicators: List[dict]
    filters: Optional[dict] = None
    prehistory_minutes: int

@router.post("/cascade")
async def cascade_filter(
    request: CascadeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Cascade-Filter: Events durch Indikator-Chain filtern"""
    from indicators.routes import _apply_cascade
    
    matched = _apply_cascade(request.events, request.indicators, request.prehistory_minutes)
    
    match_rate = round((len(matched) / len(request.events) * 100), 1) if request.events else 0
    
    return {
        "results": matched,
        "total_input": len(request.events),
        "total_matched": len(matched),
        "match_rate": match_rate
    }


@router.get("/day-open")
async def get_day_open(
    symbol: str,
    date: str,  # YYYY-MM-DD in Berlin TZ
    current_user: dict = Depends(get_current_user)
):
    """Liefert den Open-Preis der ersten Candle des Tages (00:00 Berlin) fuer ein Symbol."""
    try:
        day_start = BERLIN_TZ.localize(datetime.strptime(date, "%Y-%m-%d"))
        day_end = day_start + timedelta(days=1)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")
    
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT open_time, open FROM klines
                WHERE symbol = %s AND open_time >= %s AND open_time < %s
                ORDER BY open_time LIMIT 1
            """, (symbol, day_start, day_end))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"No klines for {symbol} on {date}")
            return {"symbol": symbol, "date": date, "open": float(row['open']), "open_time": row['open_time'].isoformat()}
