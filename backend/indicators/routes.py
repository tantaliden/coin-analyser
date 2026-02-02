"""INDICATORS ROUTES - Sets CRUD, Items, Backsearch"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import pytz
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.database import get_app_db, get_coins_db
from auth.auth import get_current_user

BERLIN_TZ = pytz.timezone('Europe/Berlin')
router = APIRouter(prefix="/api/v1/indicators", tags=["indicators"])

# === MODELS ===
class IndicatorSetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    coin_group_id: Optional[int] = None
    search_date_from: Optional[str] = None
    search_date_to: Optional[str] = None
    search_percent_min: Optional[float] = None
    search_percent_max: Optional[float] = None
    search_direction: str = "up"
    search_duration_minutes: Optional[int] = None
    prehistory_minutes: Optional[int] = None

class IndicatorSetWithFirstItem(BaseModel):
    name: str
    description: Optional[str] = None
    coin_group_id: Optional[int] = None
    search_date_from: str
    search_date_to: str
    search_percent_min: float
    search_percent_max: float
    search_direction: str = "up"
    search_duration_minutes: int
    prehistory_minutes: int
    indicator_type: str
    condition_operator: str
    condition_value: float
    condition_value2: Optional[float] = None
    time_start_minutes: int
    time_end_minutes: int
    aggregator: str = "1m"
    reference_start_minutes: Optional[int] = None
    reference_end_minutes: Optional[int] = None
    color: Optional[str] = "#3B82F6"

class IndicatorItemCreate(BaseModel):
    field: str
    operation: str
    value: Optional[float] = None
    value2: Optional[float] = None
    time_start_minutes: int
    time_end_minutes: int
    aggregator: str = "1m"
    reference_start_minutes: Optional[int] = None
    reference_end_minutes: Optional[int] = None
    color: Optional[str] = "#3B82F6"
    pattern_type: Optional[str] = None
    pattern_data: Optional[dict] = None
    pattern_count: Optional[int] = None
    pattern_consecutive: Optional[bool] = False

class IndicatorItemUpdate(BaseModel):
    pattern_data: Optional[dict] = None
    is_active: Optional[bool] = None
    time_start_minutes: Optional[int] = None
    time_end_minutes: Optional[int] = None

class CounterSearchRequest(BaseModel):
    start_date: str
    end_date: str
    symbols: Optional[list] = None
    scan_interval_minutes: int = 60

# === SETS CRUD ===
@router.get("/sets")
async def list_indicator_sets(current_user: dict = Depends(get_current_user), only_mine: bool = False):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            if only_mine:
                cur.execute("""SELECT s.*, (SELECT COUNT(*) FROM indicator_items WHERE set_id = s.set_id) as item_count
                    FROM indicator_sets s WHERE s.owner_id = %s ORDER BY s.updated_at DESC""", (current_user['user_id'],))
            else:
                cur.execute("""SELECT s.*, u.email as owner_email, (SELECT COUNT(*) FROM indicator_items WHERE set_id = s.set_id) as item_count
                    FROM indicator_sets s LEFT JOIN users u ON s.owner_id = u.user_id
                    WHERE s.is_public = TRUE ORDER BY s.current_accuracy DESC NULLS LAST, s.updated_at DESC""")
            sets = cur.fetchall()
    return {"sets": sets}

@router.post("/sets")
async def create_indicator_set(request: IndicatorSetCreate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO indicator_sets (owner_id, name, description, coin_group_id, search_date_from, search_date_to,
                search_percent_min, search_percent_max, search_direction, search_duration_minutes, prehistory_minutes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING set_id""",
                (current_user['user_id'], request.name, request.description, request.coin_group_id,
                 request.search_date_from, request.search_date_to, request.search_percent_min, request.search_percent_max,
                 request.search_direction, request.search_duration_minutes, request.prehistory_minutes))
            set_id = cur.fetchone()['set_id']
            conn.commit()
    return {"message": "Set created", "set_id": set_id}

@router.post("/sets/with-indicator")
async def create_indicator_set_with_first_item(request: IndicatorSetWithFirstItem, current_user: dict = Depends(get_current_user)):
    if request.time_end_minutes > request.prehistory_minutes - 15:
        raise HTTPException(status_code=400, detail=f"time_end muss < prehistory - 15 sein")
    if request.time_start_minutes >= request.time_end_minutes:
        raise HTTPException(status_code=400, detail="time_start muss < time_end sein")
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO indicator_sets (owner_id, name, description, coin_group_id, search_date_from, search_date_to,
                search_percent_min, search_percent_max, search_direction, search_duration_minutes, prehistory_minutes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING set_id""",
                (current_user['user_id'], request.name, request.description, request.coin_group_id,
                 request.search_date_from, request.search_date_to, request.search_percent_min, request.search_percent_max,
                 request.search_direction, request.search_duration_minutes, request.prehistory_minutes))
            set_id = cur.fetchone()['set_id']
            cur.execute("""INSERT INTO indicator_items (set_id, indicator_type, condition_operator, condition_value, condition_value2,
                time_start_minutes, time_end_minutes, aggregator, color, reference_start_minutes, reference_end_minutes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING item_id""",
                (set_id, request.indicator_type, request.condition_operator, request.condition_value, request.condition_value2,
                 request.time_start_minutes, request.time_end_minutes, request.aggregator, request.color,
                 request.reference_start_minutes, request.reference_end_minutes))
            item_id = cur.fetchone()['item_id']
            conn.commit()
    return {"message": "Set with indicator created", "set_id": set_id, "item_id": item_id, "prehistory_minutes": request.prehistory_minutes}

@router.get("/sets/{set_id}")
async def get_indicator_set(set_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT s.*, u.email as owner_email, cg.name as coin_group_name FROM indicator_sets s
                LEFT JOIN users u ON s.owner_id = u.user_id LEFT JOIN coin_groups cg ON s.coin_group_id = cg.group_id
                WHERE s.set_id = %s AND (s.is_public = TRUE OR s.owner_id = %s)""", (set_id, current_user['user_id']))
            indicator_set = cur.fetchone()
            if not indicator_set:
                raise HTTPException(status_code=404, detail="Set not found")
            cur.execute("SELECT * FROM indicator_items WHERE set_id = %s ORDER BY time_start_minutes ASC, item_id", (set_id,))
            items = cur.fetchall()
            coins = []
            if indicator_set.get('coin_group_id'):
                cur.execute("SELECT symbol FROM coin_group_members WHERE group_id = %s", (indicator_set['coin_group_id'],))
                coins = [row['symbol'] for row in cur.fetchall()]
    return {"set": indicator_set, "items": items, "coins": coins,
            "search_params": {"coin_group_id": indicator_set.get('coin_group_id'), "coin_group_name": indicator_set.get('coin_group_name'),
                "date_from": indicator_set.get('search_date_from'), "date_to": indicator_set.get('search_date_to'),
                "percent_min": indicator_set.get('search_percent_min'), "percent_max": indicator_set.get('search_percent_max'),
                "direction": indicator_set.get('search_direction'), "duration_minutes": indicator_set.get('search_duration_minutes'),
                "prehistory_minutes": indicator_set.get('prehistory_minutes'), "coins": coins}}

@router.delete("/sets/{set_id}")
async def delete_indicator_set(set_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id FROM indicator_sets WHERE set_id = %s", (set_id,))
            s = cur.fetchone()
            if not s or s["owner_id"] != current_user["user_id"]:
                raise HTTPException(status_code=404, detail="Set not found")
            cur.execute("DELETE FROM indicator_sets WHERE set_id = %s", (set_id,))
            conn.commit()
    return {"message": "Set deleted"}

# === ITEMS CRUD ===
@router.post("/sets/{set_id}/items")
async def add_indicator_item(set_id: int, request: IndicatorItemCreate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id, is_locked, prehistory_minutes FROM indicator_sets WHERE set_id = %s", (set_id,))
            s = cur.fetchone()
            if not s:
                raise HTTPException(status_code=404, detail="Set not found")
            if s['is_locked']:
                raise HTTPException(status_code=400, detail="Set is locked")
            if s['owner_id'] != current_user['user_id']:
                raise HTTPException(status_code=403, detail="Not your set")
            prehistory = s.get('prehistory_minutes')
            if request.time_start_minutes < 0:
                raise HTTPException(status_code=400, detail="time_start must be >= 0")
            if prehistory and request.time_end_minutes > prehistory - 15:
                raise HTTPException(status_code=400, detail=f"time_end must be < {prehistory - 15}")
            pattern_data_json = json.dumps(request.pattern_data) if request.pattern_data else None
            cur.execute("""INSERT INTO indicator_items (set_id, indicator_type, condition_operator, condition_value, condition_value2,
                time_start_minutes, time_end_minutes, aggregator, color, reference_start_minutes, reference_end_minutes,
                pattern_type, pattern_data, pattern_count, pattern_consecutive)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING item_id""",
                (set_id, request.field, request.operation, request.value, request.value2, request.time_start_minutes,
                 request.time_end_minutes, request.aggregator, request.color, request.reference_start_minutes,
                 request.reference_end_minutes, request.pattern_type, pattern_data_json, request.pattern_count, request.pattern_consecutive))
            item_id = cur.fetchone()['item_id']
            conn.commit()
    return {"message": "Indicator added", "item_id": item_id, "set_id": set_id}

@router.delete("/sets/{set_id}/items/{item_id}")
async def delete_indicator_item(set_id: int, item_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id FROM indicator_sets WHERE set_id = %s", (set_id,))
            s = cur.fetchone()
            if not s or s['owner_id'] != current_user['user_id']:
                raise HTTPException(status_code=403, detail="Not your set")
            cur.execute("DELETE FROM indicator_items WHERE item_id = %s AND set_id = %s RETURNING item_id", (item_id, set_id))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Item not found")
            conn.commit()
    return {"message": "Item deleted"}

@router.put("/sets/{set_id}/items/{item_id}/toggle")
async def toggle_indicator_item(set_id: int, item_id: int, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT i.is_active, s.owner_id FROM indicator_items i
                JOIN indicator_sets s ON i.set_id = s.set_id WHERE i.item_id = %s AND i.set_id = %s""", (item_id, set_id))
            item = cur.fetchone()
            if not item or item['owner_id'] != current_user['user_id']:
                raise HTTPException(status_code=404, detail="Item not found")
            new_active = not item['is_active']
            cur.execute("UPDATE indicator_items SET is_active = %s WHERE item_id = %s", (new_active, item_id))
            conn.commit()
    return {"item_id": item_id, "is_active": new_active}

@router.put("/items/{item_id}")
async def update_indicator_item(item_id: int, update: IndicatorItemUpdate, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT i.*, s.owner_id FROM indicator_items i
                JOIN indicator_sets s ON i.set_id = s.set_id WHERE i.item_id = %s""", (item_id,))
            item = cur.fetchone()
            if not item or item['owner_id'] != current_user['user_id']:
                raise HTTPException(status_code=404, detail="Item not found")
            updates, values = [], []
            if update.pattern_data is not None:
                updates.append("pattern_data = %s")
                values.append(json.dumps(update.pattern_data))
            if update.is_active is not None:
                updates.append("is_active = %s")
                values.append(update.is_active)
            if update.time_start_minutes is not None:
                updates.append("time_start_minutes = %s")
                values.append(update.time_start_minutes)
            if update.time_end_minutes is not None:
                updates.append("time_end_minutes = %s")
                values.append(update.time_end_minutes)
            if updates:
                values.append(item_id)
                cur.execute(f"UPDATE indicator_items SET {', '.join(updates)} WHERE item_id = %s", values)
                conn.commit()
    return {"item_id": item_id, "updated": True}

# === BACKSEARCH ===
@router.post("/sets/{set_id}/backsearch")
async def backsearch(set_id: int, request: CounterSearchRequest, current_user: dict = Depends(get_current_user)):
    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT s.*, g.name as coin_group_name FROM indicator_sets s
                LEFT JOIN coin_groups g ON s.coin_group_id = g.group_id
                WHERE s.set_id = %s AND (s.is_public = TRUE OR s.owner_id = %s)""", (set_id, current_user['user_id']))
            indicator_set = cur.fetchone()
            if not indicator_set:
                raise HTTPException(status_code=404, detail="Set not found")
            cur.execute("SELECT * FROM indicator_items WHERE set_id = %s AND is_active = TRUE ORDER BY time_start_minutes ASC", (set_id,))
            indicators = [dict(row) for row in cur.fetchall()]
            coins = []
            if indicator_set.get('coin_group_id'):
                cur.execute("SELECT symbol FROM coin_group_members WHERE group_id = %s", (indicator_set['coin_group_id'],))
                coins = [row['symbol'] for row in cur.fetchall()]
    if not indicators:
        raise HTTPException(status_code=400, detail="Set has no indicators")
    if len(indicators) < 3:
        raise HTTPException(status_code=400, detail="Backsearch needs at least 3 indicators")
    
    prehistory = indicator_set.get('prehistory_minutes') or 720
    duration_minutes = indicator_set.get('search_duration_minutes') or 120
    target_percent = float(indicator_set.get('search_percent_min') or 5)
    direction = indicator_set.get('search_direction', 'up')
    
    try:
        start_dt = BERLIN_TZ.localize(datetime.strptime(request.start_date, "%Y-%m-%d"))
        end_dt = BERLIN_TZ.localize(datetime.strptime(request.end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    symbols = request.symbols or coins
    if not symbols:
        raise HTTPException(status_code=400, detail="No coins defined")
    
    results, green_count, grey_count, red_count = [], 0, 0, 0
    scan_interval = request.scan_interval_minutes
    
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            for symbol in symbols:
                scan_time = start_dt
                while scan_time < end_dt:
                    zero_point = scan_time
                    event_start = zero_point + timedelta(minutes=prehistory)
                    if event_start > end_dt:
                        break
                    all_matched = _check_all_indicators(cur, symbol, zero_point, prehistory, indicators)
                    if all_matched:
                        event_end = event_start + timedelta(minutes=duration_minutes)
                        cur.execute("""SELECT open, high, low FROM klines WHERE symbol = %s AND interval = '1m'
                            AND open_time >= %s AND open_time < %s ORDER BY open_time""", (symbol, event_start, event_end))
                        event_candles = cur.fetchall()
                        if event_candles and len(event_candles) >= 2:
                            start_price = float(event_candles[0]['open'])
                            if direction == 'up':
                                max_price = max(float(c['high']) for c in event_candles)
                                actual_pct = ((max_price - start_price) / start_price) * 100
                            else:
                                min_price = min(float(c['low']) for c in event_candles)
                                actual_pct = ((start_price - min_price) / start_price) * 100
                            if actual_pct >= target_percent:
                                category, green_count = 'GREEN', green_count + 1
                            elif actual_pct >= 1:
                                category, grey_count = 'GREY', grey_count + 1
                            else:
                                category, red_count = 'RED', red_count + 1
                            results.append({"symbol": symbol, "zero_point": zero_point.isoformat(), "event_start": event_start.isoformat(),
                                "event_end": event_end.isoformat(), "actual_percent": round(actual_pct, 2), "target_percent": target_percent, "category": category})
                        scan_time += timedelta(minutes=duration_minutes)
                    else:
                        scan_time += timedelta(minutes=scan_interval)
    
    total = green_count + grey_count + red_count
    probability = round(green_count / total * 100, 1) if total > 0 else 0
    return {"set_id": set_id, "set_name": indicator_set.get('name'), "target_percent": target_percent, "duration_minutes": duration_minutes,
        "prehistory_minutes": prehistory, "direction": direction, "coins_scanned": len(symbols), "scan_interval_minutes": scan_interval,
        "results": sorted(results, key=lambda x: x["actual_percent"], reverse=True),
        "statistics": {"total_matches": total, "green_count": green_count, "green_percent": round(green_count / total * 100, 1) if total > 0 else 0,
            "grey_count": grey_count, "grey_percent": round(grey_count / total * 100, 1) if total > 0 else 0,
            "red_count": red_count, "red_percent": round(red_count / total * 100, 1) if total > 0 else 0, "probability": probability}}

@router.post("/sets/{set_id}/counter-search")
async def counter_search_alias(set_id: int, request: CounterSearchRequest, current_user: dict = Depends(get_current_user)):
    return await backsearch(set_id, request, current_user)

def _check_all_indicators(cur, symbol, zero_point, prehistory, indicators):
    for ind in indicators:
        indicator_type = ind.get('indicator_type', 'close')
        if indicator_type == 'candle_pattern':
            if not _check_candle_pattern(cur, symbol, zero_point, ind):
                return False
        else:
            if not _check_normal_indicator(cur, symbol, zero_point, ind):
                return False
    return True

def _check_candle_pattern(cur, symbol, zero_point, ind):
    pattern_data = ind.get('pattern_data', {})
    if isinstance(pattern_data, str):
        pattern_data = json.loads(pattern_data)
    if not pattern_data:
        return True
    time_start = ind.get('time_start_minutes', 0)
    time_end = ind.get('time_end_minutes', 600)
    aggregator = ind.get('aggregator', '15m')
    pattern_start_dt = zero_point + timedelta(minutes=time_start)
    pattern_end_dt = zero_point + timedelta(minutes=time_end)
    table = 'klines' if aggregator == '1m' else f'agg_{aggregator}'
    time_col = 'open_time' if table == 'klines' else 'bucket'
    interval_filter = "AND interval = '1m'" if table == 'klines' else ""
    cur.execute(f"SELECT open, close FROM {table} WHERE symbol = %s {interval_filter} AND {time_col} >= %s AND {time_col} <= %s ORDER BY {time_col} ASC",
        (symbol, pattern_start_dt, pattern_end_dt))
    candles = cur.fetchall()
    if not candles:
        return False
    for pos_str, expected_color in pattern_data.items():
        pos = int(pos_str)
        if pos >= len(candles):
            return False
        candle = candles[pos]
        actual_color = 'green' if candle['close'] >= candle['open'] else 'red'
        if actual_color != expected_color:
            return False
    return True

def _check_normal_indicator(cur, symbol, zero_point, ind):
    time_start = ind.get('time_start_minutes', 0)
    time_end = ind.get('time_end_minutes', 600)
    aggregator = ind.get('aggregator', '1m')
    operation = ind.get('condition_operator', '>')
    value = float(ind.get('condition_value') or 0)
    value2 = float(ind.get('condition_value2') or value) if ind.get('condition_value2') else None
    indicator_type = ind.get('indicator_type', 'close')
    marked_start_dt = zero_point + timedelta(minutes=time_start)
    marked_end_dt = zero_point + timedelta(minutes=time_end)
    table = 'klines' if aggregator == '1m' else f'agg_{aggregator}'
    time_col = 'open_time' if table == 'klines' else 'bucket'
    interval_filter = "AND interval = '1m'" if table == 'klines' else ""
    db_column = {'close': 'close', 'volume': 'volume', 'trades': 'trades', 'high': 'high', 'low': 'low', 'open': 'open'}.get(indicator_type, 'close')
    cur.execute(f"SELECT {db_column} as base_val FROM {table} WHERE symbol = %s {interval_filter} AND {time_col} >= %s ORDER BY {time_col} ASC LIMIT 1",
        (symbol, zero_point))
    base_result = cur.fetchone()
    if not base_result or not base_result['base_val']:
        return False
    base_val = float(base_result['base_val'])
    if base_val == 0:
        return False
    cur.execute(f"SELECT MIN({db_column}) as min_val, MAX({db_column}) as max_val FROM {table} WHERE symbol = %s {interval_filter} AND {time_col} >= %s AND {time_col} <= %s",
        (symbol, marked_start_dt, marked_end_dt))
    marked_result = cur.fetchone()
    if not marked_result or marked_result['min_val'] is None:
        return False
    normalized_min = ((float(marked_result['min_val']) - base_val) / base_val) * 100
    normalized_max = ((float(marked_result['max_val']) - base_val) / base_val) * 100
    if operation == '>':
        return normalized_min > value
    elif operation == '<':
        return normalized_max < value
    elif operation == '>=':
        return normalized_min >= value
    elif operation == '<=':
        return normalized_max <= value
    elif operation == 'between' and value2 is not None:
        lower, upper = min(value, value2), max(value, value2)
        return normalized_min >= lower and normalized_max <= upper
    return False

# === VALIDATE ===
class ValidateSetRequest(BaseModel):
    indicators: List[dict]
    main_search_events: List[dict]
    target_percent: float = 5.0
    prehistory_minutes: int = 720

@router.post("/validate")
async def validate_indicator_set(request: ValidateSetRequest, current_user: dict = Depends(get_current_user)):
    if not request.indicators or not request.main_search_events:
        return {"matched_count": 0, "ok_count": 0, "ok_percent": 0, "grey_count": 0, "grey_percent": 0,
            "fail_count": 0, "fail_percent": 0, "false_positives": 0, "hit_rate": 0, "decision": "FAIL", "message": "No indicators or events"}
    cascade_result = _apply_cascade(request.main_search_events, request.indicators, request.prehistory_minutes)
    matched_events = cascade_result
    ok_count, grey_count, fail_count = 0, 0, 0
    for event in matched_events:
        change_pct = abs(float(event.get('change_percent', 0)))
        if change_pct >= request.target_percent:
            ok_count += 1
        elif change_pct >= 1.0:
            grey_count += 1
        else:
            fail_count += 1
    matched_count = len(matched_events)
    main_count = len(request.main_search_events)
    ok_pct = (ok_count / matched_count * 100) if matched_count > 0 else 0
    grey_pct = (grey_count / matched_count * 100) if matched_count > 0 else 0
    fail_pct = (fail_count / matched_count * 100) if matched_count > 0 else 0
    hit_rate = (ok_count / main_count * 100) if main_count > 0 else 0
    decision = "FAIL" if hit_rate < 20 else "OK"
    message = f"Hit Rate {hit_rate:.0f}%"
    return {"matched_count": matched_count, "ok_count": ok_count, "ok_percent": round(ok_pct, 1),
        "grey_count": grey_count, "grey_percent": round(grey_pct, 1), "fail_count": fail_count, "fail_percent": round(fail_pct, 1),
        "false_positives": 0, "hit_rate": round(hit_rate, 1), "decision": decision, "message": message}

def _apply_cascade(events, indicators, prehistory):
    if not events or not indicators:
        return events
    filtered = events.copy()
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            for indicator in indicators:
                if not filtered:
                    break
                indicator_type = indicator.get('indicator_type') or indicator.get('field', 'close')
                newly_filtered = []
                for event in filtered:
                    symbol = event.get('symbol')
                    event_start_str = event.get('event_start')
                    if not symbol or not event_start_str:
                        continue
                    try:
                        if isinstance(event_start_str, str):
                            event_start_dt = datetime.fromisoformat(event_start_str.replace('Z', '+00:00'))
                        else:
                            event_start_dt = event_start_str
                        zero_point = event_start_dt - timedelta(minutes=prehistory)
                    except:
                        continue
                    if indicator_type == 'candle_pattern':
                        if _check_candle_pattern(cur, symbol, zero_point, indicator):
                            newly_filtered.append(event)
                    else:
                        if _check_normal_indicator(cur, symbol, zero_point, indicator):
                            newly_filtered.append(event)
                filtered = newly_filtered
    return filtered
