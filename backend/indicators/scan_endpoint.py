"""Scan-Endpoint: laedt gespeichertes Set, ruft blind_scanner auf."""
import json
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import pytz

from shared.database import get_app_db, get_coins_db
from auth.auth import get_current_user
from search.predictor_settings import SCAN_SETTINGS, INITIAL_POINT_SETTINGS
from search.candle_aggregator import load_candles_for_symbol
from search.blind_scanner import scan_symbol, attach_pct_labels
from .item_to_criterion import item_to_criterion, initial_item_to_point

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-scan"])
BERLIN_TZ = pytz.timezone('Europe/Berlin')


class ScanRequest(BaseModel):
    period_days: int  # Pflicht — kein Default
    symbols: Optional[list] = None  # None = alle HL-Coins


@router.post("/sets/{set_id}/scan")
async def scan_set(set_id: int, request: ScanRequest, current_user: dict = Depends(get_current_user)):
    """Laedt Set + Items, ruft blind_scanner, gibt Treffer mit pct-Label zurueck."""
    if request.period_days <= 0 or request.period_days > SCAN_SETTINGS['max_period_days']:
        raise HTTPException(400, f"period_days muss 1..{SCAN_SETTINGS['max_period_days']} sein")

    with get_app_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT * FROM indicator_sets
                           WHERE set_id = %s AND (is_public = TRUE OR owner_id = %s)""",
                        (set_id, current_user['user_id']))
            s = cur.fetchone()
            if not s:
                raise HTTPException(404, "Set nicht gefunden")

            cur.execute("""SELECT * FROM indicator_items
                           WHERE set_id = %s AND is_active = TRUE
                           ORDER BY sort_order, item_id""", (set_id,))
            items = [dict(r) for r in cur.fetchall()]

            coin_group_id = s.get('coin_group_id')
            coins = []
            if coin_group_id:
                cur.execute("SELECT symbol FROM coin_group_members WHERE group_id = %s", (coin_group_id,))
                coins = [r['symbol'] for r in cur.fetchall()]

    if not items:
        raise HTTPException(400, "Set hat keine aktiven Items")

    initial_points = []
    criteria = []
    for it in items:
        if it.get('is_initial_point'):
            initial_points.append(initial_item_to_point(it))
        else:
            criteria.append(item_to_criterion(it))

    ip_config_raw = s.get('initial_point_config')
    if ip_config_raw is None:
        raise HTTPException(500, "Set hat keine initial_point_config")
    ip_config = ip_config_raw if isinstance(ip_config_raw, dict) else json.loads(ip_config_raw)

    search_pattern = s.get('search_pattern')
    if search_pattern is None:
        raise HTTPException(500, "Set hat kein search_pattern (global_fuzzy fehlt)")
    sp = search_pattern if isinstance(search_pattern, dict) else json.loads(search_pattern)
    if 'global_fuzzy' not in sp:
        raise HTTPException(500, "search_pattern.global_fuzzy fehlt")
    global_fuzzy = sp['global_fuzzy']

    candle_tf = s.get('candle_timeframe')
    if candle_tf is None:
        raise HTTPException(500, "Set hat kein candle_timeframe")

    end_date = datetime.now(BERLIN_TZ)
    start_date = end_date - timedelta(days=request.period_days)

    padding = INITIAL_POINT_SETTINGS['window_minutes']
    for c in criteria:
        if c.get('time_offset_from') is not None:
            padding = max(padding, abs(c['time_offset_from']))
    padded_start = start_date - timedelta(minutes=padding)

    symbols = request.symbols or coins
    if not symbols:
        raise HTTPException(400, "Keine Coins angegeben und Set hat keine Coin-Gruppe")

    all_hits = []
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            for symbol in symbols:
                candles = load_candles_for_symbol(
                    symbol, padded_start, end_date, int(candle_tf), cur
                )
                if not candles:
                    continue
                hits = scan_symbol(
                    symbol, candles, criteria,
                    {'config': ip_config, 'points': initial_points},
                    global_fuzzy
                )
                hits = [h for h in hits if h['event_time'] >= start_date]
                all_hits.extend(hits)

            attach_pct_labels(all_hits, cur)

    all_hits.sort(key=lambda x: x['match_score'], reverse=True)

    duration = s.get('duration_minutes')
    if duration is None:
        raise HTTPException(500, "Set hat kein duration_minutes")
    target = float(s.get('target_percent') or 0)

    max_results = SCAN_SETTINGS['max_results']
    results = []
    green = grey = red = 0
    for h in all_hits[:max_results]:
        pct_label = h.get('pct_labels', {}).get(f"pct_{duration}m")
        actual = float(pct_label) if pct_label is not None else None
        cat = None
        if actual is not None:
            abs_pct = abs(actual)
            if abs_pct >= target:
                cat = 'GREEN'; green += 1
            elif abs_pct >= 1.0:
                cat = 'GREY'; grey += 1
            else:
                cat = 'RED'; red += 1

        results.append({
            'symbol': h['symbol'],
            'event_time': h['event_time'].isoformat(),
            'match_score': h['match_score'],
            'ip_score': h['ip_score'],
            'criteria_scores': h.get('criteria_scores', []),
            'pct_label': actual,
            'category': cat,
            'pct_labels': h.get('pct_labels', {}),
        })

    total = green + grey + red
    return {
        'set_id': set_id,
        'set_name': s['name'],
        'duration_minutes': duration,
        'target_percent': target,
        'period_days': request.period_days,
        'symbols_scanned': len(symbols),
        'total_found': len(all_hits),
        'results': results,
        'statistics': {
            'total_with_label': total,
            'green': green, 'grey': grey, 'red': red,
            'green_percent': round(green / total * 100, 1) if total else 0,
            'grey_percent': round(grey / total * 100, 1) if total else 0,
            'red_percent': round(red / total * 100, 1) if total else 0,
        }
    }
