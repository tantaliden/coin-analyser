"""
COUNTER SEARCH V2 — Blind-Scan ueber klines/aggs

Scannt ALLE Zeitpunkte im Zeitraum (nicht kline_metrics).
kline_metrics wird nur nachtraeglich als Label angehaengt.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
import pytz

from shared.database import get_coins_db, get_app_db
from auth.auth import get_current_user
from .predictor_settings import SCAN_SETTINGS, INITIAL_POINT_SETTINGS, FUZZY_DEFAULTS, AGG_TABLE_MAP
from .candle_aggregator import load_candles_for_symbol
from .blind_scanner import scan_symbol, attach_pct_labels

BERLIN_TZ = pytz.timezone('Europe/Berlin')

router = APIRouter(prefix="/api/v1/search/counter", tags=["counter-search"])


# --- Request Models ---

class FuzzySettings(BaseModel):
    valueTolerance: float = FUZZY_DEFAULTS['value_tolerance_pct']
    timeTolerance: float = FUZZY_DEFAULTS['time_tolerance_min']
    slopeTolerance: float = FUZZY_DEFAULTS['slope_tolerance_pct']
    ratioTolerance: float = FUZZY_DEFAULTS['ratio_tolerance_pct']
    useRange: bool = False
    rangeMin: Optional[float] = None
    rangeMax: Optional[float] = None


class Criterion(BaseModel):
    kind: str  # value|range|slope|ratio|pattern|sequence
    field: str = 'close'
    value: Optional[float] = None
    value2: Optional[float] = None
    field2: Optional[str] = None
    time_offset: int = 0
    time_offset2: Optional[int] = None
    time_offset_from: Optional[int] = None  # Fenster-Start (fuer Prae-Event)
    time_offset_to: Optional[int] = None    # Fenster-Ende
    pattern_id: Optional[str] = None
    sequence: Optional[str] = None  # z.B. 'GGRG'
    fuzzy: Optional[FuzzySettings] = None


class InitialPoint(BaseModel):
    kind: str  # value|pattern|sequence|slope|ratio
    field: str = 'close'
    value: Optional[float] = None
    value2: Optional[float] = None
    field2: Optional[str] = None
    pattern_id: Optional[str] = None
    sequence: Optional[str] = None
    fixed_offset: Optional[int] = None  # Fest bei -X Minuten (None = variabel im Fenster)
    fuzzy: Optional[FuzzySettings] = None


class InitialPointSet(BaseModel):
    points: List[InitialPoint] = []
    match_mode: str = 'all'
    match_threshold: int = 0
    enforce_sequence: bool = False
    window_minutes: int = INITIAL_POINT_SETTINGS['window_minutes']


class CounterSearchRequest(BaseModel):
    criteria: List[Criterion]
    initial_points: InitialPointSet = InitialPointSet()
    global_fuzzy: FuzzySettings = FuzzySettings()
    period_days: int = SCAN_SETTINGS['default_period_days']
    duration_minutes: int = 120  # Aus Primary Search uebernommen
    candle_timeframe: int = 1    # Minuten (1, 5, 15, 30, 60...)
    direction: str = 'both'
    hl_only: bool = True
    match_mode: str = 'all'
    match_threshold: int = 0


@router.post("/find")
async def counter_search(request: CounterSearchRequest, current_user: dict = Depends(get_current_user)):
    """Blind-Scan: Prueft Indikator-Set gegen ALLE Zeitpunkte in klines/aggs."""
    if not request.criteria and not request.initial_points.points:
        raise HTTPException(400, "Keine Kriterien oder Initialpunkte angegeben")

    end_date = datetime.now(BERLIN_TZ)
    start_date = end_date - timedelta(days=request.period_days)
    # Padding: Initialpunkt-Fenster + groesstes Prae-Event-Fenster vorher laden
    max_lookback = INITIAL_POINT_SETTINGS['window_minutes']
    for c in request.criteria:
        if c.time_offset_from is not None:
            max_lookback = max(max_lookback, abs(c.time_offset_from))
        max_lookback = max(max_lookback, abs(c.time_offset) + 60)
    padded_start = start_date - timedelta(minutes=max_lookback)

    # HL-Filter
    hl_symbols = set()
    if request.hl_only:
        with get_app_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM coin_info WHERE hl_sz_decimals IS NOT NULL")
                hl_symbols = set(row['symbol'] for row in cur.fetchall())

    # Kriterien als dicts (fuer den Scanner)
    criteria_dicts = [c.model_dump() for c in request.criteria]
    ip_config = {
        'config': {
            'match_mode': request.initial_points.match_mode,
            'match_threshold': request.initial_points.match_threshold or len(request.initial_points.points),
            'enforce_sequence': request.initial_points.enforce_sequence,
            'window_minutes': request.initial_points.window_minutes,
        },
        'points': [p.model_dump() for p in request.initial_points.points],
    }
    global_fuzzy = request.global_fuzzy.model_dump()

    all_hits = []

    with get_coins_db() as conn:
        with conn.cursor() as cur:
            # Symbole ermitteln
            table = AGG_TABLE_MAP.get(request.candle_timeframe, 'klines')
            cur.execute(f"""
                SELECT DISTINCT symbol FROM {table}
                WHERE open_time >= %s AND open_time < %s
                ORDER BY symbol
            """, (padded_start, end_date))
            symbols = [r['symbol'] for r in cur.fetchall()]
            if request.hl_only:
                symbols = [s for s in symbols if s in hl_symbols]

            for symbol in symbols:
                candles = load_candles_for_symbol(
                    symbol, padded_start, end_date,
                    request.candle_timeframe, cur
                )
                if not candles:
                    continue

                hits = scan_symbol(
                    symbol, candles, criteria_dicts, ip_config,
                    global_fuzzy, scan_step_minutes=request.candle_timeframe
                )

                # Nur Treffer im eigentlichen Scan-Zeitraum (ohne Padding)
                hits = [h for h in hits if h['event_time'] >= start_date]
                all_hits.extend(hits)

            # pct-Labels nachtraeglich anhaengen
            attach_pct_labels(all_hits, cur)

    # Sortieren nach Score
    all_hits.sort(key=lambda x: x['match_score'], reverse=True)

    # Response aufbereiten
    max_results = SCAN_SETTINGS['max_results']
    results = []
    for h in all_hits[:max_results]:
        pct_key = f"pct_{request.duration_minutes}m"
        change_pct = h.get('pct_labels', {}).get(pct_key, 0)
        results.append({
            'symbol': h['symbol'],
            'event_start': h['event_time'].isoformat(),
            'change_percent': change_pct,
            'match_score': h['match_score'],
            'matched_count': len([s for s in h.get('criteria_scores', []) if s >= 0.5]),
            'total_criteria': len(h.get('criteria_scores', [])),
            'duration_minutes': request.duration_minutes,
            'pct_labels': h.get('pct_labels', {}),
            'criteria_matched': h.get('criteria_scores', []),
        })

    return {
        'matches': results,
        'total_found': len(all_hits),
        'period_days': request.period_days,
        'criteria_count': len(request.criteria),
        'initial_points_count': len(request.initial_points.points),
        'symbols_scanned': len(symbols),
        'candle_timeframe': request.candle_timeframe,
    }
