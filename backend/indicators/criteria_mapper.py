"""Mappt ChartView-Kriterien auf das indicator_items DB-Schema.
Keine Fallbacks: wenn Pflichtfelder fehlen, wird Exception geworfen."""

import json
from search.predictor_settings import AGGREGATOR_STRINGS




# Alle erlaubten indicator_type / field Werte (zentrale Quelle, keine Hardcodes sonst)
ALLOWED_INDICATOR_FIELDS = frozenset([
    'close', 'open', 'high', 'low', 'volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'quote_asset_volume',
    'funding', 'open_interest', 'premium',
    'oracle_px', 'mark_px', 'mid_px',
    'bbo_bid_px', 'bbo_ask_px', 'bbo_bid_sz', 'bbo_ask_sz',
    'spread_bps', 'book_imbalance_5', 'book_depth_5',
])

def map_kind_to_db(crit):
    """Liefert (indicator_type, condition_operator, pattern_data) fuer DB-Insert."""
    if 'kind' not in crit.__fields__ if hasattr(crit, '__fields__') else 'kind' not in dict(crit):
        raise ValueError("Kriterium hat kein 'kind'")

    kind = crit.kind

    if kind == 'pattern':
        if not crit.pattern_id:
            raise ValueError("pattern-Kriterium braucht pattern_id")
        return 'candle_pattern', 'match', {'pattern_id': crit.pattern_id}

    if kind == 'sequence':
        if not crit.sequence:
            raise ValueError("sequence-Kriterium braucht sequence-String")
        return 'candle_pattern', 'match', {'sequence': crit.sequence}

    if kind == 'value':
        if crit.value is None:
            raise ValueError("value-Kriterium braucht value")
        if crit.field not in ALLOWED_INDICATOR_FIELDS:
            raise ValueError(f"Unbekanntes indicator-field: {crit.field}")
        return crit.field, '=', None

    if kind == 'range':
        if crit.value is None or crit.value2 is None:
            raise ValueError("range-Kriterium braucht value und value2")
        return crit.field, 'between', None

    if kind in ('slope', 'ratio'):
        if crit.value is None:
            raise ValueError(f"{kind}-Kriterium braucht value")
        return crit.field, kind, None

    raise ValueError(f"Unbekannter kind: {kind}")


def resolve_time_window(crit):
    """Ermittelt (time_start_minutes, time_end_minutes) aus einem Kriterium.
    Kein Fallback: mindestens time_offset ODER time_offset_from/to muss gesetzt sein."""
    if crit.time_offset_from is not None and crit.time_offset_to is not None:
        return crit.time_offset_from, crit.time_offset_to

    if crit.time_offset is None:
        raise ValueError("Kriterium braucht time_offset oder time_offset_from/to")

    if crit.time_offset2 is not None:
        return crit.time_offset, crit.time_offset2

    return crit.time_offset, crit.time_offset


def resolve_aggregator(timeframe_minutes):
    """Aggregator-String fuer DB. Exception wenn timeframe unbekannt."""
    if timeframe_minutes not in AGGREGATOR_STRINGS:
        raise ValueError(f"Unbekanntes candle_timeframe: {timeframe_minutes}")
    return AGGREGATOR_STRINGS[timeframe_minutes]


def serialize_fuzzy(fuzzy_model):
    """Serialisiert Fuzzy-Pydantic-Modell zu JSON-dict. Exception wenn None."""
    if fuzzy_model is None:
        raise ValueError("Fuzzy-Konfiguration fehlt")
    return fuzzy_model.model_dump()
