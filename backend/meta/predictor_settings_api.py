"""Liefert alle Predictor-Settings an das Frontend. Single Source of Truth."""
from fastapi import APIRouter

from search.predictor_settings import (
    SCAN_SETTINGS, INITIAL_POINT_SETTINGS, PATTERN_THRESHOLDS,
    FUZZY_DEFAULTS, AGG_TABLE_MAP, SET_DEFAULTS, AGGREGATOR_STRINGS
)

router = APIRouter(prefix="/api/v1/meta", tags=["meta-settings"])


@router.get("/predictor-settings")
async def get_predictor_settings():
    """Alle Backend-Defaults fuer den Predictor. Frontend laedt einmal beim Start."""
    return {
        "scan": SCAN_SETTINGS,
        "initial_point": INITIAL_POINT_SETTINGS,
        "pattern_thresholds": PATTERN_THRESHOLDS,
        "fuzzy_defaults": FUZZY_DEFAULTS,
        "agg_table_map": AGG_TABLE_MAP,
        "aggregator_strings": AGGREGATOR_STRINGS,
        "set_defaults": SET_DEFAULTS,
        "supported_kinds": ["value", "range", "slope", "ratio", "pattern", "sequence"],
        "supported_fields": ["open", "high", "low", "close", "volume", "trades",
                              "body_pct", "range_pct",
                              "open_pct_dayopen", "high_pct_dayopen",
                              "low_pct_dayopen", "close_pct_dayopen"],
    }
