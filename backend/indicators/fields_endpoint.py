"""Liefert die Liste erlaubter Indikator-Felder an den Frontend-Criterion-Editor.
Keine Hardcodes — single source in criteria_mapper.ALLOWED_INDICATOR_FIELDS."""
from fastapi import APIRouter, Depends
from auth.auth import get_current_user
from .criteria_mapper import ALLOWED_INDICATOR_FIELDS

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators-meta"])

# Lesbare Labels fuer das Frontend — eine Zeile pro Feld
FIELD_LABELS = {
    'close': 'Close',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'volume': 'Volume (Coin)',
    'trades': 'Trade-Count',
    'taker_buy_base': 'Aggressor-Buy Volume (Coin)',
    'taker_buy_quote': 'Aggressor-Buy Volume (USD)',
    'quote_asset_volume': 'Volume (USD)',
    'funding': 'Funding-Rate',
    'open_interest': 'Open Interest',
    'premium': 'Premium (Perp vs Oracle)',
    'oracle_px': 'Oracle-Preis',
    'mark_px': 'Mark-Preis',
    'mid_px': 'Mid-Preis',
    'bbo_bid_px': 'Best Bid Preis',
    'bbo_ask_px': 'Best Ask Preis',
    'bbo_bid_sz': 'Bid-Liquiditaet Top',
    'bbo_ask_sz': 'Ask-Liquiditaet Top',
    'spread_bps': 'Spread (bps)',
    'book_imbalance_5': 'Book-Imbalance Top5',
    'book_depth_5': 'Book-Depth Top5',
}

FIELD_GROUPS = [
    {"group": "OHLCV", "fields": ['close', 'open', 'high', 'low', 'volume', 'trades', 'quote_asset_volume']},
    {"group": "Order-Flow", "fields": ['taker_buy_base', 'taker_buy_quote']},
    {"group": "Derivatives", "fields": ['funding', 'open_interest', 'premium', 'mark_px', 'oracle_px', 'mid_px']},
    {"group": "Mikrostruktur", "fields": ['bbo_bid_px', 'bbo_ask_px', 'bbo_bid_sz', 'bbo_ask_sz',
                                          'spread_bps', 'book_imbalance_5', 'book_depth_5']},
]


@router.get("/fields")
async def list_fields(current_user: dict = Depends(get_current_user)):
    """Liefert alle waehlbaren indicator_type Fields mit Labels + Gruppen."""
    missing = [f for g in FIELD_GROUPS for f in g["fields"] if f not in ALLOWED_INDICATOR_FIELDS]
    if missing:
        raise RuntimeError(f"FIELD_GROUPS referenziert unbekannte Fields: {missing}")
    return {
        "fields": [{"name": f, "label": FIELD_LABELS[f]} for f in sorted(ALLOWED_INDICATOR_FIELDS)],
        "groups": FIELD_GROUPS,
    }
