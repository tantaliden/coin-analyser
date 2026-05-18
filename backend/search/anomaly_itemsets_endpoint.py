"""Itemset-Endpoint: findet Anomalie-Kombinationen die bei vielen Events GEMEINSAM auftreten.
Nutzt Apriori. Reihenfolge egal."""

from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from shared.database import get_coins_db
from auth.auth import get_current_user
from .candle_aggregator import load_candles_for_symbol
from .anomaly_detector import detect_anomalies, detect_patterns, _normalize_dt
from .itemset_miner import apriori, ItemsetExplosion
from .predictor_settings import ITEMSET_SETTINGS

router = APIRouter(prefix="/api/v1/search", tags=["anomaly"])


class ItemsetEvent(BaseModel):
    symbol: str
    event_start: str


class ItemsetRequest(BaseModel):
    # Keine Defaults — Frontend schickt alle Werte explizit (Config liefert defaults beim Laden)
    events: List[ItemsetEvent]
    prehistory_minutes: int
    candle_timeframe: int
    bucket_minutes: int
    min_support_pct: float            # Min % Events fuer Frequent-Set
    max_set_size: int                 # Max Anomalien pro Set
    min_set_size: int                 # Nur Sets mit >= so vielen Items liefern


def _parse(raw):
    return _normalize_dt(datetime.fromisoformat(raw.replace('Z', '+00:00').replace(' ', 'T')))


@router.post("/anomalies/itemsets")
async def anomaly_itemsets(req: ItemsetRequest, current_user: dict = Depends(get_current_user)):
    if req.candle_timeframe <= 0:
        raise HTTPException(400, "candle_timeframe muss > 0 sein")
    if req.prehistory_minutes <= 0:
        raise HTTPException(400, "prehistory_minutes muss > 0 sein")
    if req.bucket_minutes <= 0:
        raise HTTPException(400, "bucket_minutes muss > 0 sein")
    if not req.events:
        raise HTTPException(400, "events darf nicht leer sein")
    hard_max = ITEMSET_SETTINGS["hard_max_set_size"]
    if not 1 <= req.max_set_size <= hard_max:
        raise HTTPException(400, f"max_set_size muss 1..{hard_max} sein")
    if not 1 <= req.min_set_size <= req.max_set_size:
        raise HTTPException(400, "min_set_size muss 1..max_set_size sein")
    if not 0 < req.min_support_pct <= 100:
        raise HTTPException(400, "min_support_pct muss 0<..<=100 sein")

    # 1) Pro Event: Menge der Anomalie-Keys sammeln (metric, bucket_start)
    transactions = []      # List[set[(metric, bucket_start)]]
    event_refs = []        # parallel zu transactions: {symbol, event_start}
    scanned = 0
    with get_coins_db() as conn:
        with conn.cursor() as cur:
            for idx, ev in enumerate(req.events):
                try:
                    ev_start = _parse(ev.event_start)
                except ValueError as e:
                    raise HTTPException(400, f"Event {idx}: ungueltiges event_start ({e})")
                start = ev_start - timedelta(minutes=req.prehistory_minutes)
                candles = load_candles_for_symbol(ev.symbol, start, ev_start, req.candle_timeframe, cur)
                if not candles:
                    continue
                scanned += 1

                anomalies = set()
                for sug in detect_anomalies(candles, event_start_time=ev_start):
                    open_time = _normalize_dt(sug['open_time'])
                    if open_time is None:
                        continue
                    offset_min = (ev_start - open_time).total_seconds() / 60.0
                    bucket = int(offset_min // req.bucket_minutes) * req.bucket_minutes
                    for an in sug['anomalies']:
                        anomalies.add((an['metric'], bucket))
                for ph in detect_patterns(candles, event_start_time=ev_start):
                    open_time = _normalize_dt(ph['open_time'])
                    if open_time is None:
                        continue
                    offset_min = (ev_start - open_time).total_seconds() / 60.0
                    bucket = int(offset_min // req.bucket_minutes) * req.bucket_minutes
                    anomalies.add((f"pattern:{ph['pattern_id']}", bucket))

                transactions.append(anomalies)
                event_refs.append({"symbol": ev.symbol, "event_start": ev.event_start})

    if scanned == 0:
        return {"scanned_events": 0, "requested_events": len(req.events), "itemsets": [],
                "bucket_minutes": req.bucket_minutes,
                "events_with_anomalies": 0, "total_anomaly_hits": 0, "unique_items": 0,
                "items_meeting_support": 0, "min_support_count": 0}

    # Debug-Statistik: Items und Frequenz 1-itemsets
    events_with_anomalies = sum(1 for t in transactions if t)
    total_hits = sum(len(t) for t in transactions)
    # 1-item support zaehlen
    item_counts = {}
    for t in transactions:
        for it in t:
            item_counts[it] = item_counts.get(it, 0) + 1

    # 2) Apriori — min_count exakt, kein Fallback
    min_count = int(scanned * req.min_support_pct / 100.0)
    if min_count < 1:
        raise HTTPException(400,
            f"min_support_pct={req.min_support_pct}% bei {scanned} Events ergibt min_count<1")
    items_meeting = sum(1 for c in item_counts.values() if c >= min_count)
    top_item_count = max(item_counts.values()) if item_counts else 0
    top_item_support_pct = round(top_item_count / scanned * 100, 1) if scanned > 0 else 0.0
    try:
        frequent = apriori(transactions, min_count, req.max_set_size,
                           ITEMSET_SETTINGS["max_candidates_per_level"])
    except ItemsetExplosion as exc:
        raise HTTPException(400, str(exc))

    # 3) Nur Sets mit >= min_set_size ausgeben
    out = []
    for itemset, cnt in frequent.items():
        if len(itemset) < req.min_set_size:
            continue
        itemset_set = set(itemset)
        # Event-Refs die dieses Set komplett erfuellen
        refs = [event_refs[i] for i, t in enumerate(transactions) if itemset_set <= t]
        items_sorted = sorted(itemset_set, key=lambda it: (it[0], it[1]))
        out.append({
            "items": [
                {"metric": m, "bucket_start_min": b, "bucket_end_min": b + req.bucket_minutes}
                for (m, b) in items_sorted
            ],
            "size": len(itemset),
            "view_count": cnt,
            "frequency_pct": round(cnt / scanned * 100.0, 1),
            "event_refs": refs,
        })
    # Sortierung: Groesse desc, dann Haeufigkeit desc
    out.sort(key=lambda s: (-s["size"], -s["view_count"]))

    return {
        "scanned_events": scanned,
        "requested_events": len(req.events),
        "events_with_anomalies": events_with_anomalies,
        "total_anomaly_hits": total_hits,
        "unique_items": len(item_counts),
        "items_meeting_support": items_meeting,
        "top_item_support_pct": top_item_support_pct,
        "bucket_minutes": req.bucket_minutes,
        "prehistory_minutes": req.prehistory_minutes,
        "candle_timeframe": req.candle_timeframe,
        "min_support_pct": req.min_support_pct,
        "min_support_count": min_count,
        "itemsets": out,
    }
