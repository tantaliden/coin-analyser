"""Hyperliquid-Ingestor (Option C: WebSocket-only).
Ein Prozess, asyncio-Tasks:
  - ws_consumer: subscribe trades/activeAssetCtx/l2Book fuer alle Perps
  - flush_klines: alle X ms abgeschlossene 10s-Buckets in DB
  - meta_refresher: alle N Stunden REST /info 'meta' -> hl_meta + neue Coins resubscriben
Keine Fallbacks. Alle Konfigurationen in settings.json -> 'hyperliquid_ingest'."""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_bucket import BucketCollector
from hl_db import get_conn, insert_klines_10s, insert_asset_ctx, insert_l2_snapshot, upsert_meta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("hl_ingestor")


def load_settings():
    root = Path(__file__).resolve().parents[2]
    with open(root / 'settings.json') as f:
        return json.load(f)


class HLState:
    """Shared State zwischen Tasks."""
    def __init__(self):
        self.latest_ctx: dict = {}     # symbol -> dict der letzten ctx-Werte (fuer klines-Snapshot)
        self.latest_l2: dict = {}      # symbol -> {bids, asks} letzter L2-Snapshot
        self.subscribed: set = set()
        self.last_l2_ts: dict = {}
        self.bucket_collector: BucketCollector = None
        self.pending_ctx_rows: list = []
        self.pending_l2_rows: list = []
        self.ctx_last_flush: float = 0.0


async def fetch_meta(info_url: str) -> list:
    """REST /info meta -> universe list."""
    async with aiohttp.ClientSession() as s:
        async with s.post(info_url, json={"type": "meta"}) as r:
            r.raise_for_status()
            data = await r.json()
    return data["universe"]


async def task_meta(cfg: dict, state: HLState, on_new_symbols):
    """Periodisch Meta holen, hl_meta updaten, neue Symbole rueckmelden."""
    url = cfg["info_url"]
    interval = cfg["meta_refresh_seconds"]
    db_cfg = load_settings()
    while True:
        try:
            universe = await fetch_meta(url)
            rows = [{
                "symbol": u["name"],
                "sz_decimals": u.get("szDecimals"),
                "max_leverage": u.get("maxLeverage"),
                "margin_table_id": u.get("marginTableId"),
            } for u in universe]
            with get_conn(db_cfg) as conn:
                upsert_meta(conn, rows)
            new_syms = [r["symbol"] for r in rows if r["symbol"] not in state.subscribed]
            log.info("meta refresh: %d coins total, %d new", len(rows), len(new_syms))
            if new_syms:
                await on_new_symbols(new_syms)
        except Exception as e:
            log.exception("meta task error: %s", e)
        await asyncio.sleep(interval)


async def task_ws(cfg: dict, state: HLState):
    """WebSocket-Consumer mit Auto-Reconnect."""
    url = cfg["ws_url"]
    backoff = cfg["reconnect_backoff_seconds"]
    db_cfg = load_settings()

    async def on_new_symbols(new_syms):
        # Immer vormerken — initialer WS-Connect subscribt aus state.subscribed
        for c in new_syms:
            state.subscribed.add(c)
        # Bei aktiver Verbindung: direkt subscribe senden
        if state.ws_conn is not None:
            for c in new_syms:
                for t in ("trades", "activeAssetCtx", "l2Book"):
                    try:
                        await state.ws_conn.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": t, "coin": c}}))
                    except Exception:
                        pass

    state.ws_conn = None

    # Starte Meta-Task parallel (triggert initiales Subscriben via on_new_symbols)
    asyncio.create_task(task_meta(cfg, state, on_new_symbols))

    # Warte bis wir die initiale Coin-Liste haben
    while not state.subscribed:
        await asyncio.sleep(0.5)

    while True:
        try:
            log.info("WS connecting to %s (subscribe %d coins)", url, len(state.subscribed))
            async with websockets.connect(url, ping_interval=cfg["ws_ping_interval"], ping_timeout=cfg["ws_ping_timeout"]) as ws:
                state.ws_conn = ws
                for coin in sorted(state.subscribed):
                    for t in ("trades", "activeAssetCtx", "l2Book"):
                        await ws.send(json.dumps({"method": "subscribe",
                                                  "subscription": {"type": t, "coin": coin}}))
                async for raw in ws:
                    await handle_msg(raw, state, cfg, db_cfg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("WS disconnect: %s", e)
            state.ws_conn = None
            await asyncio.sleep(backoff)


async def handle_msg(raw: str, state: HLState, cfg: dict, db_cfg: dict):
    try:
        d = json.loads(raw)
    except Exception:
        return
    ch = d.get("channel")
    data = d.get("data")
    if ch == "trades":
        for t in data:
            price = float(t["px"])
            size = float(t["sz"])
            side = t["side"]
            time_ms = int(t["time"])
            state.bucket_collector.add_trade(t["coin"], price, size, side, time_ms)
    elif ch == "activeAssetCtx":
        coin = data["coin"]
        ctx = data["ctx"]
        impact = ctx.get("impactPxs") or [None, None]
        row = {
            "symbol": coin,
            "ts": datetime.now(timezone.utc),
            "funding": _f(ctx.get("funding")),
            "open_interest": _f(ctx.get("openInterest")),
            "premium": _f(ctx.get("premium")),
            "oracle_px": _f(ctx.get("oraclePx")),
            "mark_px": _f(ctx.get("markPx")),
            "mid_px": _f(ctx.get("midPx")),
            "impact_bid": _f(impact[0]) if len(impact) > 0 else None,
            "impact_ask": _f(impact[1]) if len(impact) > 1 else None,
            "day_ntl_vlm": _f(ctx.get("dayNtlVlm")),
            "day_base_vlm": _f(ctx.get("dayBaseVlm")),
            "prev_day_px": _f(ctx.get("prevDayPx")),
        }
        state.latest_ctx[coin] = row
        state.pending_ctx_rows.append(row)
    elif ch == "l2Book":
        ts_ms = int(data.get("time", time.time() * 1000))
        coin = data["coin"]
        levels = data.get("levels") or [[], []]
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        # Immer latest_l2 aktualisieren (fuer BBO-Features beim Bucket-Close)
        state.latest_l2[coin] = {"bids": bids, "asks": asks}
        # Throttle fuer DB-Insert der Full-Snapshots (hl_l2_snapshot)
        last = state.last_l2_ts.get(coin, 0)
        now = time.time()
        if now - last < cfg["l2_throttle_seconds"]:
            return
        state.last_l2_ts[coin] = now
        state.pending_l2_rows.append({
            "symbol": coin,
            "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            "bids": bids, "asks": asks,
        })


def _f(v):
    if v is None:
        return None
    return float(v)


async def task_flush(cfg: dict, state: HLState):
    """Alle flush_check_seconds: geschlossene Buckets + pending ctx/l2 in DB."""
    interval = cfg["flush_check_seconds"]
    ctx_debounce = cfg["ctx_insert_debounce_seconds"]
    db_cfg = load_settings()
    while True:
        try:
            now = datetime.now(timezone.utc)
            closed = state.bucket_collector.drain(
                now, state.latest_ctx, state.subscribed, grace_seconds=cfg["bucket_grace_seconds"])
            klines_rows = []
            for b in closed:
                last = state.latest_ctx.get(b.symbol, {}) or {}
                bbo = compute_bbo(state.latest_l2.get(b.symbol))
                klines_rows.append({
                    "symbol": b.symbol,
                    "interval": "10s",
                    "open_time": b.bucket_start,
                    "close_time": b.bucket_start.replace() + _td(cfg["bucket_seconds"]),
                    "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                    "volume": b.volume, "trades": b.trades,
                    "quote_asset_volume": b.quote_asset_volume,
                    "taker_buy_base": b.taker_buy_base,
                    "taker_buy_quote": b.taker_buy_quote,
                    "funding": last.get("funding"),
                    "open_interest": last.get("open_interest"),
                    "premium": last.get("premium"),
                    "oracle_px": last.get("oracle_px"),
                    "mark_px": last.get("mark_px"),
                    "mid_px": last.get("mid_px"),
                    "bbo_bid_px": bbo["bbo_bid_px"],
                    "bbo_ask_px": bbo["bbo_ask_px"],
                    "bbo_bid_sz": bbo["bbo_bid_sz"],
                    "bbo_ask_sz": bbo["bbo_ask_sz"],
                    "spread_bps": bbo["spread_bps"],
                    "book_imbalance_5": bbo["book_imbalance_5"],
                    "book_depth_5": bbo["book_depth_5"],
                })
            now_mono = time.time()
            flush_ctx = (now_mono - state.ctx_last_flush) >= ctx_debounce and state.pending_ctx_rows
            n_ctx = len(state.pending_ctx_rows)
            n_l2 = len(state.pending_l2_rows)
            if klines_rows or flush_ctx or state.pending_l2_rows:
                with get_conn(db_cfg) as conn:
                    if klines_rows:
                        insert_klines_10s(conn, klines_rows)
                    if flush_ctx:
                        insert_asset_ctx(conn, state.pending_ctx_rows)
                        state.pending_ctx_rows = []
                        state.ctx_last_flush = now_mono
                    if state.pending_l2_rows:
                        insert_l2_snapshot(conn, state.pending_l2_rows)
                        state.pending_l2_rows = []
                if klines_rows or flush_ctx or state.pending_l2_rows:
                    log.info("flushed %d klines, %d ctx, %d l2 (pending ctx=%d l2=%d)",
                             len(klines_rows),
                             n_ctx if flush_ctx else 0,
                             n_l2,
                             len(state.pending_ctx_rows), len(state.pending_l2_rows))
        except Exception as e:
            log.exception("flush task error: %s", e)
        await asyncio.sleep(interval)


def _td(seconds: int):
    from datetime import timedelta
    return timedelta(seconds=seconds)


def compute_bbo(l2: dict) -> dict:
    """Berechnet BBO/Book-Imbalance/Spread aus L2-Snapshot. None wenn L2 fehlt."""
    empty = {k: None for k in ("bbo_bid_px", "bbo_ask_px", "bbo_bid_sz", "bbo_ask_sz",
                               "spread_bps", "book_imbalance_5", "book_depth_5")}
    if not l2:
        return empty
    bids = l2.get("bids") or []
    asks = l2.get("asks") or []
    if not bids or not asks:
        return empty
    bid_px = float(bids[0]["px"])
    ask_px = float(asks[0]["px"])
    bid_sz = float(bids[0]["sz"])
    ask_sz = float(asks[0]["sz"])
    mid = (bid_px + ask_px) / 2.0
    spread_bps = (ask_px - bid_px) / mid * 10000.0 if mid > 0 else None
    top5_bid = sum(float(b["sz"]) for b in bids[:5])
    top5_ask = sum(float(a["sz"]) for a in asks[:5])
    depth5 = top5_bid + top5_ask
    imbalance5 = (top5_bid - top5_ask) / depth5 if depth5 > 0 else 0.0
    return {
        "bbo_bid_px": bid_px, "bbo_ask_px": ask_px,
        "bbo_bid_sz": bid_sz, "bbo_ask_sz": ask_sz,
        "spread_bps": spread_bps,
        "book_imbalance_5": imbalance5,
        "book_depth_5": depth5,
    }


async def main():
    s = load_settings()
    cfg = s["hyperliquid_ingest"]
    state = HLState()
    state.bucket_collector = BucketCollector(bucket_seconds=cfg["bucket_seconds"])
    await asyncio.gather(task_ws(cfg, state), task_flush(cfg, state))


if __name__ == "__main__":
    asyncio.run(main())
