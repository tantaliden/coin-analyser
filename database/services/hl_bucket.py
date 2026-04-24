"""10s-Bucket-Builder fuer HL-Trades.
Sammelt Trades pro (symbol, bucket_start) und flusht bei Bucket-Ende."""

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Bucket:
    symbol: str
    bucket_start: datetime   # UTC-aware
    interval_seconds: int
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    volume: float = 0.0
    trades: int = 0
    quote_asset_volume: float = 0.0
    taker_buy_base: float = 0.0      # side='A' (aggressor buy)
    taker_buy_quote: float = 0.0
    first_time_ms: int = None
    # BBO/L2-Snapshot am Bucket-Close-Zeitpunkt (last-known)
    bbo_bid_px: float = None
    bbo_ask_px: float = None
    bbo_bid_sz: float = None
    bbo_ask_sz: float = None
    spread_bps: float = None
    book_imbalance_5: float = None
    book_depth_5: float = None

    def add_trade(self, price: float, size: float, side: str, time_ms: int) -> None:
        notional = price * size
        if self.first_time_ms is None or time_ms < self.first_time_ms:
            self.first_time_ms = time_ms
            self.open = price
        if self.high is None or price > self.high:
            self.high = price
        if self.low is None or price < self.low:
            self.low = price
        self.close = price   # zeitlich letzter Trade landet als close (bei gleichen ts OK)
        self.volume += size
        self.trades += 1
        self.quote_asset_volume += notional
        if side == 'A':
            self.taker_buy_base += size
            self.taker_buy_quote += notional

    def bucket_end(self) -> datetime:
        return self.bucket_start.replace(microsecond=0)


def compute_bucket_start(time_ms: int, bucket_seconds: int) -> datetime:
    """Trade-Timestamp -> zugehoeriger Bucket-Start (UTC, am Grid gerastert)."""
    sec = time_ms // 1000
    bucket_sec = (sec // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(bucket_sec, tz=timezone.utc)


class BucketCollector:
    """Haelt offene Buckets je symbol. Liefert bei drain() echte + synthetic (idle) Buckets."""

    def __init__(self, bucket_seconds: int):
        self.bucket_seconds = bucket_seconds
        self.open_buckets: Dict[Tuple[str, datetime], Bucket] = {}
        # Slot-Grenze bis zu der bereits emitted wurde (End-Timestamp des letzten emitted Slots).
        self.last_emitted_slot_end = None

    def add_trade(self, symbol: str, price: float, size: float, side: str, time_ms: int) -> None:
        # Alte Trades ignorieren (HL WS schickt beim Subscribe manchmal Initial-History).
        # Cutoff: alles > 2 min in der Vergangenheit.
        import time as _time
        if _time.time() * 1000 - time_ms > 120_000:
            return
        bs = compute_bucket_start(time_ms, self.bucket_seconds)
        key = (symbol, bs)
        b = self.open_buckets.get(key)
        if b is None:
            b = Bucket(symbol=symbol, bucket_start=bs, interval_seconds=self.bucket_seconds)
            self.open_buckets[key] = b
        b.add_trade(price, size, side, time_ms)

    def drain(self, now: datetime, latest_ctx: dict, subscribed: set,
              grace_seconds: float = 1.0) -> List[Bucket]:
        """Liefert fertige Buckets (echte + synthetic fuer Coins ohne Trade im Slot).
        Synthetic Buckets nutzen mark_px aus latest_ctx, volume/trades=0."""
        from datetime import timedelta, timezone
        cutoff = now - timedelta(seconds=grace_seconds)

        # 1) Echte geschlossene Buckets rauspoppen
        real = []
        for key in list(self.open_buckets.keys()):
            b = self.open_buckets[key]
            bucket_end = b.bucket_start + timedelta(seconds=self.bucket_seconds)
            if bucket_end <= cutoff:
                real.append(b)
                del self.open_buckets[key]

        # 2) Synthetic-Filler: hoechster komplett abgelaufener Slot bestimmen
        cutoff_epoch = cutoff.timestamp()
        newest_closed_end_epoch = (int(cutoff_epoch) // self.bucket_seconds) * self.bucket_seconds
        from datetime import datetime as _dt
        newest_closed_end = _dt.fromtimestamp(newest_closed_end_epoch, tz=timezone.utc)
        if newest_closed_end_epoch <= cutoff_epoch - self.bucket_seconds:
            # Der letzte vollstaendig abgelaufene Slot
            pass

        # Initialisierung: beim ersten drain() keine Back-Fill-Historie, erst ab jetzt
        if self.last_emitted_slot_end is None:
            self.last_emitted_slot_end = newest_closed_end - timedelta(seconds=self.bucket_seconds)

        synthetic = []
        real_keys = {(b.symbol, b.bucket_start) for b in real}
        slot_end = self.last_emitted_slot_end + timedelta(seconds=self.bucket_seconds)
        while slot_end <= newest_closed_end:
            slot_start = slot_end - timedelta(seconds=self.bucket_seconds)
            for sym in subscribed:
                if (sym, slot_start) in real_keys:
                    continue
                ctx = latest_ctx.get(sym)
                if not ctx:
                    continue
                px = ctx.get("mark_px") if ctx.get("mark_px") is not None else ctx.get("mid_px")
                if px is None:
                    continue
                synthetic.append(Bucket(
                    symbol=sym, bucket_start=slot_start,
                    interval_seconds=self.bucket_seconds,
                    open=px, high=px, low=px, close=px,
                    volume=0.0, trades=0,
                    quote_asset_volume=0.0,
                    taker_buy_base=0.0, taker_buy_quote=0.0,
                ))
            slot_end = slot_end + timedelta(seconds=self.bucket_seconds)
        self.last_emitted_slot_end = newest_closed_end

        return real + synthetic

    # Backward-compat Alias (nicht mehr verwendet)
    def pop_closed(self, now, grace_seconds: float = 1.0):
        return self.drain(now, {}, set(), grace_seconds)
