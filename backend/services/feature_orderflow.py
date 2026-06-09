"""Orderflow-Features aus hl_l2_snapshot (Wall/Sweep/Velocity/Aggression).

Persistierte Tabelle hl_l2_snapshot hat pro symbol+ts:
  bids: jsonb [{n,px,sz}, ...] Top-20 Bid-Levels
  asks: jsonb [{n,px,sz}, ...] Top-20 Ask-Levels

Pro Coin werden Features in zwei Lookback-Fenstern berechnet:
  - kurz (default 60s): Velocity-Features
  - lang (default 300s): Statistik-Features (Walls, Sweeps)

NO-FALLBACK: bei zu wenig Snapshots returns None — Caller skipped Coin.
"""

from psycopg2.extras import DictCursor
import logging

log = logging.getLogger("predictor")


def _wall_ratio(sizes):
    if not sizes:
        return None
    total = sum(sizes)
    if total <= 0:
        return None
    avg = total / len(sizes)
    return max(sizes) / avg


def _level_imbalance(bid_szs, ask_szs):
    b = sum(bid_szs); a = sum(ask_szs)
    if b + a <= 0:
        return None
    return (b - a) / (b + a)


def compute_orderflow_features(coins_conn, symbol,
                                lookback_short_s=60,
                                lookback_long_s=300,
                                min_snapshots_short=3,
                                min_snapshots_long=10,
                                spoof_max_lifetime_s=30,
                                wall_factor=2.0,
                                velocity_window_s=60):
    """Berechnet Orderflow-Features für `symbol` aus hl_l2_snapshot.

    Returns dict mit Features oder None bei zu wenig Daten.
    """
    with coins_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT ts, bids, asks
            FROM hl_l2_snapshot
            WHERE symbol = %s
              AND ts > now() - %s * interval '1 second'
            ORDER BY ts ASC
        """, (symbol, lookback_long_s))
        rows = cur.fetchall()

    n = len(rows)
    if n < min_snapshots_long:
        log.error("FALLBACK_TRIGGERED compute_orderflow %s: nur %d snapshots im %ds Fenster (min=%d) -> None",
                  symbol, n, lookback_long_s, min_snapshots_long)
        return None

    # Pro Snapshot: berechne wall-ratio, top-5-imbalance, top-3-imbalance, prices der bids/asks
    bid_wall_ratios = []
    ask_wall_ratios = []
    imb5_series = []
    imb3_series = []
    spread_series = []
    last_bid_levels = None  # set von Top-5 Bid-Preisen
    last_ask_levels = None
    sweep_bid_count = 0
    sweep_ask_count = 0
    ts_series = []
    wall_lifetimes = {}  # px → first_seen_ts (für Spoof-Detection)
    spoof_count = 0
    SPOOF_MAX_LIFETIME_S = spoof_max_lifetime_s

    for r in rows:
        ts = r['ts']
        bids = r['bids'] or []
        asks = r['asks'] or []
        if not bids or not asks:
            continue
        bid_szs5 = [float(b['sz']) for b in bids[:5] if 'sz' in b]
        ask_szs5 = [float(a['sz']) for a in asks[:5] if 'sz' in a]
        if not bid_szs5 or not ask_szs5:
            continue
        bid_pxs5 = [float(b['px']) for b in bids[:5] if 'px' in b]
        ask_pxs5 = [float(a['px']) for a in asks[:5] if 'px' in a]
        if not bid_pxs5 or not ask_pxs5:
            continue

        # Wall ratio: max(size) / avg(size) im Top-5 Set
        bw = _wall_ratio(bid_szs5)
        aw = _wall_ratio(ask_szs5)
        if bw is not None and aw is not None:
            bid_wall_ratios.append(bw)
            ask_wall_ratios.append(aw)

        # Top-5 Imbalance
        i5 = _level_imbalance(bid_szs5, ask_szs5)
        if i5 is not None:
            imb5_series.append(i5)
            ts_series.append(ts)

        # Top-3 Imbalance
        i3 = _level_imbalance(bid_szs5[:3], ask_szs5[:3])
        if i3 is not None:
            imb3_series.append(i3)

        # Spread in bps relative zum mid
        bid_top = bid_pxs5[0]; ask_top = ask_pxs5[0]
        mid = (bid_top + ask_top) / 2
        if mid > 0:
            spread_series.append((ask_top - bid_top) / mid * 10000)

        # Sweep-Detection: wenn ein Bid/Ask-Level der vorherigen Top-5 nicht mehr da ist
        cur_bid_set = set(bid_pxs5)
        cur_ask_set = set(ask_pxs5)
        if last_bid_levels is not None:
            sweep_bid_count += len(last_bid_levels - cur_bid_set)
            sweep_ask_count += len(last_ask_levels - cur_ask_set)
        last_bid_levels = cur_bid_set
        last_ask_levels = cur_ask_set

        # Spoof-Detection: tracke Lebensdauer von Walls (size > wall_factor× avg)
        if bw is not None and bw > wall_factor:
            wall_px = bid_pxs5[bid_szs5.index(max(bid_szs5))]
            if wall_px not in wall_lifetimes:
                wall_lifetimes[wall_px] = ts
            else:
                lifetime = (ts - wall_lifetimes[wall_px]).total_seconds()
                if lifetime < SPOOF_MAX_LIFETIME_S and wall_px not in cur_bid_set:
                    spoof_count += 1
                    del wall_lifetimes[wall_px]

    if not imb5_series or len(imb5_series) < min_snapshots_long:
        log.error("FALLBACK_TRIGGERED compute_orderflow %s: zu wenig validate snapshots (%d) -> None",
                  symbol, len(imb5_series))
        return None

    # Statistik-Features
    feat = {}
    feat['of_bid_wall_ratio_avg_5m'] = sum(bid_wall_ratios) / len(bid_wall_ratios) if bid_wall_ratios else 1.0
    feat['of_ask_wall_ratio_avg_5m'] = sum(ask_wall_ratios) / len(ask_wall_ratios) if ask_wall_ratios else 1.0
    feat['of_bid_wall_ratio_max_5m'] = max(bid_wall_ratios) if bid_wall_ratios else 1.0
    feat['of_ask_wall_ratio_max_5m'] = max(ask_wall_ratios) if ask_wall_ratios else 1.0
    feat['of_imbalance_5_mean_5m'] = sum(imb5_series) / len(imb5_series)
    feat['of_imbalance_3_mean_5m'] = sum(imb3_series) / len(imb3_series) if imb3_series else 0.0
    feat['of_spread_bps_mean_5m'] = sum(spread_series) / len(spread_series) if spread_series else 0.0

    # Velocity: letzte 60s vs vorherige 60s im 5min-Fenster
    if len(ts_series) >= 4:
        latest_ts = ts_series[-1]
        # Mittelwert über letzte 60s
        recent_imb = [imb5_series[i] for i, t in enumerate(ts_series)
                      if (latest_ts - t).total_seconds() <= velocity_window_s]
        # Mittelwert über velocity_window_s bis 2× zurück
        older_imb = [imb5_series[i] for i, t in enumerate(ts_series)
                     if velocity_window_s < (latest_ts - t).total_seconds() <= 2 * velocity_window_s]
        if recent_imb and older_imb:
            feat['of_imbalance_velocity_60s'] = (sum(recent_imb)/len(recent_imb)
                                                  - sum(older_imb)/len(older_imb))
        else:
            feat['of_imbalance_velocity_60s'] = 0.0
    else:
        feat['of_imbalance_velocity_60s'] = 0.0

    # Sweep-Rate pro Minute
    duration_min = (ts_series[-1] - ts_series[0]).total_seconds() / 60 if len(ts_series) > 1 else 1
    feat['of_sweep_bid_per_min_5m'] = sweep_bid_count / max(duration_min, 0.1)
    feat['of_sweep_ask_per_min_5m'] = sweep_ask_count / max(duration_min, 0.1)

    # Spoof-Events pro Minute
    feat['of_spoof_events_per_min_5m'] = spoof_count / max(duration_min, 0.1)

    return feat
