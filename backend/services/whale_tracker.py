#!/usr/bin/env python3
"""Whale-Tracker (Stufe B).

Pollt HL leaderboard alle 6h, identifiziert Top-N profitable Wallets (rolling 7d
PnL > min_pnl_usd), holt user_fills alle fills_poll_seconds (default 300s),
schreibt Position-Snapshots in analyser_app.whale_positions.

Tabellen (werden beim Start angelegt wenn nicht vorhanden):
  whale_wallets (address PK, first_seen, last_seen, rolling_7d_pnl_usd, active)
  whale_positions (id PK, address FK, symbol, side, ts, qty, entry_px, mark_px)

Settings: settings.predictor.whale_tracker.{enabled, leaderboard_refresh_hours,
  fills_poll_seconds, top_n_wallets, min_rolling_7d_pnl_usd, leaderboard_endpoint}
"""

import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("whale-tracker")


def load_settings():
    with open('/opt/coin/settings.json') as f:
        return json.load(f)


def db_app(s):
    db = s['databases']['app']
    return psycopg2.connect(host=db['host'], port=db['port'], user=db['user'],
                             password=db['password'], dbname=db['name'])


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS whale_wallets (
    address TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    rolling_7d_pnl_usd DOUBLE PRECISION,
    rank_position INTEGER,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    last_polled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS whale_positions (
    id BIGSERIAL PRIMARY KEY,
    address TEXT NOT NULL REFERENCES whale_wallets(address),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    entry_px DOUBLE PRECISION,
    mark_px DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS whale_positions_addr_ts_idx ON whale_positions(address, ts DESC);
CREATE INDEX IF NOT EXISTS whale_positions_sym_ts_idx ON whale_positions(symbol, ts DESC);
"""


def ensure_schema(app):
    with app.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    app.commit()


def fetch_leaderboard(endpoint: str, timeout: int = 30):
    """HL leaderboard endpoint. Liefert {'leaderboardRows': [...]}."""
    try:
        req = urllib.request.Request(endpoint, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        log.error("FALLBACK_TRIGGERED fetch_leaderboard: HTTP/Parse error %s -> None", e)
        return None


def update_whale_list(app, leaderboard_data, top_n, min_pnl_usd):
    """Aktualisiert whale_wallets aus leaderboard."""
    if not leaderboard_data:
        log.error("FALLBACK_TRIGGERED update_whale_list: leaderboard_data None -> abort")
        return 0
    rows = leaderboard_data.get('leaderboardRows') or []
    if not rows:
        log.error("FALLBACK_TRIGGERED update_whale_list: leaderboardRows leer -> abort")
        return 0

    # Sortiere nach 7d-PnL absteigend
    parsed = []
    for r in rows:
        addr = r.get('ethAddress') or r.get('user')
        if not addr:
            continue
        wp = r.get('windowPerformances') or []
        # wp ist eine Liste [["day", {...}], ["week", {...}], ...]
        pnl_7d = None
        for item in wp:
            if isinstance(item, list) and len(item) >= 2 and item[0] == 'week':
                pnl_7d = float(item[1].get('pnl', 0))
                break
        if pnl_7d is None:
            continue
        parsed.append((addr, pnl_7d))

    parsed.sort(key=lambda x: -x[1])
    top = [p for p in parsed if p[1] >= min_pnl_usd][:top_n]

    if not top:
        log.warning("update_whale_list: keine Wallets über min_pnl=%s gefunden (n_total=%d)",
                    min_pnl_usd, len(parsed))
        return 0

    addresses = [a for a, _ in top]

    with app.cursor() as cur:
        # active=FALSE für alle die nicht mehr in Top
        cur.execute("UPDATE whale_wallets SET active=FALSE WHERE address != ALL(%s)",
                    (addresses,))
        # Upsert für die in Top
        for rank, (addr, pnl) in enumerate(top, 1):
            cur.execute("""
                INSERT INTO whale_wallets (address, rolling_7d_pnl_usd, rank_position, last_seen, active)
                VALUES (%s, %s, %s, now(), TRUE)
                ON CONFLICT (address) DO UPDATE
                  SET rolling_7d_pnl_usd = EXCLUDED.rolling_7d_pnl_usd,
                      rank_position = EXCLUDED.rank_position,
                      last_seen = now(),
                      active = TRUE
            """, (addr, pnl, rank))
    app.commit()
    log.info("update_whale_list: %d Wallets aktiv (Top-%d nach 7d-PnL >= $%s)",
             len(top), top_n, min_pnl_usd)
    return len(top)


def fetch_user_positions(address: str, timeout: int = 15):
    """HL info endpoint: GET /info { type: 'clearinghouseState', user: <address> }."""
    url = "https://api.hyperliquid.xyz/info"
    body = json.dumps({"type": "clearinghouseState", "user": address}).encode()
    try:
        req = urllib.request.Request(url, data=body, method='POST',
                                      headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        log.error("FALLBACK_TRIGGERED fetch_user_positions %s: %s -> None", address[:8], e)
        return None


def snapshot_positions(app, address: str, positions_data):
    """Schreibt aktuelle offene Positionen in whale_positions."""
    if not positions_data:
        return 0
    asset_positions = positions_data.get('assetPositions') or []
    n_inserted = 0
    with app.cursor() as cur:
        for ap in asset_positions:
            pos = ap.get('position') or {}
            coin = pos.get('coin')
            szi_str = pos.get('szi')  # signed size
            entry = pos.get('entryPx')
            try:
                szi = float(szi_str) if szi_str is not None else 0.0
            except (TypeError, ValueError):
                continue
            if szi == 0:
                continue
            side = 'long' if szi > 0 else 'short'
            qty = abs(szi)
            entry_f = float(entry) if entry else None
            unrl = pos.get('unrealizedPnl')
            unrl_f = float(unrl) if unrl else None
            cur.execute("""
                INSERT INTO whale_positions (address, symbol, side, ts, qty, entry_px, unrealized_pnl)
                VALUES (%s, %s, %s, now(), %s, %s, %s)
            """, (address, coin, side, qty, entry_f, unrl_f))
            n_inserted += 1
        cur.execute("UPDATE whale_wallets SET last_polled_at=now() WHERE address=%s",
                    (address,))
    app.commit()
    return n_inserted


def main():
    s = load_settings()
    wt_cfg = s.get('predictor', {}).get('whale_tracker', {})
    if not wt_cfg.get('enabled'):
        log.info("whale_tracker disabled in settings, exit")
        return

    leaderboard_url = wt_cfg.get('leaderboard_endpoint')
    refresh_h = float(wt_cfg.get('leaderboard_refresh_hours', 6))
    poll_s = float(wt_cfg.get('fills_poll_seconds', 300))
    top_n = int(wt_cfg.get('top_n_wallets', 50))
    min_pnl = float(wt_cfg.get('min_rolling_7d_pnl_usd', 100000))

    if not leaderboard_url:
        log.error("FALLBACK_TRIGGERED whale_tracker: leaderboard_endpoint nicht in settings -> exit")
        return

    with db_app(s) as app:
        ensure_schema(app)

    log.info("whale_tracker start. leaderboard refresh=%.1fh, fills poll=%.0fs, top_n=%d, min_pnl=$%s",
             refresh_h, poll_s, top_n, min_pnl)

    last_leaderboard = 0.0
    last_poll = 0.0

    while True:
        try:
            s = load_settings()
            wt_cfg = s.get('predictor', {}).get('whale_tracker', {})
            if not wt_cfg.get('enabled'):
                log.info("whale_tracker disabled, sleep 60s")
                time.sleep(60)
                continue

            now = time.time()

            # Leaderboard alle refresh_h Stunden
            if now - last_leaderboard >= refresh_h * 3600:
                log.info("Fetching leaderboard...")
                data = fetch_leaderboard(leaderboard_url)
                if data:
                    with db_app(s) as app:
                        update_whale_list(app, data, top_n, min_pnl)
                last_leaderboard = now

            # Polling alle poll_s
            if now - last_poll >= poll_s:
                with db_app(s) as app:
                    with app.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            SELECT address FROM whale_wallets
                            WHERE active=TRUE ORDER BY rank_position ASC LIMIT %s
                        """, (top_n,))
                        actives = [r['address'] for r in cur.fetchall()]
                if actives:
                    total = 0
                    with db_app(s) as app:
                        for addr in actives:
                            pos_data = fetch_user_positions(addr)
                            if pos_data:
                                total += snapshot_positions(app, addr, pos_data)
                    log.info("poll: %d positions snapshot von %d Wallets",
                             total, len(actives))
                last_poll = now

            time.sleep(5)
        except KeyboardInterrupt:
            log.info("interrupt")
            break
        except Exception as e:
            log.exception("loop error: %s", e)
            time.sleep(30)


if __name__ == "__main__":
    main()
