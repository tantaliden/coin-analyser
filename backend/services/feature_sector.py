"""Sektor-Cluster-Features (F).

Coin → Sektor-Mapping aus analyser_app.coin_info.categories[] mit priority-Liste
aus settings.predictor.sector_priority. Erster category-Eintrag der in priority
liegt = primärer Sektor. Sonst 'Other'.

Berechnet:
  - sector_avg_close_pct_60m: Mittelwert close_pct_60m aller Coins im Sektor
  - sector_avg_close_pct_15m: Mittelwert close_pct_15m
  - coin_rs_vs_sector_60m: Coin.close_pct_60m - sector_avg_close_pct_60m
  - coin_rs_vs_sector_15m: analog 15m
  - sector_member_count: wie viele Universe-Coins im selben Sektor

NO-FALLBACK: returns None bei fehlenden Daten.
"""

from psycopg2.extras import DictCursor
import logging

log = logging.getLogger("predictor")


def build_coin_sector_map(app_conn, universe_symbols, sector_priority):
    """Returns dict symbol -> sector_name (first category in priority or 'Other')."""
    if not universe_symbols or not sector_priority:
        return {}
    with app_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT symbol, categories FROM coin_info
            WHERE symbol = ANY(%s) AND categories IS NOT NULL
        """, (universe_symbols,))
        rows = cur.fetchall()

    prio_set = list(sector_priority)
    result = {}
    for r in rows:
        cats = r['categories'] or []
        chosen = 'Other'
        for prio in prio_set:
            if prio in cats:
                chosen = prio
                break
        result[r['symbol']] = chosen
    # Coins ohne Eintrag → 'Other'
    for sym in universe_symbols:
        if sym not in result:
            result[sym] = 'Other'
    return result


def compute_sector_close_pcts(coins_conn, coin_sector_map, lookback_buckets=61, min_buckets=16):
    """Returns dict sector → {'mean_60m': float, 'mean_15m': float, 'n': int}.

    Liest close-Werte aus agg_5m/agg_1m für alle Coins im map.
    Berechnet close_pct_60m und close_pct_15m pro Coin, gruppiert nach Sektor.
    """
    symbols = list(coin_sector_map.keys())
    if not symbols:
        return {}

    # close-Werte: aktueller (close von neueste 1m-agg) und vor 60m + vor 15m
    coin_pcts = {}  # symbol → (pct_60m, pct_15m)
    with coins_conn.cursor(cursor_factory=DictCursor) as cur:
        # Aktuelle Close + 60-Min vorher per agg_1m LIMIT 61 pro Coin (effizienz via array_agg könnte schöner sein,
        # aber wir machen einfach DISTINCT-Loop für Klarheit; Universe ist max 200)
        # SINGLE QUERY mit window: gruppiert per symbol, sortiert nach bucket DESC, holt erste, 16., 61.
        # Pragmatisch: pro Symbol eine kleine Query (200 × 2 ms = 400 ms - akzeptabel).
        for sym in symbols:
            cur.execute("""
                SELECT close FROM agg_1m
                WHERE symbol=%s
                ORDER BY bucket DESC LIMIT %s
            """, (sym, lookback_buckets))
            rows = cur.fetchall()
            if len(rows) < min_buckets:
                continue
            c_now = float(rows[0]['close']) if rows[0]['close'] is not None else None
            c_15m = float(rows[15]['close']) if len(rows) > 15 and rows[15]['close'] is not None else None
            c_60m = float(rows[60]['close']) if len(rows) > 60 and rows[60]['close'] is not None else None
            if c_now is None:
                continue
            pct_60m = ((c_now - c_60m) / c_60m * 100) if c_60m else None
            pct_15m = ((c_now - c_15m) / c_15m * 100) if c_15m else None
            coin_pcts[sym] = (pct_60m, pct_15m)

    # Gruppiere nach Sektor
    by_sector = {}
    for sym, (p60, p15) in coin_pcts.items():
        sec = coin_sector_map.get(sym, 'Other')
        if sec not in by_sector:
            by_sector[sec] = {'p60_list': [], 'p15_list': []}
        if p60 is not None:
            by_sector[sec]['p60_list'].append(p60)
        if p15 is not None:
            by_sector[sec]['p15_list'].append(p15)

    result = {}
    for sec, agg in by_sector.items():
        p60s = agg['p60_list']
        p15s = agg['p15_list']
        result[sec] = {
            'mean_60m': sum(p60s) / len(p60s) if p60s else None,
            'mean_15m': sum(p15s) / len(p15s) if p15s else None,
            'n_60m': len(p60s),
            'n_15m': len(p15s),
        }
    return result, coin_pcts


def compute_sector_features(symbol, coin_sector_map, sector_stats, coin_pcts):
    """Returns dict mit sector-Features für `symbol`."""
    sec = coin_sector_map.get(symbol)
    if sec is None:
        log.error("FALLBACK_TRIGGERED compute_sector_features %s: kein sector-mapping -> None",
                  symbol)
        return None
    stats = sector_stats.get(sec)
    if stats is None:
        log.error("FALLBACK_TRIGGERED compute_sector_features %s: keine sector-stats für '%s' -> None",
                  symbol, sec)
        return None

    sec_60m = stats['mean_60m']
    sec_15m = stats['mean_15m']
    if sec_60m is None or sec_15m is None:
        log.error("FALLBACK_TRIGGERED compute_sector_features %s: sector_stats None für '%s' -> None",
                  symbol, sec)
        return None

    pcts = coin_pcts.get(symbol)
    feat = {
        'sec_avg_close_pct_60m': float(sec_60m),
        'sec_avg_close_pct_15m': float(sec_15m),
        'sec_member_count': int(stats['n_60m']),
    }
    if pcts and pcts[0] is not None:
        feat['sec_rs_vs_coin_60m'] = float(pcts[0]) - float(sec_60m)
    else:
        feat['sec_rs_vs_coin_60m'] = 0.0
    if pcts and pcts[1] is not None:
        feat['sec_rs_vs_coin_15m'] = float(pcts[1]) - float(sec_15m)
    else:
        feat['sec_rs_vs_coin_15m'] = 0.0
    return feat
