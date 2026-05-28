"""Funding-Burst-Features (E).

Berechnet aus klines (funding-Spalte) und coins.agg_1h:
  - funding_velocity_1h: d(funding)/dh über letzte Stunde
  - funding_spike_zscore_24h: aktueller funding vs Mittel + sd der letzten 24h
  - funding_divergence_universe_1h: Coin-funding minus Universe-Median (BTC etc als Referenz)

NO-FALLBACK: bei fehlenden Daten returns None.
"""

from psycopg2.extras import DictCursor
import logging
import statistics

log = logging.getLogger("predictor")


def compute_funding_features(coins_conn, symbol, universe_funding_median=None):
    """Returns dict mit funding-burst Features oder None bei Datenmangel.
    Quelle: agg_1h.funding (1 Wert pro Coin pro Stunde, zuverlässig befüllt)."""
    with coins_conn.cursor(cursor_factory=DictCursor) as cur:
        # 24h-Window aus agg_1h (24 rows)
        cur.execute("""
            SELECT close, funding FROM agg_1h
            WHERE symbol=%s
            ORDER BY bucket DESC LIMIT 24
        """, (symbol,))
        rows_24h = cur.fetchall()

    if len(rows_24h) < 6:
        log.error("FALLBACK_TRIGGERED compute_funding %s: nur %d 1h-rows -> None",
                  symbol, len(rows_24h))
        return None
    # funding_now = neueste 1h-Zeile, funding_1h_ago = zweitneueste
    if rows_24h[0]['funding'] is None or rows_24h[1]['funding'] is None:
        log.error("FALLBACK_TRIGGERED compute_funding %s: funding NULL in latest agg_1h -> None",
                  symbol)
        return None

    feat = {}
    funding_now = float(rows_24h[0]['funding'])
    funding_1h_ago = float(rows_24h[1]['funding'])
    feat['fb_funding_velocity_1h'] = funding_now - funding_1h_ago

    # 24h Z-Score
    fundings_24h = [float(r['funding']) for r in rows_24h if r['funding'] is not None]
    if len(fundings_24h) >= 6:
        mean = statistics.mean(fundings_24h)
        try:
            sd = statistics.stdev(fundings_24h)
        except statistics.StatisticsError:
            sd = 0.0
        if sd > 1e-12:
            feat['fb_funding_zscore_24h'] = (funding_now - mean) / sd
        else:
            # sd=0 <=> alle funding-Werte identisch <=> funding_now == mean -> Abweichung exakt 0
            feat['fb_funding_zscore_24h'] = 0.0
    else:
        log.error("FALLBACK_TRIGGERED compute_funding %s: 24h-funding zu wenig -> None",
                  symbol)
        return None

    feat['fb_funding_abs'] = abs(funding_now)
    feat['fb_funding_signed'] = funding_now

    # Divergenz zu Universe-Median (kommt vom Caller)
    if universe_funding_median is not None:
        feat['fb_funding_divergence_universe'] = funding_now - universe_funding_median
    else:
        feat['fb_funding_divergence_universe'] = 0.0  # Kein Penalty hier, Caller liefert oder nicht

    return feat


def compute_universe_funding_median(coins_conn, symbols):
    """Median funding über alle universe-Coins (für Divergenz-Feature).
    Nutzt agg_1h.funding (zuverlässig 1 Wert pro Coin pro Stunde)."""
    if not symbols:
        return None
    with coins_conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (symbol) symbol, funding
            FROM agg_1h
            WHERE symbol = ANY(%s) AND funding IS NOT NULL
            ORDER BY symbol, bucket DESC
        """, (symbols,))
        vals = [float(r[1]) for r in cur.fetchall() if r[1] is not None]
    if len(vals) < 5:
        log.error("FALLBACK_TRIGGERED compute_universe_funding_median: nur %d funding-Werte (agg_1h) -> None",
                  len(vals))
        return None
    return statistics.median(vals)
