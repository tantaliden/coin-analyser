"""Whale-Features (B) — leitet Features aus whale_positions ab.

Pro Coin und Lookback-Fenster:
  - whale_net_qty_change_15m: sum(long_qty_now - long_qty_15m_ago) - sum(short_qty_now - short_qty_15m_ago)
  - whale_long_count: Anzahl Whales aktuell long
  - whale_short_count: Anzahl Whales aktuell short
  - whale_directional_score: (long_count - short_count) / max(long_count + short_count, 1)

NO-FALLBACK: returns None bei DB-Fehlern.
"""

from psycopg2.extras import DictCursor
from psycopg2.errors import UndefinedTable
import logging

log = logging.getLogger("predictor")

_TABLE_MISSING_WARNED = False


def compute_whale_features(app_conn, symbol, lookback_minutes=15):
    """Returns dict mit Whale-Features für `symbol`, leeres dict bei fehlender
    Tabelle (whale_tracker noch nie gelaufen), None bei anderen DB-Fehlern.

    lookback_minutes: Zeitfenster für aktive Whale-Positionen (Setting, vorher hart 15).

    WICHTIG: bei jedem Fehler wird die Transaction zurückgerollt, damit nachfolgende
    Queries auf derselben Connection nicht "current transaction is aborted" werfen.
    """
    global _TABLE_MISSING_WARNED
    try:
        with app_conn.cursor(cursor_factory=DictCursor) as cur:
            # Neueste Snapshot-Position pro aktiver Whale für diesen Coin
            cur.execute("""
                WITH latest AS (
                    SELECT DISTINCT ON (address) address, side, qty, ts
                    FROM whale_positions
                    WHERE symbol = %s
                      AND ts > now() - make_interval(mins => %s)
                    ORDER BY address, ts DESC
                )
                SELECT side, COUNT(*) AS n, COALESCE(SUM(qty),0) AS total_qty
                FROM latest
                GROUP BY side
            """, (symbol, lookback_minutes))
            rows = cur.fetchall()
    except UndefinedTable:
        # whale_tracker noch nicht gestartet -> Tabelle fehlt. Kein Fehler, einfach
        # leeres Feature-Set zurueckgeben. Connection rollback damit weiter genutzt.
        try: app_conn.rollback()
        except Exception: pass
        if not _TABLE_MISSING_WARNED:
            log.warning("compute_whale_features: whale_positions Tabelle existiert "
                        "nicht (whale_tracker noch nicht gestartet) -> skip Feature, "
                        "Coin laeuft ohne Whale-Daten. Diese Warnung nur einmal.")
            _TABLE_MISSING_WARNED = True
        return {}
    except Exception as e:
        try: app_conn.rollback()
        except Exception: pass
        log.error("FALLBACK_TRIGGERED compute_whale_features %s: DB error %s -> None",
                  symbol, e)
        return None

    long_n = 0; short_n = 0
    long_qty = 0.0; short_qty = 0.0
    for r in rows:
        if r['side'] == 'long':
            long_n = int(r['n']); long_qty = float(r['total_qty'])
        elif r['side'] == 'short':
            short_n = int(r['n']); short_qty = float(r['total_qty'])

    total_n = long_n + short_n
    feat = {
        'wh_long_count_15m': long_n,
        'wh_short_count_15m': short_n,
        'wh_total_count_15m': total_n,
        'wh_long_qty_15m': long_qty,
        'wh_short_qty_15m': short_qty,
    }
    if total_n > 0:
        feat['wh_directional_score_15m'] = (long_n - short_n) / total_n
    else:
        feat['wh_directional_score_15m'] = 0.0
    return feat
