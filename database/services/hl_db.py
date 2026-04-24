"""DB-Helper fuer HL-Ingestor. Keine Fallbacks — Fehler propagieren."""
import json
import psycopg2
from psycopg2.extras import execute_values


def get_conn(settings: dict):
    db = settings["databases"]["coins"]
    return psycopg2.connect(
        host=db["host"], port=db["port"], dbname=db["name"],
        user=db["user"], password=db["password"]
    )


def insert_klines_10s(conn, rows):
    """rows: list[dict] mit allen klines-Feldern inkl. ctx-Snapshots."""
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO klines
              (symbol, interval, open_time, close_time,
               open, high, low, close, volume, trades,
               quote_asset_volume, taker_buy_base, taker_buy_quote,
               funding, open_interest, premium, oracle_px, mark_px, mid_px,
               bbo_bid_px, bbo_ask_px, bbo_bid_sz, bbo_ask_sz,
               spread_bps, book_imbalance_5, book_depth_5)
            VALUES %s
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
              close_time=EXCLUDED.close_time, open=EXCLUDED.open,
              high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
              volume=EXCLUDED.volume, trades=EXCLUDED.trades,
              quote_asset_volume=EXCLUDED.quote_asset_volume,
              taker_buy_base=EXCLUDED.taker_buy_base,
              taker_buy_quote=EXCLUDED.taker_buy_quote,
              funding=EXCLUDED.funding, open_interest=EXCLUDED.open_interest,
              premium=EXCLUDED.premium, oracle_px=EXCLUDED.oracle_px,
              mark_px=EXCLUDED.mark_px, mid_px=EXCLUDED.mid_px,
              bbo_bid_px=EXCLUDED.bbo_bid_px, bbo_ask_px=EXCLUDED.bbo_ask_px,
              bbo_bid_sz=EXCLUDED.bbo_bid_sz, bbo_ask_sz=EXCLUDED.bbo_ask_sz,
              spread_bps=EXCLUDED.spread_bps,
              book_imbalance_5=EXCLUDED.book_imbalance_5,
              book_depth_5=EXCLUDED.book_depth_5
        """, [(
            r["symbol"], r["interval"], r["open_time"], r["close_time"],
            r["open"], r["high"], r["low"], r["close"], r["volume"], r["trades"],
            r["quote_asset_volume"], r["taker_buy_base"], r["taker_buy_quote"],
            r["funding"], r["open_interest"], r["premium"],
            r["oracle_px"], r["mark_px"], r["mid_px"],
            r["bbo_bid_px"], r["bbo_ask_px"], r["bbo_bid_sz"], r["bbo_ask_sz"],
            r["spread_bps"], r["book_imbalance_5"], r["book_depth_5"]
        ) for r in rows])
    conn.commit()


def insert_asset_ctx(conn, rows):
    # LEAN (Test-Server Backup): keine hl_asset_ctx-Tabelle.
    return


def insert_l2_snapshot(conn, rows):
    # LEAN (Test-Server Backup): keine hl_l2_snapshot-Tabelle.
    return


def upsert_meta(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO hl_meta (symbol, sz_decimals, max_leverage, margin_table_id)
            VALUES %s
            ON CONFLICT (symbol) DO UPDATE SET
              sz_decimals=EXCLUDED.sz_decimals,
              max_leverage=EXCLUDED.max_leverage,
              margin_table_id=EXCLUDED.margin_table_id,
              updated_at=now()
        """, [(r["symbol"], r["sz_decimals"], r["max_leverage"], r["margin_table_id"]) for r in rows])
    conn.commit()
