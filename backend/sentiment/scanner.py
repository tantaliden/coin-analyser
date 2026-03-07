#!/usr/bin/env python3
"""
Sentiment Scanner — Sammelt News-Sentiment für den RL-Agent.

Quellen:
  1. RSS-Feeds (12 Feeds, 7 Sprachen) → FinBERT (EN) + Keywords (andere)
  2. Fear & Greed Index (global)

Scores werden in sentiment_scores gespeichert (4h Gültigkeitsdauer).
Abgelaufene Einträge werden automatisch gelöscht.
"""

import json
import logging
import signal
import sys
import time
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import psycopg2
import psycopg2.extras
import requests

# FinBERT — wird beim Start geladen
from transformers import pipeline as hf_pipeline

from keywords import keyword_score

# ── Config ──────────────────────────────────────────────────────────────────

SETTINGS_PATH = "/opt/coin/settings.json"
LOG_FILE = "/opt/coin/logs/sentiment_scanner.log"
HEARTBEAT_FILE = "/opt/coin/logs/.sentiment_scanner_heartbeat"

SCORE_TTL_HOURS = 4
POLL_INTERVAL_RSS = 900       # 15 Minuten
POLL_INTERVAL_FEAR_GREED = 21600  # 6 Stunden

# RSS Feeds mit Sprach-Tag
RSS_FEEDS = [
    # Englisch
    {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "lang": "en", "name": "CoinDesk"},
    {"url": "https://cointelegraph.com/rss", "lang": "en", "name": "CoinTelegraph"},
    {"url": "https://decrypt.co/feed", "lang": "en", "name": "Decrypt"},
    {"url": "https://u.today/rss", "lang": "en", "name": "U.Today"},
    {"url": "https://cryptonews.com/news/feed/", "lang": "en", "name": "CryptoNews"},
    # Japanisch
    {"url": "https://coinpost.jp/?feed=rss2", "lang": "ja", "name": "CoinPost JP"},
    # Chinesisch
    {"url": "https://www.blocktempo.com/feed/", "lang": "zh", "name": "BlockTempo ZH"},
    # Koreanisch
    {"url": "https://www.blockmedia.co.kr/feed/", "lang": "ko", "name": "BlockMedia KR"},
    # Russisch
    {"url": "https://forklog.com/feed/", "lang": "ru", "name": "ForkLog RU"},
    # Türkisch
    {"url": "https://koinbulteni.com/feed/", "lang": "tr", "name": "KoinBulteni TR"},
    # Spanisch
    {"url": "https://www.criptonoticias.com/feed/", "lang": "es", "name": "CriptoNoticias ES"},
    # Weitere EN
    {"url": "https://news.bitcoin.com/feed/", "lang": "en", "name": "Bitcoin.com"},
]

# Coin-Name zu Symbol Mapping (häufigste Coins)
COIN_ALIASES = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "ripple": "XRP", "xrp": "XRP",
    "cardano": "ADA", "ada": "ADA",
    "dogecoin": "DOGE", "doge": "DOGE",
    "polkadot": "DOT", "dot": "DOT",
    "avalanche": "AVAX", "avax": "AVAX",
    "chainlink": "LINK", "link": "LINK",
    "polygon": "MATIC", "matic": "MATIC", "pol": "POL",
    "litecoin": "LTC", "ltc": "LTC",
    "uniswap": "UNI", "uni": "UNI",
    "aave": "AAVE",
    "cosmos": "ATOM", "atom": "ATOM",
    "near": "NEAR", "near protocol": "NEAR",
    "aptos": "APT", "apt": "APT",
    "sui": "SUI",
    "arbitrum": "ARB", "arb": "ARB",
    "optimism": "OP",
    "filecoin": "FIL", "fil": "FIL",
    "render": "RENDER",
    "injective": "INJ", "inj": "INJ",
    "sei": "SEI",
    "celestia": "TIA", "tia": "TIA",
    "pepe": "PEPE",
    "shiba": "SHIB", "shib": "SHIB", "shiba inu": "SHIB",
    "bonk": "BONK",
    "tron": "TRX", "trx": "TRX",
    "bnb": "BNB", "binance coin": "BNB",
    "toncoin": "TON", "ton": "TON",
    "stellar": "XLM", "xlm": "XLM",
    "hedera": "HBAR", "hbar": "HBAR",
    "algorand": "ALGO", "algo": "ALGO",
    "monero": "XMR", "xmr": "XMR",
    # Chinesische/Japanische Aliases
    "比特币": "BTC", "ビットコイン": "BTC",
    "以太坊": "ETH", "イーサリアム": "ETH",
    "瑞波": "XRP", "リップル": "XRP",
    "莱特币": "LTC", "ライトコイン": "LTC",
    "狗狗币": "DOGE", "ドージコイン": "DOGE",
    "솔라나": "SOL", "비트코인": "BTC", "이더리움": "ETH",
}

# ── Logging ─────────────────────────────────────────────────────────────────

logger = logging.getLogger("sentiment_scanner")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(sh)

# ── Globals ─────────────────────────────────────────────────────────────────

running = True
finbert = None
db_config = None
known_symbols = set()  # Gültige Symbole aus der DB


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ── DB ──────────────────────────────────────────────────────────────────────

def get_app_conn():
    return psycopg2.connect(
        dbname=db_config["name"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_coins_conn():
    settings = json.load(open(SETTINGS_PATH))
    cdb = settings["databases"]["coins"]
    return psycopg2.connect(
        dbname=cdb["name"],
        user=cdb["user"],
        password=cdb["password"],
        host=cdb["host"],
        port=cdb["port"],
    )


def load_known_symbols():
    """Lädt alle bekannten Coin-Symbole aus der Klines-DB."""
    global known_symbols
    try:
        conn = get_coins_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM klines")
        raw = {r[0] for r in cur.fetchall()}
        # Symbole sind z.B. "BTCUSDC" — wir brauchen den Base-Teil
        known_symbols = set()
        for s in raw:
            for suffix in ["USDC", "USDT", "BTC", "ETH", "BUSD"]:
                if s.endswith(suffix):
                    base = s[: -len(suffix)]
                    if base:
                        known_symbols.add(base)
                    break
        conn.close()
        logger.info(f"Loaded {len(known_symbols)} known coin symbols")
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")


def upsert_score(symbol: str, source: str, score: float, news_count: int = 1):
    """Schreibt oder aktualisiert einen Sentiment-Score."""
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=SCORE_TTL_HOURS)
    try:
        conn = get_app_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sentiment_scores (symbol, source, score, news_count, updated_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, source) DO UPDATE SET
                score = EXCLUDED.score,
                news_count = EXCLUDED.news_count,
                updated_at = EXCLUDED.updated_at,
                expires_at = EXCLUDED.expires_at
            """,
            (symbol, source, score, news_count, now, expires),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB upsert failed for {symbol}/{source}: {e}")


def cleanup_expired():
    """Löscht abgelaufene Scores."""
    try:
        conn = get_app_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM sentiment_scores WHERE expires_at < NOW()")
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        if deleted > 0:
            logger.info(f"Cleanup: {deleted} expired scores deleted")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


# ── Coin Detection ──────────────────────────────────────────────────────────

def detect_coins(text: str) -> list[str]:
    """Erkennt Coin-Symbole in einem Text. Gibt USDC-Suffixed Symbole zurück."""
    text_lower = text.lower()
    found = set()

    # 1. Alias-Matching (Name → Symbol)
    for alias, symbol in COIN_ALIASES.items():
        if alias in text_lower:
            if symbol in known_symbols:
                found.add(symbol + "USDC")

    # 2. Direct Symbol Matching (z.B. "BTC", "ETH" im Text)
    # Nur Symbole mit 3+ Zeichen um Fehlmatches zu vermeiden
    words = set(re.findall(r'\b[A-Z]{3,10}\b', text))
    for word in words:
        if word in known_symbols:
            found.add(word + "USDC")

    return list(found)


# ── FinBERT ─────────────────────────────────────────────────────────────────

def finbert_score(text: str) -> float:
    """Bewertet englischen Text mit FinBERT. Returns -1.0 bis +1.0."""
    result = finbert(text[:512])[0]
    label = result["label"]    # positive, negative, neutral
    conf = result["score"]     # 0.0 - 1.0

    if label == "positive":
        return conf
    elif label == "negative":
        return -conf
    else:
        return 0.0


# ── RSS Feed Processing ────────────────────────────────────────────────────

def process_feeds():
    """Holt und bewertet alle RSS Feeds."""
    # Sammle Scores pro Coin
    coin_scores = {}  # symbol -> [(score, weight)]

    for feed_cfg in RSS_FEEDS:
        if not running:
            break
        try:
            feed = feedparser.parse(feed_cfg["url"])
            lang = feed_cfg["lang"]
            name = feed_cfg["name"]

            if not feed.entries:
                logger.warning(f"No entries from {name}")
                continue

            # Nur Headlines der letzten 4h
            cutoff = datetime.now(timezone.utc) - timedelta(hours=SCORE_TTL_HOURS)
            processed = 0

            for entry in feed.entries:
                # Publish-Date prüfen
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

                if pub_date and pub_date < cutoff:
                    continue

                title = entry.get("title", "")
                if not title:
                    continue

                # Score berechnen
                if lang == "en":
                    score = finbert_score(title)
                else:
                    score = keyword_score(title, lang)
                    if score is None:
                        continue  # Keine Keywords gefunden, skip

                # Coins erkennen
                coins = detect_coins(title)
                if not coins:
                    continue

                # Weight: EN feeds haben höheres Gewicht (FinBERT ist genauer)
                weight = 1.5 if lang == "en" else 1.0

                for coin in coins:
                    if coin not in coin_scores:
                        coin_scores[coin] = []
                    coin_scores[coin].append((score, weight))

                processed += 1

            if processed > 0:
                logger.info(f"  {name}: {processed} headlines processed")

        except Exception as e:
            logger.error(f"Feed error {feed_cfg['name']}: {e}")

    # Aggregierte Scores in DB schreiben
    for symbol, scores in coin_scores.items():
        if not scores:
            continue
        # Gewichteter Durchschnitt
        total_weight = sum(w for _, w in scores)
        weighted_score = sum(s * w for s, w in scores) / total_weight
        # Auf -1.0 bis +1.0 clampen
        weighted_score = max(-1.0, min(1.0, weighted_score))

        source = "finbert" if any(True for _ in scores) else "keywords"
        upsert_score(symbol, "news", round(weighted_score, 4), len(scores))

    logger.info(f"RSS: {len(coin_scores)} coins scored from {sum(len(v) for v in coin_scores.values())} headlines")


# ── Fear & Greed Index ──────────────────────────────────────────────────────

def fetch_fear_greed():
    """Holt Fear & Greed Index von alternative.me."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        value = int(data["data"][0]["value"])  # 0-100
        classification = data["data"][0]["value_classification"]

        # Normalisieren auf -1.0 bis +1.0 (50 = neutral)
        score = (value - 50) / 50.0

        upsert_score("GLOBAL", "fear_greed", round(score, 4), 1)
        logger.info(f"Fear & Greed: {value} ({classification}) → score {score:.2f}")

    except Exception as e:
        logger.error(f"Fear & Greed fetch failed: {e}")


# ── Heartbeat ───────────────────────────────────────────────────────────────

def write_heartbeat():
    try:
        Path(HEARTBEAT_FILE).write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass


# ── Main Loop ───────────────────────────────────────────────────────────────

def main():
    global finbert, db_config

    logger.info("=" * 60)
    logger.info("Sentiment Scanner starting...")

    # Config laden
    settings = json.load(open(SETTINGS_PATH))
    db_config = settings["databases"]["app"]

    # Symbole laden
    load_known_symbols()

    # FinBERT laden
    logger.info("Loading FinBERT model...")
    t0 = time.time()
    finbert = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        max_length=128,
        truncation=True,
    )
    logger.info(f"FinBERT loaded in {time.time() - t0:.1f}s")

    # Initialer Lauf
    logger.info("Initial fetch...")
    fetch_fear_greed()
    process_feeds()
    cleanup_expired()
    write_heartbeat()

    last_rss = time.time()
    last_fg = time.time()

    logger.info(f"Running. RSS every {POLL_INTERVAL_RSS}s, Fear&Greed every {POLL_INTERVAL_FEAR_GREED}s")

    while running:
        time.sleep(30)  # Check-Interval

        now = time.time()

        # RSS Feeds
        if now - last_rss >= POLL_INTERVAL_RSS:
            logger.info("RSS poll...")
            process_feeds()
            cleanup_expired()
            write_heartbeat()
            last_rss = now

        # Fear & Greed
        if now - last_fg >= POLL_INTERVAL_FEAR_GREED:
            logger.info("Fear & Greed poll...")
            fetch_fear_greed()
            last_fg = now

    logger.info("Sentiment Scanner stopped.")


if __name__ == "__main__":
    main()
