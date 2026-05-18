"""
Stalker-Predictor — Coin-Identity-Features
============================================
Pro Coin: Hash-Trick-Slots (Symbol/Network) + Binary-Categorie-Features
basierend auf analyser_app.coin_info.

Reine Funktion: 1× pro Process beim Start coin_info laden (cache), dann
per scan-Aufruf Lookup.

Settings: predictor.stalker.coin_identity.{hash_slots, network_hash_slots, categories_top_k}
"""
import hashlib
import logging
import time

log = logging.getLogger(__name__)

_CACHE = {"coin_info": None, "loaded_at": 0.0}
_CACHE_TTL_S = 600  # 10 min — refresh coin_info aus DB


def _hash_slot(value: str, n_slots: int) -> int:
    """Stabiler MD5-Hash → Slot 0..n-1."""
    h = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % n_slots


def _must(cfg, key):
    v = cfg.get(key)
    if v is None:
        log.error(f"FALLBACK_TRIGGERED feature_coin_identity: '{key}' fehlt in predictor.stalker.coin_identity")
        raise RuntimeError(f"settings.predictor.stalker.coin_identity.{key} missing")
    return v


def _load_coin_info(app_conn):
    """Liest coin_info vollständig ein, returnt {base_asset -> (network, categories_set)}.

    base_asset ist der Schlüssel, der zu agg_1m.symbol/hl_meta.symbol passt
    ('BTC', 'ETH', ...). coin_info.symbol enthält dagegen Binance-style Pairs
    ('BTCUSDC').
    """
    with app_conn.cursor() as cur:
        cur.execute(
            "SELECT base_asset, network, categories FROM coin_info "
            "WHERE base_asset IS NOT NULL"
        )
        out = {}
        for base, net, cats in cur.fetchall():
            out[base] = (net or "", set(c.lower() for c in (cats or [])))
        return out


def get_coin_info_map(app_conn):
    now = time.time()
    if _CACHE["coin_info"] is None or (now - _CACHE["loaded_at"]) > _CACHE_TTL_S:
        _CACHE["coin_info"] = _load_coin_info(app_conn)
        _CACHE["loaded_at"] = now
        log.info(f"coin_info cache reload: {len(_CACHE['coin_info'])} coins")
    return _CACHE["coin_info"]


def compute_identity_features(symbol: str, app_conn, identity_cfg: dict) -> dict:
    """
    Returnt dict mit:
      id_sym_hash_0..N-1   (Binary, genau 1× 1)
      id_net_hash_0..M-1   (Binary, genau 1× 1)
      id_cat_<cat>         (Binary 0/1) je Kategorie aus categories_top_k
    """
    sym_slots = int(_must(identity_cfg, "hash_slots"))
    net_slots = int(_must(identity_cfg, "network_hash_slots"))
    cats_top_k = _must(identity_cfg, "categories_top_k")

    info = get_coin_info_map(app_conn)
    network, cat_set = info.get(symbol, ("", set()))

    feat = {}
    s_idx = _hash_slot(symbol, sym_slots)
    for i in range(sym_slots):
        feat[f"id_sym_hash_{i}"] = 1.0 if i == s_idx else 0.0

    if network:
        n_idx = _hash_slot(network, net_slots)
        for i in range(net_slots):
            feat[f"id_net_hash_{i}"] = 1.0 if i == n_idx else 0.0
    else:
        for i in range(net_slots):
            feat[f"id_net_hash_{i}"] = 0.0

    for cat in cats_top_k:
        feat[f"id_cat_{cat}"] = 1.0 if cat.lower() in cat_set else 0.0

    return feat


def feature_keys(identity_cfg: dict) -> list:
    """Feature-Namen-Liste in deterministischer Reihenfolge — für vectorize()."""
    sym_slots = int(_must(identity_cfg, "hash_slots"))
    net_slots = int(_must(identity_cfg, "network_hash_slots"))
    cats_top_k = _must(identity_cfg, "categories_top_k")
    keys = [f"id_sym_hash_{i}" for i in range(sym_slots)]
    keys += [f"id_net_hash_{i}" for i in range(net_slots)]
    keys += [f"id_cat_{cat}" for cat in cats_top_k]
    return keys
