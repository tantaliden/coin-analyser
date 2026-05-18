"""
Stalker-Predictor — Cross-Coin-Korrelations- und Lead/Lag-Features
====================================================================
Pro scan-pass: Returns für Reference-Coins (BTC, ETH, SOL, BNB) und Target
über mehrere Fenster vergleichen → Pearson-Korrelationen + Lead/Lag-Marker.

Reference-Returns werden 1× pro scan-pass für alle Coins gecached
(`scan_state` Dict).

Settings: predictor.stalker.cross_coin.{reference_coins, corr_windows_minutes,
          lead_lag_window_minutes, lead_lag_threshold_pct}
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


def _must(cfg, key):
    v = cfg.get(key)
    if v is None:
        log.error(f"FALLBACK_TRIGGERED feature_cross_coin: '{key}' fehlt")
        raise RuntimeError(f"settings.predictor.stalker.cross_coin.{key} missing")
    return v


def _fetch_closes(coins_conn, symbol, n_minutes):
    """Letzte n agg_1m closes, älteste zuerst. Returnt None wenn zu wenig Daten."""
    with coins_conn.cursor() as cur:
        cur.execute(
            "SELECT close FROM agg_1m WHERE symbol=%s ORDER BY bucket DESC LIMIT %s",
            (symbol, n_minutes + 1),
        )
        rows = [r[0] for r in cur.fetchall()]
    if len(rows) < n_minutes + 1:
        return None
    rows = list(reversed(rows))
    return rows


def _returns(closes):
    out = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev and cur is not None and prev != 0:
            out.append((cur - prev) / prev)
        else:
            out.append(0.0)
    return np.array(out, dtype=float)


def build_reference_cache(coins_conn, cross_cfg):
    """1× pro scan-pass: alle Reference-Coins für alle benötigten Fenster holen.

    Returnt dict: {ref_symbol -> {window_min -> np.array(returns)}}
    Plus: {ref_symbol -> {'last_pct_<lag>m'}} für lead/lag.
    """
    refs = _must(cross_cfg, "reference_coins")
    windows = _must(cross_cfg, "corr_windows_minutes")
    lag_min = _must(cross_cfg, "lead_lag_window_minutes")
    max_window = max(max(windows), lag_min + 1)

    cache = {}
    for ref in refs:
        closes = _fetch_closes(coins_conn, ref, max_window)
        if closes is None:
            log.warning(f"feature_cross_coin: reference {ref} unzureichende agg_1m-Daten")
            cache[ref] = None
            continue
        per_win = {}
        for w in windows:
            # returns für letzte w-Minuten:
            sub = closes[-(w + 1):]
            per_win[w] = _returns(sub)
        last_close = closes[-1]
        old_close = closes[-(lag_min + 1)]
        lead_lag_pct = (
            (last_close - old_close) / old_close * 100.0 if (old_close and last_close is not None) else 0.0
        )
        cache[ref] = {"per_win": per_win, "lead_lag_pct": lead_lag_pct, "closes": closes}
    return cache


def compute_cross_coin_features(coins_conn, symbol, cross_cfg, ref_cache):
    """Returnt dict mit corr_<ref>_<w>m und ll_<ref>_<lag>m_aufholbedarf."""
    refs = _must(cross_cfg, "reference_coins")
    windows = _must(cross_cfg, "corr_windows_minutes")
    lag_min = _must(cross_cfg, "lead_lag_window_minutes")
    lag_th = _must(cross_cfg, "lead_lag_threshold_pct")
    max_window = max(max(windows), lag_min + 1)

    own_closes = _fetch_closes(coins_conn, symbol, max_window)
    feat = {}

    for ref in refs:
        rdata = ref_cache.get(ref)
        # Corr-Features
        for w in windows:
            key_c = f"corr_{ref.lower()}_{w}m"
            if own_closes is None or rdata is None:
                feat[key_c] = 0.0
                continue
            sub_own = own_closes[-(w + 1):]
            r_own = _returns(sub_own)
            r_ref = rdata["per_win"][w]
            if len(r_own) != len(r_ref) or len(r_own) < 5:
                feat[key_c] = 0.0
                continue
            if np.std(r_own) == 0 or np.std(r_ref) == 0:
                feat[key_c] = 0.0
                continue
            c = float(np.corrcoef(r_own, r_ref)[0, 1])
            feat[key_c] = c if np.isfinite(c) else 0.0
        # Lead/Lag
        key_ll = f"ll_{ref.lower()}_{lag_min}m"
        if own_closes is None or rdata is None:
            feat[key_ll] = 0.0
            continue
        own_last = own_closes[-1]
        own_old = own_closes[-(lag_min + 1)]
        own_lag_pct = (
            (own_last - own_old) / own_old * 100.0
            if (own_old and own_last is not None)
            else 0.0
        )
        ref_lag_pct = rdata["lead_lag_pct"]
        # Aufholbedarf: ref bewegt sich stark, wir nicht
        if abs(ref_lag_pct) >= lag_th:
            feat[key_ll] = ref_lag_pct - own_lag_pct
        else:
            feat[key_ll] = 0.0

    return feat


def feature_keys(cross_cfg: dict) -> list:
    refs = _must(cross_cfg, "reference_coins")
    windows = _must(cross_cfg, "corr_windows_minutes")
    lag_min = _must(cross_cfg, "lead_lag_window_minutes")
    keys = []
    for ref in refs:
        for w in windows:
            keys.append(f"corr_{ref.lower()}_{w}m")
        keys.append(f"ll_{ref.lower()}_{lag_min}m")
    return keys
