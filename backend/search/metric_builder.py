"""Baut pro Candle die Werte-Arrays fuer alle metrics (base, indicators, window).
Keine Fallbacks — None-Werte sind erlaubt wo noch keine Historie (slopes auf rsi am Anfang)."""

from .indicators import rsi, macd_hist, bollinger_pos, atr
from .window_metrics import rolling_sum, window_pct_change, slope_over
from .predictor_settings import WINDOW_METRIC_SETTINGS

BASE_METRICS = {"volume", "trades", "body_pct", "range_pct",
                "upper_wick_pct", "lower_wick_pct", "close_delta_pct"}
INDICATOR_METRICS = {"rsi_14", "macd_hist", "bollinger_pos", "atr_14"}

# Operationen fuer Window-Familien
_OPS = {
    "sum": rolling_sum,
    "pct": window_pct_change,
    "slope": slope_over,
}


def _candle_basic(c, prev_c):
    o, h, l, cl = float(c['open']), float(c['high']), float(c['low']), float(c['close'])
    vol = float(c.get('volume', 0))
    tr = float(c.get('trades', 0))
    body = abs(cl - o)
    rng = h - l
    if rng == 0:
        body_pct = upper_w = lower_w = 0.0
    else:
        body_pct = body / rng * 100
        upper_w = (h - max(o, cl)) / rng * 100
        lower_w = (min(o, cl) - l) / rng * 100
    range_pct = (rng / l * 100) if l > 0 else 0.0
    if prev_c is not None and float(prev_c['close']) > 0:
        close_delta = (cl - float(prev_c['close'])) / float(prev_c['close']) * 100
    else:
        close_delta = 0.0
    return {
        'volume': vol, 'trades': tr,
        'body_pct': body_pct, 'range_pct': range_pct,
        'upper_wick_pct': upper_w, 'lower_wick_pct': lower_w,
        'close_delta_pct': close_delta,
    }


def build_metric_arrays(candles, requested_metrics):
    """Rueckgabe: dict[metric_name, list[value_per_candle | None]]."""
    arrays = {}

    # 1) Basis-Metriken pro Candle
    base_needed = [m for m in requested_metrics if m in BASE_METRICS]
    if base_needed:
        for m in base_needed:
            arrays[m] = []
        prev = None
        for c in candles:
            vals = _candle_basic(c, prev)
            for m in base_needed:
                arrays[m].append(vals[m])
            prev = c

    # 2) Indikatoren
    closes = [float(c['close']) for c in candles]
    if 'rsi_14' in requested_metrics:
        arrays['rsi_14'] = rsi(closes, 14)
    if 'macd_hist' in requested_metrics:
        arrays['macd_hist'] = macd_hist(closes)
    if 'bollinger_pos' in requested_metrics:
        arrays['bollinger_pos'] = bollinger_pos(closes)
    if 'atr_14' in requested_metrics:
        arrays['atr_14'] = atr(candles, 14)

    # 3) Window-Metriken (family_window wie 'volume_sum_5')
    families = WINDOW_METRIC_SETTINGS["families"]
    sizes = WINDOW_METRIC_SETTINGS["window_sizes"]
    for fam_name, fam_cfg in families.items():
        source = fam_cfg["source"]
        op = _OPS[fam_cfg["op"]]
        for w in sizes:
            key = f"{fam_name}_{w}"
            if key not in requested_metrics:
                continue
            # Source-Array besorgen
            if source == "close":
                src_arr = closes
            elif source in BASE_METRICS:
                # Stelle sicher dass Basis da ist (laden wenn nicht requested)
                if source not in arrays:
                    src_arr = []
                    prev = None
                    for c in candles:
                        src_arr.append(_candle_basic(c, prev)[source])
                        prev = c
                else:
                    src_arr = arrays[source]
            elif source in INDICATOR_METRICS:
                if source not in arrays:
                    if source == "rsi_14":
                        src_arr = rsi(closes, 14)
                    elif source == "macd_hist":
                        src_arr = macd_hist(closes)
                    elif source == "bollinger_pos":
                        src_arr = bollinger_pos(closes)
                    elif source == "atr_14":
                        src_arr = atr(candles, 14)
                    else:
                        raise ValueError(f"Unbekannte indicator source: {source}")
                else:
                    src_arr = arrays[source]
            else:
                raise ValueError(f"Unbekannte Window-Source: {source}")

            arrays[key] = op(src_arr, w)

    return arrays
