"""Scored einzelne Kriterien gegen in-memory Candle-Daten. Keine Fallbacks — fehlende Felder = Exception."""

from datetime import datetime, timedelta
from .window_slope_scorer import score_slope_with_window
from .candle_aggregator import get_candle_at_offset, find_candles_in_window, get_day_open_from_candles


REQUIRED_CRITERION_KEYS = {'kind', 'field'}


def _require(d, keys, ctx=''):
    """Wirft ValueError wenn eines der keys fehlt oder None ist."""
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(f"{ctx}: Pflichtfeld fehlt oder ist None: {k}")


def score_criterion(crit, candles, event_time, global_fuzzy, tz):
    """Haupteinstieg: scored ein Kriterium."""
    _require(crit, REQUIRED_CRITERION_KEYS, 'criterion')
    if not isinstance(global_fuzzy, dict):
        raise TypeError("global_fuzzy muss dict sein")

    kind = crit['kind']
    fz = crit.get('fuzzy') if crit.get('fuzzy') is not None else global_fuzzy

    if kind == 'value':
        return score_value(crit, fz, candles, event_time, tz)
    if kind == 'range':
        return score_range(crit, fz, candles, event_time, tz)
    if kind == 'slope':
        if crit.get('end_offset') is not None and crit.get('duration_minutes') is not None:
            score, _meta = score_slope_with_window(crit, fz, candles, event_time)
            return score
        return score_slope(crit, fz, candles, event_time)
    if kind == 'ratio':
        return score_ratio(crit, fz, candles, event_time)

    raise ValueError(f"Unbekannter kind: {kind}")


def extract_field(field, candle, day_open=None):
    """Extrahiert Wert aus Candle. Wirft Exception wenn Feld fehlt."""
    if candle is None:
        raise ValueError("extract_field: candle ist None")
    if field in ('open', 'high', 'low', 'close', 'volume', 'trades'):
        if field not in candle or candle[field] is None:
            raise ValueError(f"Candle hat Feld nicht: {field}")
        return float(candle[field])
    if field.endswith('_pct_dayopen'):
        if day_open is None or day_open == 0:
            raise ValueError(f"day_open fehlt oder 0 fuer Feld {field}")
        base_field = field.replace('_pct_dayopen', '')
        if base_field not in candle or candle[base_field] is None:
            raise ValueError(f"Candle hat base_field nicht: {base_field}")
        return ((float(candle[base_field]) - day_open) / day_open) * 100
    if field == 'body_pct':
        o, c = float(candle['open']), float(candle['close'])
        if o == 0:
            raise ValueError("body_pct: open ist 0")
        return (c - o) / o * 100
    if field == 'range_pct':
        h, l = float(candle['high']), float(candle['low'])
        if l == 0:
            raise ValueError("range_pct: low ist 0")
        return (h - l) / l * 100
    raise ValueError(f"Unbekanntes field: {field}")


def _fz(fz, key):
    """Holt Fuzzy-Wert. Wirft Exception wenn fehlt."""
    if isinstance(fz, dict):
        if key not in fz or fz[key] is None:
            raise ValueError(f"Fuzzy-Schluessel fehlt: {key}")
        return fz[key]
    val = getattr(fz, key, None)
    if val is None:
        raise ValueError(f"Fuzzy-Schluessel fehlt: {key}")
    return val


def score_value(crit, fz, candles, event_time, tz):
    """Wert bei Zeitpunkt oder Fenster."""
    _require(crit, {'field', 'value'}, 'score_value')
    time_tol = _fz(fz, 'timeTolerance')

    from_off = crit.get('time_offset_from')
    to_off = crit.get('time_offset_to')
    if from_off is not None and to_off is not None:
        window = find_candles_in_window(candles, event_time, from_off, to_off)
        if not window:
            return 0.0
        return max(_score_single_value(crit, fz, c, candles, event_time, tz) for c in window)

    if 'time_offset' not in crit or crit['time_offset'] is None:
        raise ValueError("score_value: time_offset oder time_offset_from/to muss gesetzt sein")

    candle = get_candle_at_offset(candles, event_time, crit['time_offset'], time_tol)
    if not candle:
        return 0.0
    return _score_single_value(crit, fz, candle, candles, event_time, tz)


def _score_single_value(crit, fz, candle, candles, event_time, tz):
    day_open = None
    if 'pct_dayopen' in crit['field']:
        day_open = get_day_open_from_candles(candles, event_time, tz)
        if day_open is None:
            return 0.0

    actual = extract_field(crit['field'], candle, day_open)
    target = crit['value']
    val_tol = _fz(fz, 'valueTolerance') / 100.0

    if target == 0:
        return 1.0 if abs(actual) < 0.0001 else 0.0
    diff = abs(actual - target) / abs(target)
    if val_tol <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / val_tol)


def score_range(crit, fz, candles, event_time, tz):
    _require(crit, {'field'}, 'score_range')
    time_tol = _fz(fz, 'timeTolerance')

    from_off = crit.get('time_offset_from')
    to_off = crit.get('time_offset_to')
    if from_off is not None and to_off is not None:
        window = find_candles_in_window(candles, event_time, from_off, to_off)
        if not window:
            return 0.0
        return max(_score_single_range(crit, fz, c, candles, event_time, tz) for c in window)

    if 'time_offset' not in crit or crit['time_offset'] is None:
        raise ValueError("score_range: time_offset oder time_offset_from/to muss gesetzt sein")

    candle = get_candle_at_offset(candles, event_time, crit['time_offset'], time_tol)
    if not candle:
        return 0.0
    return _score_single_range(crit, fz, candle, candles, event_time, tz)


def _score_single_range(crit, fz, candle, candles, event_time, tz):
    day_open = None
    if 'pct_dayopen' in crit['field']:
        day_open = get_day_open_from_candles(candles, event_time, tz)
        if day_open is None:
            return 0.0

    actual = extract_field(crit['field'], candle, day_open)

    lo = _fz(fz, 'rangeMin') if 'rangeMin' in fz else float('-inf')
    hi = _fz(fz, 'rangeMax') if 'rangeMax' in fz else float('inf')
    if lo is None:
        lo = float('-inf')
    if hi is None:
        hi = float('inf')

    if lo <= actual <= hi:
        return 1.0

    val_tol = _fz(fz, 'valueTolerance') / 100.0
    if actual < lo:
        diff = (lo - actual) / (abs(lo) if lo != 0 else 1)
    else:
        diff = (actual - hi) / (abs(hi) if hi != 0 else 1)
    if val_tol <= 0:
        return 0.0
    return max(0.0, 1.0 - diff / val_tol)


def score_slope(crit, fz, candles, event_time):
    _require(crit, {'field', 'value', 'time_offset', 'time_offset2'}, 'score_slope')
    time_tol = _fz(fz, 'timeTolerance')

    c1 = get_candle_at_offset(candles, event_time, crit['time_offset'], time_tol)
    c2 = get_candle_at_offset(candles, event_time, crit['time_offset2'], time_tol)
    if not c1 or not c2:
        return 0.0

    v1 = extract_field(crit['field'], c1)
    v2 = extract_field(crit['field'], c2)
    if v1 == 0:
        raise ValueError("score_slope: v1 ist 0")

    actual_slope = ((v2 - v1) / v1) * 100
    target_slope = crit['value']

    slope_tol = _fz(fz, 'slopeTolerance')
    diff = abs(actual_slope - target_slope)
    if slope_tol <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / slope_tol)


def score_ratio(crit, fz, candles, event_time):
    _require(crit, {'field', 'field2', 'value', 'time_offset'}, 'score_ratio')
    time_tol = _fz(fz, 'timeTolerance')
    candle = get_candle_at_offset(candles, event_time, crit['time_offset'], time_tol)
    if not candle:
        return 0.0

    v1 = extract_field(crit['field'], candle)
    v2 = extract_field(crit['field2'], candle)
    if v2 == 0:
        raise ValueError("score_ratio: v2 ist 0")

    actual = v1 / v2
    target = crit['value']

    ratio_tol = _fz(fz, 'ratioTolerance') / 100.0
    if target == 0:
        return 1.0 if abs(actual) < 0.0001 else 0.0
    diff = abs(actual - target) / abs(target)
    if ratio_tol <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / ratio_tol)
