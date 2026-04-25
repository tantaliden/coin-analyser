"""Echte Candle-Pattern-Erkennung. Arbeitet auf In-Memory Candle-Arrays."""

from .predictor_settings import PATTERN_THRESHOLDS as T


def _body(c):
    return abs(c['close'] - c['open'])

def _range(c):
    return c['high'] - c['low']

def _body_ratio(c):
    r = _range(c)
    return _body(c) / r if r > 0 else 0

def _upper_wick(c):
    return c['high'] - max(c['open'], c['close'])

def _lower_wick(c):
    return min(c['open'], c['close']) - c['low']

def _is_green(c):
    return c['close'] > c['open']

def _is_red(c):
    return c['close'] < c['open']


def detect_pattern(pattern_id, candles, target_idx):
    """Prueft ob pattern_id an target_idx vorliegt. Gibt Score 0.0-1.0 zurueck."""
    if target_idx < 0 or target_idx >= len(candles):
        return 0.0

    c = candles[target_idx]
    r = _range(c)
    if r == 0:
        return 0.0

    # Single-candle patterns
    if pattern_id == 'doji':
        return 1.0 if _body_ratio(c) <= T['doji_body_ratio'] else 0.0

    if pattern_id == 'hammer':
        lw = _lower_wick(c)
        b = _body(c)
        if b == 0:
            return 0.0
        return 1.0 if (lw / b >= T['hammer_wick_ratio'] and _upper_wick(c) / r < 0.15) else 0.0

    if pattern_id == 'inverted_hammer':
        uw = _upper_wick(c)
        b = _body(c)
        if b == 0:
            return 0.0
        return 1.0 if (uw / b >= T['hammer_wick_ratio'] and _lower_wick(c) / r < 0.15) else 0.0

    if pattern_id == 'spinning_top':
        br = _body_ratio(c)
        return 1.0 if (br <= T['spinning_top_body_ratio'] and _upper_wick(c) > 0 and _lower_wick(c) > 0) else 0.0

    if pattern_id == 'marubozu_bull':
        return 1.0 if (_is_green(c) and _upper_wick(c) / r <= T['marubozu_wick_ratio']
                       and _lower_wick(c) / r <= T['marubozu_wick_ratio']) else 0.0

    if pattern_id == 'marubozu_bear':
        return 1.0 if (_is_red(c) and _upper_wick(c) / r <= T['marubozu_wick_ratio']
                       and _lower_wick(c) / r <= T['marubozu_wick_ratio']) else 0.0

    # Two-candle patterns (brauchen Vorgaenger)
    if target_idx < 1:
        return 0.0
    prev = candles[target_idx - 1]

    if pattern_id == 'engulfing_bull':
        if _is_red(prev) and _is_green(c):
            if c['open'] <= prev['close'] and c['close'] >= prev['open']:
                if _body(c) > _body(prev) * T['engulfing_min_body']:
                    return 1.0
        return 0.0

    if pattern_id == 'engulfing_bear':
        if _is_green(prev) and _is_red(c):
            if c['open'] >= prev['close'] and c['close'] <= prev['open']:
                if _body(c) > _body(prev) * T['engulfing_min_body']:
                    return 1.0
        return 0.0

    if pattern_id == 'harami_bull':
        if _is_red(prev) and _is_green(c):
            if c['open'] >= prev['close'] and c['close'] <= prev['open'] and _body(c) < _body(prev):
                return 1.0
        return 0.0

    if pattern_id == 'harami_bear':
        if _is_green(prev) and _is_red(c):
            if c['open'] <= prev['close'] and c['close'] >= prev['open'] and _body(c) < _body(prev):
                return 1.0
        return 0.0

    if pattern_id == 'tweezer_top':
        tol = T['tweezer_tolerance'] * max(prev['high'], c['high'])
        return 1.0 if abs(prev['high'] - c['high']) <= tol else 0.0

    if pattern_id == 'tweezer_bottom':
        tol = T['tweezer_tolerance'] * max(prev['low'], c['low']) if prev['low'] > 0 else 0.001
        return 1.0 if abs(prev['low'] - c['low']) <= tol else 0.0

    if pattern_id == 'piercing':
        if _is_red(prev) and _is_green(c):
            mid = (prev['open'] + prev['close']) / 2
            if c['open'] < prev['close'] and c['close'] > mid and c['close'] < prev['open']:
                return 1.0
        return 0.0

    if pattern_id == 'dark_cloud':
        if _is_green(prev) and _is_red(c):
            mid = (prev['open'] + prev['close']) / 2
            if c['open'] > prev['close'] and c['close'] < mid and c['close'] > prev['open']:
                return 1.0
        return 0.0

    # Three-candle patterns
    if target_idx < 2:
        return 0.0
    pp = candles[target_idx - 2]

    if pattern_id == 'morning_star':
        if _is_red(pp) and _body_ratio(prev) <= T['doji_body_ratio'] and _is_green(c):
            if c['close'] > (pp['open'] + pp['close']) / 2:
                return 1.0
        return 0.0

    if pattern_id == 'evening_star':
        if _is_green(pp) and _body_ratio(prev) <= T['doji_body_ratio'] and _is_red(c):
            if c['close'] < (pp['open'] + pp['close']) / 2:
                return 1.0
        return 0.0

    if pattern_id == 'three_white':
        if _is_green(pp) and _is_green(prev) and _is_green(c):
            if prev['close'] > pp['close'] and c['close'] > prev['close']:
                return 1.0
        return 0.0

    if pattern_id == 'three_black':
        if _is_red(pp) and _is_red(prev) and _is_red(c):
            if prev['close'] < pp['close'] and c['close'] < prev['close']:
                return 1.0
        return 0.0

    return 0.0


def detect_candle_sequence(sequence_str, candles, target_idx):
    """Prueft Candle-Folge wie 'GGRG' (Green/Green/Red/Green).
    target_idx ist der letzte Candle-Index der Folge."""
    seq_len = len(sequence_str)
    start = target_idx - seq_len + 1
    if start < 0:
        return 0.0

    for i, expected in enumerate(sequence_str.upper()):
        c = candles[start + i]
        if expected == 'G' and not _is_green(c):
            return 0.0
        if expected == 'R' and not _is_red(c):
            return 0.0
        # 'D' = Doji
        if expected == 'D' and _body_ratio(c) > PATTERN_THRESHOLDS['doji_body_ratio']:
            return 0.0

    return 1.0
