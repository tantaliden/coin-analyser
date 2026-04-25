"""
BLIND SCANNER — Scannt alle Zeitpunkte in klines/aggs.
Keine Fallbacks. Kriterien muessen alle Pflichtfelder haben.
"""

from datetime import datetime, timedelta
import pytz

from .predictor_settings import SCAN_SETTINGS, INITIAL_POINT_SETTINGS
from .pattern_scorer import detect_pattern, detect_candle_sequence
from .criterion_scorer import score_criterion

BERLIN_TZ = pytz.timezone('Europe/Berlin')


def scan_symbol(symbol, candles, criteria, initial_points_cfg, global_fuzzy):
    """Scannt alle Zeitpunkte eines Symbols.

    initial_points_cfg: dict mit keys 'config' (match_mode, threshold, enforce_sequence, window_minutes)
                       und 'points' (Liste von Initialpunkt-Kriterien)
    """
    if not isinstance(initial_points_cfg, dict):
        raise TypeError("initial_points_cfg muss dict sein")
    if 'config' not in initial_points_cfg or 'points' not in initial_points_cfg:
        raise ValueError("initial_points_cfg braucht 'config' und 'points'")
    if not isinstance(global_fuzzy, dict):
        raise TypeError("global_fuzzy muss dict sein")

    if not candles or len(candles) < 2:
        return []

    hits = []
    cfg = initial_points_cfg['config']
    points = initial_points_cfg['points']

    for required in ('match_mode', 'match_threshold', 'enforce_sequence', 'window_minutes'):
        if required not in cfg:
            raise ValueError(f"initial_points_cfg.config fehlt: {required}")

    for i, candle in enumerate(candles):
        event_time = candle['open_time']

        if points:
            ip_pass, ip_score = check_initial_points(
                candles, i, event_time, points,
                cfg['match_mode'], cfg['match_threshold'],
                cfg['enforce_sequence'], cfg['window_minutes'],
                global_fuzzy
            )
            if not ip_pass:
                continue
        else:
            ip_score = 1.0

        crit_scores = []
        all_pass = True
        min_score = SCAN_SETTINGS['per_criterion_min_score']
        for crit in criteria:
            s = score_criterion(crit, candles, event_time, global_fuzzy, BERLIN_TZ)
            crit_scores.append(s)
            if s < min_score:
                all_pass = False
                break

        if not all_pass:
            continue

        if crit_scores:
            avg_score = sum(crit_scores) / len(crit_scores)
            final_score = (ip_score + avg_score) / 2
        else:
            final_score = ip_score

        hits.append({
            'symbol': symbol,
            'event_time': event_time,
            'match_score': round(final_score, 3),
            'ip_score': round(ip_score, 3),
            'criteria_scores': [round(s, 3) for s in crit_scores],
        })

    return hits


def check_initial_points(candles, event_idx, event_time, points,
                          match_mode, threshold, enforce_sequence, window_min, global_fuzzy):
    """Initialpunkt-Set im Fenster [-window, 0] vor event_time pruefen."""
    start_idx = max(0, event_idx - window_min)
    window_candles = [(j, candles[j]) for j in range(start_idx, event_idx + 1)]
    if not window_candles:
        return False, 0.0

    min_per_crit = SCAN_SETTINGS['per_criterion_min_score']
    scores = []
    match_times = []

    for ip in points:
        best_score = 0.0
        best_time = None

        if ip.get('fixed_offset') is not None:
            target_idx = event_idx + ip['fixed_offset']
            if 0 <= target_idx < len(candles):
                best_score = eval_point(ip, candles, target_idx, global_fuzzy)
                if best_score > 0:
                    best_time = candles[target_idx]['open_time']
        else:
            for j, c in window_candles:
                s = eval_point(ip, candles, j, global_fuzzy)
                if s > best_score:
                    best_score = s
                    best_time = c['open_time']

        scores.append(best_score)
        match_times.append(best_time)

    if enforce_sequence and len([t for t in match_times if t is not None]) > 1:
        valid = [t for t in match_times if t is not None]
        if valid != sorted(valid):
            return False, 0.0

    matched = sum(1 for s in scores if s >= min_per_crit)
    if match_mode == 'all':
        passes = matched == len(scores)
    elif match_mode == 'atleast':
        passes = matched >= threshold
    else:
        raise ValueError(f"Unbekannter match_mode: {match_mode}")

    avg = sum(scores) / len(scores) if scores else 0
    return passes, avg


def eval_point(ip, candles, idx, global_fuzzy):
    """Bewertet einen Initialpunkt an einem Candle-Index."""
    if 'kind' not in ip:
        raise ValueError("Initialpunkt hat kein 'kind'")
    kind = ip['kind']

    if kind == 'pattern':
        if 'pattern_id' not in ip or ip['pattern_id'] is None:
            raise ValueError("pattern-Initialpunkt braucht pattern_id")
        return detect_pattern(ip['pattern_id'], candles, idx)

    if kind == 'sequence':
        if 'sequence' not in ip or ip['sequence'] is None:
            raise ValueError("sequence-Initialpunkt braucht sequence")
        return detect_candle_sequence(ip['sequence'], candles, idx)

    return score_criterion(ip, candles, candles[idx]['open_time'], global_fuzzy, BERLIN_TZ)


def attach_pct_labels(hits, cursor):
    """Haengt kline_metrics pct-Werte als Labels an."""
    if not hits:
        return hits

    for hit in hits:
        cursor.execute("""
            SELECT pct_30m, pct_60m, pct_90m, pct_120m, pct_180m, pct_240m,
                   pct_300m, pct_360m, pct_480m, pct_600m
            FROM kline_metrics
            WHERE symbol = %s AND open_time = %s
        """, (hit['symbol'], hit['event_time']))
        row = cursor.fetchone()
        if row:
            hit['pct_labels'] = {k: round(float(v), 2) for k, v in dict(row).items() if v is not None}
        else:
            hit['pct_labels'] = {}

    return hits
