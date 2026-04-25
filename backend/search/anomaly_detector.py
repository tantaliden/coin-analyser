"""Rolling-Window Anomaly-Detektor. z-score ueber mehrere Metriken.
Liefert Kandidaten-Vorschlaege fuer Initialpunkt/Kriterien.
Keine Fallbacks — fehlende Candles erzeugen leere Ergebnisse explizit."""

import math
from .predictor_settings import ANOMALY_SETTINGS, ALL_PATTERN_IDS
from .metric_builder import build_metric_arrays
from .pattern_scorer import detect_pattern



def _candle_metrics(c, prev_c):
    """Berechnet Metriken-Dict fuer eine Candle. prev_c fuer close_delta_pct."""
    o, h, l, cl = float(c['open']), float(c['high']), float(c['low']), float(c['close'])
    vol = float(c.get('volume', 0))
    tr = float(c.get('trades', 0))

    body = abs(cl - o)
    rng = h - l
    if rng == 0:
        body_pct = 0.0
        upper_w_pct = 0.0
        lower_w_pct = 0.0
    else:
        body_pct = body / rng * 100
        upper_w_pct = (h - max(o, cl)) / rng * 100
        lower_w_pct = (min(o, cl) - l) / rng * 100

    range_pct = (rng / l * 100) if l > 0 else 0.0

    if prev_c is not None:
        pc = float(prev_c['close'])
        close_delta_pct = ((cl - pc) / pc * 100) if pc > 0 else 0.0
    else:
        close_delta_pct = 0.0

    return {
        'volume': vol, 'trades': tr,
        'body_pct': body_pct, 'range_pct': range_pct,
        'upper_wick_pct': upper_w_pct, 'lower_wick_pct': lower_w_pct,
        'close_delta_pct': close_delta_pct,
    }


def _z_score(value, baseline_values):
    if len(baseline_values) < 2:
        return 0.0
    mean = sum(baseline_values) / len(baseline_values)
    variance = sum((v - mean) ** 2 for v in baseline_values) / len(baseline_values)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (value - mean) / std


def _normalize_dt(dt):
    """Entfernt tzinfo (DB-datetimes sind naive, Vergleiche brauchen gleichen Typ)."""
    if dt is None:
        return None
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def detect_anomalies(candles, event_start_time=None):
    """Gibt Liste von Anomalie-Vorschlaegen sortiert nach Gewicht zurueck.

    candles: chronologisch sortierte Liste (aelteste zuerst)
    event_start_time: optional — wenn gesetzt, nur Candles VOR event_start beruecksichtigen
    """
    event_start_time = _normalize_dt(event_start_time)
    window = ANOMALY_SETTINGS['rolling_window_candles']
    min_z = ANOMALY_SETTINGS['min_z_score']
    strong_z = ANOMALY_SETTINGS['strong_z_score']
    metrics = ANOMALY_SETTINGS['metrics']
    weights = ANOMALY_SETTINGS['metric_weights']
    max_sugg = ANOMALY_SETTINGS['max_suggestions']

    if len(candles) <= window:
        return []

    # Alle Metrik-Arrays ueber zentralen Builder
    metric_arrays = build_metric_arrays(candles, set(metrics))

    suggestions = []
    for i in range(window, len(candles)):
        if event_start_time is not None and _normalize_dt(candles[i]['open_time']) >= event_start_time:
            break

        anomalies = []
        total_weight = 0.0
        for key in metrics:
            cur = metric_arrays[key][i]
            if cur is None:
                continue
            baseline_raw = metric_arrays[key][i - window:i]
            baseline = [v for v in baseline_raw if v is not None]
            if len(baseline) < 2:
                continue
            z = _z_score(cur, baseline)
            if abs(z) >= min_z:
                anomalies.append({
                    'metric': key,
                    'z_score': round(z, 2),
                    'value': round(cur, 4),
                    'baseline_mean': round(sum(baseline) / len(baseline), 4),
                    'weight': round(weights[key] * abs(z), 3),
                    'strength': 'strong' if abs(z) >= strong_z else 'moderate',
                })
                total_weight += weights[key] * abs(z)

        if anomalies:
            suggestions.append({
                'candle_index': i,
                'open_time': candles[i]['open_time'],
                'total_weight': round(total_weight, 3),
                'anomalies': anomalies,
                'candle': {
                    'open': float(candles[i]['open']),
                    'high': float(candles[i]['high']),
                    'low': float(candles[i]['low']),
                    'close': float(candles[i]['close']),
                },
            })

    suggestions.sort(key=lambda s: s['total_weight'], reverse=True)
    return suggestions[:max_sugg]


def detect_patterns(candles, event_start_time=None):
    """Gibt Liste von {open_time, pattern_id, score} fuer erkannte Candle-Muster zurueck.
    Nur Candles VOR event_start_time werden geprueft."""
    event_start_time = _normalize_dt(event_start_time)
    hits = []
    for i in range(len(candles)):
        if event_start_time is not None and _normalize_dt(candles[i]['open_time']) >= event_start_time:
            break
        for pid in ALL_PATTERN_IDS:
            score = detect_pattern(pid, candles, i)
            if score > 0:
                hits.append({
                    'open_time': candles[i]['open_time'],
                    'pattern_id': pid,
                    'score': float(score),
                })
    return hits
