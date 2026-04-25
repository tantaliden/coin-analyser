"""Slope-Scoring mit Dauer- und Positions-Unschaerfe.
Iteriert ueber alle moeglichen Sub-Fenster innerhalb der konfigurierten Toleranzen.
Keine Fallbacks — Pflichtfelder erzwungen."""

from .candle_aggregator import get_candle_at_offset, find_candles_in_window
from .predictor_settings import WINDOW_SCAN_SETTINGS


def _require(d, keys, ctx):
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(f"{ctx}: Pflichtfeld fehlt: {k}")


def score_slope_with_window(crit, fz, candles, event_time):
    """Erweiterter Slope-Score mit Dauer- und Positions-Unschaerfe.

    crit-Felder:
      - field: close/high/low/open/volume/trades
      - value: Ziel-Prozent (oder Range-Mitte)
      - value2 (optional): fuer Range-Modus (z.B. 20-50% Anstieg)
      - end_offset: Fenster-ENDE (naeher am Initialpunkt, negativer Wert oder 0)
      - duration_minutes: Dauer des Anstieg-Fensters
    fz-Felder:
      - positionTolerance (Min): wie weit das end_offset verschoben werden darf (+/-)
      - durationTolerance (Min): wie weit duration_minutes variieren darf (+/-)
      - slopeTolerance: Prozent-Abweichung fuer Match-Bewertung
    """
    _require(crit, {'field', 'value', 'end_offset', 'duration_minutes'}, 'score_slope_with_window')
    _require(fz, {'slopeTolerance', 'positionTolerance', 'durationTolerance'}, 'fuzzy')

    field = crit['field']
    target = crit['value']
    target_max = crit.get('value2')  # optional: Range-Modus
    end_off = crit['end_offset']
    duration = crit['duration_minutes']

    pos_tol = fz['positionTolerance']
    dur_tol = fz['durationTolerance']
    slope_tol = fz['slopeTolerance']

    # Alle moeglichen Fenster iterieren
    pos_step = WINDOW_SCAN_SETTINGS['position_step_candles']
    dur_step = WINDOW_SCAN_SETTINGS['duration_step_candles']

    best_score = 0.0
    best_meta = None

    end_min = end_off - pos_tol
    end_max = end_off + pos_tol
    dur_min = max(1, duration - dur_tol)
    dur_max = duration + dur_tol

    end_cur = end_min
    while end_cur <= end_max:
        d = dur_min
        while d <= dur_max:
            start_off = end_cur - d
            c_start = get_candle_at_offset(candles, event_time, start_off, tolerance_minutes=0)
            c_end = get_candle_at_offset(candles, event_time, end_cur, tolerance_minutes=0)
            if c_start and c_end:
                v1 = float(c_start[field]) if field in c_start else None
                v2 = float(c_end[field]) if field in c_end else None
                if v1 is not None and v2 is not None and v1 != 0:
                    actual = ((v2 - v1) / v1) * 100

                    # Range-Modus: innerhalb [target, target_max]?
                    if target_max is not None:
                        lo, hi = min(target, target_max), max(target, target_max)
                        if lo <= actual <= hi:
                            score = 1.0
                        else:
                            diff = min(abs(actual - lo), abs(actual - hi))
                            score = max(0.0, 1.0 - diff / slope_tol) if slope_tol > 0 else 0.0
                    else:
                        diff = abs(actual - target)
                        score = max(0.0, 1.0 - diff / slope_tol) if slope_tol > 0 else (1.0 if diff == 0 else 0.0)

                    if score > best_score:
                        best_score = score
                        best_meta = {
                            'actual_slope': round(actual, 3),
                            'duration': d,
                            'end_offset': end_cur,
                            'start_offset': start_off,
                        }
            d += dur_step
        end_cur += pos_step

    return best_score, best_meta
