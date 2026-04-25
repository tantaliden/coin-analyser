"""Window-basierte Metriken: rollende Summen, %-Spruenge ueber N Candles, Slopes.
Keine Fallbacks — partielle Fenster am Anfang liefern partielle Summen (mathematisch korrekt)."""


def rolling_sum(values, window):
    """Rollende Summe. Am Anfang partielle Summe (idx 0 = values[0], ..., idx window-1 = full)."""
    n = len(values)
    result = []
    running = 0.0
    for i in range(n):
        running += float(values[i])
        if i >= window:
            running -= float(values[i - window])
        result.append(running)
    return result


def window_pct_change(values, window):
    """values[i] / values[i-window] - 1 als Prozent.
    Fuer i < window: Referenz ist values[0] (cumulative return from start).
    Wirft ValueError wenn Referenzwert 0 ist."""
    n = len(values)
    result = []
    for i in range(n):
        ref_idx = max(0, i - window)
        prev = float(values[ref_idx])
        cur = float(values[i])
        if prev == 0:
            raise ValueError(f"window_pct_change: values[{ref_idx}] == 0")
        result.append((cur / prev - 1) * 100)
    return result


def slope_over(values, window):
    """values[i] - values[i-window]. Fuer None-Einsprung-Werte: None im Ergebnis."""
    n = len(values)
    result = []
    for i in range(n):
        ref_idx = max(0, i - window)
        prev = values[ref_idx]
        cur = values[i]
        if prev is None or cur is None:
            result.append(None)
        else:
            result.append(float(cur) - float(prev))
    return result
