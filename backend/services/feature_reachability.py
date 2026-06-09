"""Erreichbarkeits-Kopf (Easy-Run-Filter, 29.05.2026, Volker-Vision).

Prüft VOR dem Deploy einer Prediction, ob sie ein "Easy Run" ist: Liefen in der
Coin-Historie vergleichbare Situationen (gleiche Vorboten) zu >= reach_rate_target
so, dass das gesetzte TP (peak_share × erwarteter Peak) erreicht wurde UND dabei
nah am realen Peak lag? Nur dann deployen.

Datenbasis: agg_1m des Coins selbst (letzte lookback_days), NICHT die Predictions
(zu dünnes, verzerrtes Sample). Vorboten = Dynamik der profile_window_minutes vor
dem Move (analog event_finder, das Original bleibt unangetastet — dies ist die Kopie).

KEIN silent fallback: returns dict mit 'pass' (bool) + Diagnose, oder None bei zu
wenig Daten / Fehler (Caller behandelt None = skip, wenn Kopf aktiv).

Gated durch settings predictor.reachability.enabled — default AUS.
"""

from psycopg2.extras import RealDictCursor
import logging
import math

log = logging.getLogger("predictor")

# Vorboten-Profil-Keys (analog event_finder.analyze_pre_event)
PROFILE_KEYS = ['vol_trend', 'taker_shift', 'cumulative_pct', 'last30_pct', 'range_expansion']


def _profile_from_block(block):
    """block = Liste von agg_1m-Dicts (chronologisch) über das Vorboten-Fenster.
    Returns dict mit PROFILE_KEYS oder None bei zu wenig/ungültigen Daten."""
    n = len(block)
    if n < 12:
        return None
    half = n // 2
    first, second = block[:half], block[half:]

    def _sum(rows, k):
        return sum(float(r[k] or 0) for r in rows)

    vol_first = _sum(first, 'volume') or 1.0
    vol_second = _sum(second, 'volume') or 1.0
    vol_trend = vol_second / vol_first

    # Taker-Ratio (taker_buy_base / volume) je Hälfte
    def _taker_ratio(rows):
        v = _sum(rows, 'volume')
        tb = _sum(rows, 'taker_buy_base')
        return (tb / v * 100.0) if v > 0 else 50.0
    taker_shift = _taker_ratio(second) - _taker_ratio(first)

    # Momentum: pct_change je 1m-Bucket aufsummiert
    pcts = []
    for r in block:
        o = float(r['open'] or 0); c = float(r['close'] or 0)
        pcts.append(((c - o) / o * 100.0) if o > 0 else 0.0)
    cumulative_pct = sum(pcts)
    last30 = block[-30:] if n >= 30 else block
    last30_pct = sum(((float(r['close'] or 0) - float(r['open'] or 0)) / float(r['open'] or 1) * 100.0)
                     for r in last30 if float(r['open'] or 0) > 0)

    def _range_avg(rows):
        rs = []
        for r in rows:
            hi = float(r['high'] or 0); lo = float(r['low'] or 0)
            if lo > 0:
                rs.append((hi - lo) / lo * 100.0)
        return (sum(rs) / len(rs)) if rs else 0.0
    rf, rs = _range_avg(first), _range_avg(second)
    range_expansion = (rs / rf) if rf > 0 else 1.0

    return {
        'vol_trend': round(vol_trend, 4),
        'taker_shift': round(taker_shift, 4),
        'cumulative_pct': round(cumulative_pct, 4),
        'last30_pct': round(last30_pct, 4),
        'range_expansion': round(range_expansion, 4),
    }


def check_reachability(coins_conn, symbol, side, predicted_peak_pct, cfg):
    """Haupt-Check. cfg = predictor.reachability-Block.

    side: 'long'/'short'. predicted_peak_pct: erwarteter Peak in % (tp_raw, vor peak_share).
    Returns dict {pass, reach_rate, n_cases, n_match, set_tp_pct} oder None (zu wenig Daten).
    """
    lookback_days = int(cfg.get("lookback_days", 12))
    profile_window = int(cfg.get("profile_window_minutes", 120))
    sample_step = int(cfg.get("sample_step_minutes", 5))
    forward_window = int(cfg.get("forward_window_minutes", 60))
    distance_threshold = float(cfg.get("distance_threshold", 1.5))
    min_cases = int(cfg.get("min_cases", 20))
    peak_share = float(cfg.get("peak_share", 0.8))
    reach_rate_target = float(cfg.get("reach_rate_target", 0.8))
    min_tp_pct = float(cfg.get("min_tp_pct", 0.7))

    set_tp_pct = peak_share * float(predicted_peak_pct)
    if set_tp_pct < min_tp_pct:
        return {'pass': False, 'reason': 'tp_below_min', 'set_tp_pct': round(set_tp_pct, 3),
                'reach_rate': None, 'n_cases': 0, 'n_match': 0}

    try:
        with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT bucket, open, high, low, close, volume, taker_buy_base
                FROM agg_1m WHERE symbol=%s
                  AND bucket > now() - make_interval(days => %s)
                ORDER BY bucket ASC
            """, (symbol, lookback_days))
            rows = cur.fetchall()
    except Exception as e:
        log.error("FALLBACK_TRIGGERED check_reachability %s: DB error %s -> None", symbol, e)
        try: coins_conn.rollback()
        except Exception: pass
        return None

    n = len(rows)
    if n < profile_window + forward_window + min_cases:
        return None  # zu wenig Historie (neuer Coin / Lücke) -> Caller skip

    # Aktuelles Profil = letztes vollständiges Vorboten-Fenster
    cur_profile = _profile_from_block(rows[-profile_window:])
    if cur_profile is None:
        return None

    # Historische Profile (gesampelt) + Forward-Peak in Gewinnrichtung
    hist = []  # (profile, peak_pct in Gewinnrichtung)
    last_idx = n - forward_window  # ab hier kein volles Forward-Fenster mehr
    i = profile_window
    while i < last_idx:
        block = rows[i - profile_window:i]
        prof = _profile_from_block(block)
        if prof is not None:
            entry = float(rows[i]['close'] or 0)
            if entry > 0:
                fwd = rows[i:i + forward_window]
                if side == 'long':
                    peak = max((float(r['high'] or 0) for r in fwd), default=entry)
                    peak_pct = (peak - entry) / entry * 100.0
                else:
                    trough = min((float(r['low'] or 0) for r in fwd if float(r['low'] or 0) > 0), default=entry)
                    peak_pct = (entry - trough) / entry * 100.0
                hist.append((prof, peak_pct))
        i += sample_step

    if len(hist) < min_cases:
        return None

    # Z-Normalisierung pro Feature über die historischen Profile
    stats = {}
    for k in PROFILE_KEYS:
        vals = [p[k] for p, _ in hist]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1.0
        stats[k] = (mean, std)

    def _dist(p):
        s = 0.0
        for k in PROFILE_KEYS:
            mean, std = stats[k]
            s += ((p[k] - mean) / std - (cur_profile[k] - mean) / std) ** 2
        return math.sqrt(s)

    # Vergleichsfälle: Profil-Distanz < threshold
    matched = [(p, peak_pct) for p, peak_pct in hist if _dist(p) <= distance_threshold]
    n_match = len(matched)
    if n_match < min_cases:
        return {'pass': False, 'reason': 'too_few_matches', 'reach_rate': None,
                'n_cases': len(hist), 'n_match': n_match, 'set_tp_pct': round(set_tp_pct, 3)}

    # Counterfactual: in wie viel % der Matches galt BEIDES:
    #   (a) realer Peak >= gesetztes TP  (getroffen)
    #   (b) gesetztes TP >= peak_share × realem Peak  (nah dran, real in [tp, tp/peak_share])
    hits = 0
    for _, real_peak in matched:
        if real_peak >= set_tp_pct and set_tp_pct >= peak_share * real_peak:
            hits += 1
    reach_rate = hits / n_match

    return {
        'pass': reach_rate >= reach_rate_target,
        'reach_rate': round(reach_rate, 3),
        'n_cases': len(hist),
        'n_match': n_match,
        'set_tp_pct': round(set_tp_pct, 3),
    }
