"""Reach-Predictor (eigenständig, 09.06.2026, Volker-Vision).

Ersetzt den abgeschalteten Multi-Head. KEIN trainiertes Modell, KEINE geerbte
Richtung — die Richtung kommt aus der Coin-Historie selbst:

  Für Coin C JETZT:
   1. Voller Vorboten-Vektor (alle verfügbaren Mikrostruktur- + Momentum-Features,
      JEDER in JEDEM Vergleich — kein Subsampling, kein Random) aus agg_1m + kline_metrics.
   2. Ähnliche Momente in den letzten lookback_days von C (deterministische
      z-normalisierte Distanz < Schwelle, Mindest-Trefferzahl — kein top-k/random).
   3. Forward-Ausgang JE RICHTUNG messen (long ↑ / short ↓):
        (a) realer Peak >= gesetztes TP          (erreichbar)
        (b) gesetztes TP >= peak_share*realer Peak (nah dran, kein Mikrogewinn)
        (c) SL-Survival: +TP wurde VOR -SL erreicht (Point-of-no-Return aus Verteilung)
   4. reach_rate = Anteil (a∧b∧c). Seite mit reach_rate >= Ziel (und höher) wird
      getradet; keine >= Ziel -> skip. conf = reach_rate (per Konstruktion kalibriert).

Datenbasis: Coin-Historie (agg_1m/kline_metrics), NICHT vergangene Predictions.
Forward-Ausgang aus rohen agg_1m-High/Low (NICHT kline_metrics -> kein Zirkelschluss).
RULE_6: keine silent Fallbacks. Zu wenig Daten/Matches -> None/skip + Diagnose.
"""

from psycopg2.extras import RealDictCursor
import logging
import math

log = logging.getLogger("reach_predictor")

# --- Voller Vorboten-Feature-Vektor. ALLE Keys werden IMMER in der Distanz benutzt. ---
# Mikrostruktur-Dynamik aus dem profile_window (agg_1m):
PROFILE_KEYS = [
    'vol_trend', 'trades_trend', 'taker_ratio', 'taker_shift',
    'cumulative_pct', 'last30_pct', 'range_expansion',
    'book_imbalance', 'book_imb_shift', 'spread', 'funding', 'oi_delta', 'premium',
]
# Multi-Horizont-Rückwärts-Momentum aus kline_metrics am Anchor (Vorboten, kein Leak):
METRIC_KEYS = ['pct_30m', 'pct_60m', 'pct_120m', 'pct_240m', 'pct_360m']
ALL_KEYS = PROFILE_KEYS + METRIC_KEYS


def _f(v):
    return 0.0 if v is None else float(v)


def _profile_from_block(block, metric_row):
    """block = agg_1m-Dicts (chronologisch) über das Vorboten-Fenster.
    metric_row = kline_metrics-Dict am Anchor (kann None sein).
    Returns dict mit ALL_KEYS oder None bei zu wenig Daten."""
    n = len(block)
    if n < 12:
        return None
    half = n // 2
    first, second = block[:half], block[half:]

    def _sum(rows, k):
        return sum(_f(r.get(k)) for r in rows)

    vol_first = _sum(first, 'volume') or 1.0
    vol_second = _sum(second, 'volume') or 1.0
    vol_trend = vol_second / vol_first

    trd_first = _sum(first, 'trades') or 1.0
    trd_second = _sum(second, 'trades') or 1.0
    trades_trend = trd_second / trd_first

    def _taker_ratio(rows):
        v = _sum(rows, 'volume')
        tb = _sum(rows, 'taker_buy_base')
        return (tb / v * 100.0) if v > 0 else 50.0
    taker_2nd = _taker_ratio(second)
    taker_shift = taker_2nd - _taker_ratio(first)

    pcts = []
    for r in block:
        o = _f(r.get('open')); c = _f(r.get('close'))
        pcts.append(((c - o) / o * 100.0) if o > 0 else 0.0)
    cumulative_pct = sum(pcts)
    last30 = block[-30:] if n >= 30 else block
    last30_pct = sum(((_f(r.get('close')) - _f(r.get('open'))) / _f(r.get('open')) * 100.0)
                     for r in last30 if _f(r.get('open')) > 0)

    def _range_avg(rows):
        rs = []
        for r in rows:
            hi = _f(r.get('high')); lo = _f(r.get('low'))
            if lo > 0:
                rs.append((hi - lo) / lo * 100.0)
        return (sum(rs) / len(rs)) if rs else 0.0
    rf, rs2 = _range_avg(first), _range_avg(second)
    range_expansion = (rs2 / rf) if rf > 0 else 1.0

    def _avg(rows, k):
        vals = [_f(r.get(k)) for r in rows if r.get(k) is not None]
        return (sum(vals) / len(vals)) if vals else 0.0
    book_imbalance = _avg(second, 'book_imbalance_5')
    book_imb_shift = book_imbalance - _avg(first, 'book_imbalance_5')
    spread = _avg(second, 'spread_bps')
    funding = _f(block[-1].get('funding'))
    premium = _f(block[-1].get('premium'))

    oi_start = _f(block[0].get('open_interest'))
    oi_end = _f(block[-1].get('open_interest'))
    oi_delta = ((oi_end - oi_start) / oi_start * 100.0) if oi_start > 0 else 0.0

    prof = {
        'vol_trend': vol_trend, 'trades_trend': trades_trend,
        'taker_ratio': taker_2nd, 'taker_shift': taker_shift,
        'cumulative_pct': cumulative_pct, 'last30_pct': last30_pct,
        'range_expansion': range_expansion,
        'book_imbalance': book_imbalance, 'book_imb_shift': book_imb_shift,
        'spread': spread, 'funding': funding, 'oi_delta': oi_delta, 'premium': premium,
    }
    for k in METRIC_KEYS:
        prof[k] = _f(metric_row.get(k)) if metric_row else 0.0
    return prof


def _forward_peaks(rows, anchor_idx, fwd_n):
    """Realer Forward-Peak ↑/↓ + geordnete TP/SL-Pfade aus rohen High/Low.
    Returns (peak_up_pct, peak_down_pct) oder None."""
    entry = _f(rows[anchor_idx].get('close'))
    if entry <= 0:
        return None
    fwd = rows[anchor_idx + 1: anchor_idx + 1 + fwd_n]
    if not fwd:
        return None
    highs = [_f(r.get('high')) for r in fwd if r.get('high') is not None]
    lows = [_f(r.get('low')) for r in fwd if r.get('low') is not None and _f(r.get('low')) > 0]
    if not highs or not lows:
        return None
    peak_up = (max(highs) - entry) / entry * 100.0
    peak_down = (entry - min(lows)) / entry * 100.0
    return peak_up, peak_down


def _ordered_hit(rows, anchor_idx, fwd_n, side, tp_pct, sl_pct):
    """Erreicht +TP VOR -SL (SL-Survival)? Geht die rohen Forward-Bars in Reihenfolge
    durch. Returns True (TP zuerst), False (SL zuerst oder keiner)."""
    entry = _f(rows[anchor_idx].get('close'))
    if entry <= 0:
        return False
    if side == 'long':
        tp_lvl = entry * (1 + tp_pct / 100.0); sl_lvl = entry * (1 - sl_pct / 100.0)
    else:
        tp_lvl = entry * (1 - tp_pct / 100.0); sl_lvl = entry * (1 + sl_pct / 100.0)
    for r in rows[anchor_idx + 1: anchor_idx + 1 + fwd_n]:
        hi = _f(r.get('high')); lo = _f(r.get('low'))
        if hi <= 0 or lo <= 0:
            continue
        if side == 'long':
            if lo <= sl_lvl:
                return False
            if hi >= tp_lvl:
                return True
        else:
            if hi >= sl_lvl:
                return False
            if lo <= tp_lvl:
                return True
    return False


def _median(xs):
    s = sorted(xs); n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _quantile(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def _eval_side(rows, matched, side, cfg):
    """Bewertet eine Richtung über die gematchten Anchor (Indizes).
    Gate = direkte Netto-Expectancy nach Gebühr (Gate-1-Befund 09.06.: Reach-Rate
    allein ist kein Profit-Kriterium — Mini-TP/grosser-SL kippt die R/R).
    Returns dict {reach_rate, set_tp_pct, sl_pct, net_exp_pct, n, pass} oder None."""
    peak_share = float(cfg["peak_share"])
    sl_q = float(cfg["sl_survival_quantile"])
    min_tp = float(cfg["min_tp_pct"]); max_tp = float(cfg["max_tp_pct"])
    fwd_n = int(cfg["forward_window_minutes"])
    fee = float(cfg["fee_roundtrip_pct"])
    min_net = float(cfg["min_net_expectancy_pct"])
    min_reach = float(cfg["min_reach_rate"])

    peaks = []  # realer Peak in Gewinnrichtung
    adverse = []  # Gegenbewegung in Verlustrichtung
    for idx, pu, pd in matched:
        peaks.append(pu if side == 'long' else pd)
        adverse.append(pd if side == 'long' else pu)

    # TP = peak_share × erwarteter (Median-)Peak, gefloort/gecappt.
    set_tp = max(min_tp, min(max_tp, peak_share * _median(peaks)))

    # SL = Point-of-no-Return: Level, unter dem (1-sl_q) der Fälle blieben.
    # sl_survival_quantile=0.8 -> 80%-Perzentil der Gegenbewegungen.
    sl_pct = _quantile(adverse, sl_q)
    if sl_pct <= 0:
        return None

    # Erfolg = +TP VOR -SL erreicht (SL-Survival). Das selbst-strangulierende
    # 'nah-dran'-Band (b) ist RAUS — es maß Peak-Streuung, nicht Verlässlichkeit.
    hits = sum(1 for idx, _, _ in matched if _ordered_hit(rows, idx, fwd_n, side, set_tp, sl_pct))
    reach_rate = hits / len(matched)

    # Direkter Profit-Gate: Netto-Erwartung pro Trade nach Gebühr.
    net_exp = reach_rate * set_tp - (1 - reach_rate) * sl_pct - fee
    return {
        'reach_rate': round(reach_rate, 3),
        'set_tp_pct': round(set_tp, 3),
        'sl_pct': round(sl_pct, 3),
        'net_exp_pct': round(net_exp, 4),
        'n': len(matched),
        'pass': (net_exp >= min_net) and (reach_rate >= min_reach),
    }


def compute_signal(coins_conn, symbol, cfg):
    """Haupt-Entry. cfg = reach_model.match ∪ reach_model.target (flach).
    Returns dict {side, tp_pct, sl_pct, conf, diag} oder None (skip/zu wenig Daten)."""
    lookback_days = int(cfg["lookback_days"])
    profile_window = int(cfg["profile_window_minutes"])
    sample_step = int(cfg["sample_step_minutes"])
    forward_window = int(cfg["forward_window_minutes"])
    distance_threshold = float(cfg["distance_threshold"])
    min_matches = int(cfg["min_matches"])

    try:
        with coins_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.bucket, a.open, a.high, a.low, a.close, a.volume, a.trades,
                       a.taker_buy_base, a.funding, a.open_interest, a.premium,
                       a.spread_bps, a.book_imbalance_5,
                       m.pct_30m, m.pct_60m, m.pct_120m, m.pct_240m, m.pct_360m
                FROM agg_1m a
                LEFT JOIN kline_metrics m ON m.symbol = a.symbol AND m.open_time = a.bucket
                WHERE a.symbol = %s AND a.bucket > now() - make_interval(days => %s)
                ORDER BY a.bucket ASC
            """, (symbol, lookback_days))
            rows = cur.fetchall()
    except Exception as e:
        log.error("FALLBACK_TRIGGERED compute_signal %s: DB error %s -> None", symbol, e)
        try: coins_conn.rollback()
        except Exception: pass
        return None

    n = len(rows)
    if n < profile_window + forward_window + min_matches:
        return None  # zu wenig Historie (neuer Coin / Lücke) -> skip

    cur_profile = _profile_from_block(rows[-profile_window:], rows[-1])
    if cur_profile is None:
        return None

    # Historische Profile + Forward-Peaks (ein Pass)
    hist = []  # (profile, anchor_idx, peak_up, peak_down)
    last_idx = n - forward_window
    i = profile_window
    while i < last_idx:
        prof = _profile_from_block(rows[i - profile_window:i], rows[i])
        if prof is not None:
            fp = _forward_peaks(rows, i, forward_window)
            if fp is not None:
                hist.append((prof, i, fp[0], fp[1]))
        i += sample_step

    if len(hist) < min_matches:
        return None

    # z-Normalisierung pro Feature über die Historie (ALLE Keys)
    stats = {}
    for k in ALL_KEYS:
        vals = [p[k] for p, _, _, _ in hist]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1.0
        stats[k] = (mean, std)

    def _dist(p):
        s = 0.0
        for k in ALL_KEYS:
            mean, std = stats[k]
            s += ((p[k] - mean) / std - (cur_profile[k] - mean) / std) ** 2
        return math.sqrt(s)

    matched = [(idx, pu, pd) for p, idx, pu, pd in hist if _dist(p) <= distance_threshold]
    if len(matched) < min_matches:
        return {'side': None, 'skip': 'too_few_matches',
                'diag': {'n_hist': len(hist), 'n_match': len(matched)}}

    long_r = _eval_side(rows, matched, 'long', cfg)
    short_r = _eval_side(rows, matched, 'short', cfg)
    diag = {'n_hist': len(hist), 'n_match': len(matched), 'long': long_r, 'short': short_r}

    cands = [(s, r) for s, r in (('long', long_r), ('short', short_r)) if r and r['pass']]
    if not cands:
        return {'side': None, 'skip': 'no_side_positive_expectancy', 'diag': diag}

    # Seite mit der höchsten Netto-Expectancy (nicht Reach-Rate).
    side, r = max(cands, key=lambda x: x[1]['net_exp_pct'])
    return {
        'side': side,
        'tp_pct': r['set_tp_pct'],
        'sl_pct': r['sl_pct'],
        'conf': r['reach_rate'],
        'net_exp': r['net_exp_pct'],
        'n_match': r['n'],
        'diag': diag,
    }
