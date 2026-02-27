#!/usr/bin/env python3
"""
GEGENPROBE — Counter-Check für Precision Scanner Filter-Sets

Testet die besten Filter-Sets aus dem v2-Lauf gegen ALLE 2h-Zeitpunkte
aller Symbole (nicht nur bekannte Events).

Wenn Precision niedrig: sucht automatisch zusätzliche Filter die
False Positives eliminieren, so dass nur ≥5% Moves übrig bleiben.

Usage: python3 gegenprobe.py
"""

import sys, os, logging, time
import numpy as np
from datetime import datetime

sys.path.insert(0, '/opt/coin/backend/momentum')
from precision_scanner import (coins_db, compute_indicators, match_rule,
                                _rsi, _bb_pos, _range_pos, _atr)

LOG_FILE = '/opt/coin/logs/gegenprobe.log'
REPORT_DIR = '/opt/coin/database/data'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('gegenprobe')

# ========================================================================
# FILTER SETS FROM V2 LOG (beste 2er-Ketten)
# ========================================================================

FILTER_SETS = [
    {
        'name': 'LONG-A',
        'direction': 'long',
        'rules': [
            {'ind': 'range_pos_7d', 'op': '>=', 'thr': 0.838},
            {'ind': 'atr_1h_pct', 'op': '<', 'thr': 0.929},
        ],
        'source': 'v2: 10h/12h durations, n=175, WR=100%',
    },
    {
        'name': 'SHORT-A',
        'direction': 'short',
        'rules': [
            {'ind': 'bb_position_4h', 'op': '<', 'thr': -0.202},
            {'ind': 'atr_1h_pct', 'op': '<', 'thr': 0.99},
        ],
        'source': 'v2: 6h durations, n=142, WR=100%',
    },
    {
        'name': 'LONG-B',
        'direction': 'long',
        'rules': [
            {'ind': 'range_pos_7d', 'op': '>=', 'thr': 0.838},
            {'ind': 'atr_1h_pct', 'op': '<', 'thr': 0.99},
        ],
        'source': 'v2: 6h durations, n=56, WR=100%',
    },
    {
        'name': 'SHORT-B',
        'direction': 'short',
        'rules': [
            {'ind': 'range_pos_7d', 'op': '<', 'thr': 0.064},
            {'ind': 'consec_downs_5m', 'op': '<', 'thr': 3.0},
        ],
        'source': 'v2: 8h-12h, n=27-38, WR=92-97%',
    },
]

MIN_MOVE_PCT = 5.0
LOOKBACK_DAYS = 30
CHECK_WINDOW_MIN = 720  # 12h nach Filter-Match


# ========================================================================
# QUICK FILTER CHECK (nur benötigte Indikatoren berechnen)
# ========================================================================

def quick_filter_check(cur, symbol, tp_time, rules):
    """
    Berechnet NUR die Indikatoren die für die Filter-Regeln gebraucht werden.
    Viel schneller als compute_indicators() (2-3 Queries statt 4+Berechnungen).
    Returns dict mit berechneten Indikatoren oder None.
    """
    needed = {r['ind'] for r in rules}
    ind = {}

    if 'range_pos_7d' in needed:
        cur.execute("""
            SELECT high, low, close FROM agg_1d
            WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 7
        """, (symbol, tp_time))
        rows = list(reversed(cur.fetchall()))
        if len(rows) < 7:
            return None
        h = [float(r['high']) for r in rows]
        l = [float(r['low']) for r in rows]
        c = [float(r['close']) for r in rows]
        rp = _range_pos(h, l, c, 7)
        if rp is None:
            return None
        ind['range_pos_7d'] = round(rp, 3)

    if 'atr_1h_pct' in needed:
        cur.execute("""
            SELECT high, low, close FROM agg_1h
            WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 15
        """, (symbol, tp_time))
        rows = list(reversed(cur.fetchall()))
        if len(rows) < 15:
            return None
        h = [float(r['high']) for r in rows]
        l = [float(r['low']) for r in rows]
        c = [float(r['close']) for r in rows]
        atr = _atr(h, l, c, 14)
        if atr is None or c[-1] <= 0:
            return None
        ind['atr_1h_pct'] = round(atr / c[-1] * 100, 3)

    if 'bb_position_4h' in needed:
        cur.execute("""
            SELECT close FROM agg_4h
            WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 20
        """, (symbol, tp_time))
        rows = list(reversed(cur.fetchall()))
        if len(rows) < 20:
            return None
        c = [float(r['close']) for r in rows]
        ind['bb_position_4h'] = round(_bb_pos(c), 3)

    if 'consec_downs_5m' in needed or 'consec_ups_5m' in needed:
        cur.execute("""
            SELECT close FROM agg_5m
            WHERE symbol=%s AND bucket<%s ORDER BY bucket DESC LIMIT 11
        """, (symbol, tp_time))
        rows = list(reversed(cur.fetchall()))
        if len(rows) < 11:
            return None
        c = [float(r['close']) for r in rows]
        ups = sum(1 for i in range(1, 11) if c[i] > c[i - 1])
        ind['consec_ups_5m'] = ups
        ind['consec_downs_5m'] = 10 - ups

    return ind


def check_rules(ind, rules):
    """Prüft ob alle Regeln auf die Indikatoren matchen"""
    for r in rules:
        v = ind.get(r['ind'])
        if v is None:
            return False
        if r['op'] == '>=' and v < r['thr']:
            return False
        if r['op'] == '<' and v >= r['thr']:
            return False
    return True


# ========================================================================
# GEGENPROBE PRO FILTER-SET
# ========================================================================

def run_gegenprobe(conn, filter_set, days, min_pct):
    name = filter_set['name']
    direction = filter_set['direction']
    rules = filter_set['rules']

    logger.info("")
    logger.info("=" * 70)
    logger.info("GEGENPROBE: %s (%s)", name, direction.upper())
    for r in rules:
        logger.info("  Filter: %s %s %s", r['ind'], r['op'], r['thr'])
    logger.info("  Source: %s", filter_set['source'])
    logger.info("=" * 70)

    cur = conn.cursor()

    # Alle Symbole
    cur.execute("SELECT DISTINCT symbol FROM agg_2h WHERE bucket >= NOW() - INTERVAL '%s days'", (days,))
    symbols = [r['symbol'] for r in cur.fetchall()]
    logger.info("  %d symbols", len(symbols))

    t0 = time.time()
    total_samples = 0
    filter_matches = 0
    matches = []

    for si, symbol in enumerate(symbols):
        if si % 50 == 0:
            elapsed = time.time() - t0
            n_tp = sum(1 for m in matches if m['is_tp'])
            n_fp = sum(1 for m in matches if not m['is_tp'])
            logger.info("  [%d/%d] samples=%d matches=%d TP=%d FP=%d (%.0fs)",
                        si, len(symbols), total_samples, filter_matches, n_tp, n_fp, elapsed)

        # Alle 2h-Zeitpunkte
        cur.execute("""
            SELECT bucket FROM agg_2h
            WHERE symbol = %s AND bucket >= NOW() - INTERVAL '%s days'
            ORDER BY bucket
        """, (symbol, days))
        timepoints = [r['bucket'] for r in cur.fetchall()]
        total_samples += len(timepoints)

        for tp_time in timepoints:
            # Quick Check: nur Filter-Indikatoren berechnen
            qind = quick_filter_check(cur, symbol, tp_time, rules)
            if qind is None:
                continue

            if not check_rules(qind, rules):
                continue

            # Filter matcht! Jetzt volle Indikatoren berechnen (fuer Nachfilterung)
            filter_matches += 1
            full_ind = compute_indicators(conn, symbol, tp_time)
            if full_ind is None:
                full_ind = qind

            # Outcome pruefen: folgt ein >=min_pct% Move?
            cur.execute("""
                SELECT open, high, low, close FROM agg_5m
                WHERE symbol = %s AND bucket >= %s
                  AND bucket < %s + INTERVAL '%s minutes'
                ORDER BY bucket
            """, (symbol, tp_time, tp_time, CHECK_WINDOW_MIN))
            candles = cur.fetchall()

            if len(candles) < 2:
                matches.append({
                    'symbol': symbol, 'time': tp_time,
                    'indicators': full_ind, 'is_tp': False,
                    'max_move': 0,
                })
                continue

            ep = float(candles[0]['open'])
            if ep <= 0:
                continue

            max_up = max((float(c['high']) - ep) / ep * 100 for c in candles)
            max_down = max((ep - float(c['low'])) / ep * 100 for c in candles)

            if direction == 'long':
                is_tp = max_up >= min_pct
                max_move = max_up
            else:
                is_tp = max_down >= min_pct
                max_move = max_down

            matches.append({
                'symbol': symbol, 'time': tp_time,
                'indicators': full_ind, 'is_tp': is_tp,
                'max_move': round(max_move, 2),
            })

    elapsed = time.time() - t0
    n_tp = sum(1 for m in matches if m['is_tp'])
    n_fp = sum(1 for m in matches if not m['is_tp'])
    precision = n_tp / len(matches) * 100 if matches else 0

    logger.info("")
    logger.info("  ERGEBNIS %s:", name)
    logger.info("  Samples: %d | Filter-Matches: %d", total_samples, filter_matches)
    logger.info("  TP: %d | FP: %d | PRECISION: %.1f%%", n_tp, n_fp, precision)
    logger.info("  Time: %.0fs (%.1fmin)", elapsed, elapsed / 60)

    if matches:
        moves = [m['max_move'] for m in matches]
        logger.info("  Max-Move Verteilung: min=%.1f%% median=%.1f%% max=%.1f%% mean=%.1f%%",
                    min(moves), np.median(moves), max(moves), np.mean(moves))

    result = {
        'name': name, 'direction': direction, 'rules': list(rules),
        'total_samples': total_samples, 'filter_matches': filter_matches,
        'tp': n_tp, 'fp': n_fp, 'precision': round(precision, 1),
        'elapsed': round(elapsed, 0),
        'move_stats': None,
    }

    if matches:
        moves = [m['max_move'] for m in matches]
        result['move_stats'] = {
            'min': round(min(moves), 1), 'median': round(float(np.median(moves)), 1),
            'max': round(max(moves), 1), 'mean': round(float(np.mean(moves)), 1),
        }

    # Wenn Precision < 80%: zusaetzliche Filter suchen
    if precision < 80 and len(matches) >= 20 and n_fp >= 5:
        logger.info("")
        logger.info("  Precision %.1f%% < 80%% → NACHFILTERUNG...", precision)
        refined = refine_filters(matches, direction, min_pct)
        result['refinement'] = refined
    elif precision >= 80:
        logger.info("  Precision %.1f%% >= 80%% — OK!", precision)
    else:
        logger.info("  Zu wenig Daten zum Nachfiltern")

    return result


# ========================================================================
# NACHFILTERUNG: zusaetzliche Filter suchen die FP eliminieren
# ========================================================================

def refine_filters(matches, direction, min_pct):
    """
    Nimmt alle Matches (TP + FP) mit vollen Indikatoren.
    Sucht iterativ zusaetzliche Filter-Regeln die FP rauswerfen
    aber TPs behalten. Ziel: nur >= min_pct% Moves uebrig.
    """
    # Alle verfuegbaren Indikatoren sammeln
    indicators = set()
    for m in matches:
        for k, v in m['indicators'].items():
            if v is not None:
                indicators.add(k)

    logger.info("  Refining: %d matches, %d Indikatoren verfuegbar", len(matches), len(indicators))

    # Kandidaten-Regeln aus Percentilen
    rule_candidates = []
    for ind_name in sorted(indicators):
        vals = [m['indicators'].get(ind_name) for m in matches
                if m['indicators'].get(ind_name) is not None]
        if len(vals) < 10:
            continue
        for p in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            t = float(np.percentile(vals, p))
            rule_candidates.append({'ind': ind_name, 'op': '>=', 'thr': round(t, 4)})
            rule_candidates.append({'ind': ind_name, 'op': '<', 'thr': round(t, 4)})

    logger.info("  %d Kandidaten-Regeln", len(rule_candidates))

    # Greedy: iterativ beste Regel finden die Precision verbessert
    # Kein used_inds Lock — gleicher Indikator darf mehrfach (baut Ranges: >= X UND < Y)
    chosen = []
    remaining = list(matches)

    for step in range(10):  # max 10 zusaetzliche Filter (Ranges brauchen 2 pro Indikator)
        n_tp_now = sum(1 for m in remaining if m['is_tp'])
        n_fp_now = sum(1 for m in remaining if not m['is_tp'])

        if n_fp_now == 0:
            logger.info("    Step %d: Keine FP mehr!", step + 1)
            break

        best_rule = None
        best_score = -1
        best_kept = None

        for r in rule_candidates:
            # Exakt gleiche Regel nicht nochmal (aber gleicher Indikator mit anderem Threshold OK)
            if any(r['ind'] == c['ind'] and r['op'] == c['op'] and r['thr'] == c['thr'] for c in chosen):
                continue

            kept = []
            for m in remaining:
                v = m['indicators'].get(r['ind'])
                if v is None:
                    continue
                if r['op'] == '>=' and v >= r['thr']:
                    kept.append(m)
                elif r['op'] == '<' and v < r['thr']:
                    kept.append(m)

            if len(kept) < 5:
                continue

            tp_kept = sum(1 for m in kept if m['is_tp'])
            fp_kept = sum(1 for m in kept if not m['is_tp'])
            total_kept = tp_kept + fp_kept

            if total_kept == 0 or tp_kept == 0:
                continue

            prec = tp_kept / total_kept * 100
            # Score: Precision * TP-Retention * sqrt(n)
            tp_retention = tp_kept / n_tp_now if n_tp_now > 0 else 0
            score = prec * tp_retention * np.sqrt(tp_kept)

            if score > best_score:
                best_score = score
                best_rule = r
                best_kept = kept

        if best_rule is None:
            logger.info("    Step %d: Kein verbessernder Filter gefunden", step + 1)
            break

        chosen.append(dict(best_rule))
        remaining = best_kept

        tp_kept = sum(1 for m in remaining if m['is_tp'])
        fp_kept = sum(1 for m in remaining if not m['is_tp'])
        prec = tp_kept / (tp_kept + fp_kept) * 100 if (tp_kept + fp_kept) > 0 else 0

        logger.info("    Step %d: + %s %s %s", step + 1,
                    best_rule['ind'], best_rule['op'], best_rule['thr'])
        logger.info("             -> TP=%d FP=%d Precision=%.1f%% (n=%d)",
                    tp_kept, fp_kept, prec, len(remaining))

        if prec >= 90:
            logger.info("    Precision >= 90%% erreicht!")
            break

    tp_final = sum(1 for m in remaining if m['is_tp'])
    fp_final = sum(1 for m in remaining if not m['is_tp'])
    final_prec = tp_final / (tp_final + fp_final) * 100 if (tp_final + fp_final) > 0 else 0

    return {
        'additional_rules': chosen,
        'final_tp': tp_final,
        'final_fp': fp_final,
        'final_precision': round(final_prec, 1),
        'final_n': len(remaining),
    }


# ========================================================================
# MAIN
# ========================================================================

def main():
    T0 = time.time()

    logger.info("=" * 70)
    logger.info("GEGENPROBE — Counter-Check fuer Precision Scanner Filter")
    logger.info("Period: %d days | Min move: +/-%.1f%%", LOOKBACK_DAYS, MIN_MOVE_PCT)
    logger.info("Check window: %dmin (%dh)", CHECK_WINDOW_MIN, CHECK_WINDOW_MIN // 60)
    logger.info("Filter-Sets: %d", len(FILTER_SETS))
    logger.info("=" * 70)

    conn = coins_db()
    results = []

    for fs in FILTER_SETS:
        r = run_gegenprobe(conn, fs, LOOKBACK_DAYS, MIN_MOVE_PCT)
        results.append(r)

    conn.close()

    # ========================================================================
    # REPORT
    # ========================================================================
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORT_DIR, 'gegenprobe_{}.txt'.format(ts))
    lines = []

    def w(s=''):
        lines.append(s)

    w("=" * 80)
    w("GEGENPROBE REPORT")
    w("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    w("Period: {} days | Min move: +/-{}%".format(LOOKBACK_DAYS, MIN_MOVE_PCT))
    w("Check window: {}h".format(CHECK_WINDOW_MIN // 60))
    w("Runtime: {:.0f}s ({:.1f}min)".format(time.time() - T0, (time.time() - T0) / 60))
    w("=" * 80)

    for r in results:
        w("")
        w("-" * 80)
        w("{} ({})".format(r['name'], r['direction'].upper()))
        rules_str = ' AND '.join("{} {} {}".format(rl['ind'], rl['op'], rl['thr']) for rl in r['rules'])
        w("  Filter: {}".format(rules_str))
        w("  Samples: {} | Matches: {}".format(r['total_samples'], r['filter_matches']))
        w("  TP: {} | FP: {} | PRECISION: {}%".format(r['tp'], r['fp'], r['precision']))
        if r['move_stats']:
            ms = r['move_stats']
            w("  Moves: min={}% median={}% mean={}% max={}%".format(
                ms['min'], ms['median'], ms['mean'], ms['max']))
        w("  Time: {:.0f}s".format(r['elapsed']))

        ref = r.get('refinement')
        if ref:
            w("")
            w("  NACHFILTERUNG (zusaetzliche Filter um FP zu eliminieren):")
            for rl in ref['additional_rules']:
                w("    + {} {} {}".format(rl['ind'], rl['op'], rl['thr']))
            w("  -> Final: TP={} FP={} Precision={}% (n={})".format(
                ref['final_tp'], ref['final_fp'], ref['final_precision'], ref['final_n']))
            w("")
            w("  KOMPLETTES FILTERSET:")
            all_rules = r['rules'] + ref['additional_rules']
            for i, rl in enumerate(all_rules, 1):
                w("    {}. {} {} {}".format(i, rl['ind'], rl['op'], rl['thr']))
        else:
            if r['precision'] >= 80:
                w("  -> Kein Nachfiltern noetig")

    w("")
    w("=" * 80)
    w("END OF REPORT")
    w("=" * 80)

    report_text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Report auch ins Log
    for line in lines:
        logger.info(line)

    logger.info("")
    logger.info("Report: %s", report_path)
    logger.info("Total runtime: %.0fs (%.1fmin)", time.time() - T0, (time.time() - T0) / 60)


if __name__ == '__main__':
    main()
