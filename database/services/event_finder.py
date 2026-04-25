#!/usr/bin/env python3
"""
Event Finder v2 - Retrospektive Pattern-Analyse mit Long/Short Trennung

Findet ≥5% Moves, analysiert Vorgeschichte GETRENNT für Long und Short,
testet Precision der gefundenen Muster, schreibt detaillierten Report.

Phasen:
  1. Events identifizieren (kline_metrics: pct_60m..pct_240m >= ±threshold)
  2. Vorgeschichte analysieren (klines: 120min vor Event) + pct_30m/pct_60m
  3. Baseline erstellen (Nicht-Events als Vergleich)
  4. Abweichungen berechnen - GETRENNT Long vs Short
  5. Precision-Test: Wenn Muster X → wie oft folgt ≥threshold?
  6. Gruppierung (per Coin, per Kategorie)
  7. Report schreiben

Läuft als:
  --once           Einmaliger Scan
  --loop           Dauerhafter Service (alle 6h)
  --start DATE     Startdatum (default: 3 Monate zurück)
  --end DATE       Enddatum (default: gestern)
  --threshold N    Min-Prozent für Event (default: 5.0)
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json, sys, os, time, signal, random, math
import pytz

BERLIN_TZ = pytz.timezone('Europe/Berlin')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, '..', 'settings.json')) as f:
    SETTINGS = json.load(f)

COINS_DB = {
    'host': SETTINGS['databases']['coins']['host'],
    'port': SETTINGS['databases']['coins']['port'],
    'dbname': SETTINGS['databases']['coins']['name'],
    'user': SETTINGS['databases']['coins']['user'],
    'password': SETTINGS['databases']['coins']['password']
}

APP_DB = {
    'host': SETTINGS['databases']['app']['host'],
    'port': SETTINGS['databases']['app']['port'],
    'dbname': SETTINGS['databases']['app']['name'],
    'user': SETTINGS['databases']['app']['user'],
    'password': SETTINGS['databases']['app']['password']
}

LOG_DIR = '/opt/coin/database/logs'
LOG_FILE = os.path.join(LOG_DIR, 'event_finder.log')
REPORT_DIR = '/opt/coin/database/data'

PCT_COLUMNS = ['pct_60m', 'pct_120m', 'pct_180m', 'pct_240m']
PRE_EVENT_MINUTES = 120
BLOCK_SIZE_MINUTES = 5

running = True

def handle_signal(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

def log(msg):
    ts = datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass

def coins_conn():
    return psycopg2.connect(**COINS_DB)

def app_conn():
    return psycopg2.connect(**APP_DB)


# ============================================================
# PHASE 1: Events identifizieren
# ============================================================

def find_events(conn, start_date, end_date, threshold=5.0):
    """
    Findet alle Zeitpunkte wo mindestens ein pct_Xm >= threshold oder <= -threshold.
    Holt gleichzeitig pct_30m und pct_60m VOR dem Event aus kline_metrics.
    """
    log(f"Phase 1: Finding ≥{threshold}% events from {start_date} to {end_date}")

    conditions = " OR ".join([f"ABS({col}) >= {threshold}" for col in PCT_COLUMNS])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT symbol, open_time,
                   pct_30m, pct_60m, pct_90m, pct_120m, pct_180m, pct_240m
            FROM kline_metrics
            WHERE open_time >= %s AND open_time < %s
              AND ({conditions})
            ORDER BY symbol, open_time
        """, (start_date, end_date))
        rows = cur.fetchall()

    log(f"  Raw rows with ≥{threshold}% move: {len(rows)}")

    events = []
    last_event = {}

    for row in rows:
        sym = row['symbol']
        ot = row['open_time']

        if sym in last_event and (ot - last_event[sym]).total_seconds() < 3600:
            continue

        best_pct = 0
        best_tf = None
        for col in PCT_COLUMNS:
            val = row[col]
            if val is not None and abs(val) > abs(best_pct):
                best_pct = val
                best_tf = col

        if best_tf is None:
            continue

        direction = 'LONG' if best_pct > 0 else 'SHORT'

        fastest_tf = None
        for col in PCT_COLUMNS:
            val = row[col]
            if val is not None and abs(val) >= threshold:
                fastest_tf = col
                break

        # pct_30m VOR dem Event = das Momentum beim Einstieg
        # (pct_30m in kline_metrics = % change der letzten 30min BIS zu diesem Zeitpunkt)
        pre_pct_30m = float(row['pct_30m'] or 0)
        pre_pct_60m = float(row['pct_60m'] or 0)

        events.append({
            'symbol': sym,
            'open_time': ot,
            'direction': direction,
            'max_pct': round(best_pct, 2),
            'best_tf': best_tf,
            'fastest_tf': fastest_tf,
            'pre_pct_30m': round(pre_pct_30m, 2),
            'pre_pct_60m': round(pre_pct_60m, 2),
            'pct_60m': round(row['pct_60m'] or 0, 2),
            'pct_120m': round(row['pct_120m'] or 0, 2),
            'pct_180m': round(row['pct_180m'] or 0, 2),
            'pct_240m': round(row['pct_240m'] or 0, 2),
        })
        last_event[sym] = ot

    log(f"  Deduplicated events: {len(events)}")
    long_events = [e for e in events if e['direction'] == 'LONG']
    short_events = [e for e in events if e['direction'] == 'SHORT']
    log(f"  LONG: {len(long_events)}, SHORT: {len(short_events)}")

    return events


# ============================================================
# PHASE 2: Vorgeschichte analysieren
# ============================================================

def analyze_pre_event(conn, symbol, event_time):
    """Analysiert 120min vor Event: Volume, Trades, Taker, Preis-Momentum"""
    pre_start = event_time - timedelta(minutes=PRE_EVENT_MINUTES)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT open_time, open, high, low, close, volume,
                   trades, taker_buy_base, taker_buy_quote, quote_asset_volume
            FROM klines
            WHERE symbol = %s AND interval = '1m'
              AND open_time >= %s AND open_time < %s
            ORDER BY open_time
        """, (symbol, pre_start, event_time))
        candles = cur.fetchall()

    if len(candles) < 60:
        return None

    blocks = []
    for i in range(0, PRE_EVENT_MINUTES, BLOCK_SIZE_MINUTES):
        block_start = pre_start + timedelta(minutes=i)
        block_end = block_start + timedelta(minutes=BLOCK_SIZE_MINUTES)
        block_candles = [c for c in candles if block_start <= c['open_time'] < block_end]

        if not block_candles:
            blocks.append(None)
            continue

        vol = sum(float(c['volume'] or 0) for c in block_candles)
        trades = sum(int(c['trades'] or 0) for c in block_candles)
        taker_buy = sum(float(c['taker_buy_base'] or 0) for c in block_candles)
        high = max(float(c['high'] or 0) for c in block_candles)
        low = min(float(c['low'] or 0) for c in block_candles if c['low'])
        open_p = float(block_candles[0]['open'] or 0)
        close_p = float(block_candles[-1]['close'] or 0)

        taker_ratio = (taker_buy / vol * 100) if vol > 0 else 50.0
        range_pct = ((high - low) / low * 100) if low > 0 else 0
        pct_change = ((close_p - open_p) / open_p * 100) if open_p > 0 else 0

        blocks.append({
            'minutes_before': PRE_EVENT_MINUTES - i,
            'volume': vol, 'trades': trades,
            'taker_ratio': round(taker_ratio, 2),
            'range_pct': round(range_pct, 4),
            'pct_change': round(pct_change, 4),
        })

    valid_blocks = [b for b in blocks if b is not None]
    if len(valid_blocks) < 12:
        return None

    # Aggregation
    half = len(valid_blocks) // 2
    first_half = valid_blocks[:half]
    second_half = valid_blocks[half:]

    vol_first = sum(b['volume'] for b in first_half) or 1
    vol_second = sum(b['volume'] for b in second_half) or 1
    vol_trend = vol_second / vol_first

    trades_first = sum(b['trades'] for b in first_half) or 1
    trades_second = sum(b['trades'] for b in second_half) or 1
    trades_trend = trades_second / trades_first

    last_6 = [b for b in valid_blocks if b['minutes_before'] <= 30]
    rest = [b for b in valid_blocks if b['minutes_before'] > 30]

    vol_last30 = sum(b['volume'] for b in last_6) if last_6 else 0
    vol_rest = sum(b['volume'] for b in rest) or 1
    vol_last30_ratio = (vol_last30 / vol_rest * (len(rest) / max(len(last_6), 1))) if vol_rest > 0 else 1.0

    avg_taker = sum(b['taker_ratio'] for b in valid_blocks) / len(valid_blocks)
    taker_last30 = sum(b['taker_ratio'] for b in last_6) / len(last_6) if last_6 else 50
    taker_rest = sum(b['taker_ratio'] for b in rest) / len(rest) if rest else 50

    # Preis-Momentum: Gesamtbewegung der 120min
    all_pct = [b['pct_change'] for b in valid_blocks]
    cumulative_pct = sum(all_pct)
    last30_pct = sum(b['pct_change'] for b in last_6)

    # Range-Expansion: wird die Range größer?
    range_first = sum(b['range_pct'] for b in first_half) / len(first_half) if first_half else 0
    range_second = sum(b['range_pct'] for b in second_half) / len(second_half) if second_half else 0
    range_expansion = (range_second / range_first) if range_first > 0 else 1.0

    return {
        'vol_trend': round(vol_trend, 3),
        'trades_trend': round(trades_trend, 3),
        'vol_last30_ratio': round(vol_last30_ratio, 3),
        'avg_taker_ratio': round(avg_taker, 2),
        'taker_shift': round(taker_last30 - taker_rest, 2),
        'cumulative_pct': round(cumulative_pct, 3),
        'last30_pct': round(last30_pct, 3),
        'range_expansion': round(range_expansion, 3),
    }


# ============================================================
# PHASE 3: Baseline (Nicht-Events)
# ============================================================

def create_baseline(conn, symbols, start_date, end_date, event_times_by_symbol, count=500):
    log("Phase 3: Creating baseline profiles")
    baselines = []
    total_days = (end_date - start_date).days
    if total_days < 7:
        return baselines

    sym_list = list(symbols)[:50]
    per_sym = max(count // len(sym_list), 5)

    for sym in sym_list:
        event_times = event_times_by_symbol.get(sym, set())
        found = 0
        for _ in range(per_sym * 5):
            if found >= per_sym:
                break
            rand_time = start_date + timedelta(
                days=random.randint(0, total_days - 1),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59))

            too_close = any(abs((rand_time - et).total_seconds()) < 7200 for et in event_times)
            if too_close:
                continue

            profile = analyze_pre_event(conn, sym, rand_time)
            if profile:
                baselines.append(profile)
                found += 1

    log(f"  Baseline profiles: {len(baselines)}")
    return baselines


# ============================================================
# PHASE 4: Abweichungen GETRENNT Long vs Short
# ============================================================

def compute_deviations(long_profiles, short_profiles, baseline_profiles):
    """Berechnet Z-Scores getrennt für Long-Events, Short-Events vs Baseline"""
    log("Phase 4: Computing deviations (Long vs Short vs Baseline)")

    if not baseline_profiles:
        log("  No baseline, skipping")
        return {}

    metrics = ['vol_trend', 'trades_trend', 'vol_last30_ratio',
               'avg_taker_ratio', 'taker_shift', 'cumulative_pct',
               'last30_pct', 'range_expansion']

    def calc_stats(profiles, label):
        result = {}
        for m in metrics:
            vals = [p[m] for p in profiles if m in p and p[m] is not None]
            if vals:
                avg = sum(vals) / len(vals)
                std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
                result[m] = {'avg': avg, 'std': max(std, 0.001)}
        return result

    baseline_stats = calc_stats(baseline_profiles, 'baseline')
    long_stats = calc_stats(long_profiles, 'long')
    short_stats = calc_stats(short_profiles, 'short')

    def z_scores(event_stats, baseline_stats):
        zs = {}
        for m in metrics:
            if m in event_stats and m in baseline_stats:
                z = (event_stats[m]['avg'] - baseline_stats[m]['avg']) / baseline_stats[m]['std']
                zs[m] = round(z, 2)
        return zs

    long_z = z_scores(long_stats, baseline_stats)
    short_z = z_scores(short_stats, baseline_stats)

    # Was unterscheidet Long von Short?
    diff_z = {}
    for m in metrics:
        if m in long_stats and m in short_stats:
            combined_std = max(long_stats[m]['std'], short_stats[m]['std'], 0.001)
            diff = (long_stats[m]['avg'] - short_stats[m]['avg']) / combined_std
            diff_z[m] = round(diff, 2)

    for direction, zs in [('LONG', long_z), ('SHORT', short_z)]:
        ranked = sorted(zs.items(), key=lambda x: abs(x[1]), reverse=True)
        for m, z in ranked:
            star = " ***" if abs(z) > 2 else " **" if abs(z) > 1 else ""
            log(f"  {direction} {m}: z={z}{star}")

    log(f"  LONG vs SHORT differences:")
    for m, z in sorted(diff_z.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(z) > 0.3:
            log(f"    {m}: {z:+.2f} ({'Long höher' if z > 0 else 'Short höher'})")

    return {
        'baseline': {m: round(baseline_stats[m]['avg'], 4) for m in metrics if m in baseline_stats},
        'long_avgs': {m: round(long_stats[m]['avg'], 4) for m in metrics if m in long_stats},
        'short_avgs': {m: round(short_stats[m]['avg'], 4) for m in metrics if m in short_stats},
        'long_z': long_z,
        'short_z': short_z,
        'diff_z': diff_z,
    }


# ============================================================
# PHASE 5: Precision-Test
# ============================================================

def run_precision_test(conn, start_date, end_date, threshold):
    """
    Testet: Wenn Muster X auftritt, wie oft folgt tatsächlich ≥threshold% Move?
    Vergleicht mit Baseline (random).
    """
    log(f"Phase 5: Precision test (threshold={threshold}%)")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Baseline: random Zeitpunkte
        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE ABS(pct_60m) >= %s) as h60,
                   COUNT(*) FILTER (WHERE ABS(pct_120m) >= %s) as h120,
                   COUNT(*) FILTER (WHERE ABS(pct_240m) >= %s) as h240
            FROM kline_metrics TABLESAMPLE SYSTEM(0.1)
            WHERE open_time >= %s AND open_time < %s
        """, (threshold, threshold, threshold, start_date, end_date))
        bl = cur.fetchone()
        bl_total = bl['total'] or 1

    results = {'baseline': {
        'n': bl_total,
        'hit_60m_pct': round(bl['h60'] / bl_total * 100, 2),
        'hit_120m_pct': round(bl['h120'] / bl_total * 100, 2),
        'hit_240m_pct': round(bl['h240'] / bl_total * 100, 2),
    }}

    log(f"  Baseline (random): {bl_total} samples, "
        f"≥{threshold}% in 120m: {results['baseline']['hit_120m_pct']}%")

    # Tests mit verschiedenen pct_30m Schwellen
    for direction, dir_label, sign_op, sign_tp in [
        ('long', 'LONG', '>=', '>='),
        ('short', 'SHORT', '<=', '<=')
    ]:
        results[direction] = []
        for pct_thresh in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            pct_val = pct_thresh if direction == 'long' else -pct_thresh

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if direction == 'long':
                    cur.execute(f"""
                        SELECT COUNT(*) as total,
                               COUNT(*) FILTER (WHERE pct_60m >= %s) as tp_60m,
                               COUNT(*) FILTER (WHERE pct_60m <= -%s) as sl_60m,
                               COUNT(*) FILTER (WHERE pct_120m >= %s) as tp_120m,
                               COUNT(*) FILTER (WHERE pct_120m <= -%s) as sl_120m,
                               COUNT(*) FILTER (WHERE pct_240m >= %s) as tp_240m,
                               ROUND(AVG(pct_60m)::numeric, 3) as avg_60m,
                               ROUND(AVG(pct_120m)::numeric, 3) as avg_120m
                        FROM kline_metrics TABLESAMPLE SYSTEM(1)
                        WHERE open_time >= %s AND open_time < %s
                          AND pct_30m >= %s
                    """, (threshold, threshold, threshold, threshold, threshold,
                          start_date, end_date, pct_thresh))
                else:
                    cur.execute(f"""
                        SELECT COUNT(*) as total,
                               COUNT(*) FILTER (WHERE pct_60m <= -%s) as tp_60m,
                               COUNT(*) FILTER (WHERE pct_60m >= %s) as sl_60m,
                               COUNT(*) FILTER (WHERE pct_120m <= -%s) as tp_120m,
                               COUNT(*) FILTER (WHERE pct_120m >= %s) as sl_120m,
                               COUNT(*) FILTER (WHERE pct_240m <= -%s) as tp_240m,
                               ROUND(AVG(pct_60m)::numeric, 3) as avg_60m,
                               ROUND(AVG(pct_120m)::numeric, 3) as avg_120m
                        FROM kline_metrics TABLESAMPLE SYSTEM(1)
                        WHERE open_time >= %s AND open_time < %s
                          AND pct_30m <= -%s
                    """, (threshold, threshold, threshold, threshold, threshold,
                          start_date, end_date, pct_thresh))

                r = cur.fetchone()

            t = r['total'] or 1
            entry = {
                'pct_30m_thresh': pct_thresh,
                'n': t,
                'tp_60m_pct': round(r['tp_60m'] / t * 100, 1),
                'sl_60m_pct': round(r['sl_60m'] / t * 100, 1),
                'tp_120m_pct': round(r['tp_120m'] / t * 100, 1),
                'sl_120m_pct': round(r['sl_120m'] / t * 100, 1),
                'tp_240m_pct': round(r['tp_240m'] / t * 100, 1),
                'avg_60m': float(r['avg_60m'] or 0),
                'avg_120m': float(r['avg_120m'] or 0),
            }
            results[direction].append(entry)
            log(f"  {dir_label} pct_30m {'≥' if direction=='long' else '≤'}{'-' if direction=='short' else '+'}{pct_thresh}%: "
                f"n={t}, TP(120m)={entry['tp_120m_pct']}%, SL(120m)={entry['sl_120m_pct']}%")

    # Zusätzlich: 2% TP Test (für aktuellen Scanner)
    log(f"  --- 2% TP Precision ---")
    results['tp2'] = {}
    for direction in ['long', 'short']:
        results['tp2'][direction] = []
        for pct_thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if direction == 'long':
                    cur.execute("""
                        SELECT COUNT(*) as total,
                               COUNT(*) FILTER (WHERE pct_60m >= 2) as tp2_60m,
                               COUNT(*) FILTER (WHERE pct_60m <= -2) as sl2_60m,
                               COUNT(*) FILTER (WHERE pct_120m >= 2) as tp2_120m,
                               COUNT(*) FILTER (WHERE pct_120m <= -2) as sl2_120m
                        FROM kline_metrics TABLESAMPLE SYSTEM(1)
                        WHERE open_time >= %s AND open_time < %s
                          AND pct_30m >= %s
                    """, (start_date, end_date, pct_thresh))
                else:
                    cur.execute("""
                        SELECT COUNT(*) as total,
                               COUNT(*) FILTER (WHERE pct_60m <= -2) as tp2_60m,
                               COUNT(*) FILTER (WHERE pct_60m >= 2) as sl2_60m,
                               COUNT(*) FILTER (WHERE pct_120m <= -2) as tp2_120m,
                               COUNT(*) FILTER (WHERE pct_120m >= 2) as sl2_120m
                        FROM kline_metrics TABLESAMPLE SYSTEM(1)
                        WHERE open_time >= %s AND open_time < %s
                          AND pct_30m <= -%s
                    """, (start_date, end_date, pct_thresh))

                r = cur.fetchone()

            t = r['total'] or 1
            entry = {
                'pct_30m_thresh': pct_thresh,
                'n': t,
                'tp2_60m_pct': round(r['tp2_60m'] / t * 100, 1),
                'sl2_60m_pct': round(r['sl2_60m'] / t * 100, 1),
                'tp2_120m_pct': round(r['tp2_120m'] / t * 100, 1),
                'sl2_120m_pct': round(r['sl2_120m'] / t * 100, 1),
            }
            results['tp2'][direction].append(entry)
            log(f"  2%TP {direction.upper()} pct_30m≥{pct_thresh}%: n={t}, "
                f"TP(60m)={entry['tp2_60m_pct']}%, SL(60m)={entry['sl2_60m_pct']}%")

    return results


# ============================================================
# PHASE 6: Gruppierung
# ============================================================

def load_categories():
    try:
        conn = app_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT symbol, name, categories, network FROM coin_info WHERE categories IS NOT NULL")
            rows = cur.fetchall()
        conn.close()
        return {r['symbol']: {'name': r['name'], 'categories': r['categories'] or [], 'network': r['network']} for r in rows}
    except Exception as e:
        log(f"  Warning: Could not load categories: {e}")
        return {}


def group_events(events, event_profiles, cat_map):
    log("Phase 6: Grouping events")

    by_coin = {}
    for i, ev in enumerate(events):
        sym = ev['symbol']
        if sym not in by_coin:
            by_coin[sym] = {'long': [], 'short': []}
        d = 'long' if ev['direction'] == 'LONG' else 'short'
        by_coin[sym][d].append(ev)

    by_category = {}
    for sym, data in by_coin.items():
        cats = cat_map.get(sym, {}).get('categories', ['Unknown']) or ['Unknown']
        for cat in cats:
            if cat not in by_category:
                by_category[cat] = {'long': [], 'short': [], 'symbols': set()}
            by_category[cat]['long'].extend(data['long'])
            by_category[cat]['short'].extend(data['short'])
            by_category[cat]['symbols'].add(sym)

    log(f"  {len(by_coin)} coins, {len(by_category)} categories")
    return by_coin, by_category


# ============================================================
# PHASE 7: Report schreiben
# ============================================================

def write_report(events, deviations, precision, by_coin, by_category,
                 cat_map, threshold, start_date, end_date, baseline_count):

    ts = datetime.now(BERLIN_TZ).strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(REPORT_DIR, f'event_report_{ts}.txt')

    long_events = [e for e in events if e['direction'] == 'LONG']
    short_events = [e for e in events if e['direction'] == 'SHORT']

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVENT FINDER v2 REPORT\n")
        f.write(f"Generated: {datetime.now(BERLIN_TZ):%Y-%m-%d %H:%M:%S} (Europe/Berlin)\n")
        f.write(f"Period: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}\n")
        f.write(f"Threshold: ≥{threshold}% move in 60-240 min\n")
        f.write("=" * 80 + "\n\n")

        # === SUMMARY ===
        f.write("SUMMARY\n" + "-" * 40 + "\n")
        f.write(f"Total events: {len(events)}\n")
        f.write(f"  LONG:  {len(long_events)}\n")
        f.write(f"  SHORT: {len(short_events)}\n")
        f.write(f"Baseline samples: {baseline_count}\n")
        f.write(f"Unique coins: {len(by_coin)}\n\n")

        # === PRE-EVENT MOMENTUM (pct_30m) ===
        f.write("PRE-EVENT MOMENTUM (pct_30m at event start)\n" + "-" * 60 + "\n")
        for label, evts in [("LONG", long_events), ("SHORT", short_events)]:
            vals = [e['pre_pct_30m'] for e in evts if e['pre_pct_30m'] != 0]
            if vals:
                avg = sum(vals) / len(vals)
                pos = sum(1 for v in vals if v > 0)
                neg = sum(1 for v in vals if v < 0)
                f.write(f"  {label}: avg pct_30m = {avg:+.2f}% "
                        f"(positive: {pos}/{len(vals)}, negative: {neg}/{len(vals)})\n")
        f.write("\n")

        # === SIGNAL RANKING (Long vs Short vs Baseline) ===
        if deviations:
            metrics_list = ['vol_trend', 'trades_trend', 'vol_last30_ratio',
                           'avg_taker_ratio', 'taker_shift', 'cumulative_pct',
                           'last30_pct', 'range_expansion']

            f.write("SIGNAL STRENGTH: LONG vs Baseline\n" + "-" * 70 + "\n")
            f.write(f"{'Metric':<22} {'Z-Score':>8} {'Long Avg':>10} {'Baseline':>10}  Strength\n")
            f.write("-" * 70 + "\n")
            for m, z in sorted(deviations['long_z'].items(), key=lambda x: abs(x[1]), reverse=True):
                star = " ***" if abs(z) > 2 else " **" if abs(z) > 1 else ""
                la = deviations['long_avgs'].get(m, 0)
                ba = deviations['baseline'].get(m, 0)
                f.write(f"  {m:<20} {z:>8.2f} {la:>10.3f} {ba:>10.3f}{star}\n")
            f.write("\n")

            f.write("SIGNAL STRENGTH: SHORT vs Baseline\n" + "-" * 70 + "\n")
            f.write(f"{'Metric':<22} {'Z-Score':>8} {'Short Avg':>10} {'Baseline':>10}  Strength\n")
            f.write("-" * 70 + "\n")
            for m, z in sorted(deviations['short_z'].items(), key=lambda x: abs(x[1]), reverse=True):
                star = " ***" if abs(z) > 2 else " **" if abs(z) > 1 else ""
                sa = deviations['short_avgs'].get(m, 0)
                ba = deviations['baseline'].get(m, 0)
                f.write(f"  {m:<20} {z:>8.2f} {sa:>10.3f} {ba:>10.3f}{star}\n")
            f.write("\n")

            f.write("LONG vs SHORT DIFFERENCES\n" + "-" * 60 + "\n")
            f.write("(positive = Long higher, negative = Short higher)\n")
            for m, z in sorted(deviations['diff_z'].items(), key=lambda x: abs(x[1]), reverse=True):
                la = deviations['long_avgs'].get(m, 0)
                sa = deviations['short_avgs'].get(m, 0)
                f.write(f"  {m:<22} diff={z:>+6.2f}  (L={la:.3f}  S={sa:.3f})\n")
            f.write("\n")

        # === PRECISION TEST ===
        if precision:
            f.write("=" * 80 + "\n")
            f.write(f"PRECISION TEST: pct_30m threshold → ≥{threshold}% move follows?\n")
            f.write("=" * 80 + "\n\n")

            bl = precision['baseline']
            f.write(f"Baseline (random): ≥{threshold}% in 60m={bl['hit_60m_pct']}%, "
                    f"120m={bl['hit_120m_pct']}%, 240m={bl['hit_240m_pct']}%\n\n")

            for direction in ['long', 'short']:
                symbol = "+" if direction == 'long' else "-"
                f.write(f"{direction.upper()} — pct_30m threshold → ≥{threshold}% TP hit rate\n")
                f.write(f"{'pct_30m':<12} {'n':>7} {'TP 60m':>8} {'SL 60m':>8} {'TP 120m':>9} {'SL 120m':>9} {'TP 240m':>9}\n")
                f.write("-" * 65 + "\n")
                for entry in precision[direction]:
                    f.write(f"  {symbol}{entry['pct_30m_thresh']}%{'':<5} "
                            f"{entry['n']:>7} {entry['tp_60m_pct']:>7.1f}% {entry['sl_60m_pct']:>7.1f}% "
                            f"{entry['tp_120m_pct']:>8.1f}% {entry['sl_120m_pct']:>8.1f}% "
                            f"{entry['tp_240m_pct']:>8.1f}%\n")
                f.write("\n")

            f.write(f"--- 2% TP PRECISION (relevant für aktuellen Scanner) ---\n\n")
            for direction in ['long', 'short']:
                symbol = "+" if direction == 'long' else "-"
                f.write(f"{direction.upper()} — pct_30m threshold → ≥2% TP hit rate\n")
                f.write(f"{'pct_30m':<12} {'n':>7} {'TP 60m':>8} {'SL 60m':>8} {'TP 120m':>9} {'SL 120m':>9}\n")
                f.write("-" * 55 + "\n")
                for entry in precision['tp2'][direction]:
                    f.write(f"  {symbol}{entry['pct_30m_thresh']}%{'':<5} "
                            f"{entry['n']:>7} {entry['tp2_60m_pct']:>7.1f}% {entry['sl2_60m_pct']:>7.1f}% "
                            f"{entry['tp2_120m_pct']:>8.1f}% {entry['sl2_120m_pct']:>8.1f}%\n")
                f.write("\n")

        # === TOP COINS ===
        f.write("TOP COINS BY EVENT COUNT\n" + "-" * 70 + "\n")
        f.write(f"{'Symbol':<15} {'Total':>6} {'LONG':>6} {'SHORT':>6} {'L/S':>6} {'Avg%':>7} {'Name'}\n")
        f.write("-" * 70 + "\n")
        sorted_coins = sorted(by_coin.items(),
                              key=lambda x: len(x[1]['long']) + len(x[1]['short']), reverse=True)[:30]
        for sym, data in sorted_coins:
            total = len(data['long']) + len(data['short'])
            all_evts = data['long'] + data['short']
            avg_pct = sum(abs(e['max_pct']) for e in all_evts) / len(all_evts) if all_evts else 0
            ratio = f"{len(data['long'])/len(data['short']):.1f}" if data['short'] else "∞"
            name = cat_map.get(sym, {}).get('name', '')[:25]
            f.write(f"{sym:<15} {total:>6} {len(data['long']):>6} {len(data['short']):>6} "
                    f"{ratio:>6} {avg_pct:>6.1f}% {name}\n")
        f.write("\n")

        # === TOP CATEGORIES ===
        f.write("TOP CATEGORIES\n" + "-" * 70 + "\n")
        sorted_cats = sorted(by_category.items(),
                             key=lambda x: len(x[1]['long']) + len(x[1]['short']), reverse=True)[:15]
        for cat, data in sorted_cats:
            total = len(data['long']) + len(data['short'])
            ratio = f"{len(data['long'])/max(len(data['short']),1):.2f}"
            f.write(f"\n  {cat}: {total} events ({len(data['symbols'])} coins) "
                    f"L:{len(data['long'])} S:{len(data['short'])} (L/S={ratio})\n")
        f.write("\n")

        # === HOURLY DISTRIBUTION ===
        f.write("HOURLY DISTRIBUTION (UTC)\n" + "-" * 70 + "\n")
        f.write(f"{'Hour':<7} {'LONG':>6} {'SHORT':>6} {'Total':>7}  Visual\n")
        long_hours = [0] * 24
        short_hours = [0] * 24
        for e in events:
            h = e['open_time'].hour
            if e['direction'] == 'LONG':
                long_hours[h] += 1
            else:
                short_hours[h] += 1
        max_h = max(long_hours[h] + short_hours[h] for h in range(24)) or 1
        for h in range(24):
            total = long_hours[h] + short_hours[h]
            bar_l = "▓" * int(long_hours[h] / max_h * 30)
            bar_s = "░" * int(short_hours[h] / max_h * 30)
            f.write(f"  {h:02d}:00 {long_hours[h]:>6} {short_hours[h]:>6} {total:>7}  {bar_l}{bar_s}\n")
        f.write("  (▓ = Long, ░ = Short)\n\n")

        # === FASTEST TIMEFRAME ===
        f.write("FASTEST TIMEFRAME DISTRIBUTION\n" + "-" * 50 + "\n")
        for label, evts in [("LONG", long_events), ("SHORT", short_events)]:
            tf_counts = {}
            for e in evts:
                tf = e['fastest_tf'] or 'unknown'
                tf_counts[tf] = tf_counts.get(tf, 0) + 1
            f.write(f"  {label}:\n")
            for tf, cnt in sorted(tf_counts.items()):
                pct = cnt / len(evts) * 100 if evts else 0
                f.write(f"    {tf:<12} {cnt:>6} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")

    log(f"Report written: {report_file}")
    return report_file


# ============================================================
# MAIN
# ============================================================

def run_scan(start_date=None, end_date=None, threshold=5.0):
    log("=" * 60)
    log("EVENT FINDER v2 SCAN START")
    log("=" * 60)

    now = datetime.now(BERLIN_TZ)
    if start_date is None:
        start_date = (now - timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if end_date is None:
        end_date = (now - timedelta(days=1)).replace(hour=23, minute=59, second=0, microsecond=0, tzinfo=None)

    conn = coins_conn()

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM kline_metrics WHERE open_time >= %s", (start_date,))
        mn, mx, cnt = cur.fetchone()

    if cnt < 1000:
        log(f"Not enough data: {cnt} rows")
        conn.close()
        return None

    log(f"Data range: {mn} to {mx} ({cnt} rows)")

    # Phase 1
    events = find_events(conn, start_date, end_date, threshold)
    if not events:
        log("No events found")
        conn.close()
        return None

    # Phase 2
    log(f"Phase 2: Analyzing pre-event profiles ({len(events)} events)")
    long_profiles = []
    short_profiles = []
    analyzed = 0
    for i, ev in enumerate(events):
        profile = analyze_pre_event(conn, ev['symbol'], ev['open_time'])
        if profile:
            analyzed += 1
            if ev['direction'] == 'LONG':
                long_profiles.append(profile)
            else:
                short_profiles.append(profile)
        if (i + 1) % 500 == 0:
            log(f"  Analyzed {i+1}/{len(events)} ({analyzed} valid)")
    log(f"  Total analyzed: {analyzed}/{len(events)} (Long: {len(long_profiles)}, Short: {len(short_profiles)})")

    # Phase 3
    symbols = list(set(e['symbol'] for e in events))
    event_times_by_symbol = {}
    for ev in events:
        s = ev['symbol']
        if s not in event_times_by_symbol:
            event_times_by_symbol[s] = set()
        event_times_by_symbol[s].add(ev['open_time'])

    baseline_profiles = create_baseline(conn, symbols, start_date, end_date, event_times_by_symbol)

    # Phase 4
    deviations = compute_deviations(long_profiles, short_profiles, baseline_profiles)

    # Phase 5
    precision = run_precision_test(conn, start_date, end_date, threshold)

    # Phase 6
    cat_map = load_categories()
    by_coin, by_category = group_events(events, [], cat_map)

    # Phase 7
    report_file = write_report(events, deviations, precision, by_coin, by_category,
                                cat_map, threshold, start_date, end_date, len(baseline_profiles))

    conn.close()
    log("EVENT FINDER v2 SCAN COMPLETE")
    return report_file


def run_loop():
    log("Event Finder v2 starting in loop mode (every 6h)")
    while running:
        try:
            run_scan()
        except Exception as e:
            log(f"ERROR in scan: {e}")
            import traceback
            log(traceback.format_exc())

        for _ in range(720):
            if not running:
                break
            time.sleep(30)
    log("Event Finder v2 stopped")


if __name__ == '__main__':
    args = sys.argv[1:]

    threshold = 5.0
    start_date = None
    end_date = None

    for i, arg in enumerate(args):
        if arg == '--threshold' and i + 1 < len(args):
            threshold = float(args[i + 1])
        if arg == '--start' and i + 1 < len(args):
            start_date = datetime.strptime(args[i + 1], '%Y-%m-%d')
        if arg == '--end' and i + 1 < len(args):
            end_date = datetime.strptime(args[i + 1], '%Y-%m-%d')

    if '--once' in args:
        run_scan(start_date, end_date, threshold)
    elif '--loop' in args:
        run_loop()
    else:
        print("Usage:")
        print("  python event_finder.py --once [--threshold 5.0] [--start 2025-11-01] [--end 2026-02-17]")
        print("  python event_finder.py --loop")
