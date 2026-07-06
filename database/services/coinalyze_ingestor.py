#!/usr/bin/env python3
"""Coinalyze-Ingestor (20.06.2026): cross-exchange Liquidationen + OI + Funding.
Liquidationen + OI aggregiert ueber Binance(A)/Bybit(6)/OKX(3) je base_asset (HL liefert
keine Liq an Coinalyze). Funding = Mittel ueber die Boersen. 5min-Takt, beide Server.
Keine Fallbacks: fehlt der Key -> raise. Quelle settings.coinalyze."""
import json, time, logging
import urllib.request, urllib.parse
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("coinalyze")
SETTINGS = "/opt/coin/settings.json"

def load():
    s = json.load(open(SETTINGS)); c = s["coinalyze"]
    if not c.get("api_key"): raise RuntimeError("settings.coinalyze.api_key fehlt")
    return s, c

def dbc(s):
    d = s["databases"]["coins"]
    cn = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])
    cn.autocommit = True; return cn

def api(cfg, path, params):
    url = f"{cfg['base_url']}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"api_key": cfg["api_key"]})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=int(cfg["request_timeout_seconds"])) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                wait = int(float(e.headers.get("Retry-After") or 0)) or (10 * (attempt + 1))
                log.warning("429 on %s -> wait %ds (try %d)", path, wait, attempt + 1)
                time.sleep(wait); continue
            raise

def top_coins(coins, n):
    with coins.cursor() as cur:
        cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
                       GROUP BY symbol ORDER BY SUM(quote_asset_volume) DESC NULLS LAST LIMIT %s""", (n,))
        return [r[0] for r in cur.fetchall()]

SYMMAP_CACHE = "/opt/coin/database/data/coinalyze_symmap.json"

def build_symbol_map(cfg, bases):
    """base_asset -> Coinalyze-Perp-Symbole. future-markets (4550 Maerkte) frisst fast das
    ganze Minutenbudget -> Ergebnis 7 Tage cachen, nicht bei jedem Start neu ziehen."""
    import os
    if os.path.exists(SYMMAP_CACHE) and (time.time() - os.path.getmtime(SYMMAP_CACHE)) < 7*86400:
        cached = json.load(open(SYMMAP_CACHE))
        out = {b: cached[b] for b in bases if b in cached}
        if len(out) >= len(bases) * 0.8:
            log.info("symbol-map aus Cache (%d coins)", len(out)); return out
    mk = api(cfg, "future-markets", {})
    want = set(cfg["exchanges"]); bset = set(bases)
    m = {}
    for row in mk:
        if not row.get("is_perpetual"): continue
        if row["exchange"] not in want: continue
        b = row["base_asset"]
        if b not in bset: continue
        if row["quote_asset"] not in ("USDT", "USD", "USDC"): continue
        m.setdefault(b, []).append((row["exchange"], row["quote_asset"], row["symbol"]))
    # je Boerse genau ein Symbol (USDT > USDC > USD)
    out = {}
    pri = {"USDT": 0, "USDC": 1, "USD": 2}
    for b, lst in m.items():
        byex = {}
        for ex, q, sym in sorted(lst, key=lambda x: pri.get(x[1], 9)):
            if ex not in byex: byex[ex] = sym
        out[b] = list(byex.values())
    try: json.dump(out, open(SYMMAP_CACHE, "w"))
    except Exception as e: log.warning("symmap-cache write failed: %s", e)
    return out

def chunked(it, n):
    it = list(it)
    for i in range(0, len(it), n): yield it[i:i+n]

def main():
    s, cfg = load()
    coins = dbc(s)
    bases = top_coins(coins, int(cfg["top_n_coins"]))
    symmap = build_symbol_map(cfg, bases)
    sym2base = {sym: b for b, syms in symmap.items() for sym in syms}
    allsyms = list(sym2base.keys())
    log.info("Coinalyze start: %d coins, %d symbols ueber %s", len(symmap), len(allsyms), cfg["exchanges"])
    iv = int(cfg["interval_seconds"]); lookback = int(cfg["liq_lookback_hours"]) * 3600
    time.sleep(8)  # Minutenfenster nach future-markets entlasten
    while True:
        t0 = time.time()
        try:
            coins2 = dbc(s)
            now = int(time.time()); frm = now - lookback
            # --- Liquidationen (5min) je Symbol -> aggregiert je base/bucket ---
            liq = {}  # (base, bucket_epoch) -> [long, short]
            for batch in chunked(allsyms, 70):
                data = api(cfg, "liquidation-history", {"symbols": ",".join(batch),
                           "interval": "5min", "from": frm, "to": now})
                for srs in data:
                    b = sym2base.get(srs["symbol"])
                    if not b: continue
                    for h in srs.get("history", []):
                        k = (b, h["t"])
                        a = liq.setdefault(k, [0.0, 0.0])
                        a[0] += h.get("l", 0) or 0; a[1] += h.get("s", 0) or 0
                time.sleep(12.0)  # 40/min Rate-Limit (konservativ, gemeinsam mit Live-Server)
            if liq:
                rows = [(b, datetime.fromtimestamp(t, timezone.utc), v[0], v[1]) for (b, t), v in liq.items()]
                with coins2.cursor() as cur:
                    execute_values(cur, """INSERT INTO ext_liquidations(base_asset,bucket,long_usd,short_usd)
                        VALUES %s ON CONFLICT(base_asset,bucket) DO UPDATE SET long_usd=EXCLUDED.long_usd,short_usd=EXCLUDED.short_usd""", rows)
            # --- OI aktuell je base (Summe ueber Boersen) ---
            oi = {}
            for batch in chunked(allsyms, 70):
                data = api(cfg, "open-interest", {"symbols": ",".join(batch)})
                for row in data:
                    b = sym2base.get(row["symbol"])
                    if b: oi[b] = oi.get(b, 0.0) + (row.get("value") or 0)
                time.sleep(12.0)
            tsnow = datetime.now(timezone.utc)
            if oi:
                with coins2.cursor() as cur:
                    execute_values(cur, "INSERT INTO ext_open_interest(base_asset,ts,oi_value) VALUES %s ON CONFLICT DO NOTHING",
                                   [(b, tsnow, v) for b, v in oi.items()])
            # --- Funding aktuell je base (Mittel ueber Boersen) ---
            fund = {}
            for batch in chunked(allsyms, 70):
                data = api(cfg, "predicted-funding-rate", {"symbols": ",".join(batch)})
                for row in data:
                    b = sym2base.get(row["symbol"])
                    if b and row.get("value") is not None: fund.setdefault(b, []).append(row["value"])
                time.sleep(12.0)
            if fund:
                with coins2.cursor() as cur:
                    execute_values(cur, "INSERT INTO ext_funding(base_asset,ts,funding_avg) VALUES %s ON CONFLICT DO NOTHING",
                                   [(b, tsnow, sum(v)/len(v)) for b, v in fund.items()])
            coins2.close()
            log.info("ingest ok: liq %d buckets, oi %d, funding %d coins", len(liq), len(oi), len(fund))
        except Exception as e:
            log.exception("ingest-pass failed: %s", e)
        time.sleep(max(5, iv - (time.time() - t0)))

if __name__ == "__main__":
    main()
