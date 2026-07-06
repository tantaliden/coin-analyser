#!/usr/bin/env python3
"""Coinalyze-Backfill (einmalig, 21.06.2026): holt ~89 Tage Historie fuer Liquidationen,
Open-Interest und Funding je base_asset (aggregiert ueber Binance/Bybit/OKX) in die
ext_-Tabellen. + Deribit DVOL-Historie (BTC/ETH) in ext_dvol. 1h-Aufloesung.
Schonend getaktet (Coinalyze ~4 liq-calls/min): grosse Batches, lange Pausen. nohup."""
import json, time, logging, urllib.request, urllib.parse
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("cz-backfill")
SETTINGS = "/opt/coin/settings.json"
DAYS = 89
PACE = 18          # s zwischen Calls (sicher unter liq ~4/min)
BATCH = 40

def load():
    s = json.load(open(SETTINGS)); return s, s["coinalyze"], s["deribit"]

def dbc(s):
    d = s["databases"]["coins"]
    cn = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])
    cn.autocommit = True; return cn

def api(base, path, params, key=None, timeout=25):
    url = f"{base}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=({"api_key": key} if key else {}))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                w = int(float(e.headers.get("Retry-After") or 0)) or 60
                log.warning("429 %s -> wait %ds", path, w); time.sleep(w); continue
            raise

def chunked(it, n):
    it = list(it)
    for i in range(0, len(it), n): yield it[i:i+n]

def main():
    s, cz, dv = load()
    coins = dbc(s)
    symmap = json.load(open("/opt/coin/database/data/coinalyze_symmap.json"))
    sym2base = {sym: b for b, syms in symmap.items() for sym in syms}
    allsyms = list(sym2base.keys())
    now = int(time.time()); frm = now - DAYS * 86400
    base = cz["base_url"]; key = cz["api_key"]
    log.info("Backfill %d Tage, %d Symbole, %d Coins", DAYS, len(allsyms), len(symmap))

    # --- Liquidationen (aggregiert je base/bucket) ---
    liq = {}
    for batch in chunked(allsyms, BATCH):
        d = api(base, "liquidation-history", {"symbols": ",".join(batch), "interval": "1hour", "from": frm, "to": now}, key)
        for srs in d:
            b = sym2base.get(srs["symbol"])
            if not b: continue
            for h in srs.get("history", []):
                k = (b, h["t"]); a = liq.setdefault(k, [0.0, 0.0])
                a[0] += h.get("l", 0) or 0; a[1] += h.get("s", 0) or 0
        time.sleep(PACE)
    rows = [(b, datetime.fromtimestamp(t, timezone.utc), v[0], v[1]) for (b, t), v in liq.items()]
    with coins.cursor() as c:
        execute_values(c, """INSERT INTO ext_liquidations(base_asset,bucket,long_usd,short_usd) VALUES %s
            ON CONFLICT(base_asset,bucket) DO UPDATE SET long_usd=EXCLUDED.long_usd,short_usd=EXCLUDED.short_usd""", rows)
    log.info("Liq: %d Zeilen", len(rows))

    # --- Open-Interest (Summe je base/bucket, c=Close des Buckets) ---
    oi = {}
    for batch in chunked(allsyms, BATCH):
        d = api(base, "open-interest-history", {"symbols": ",".join(batch), "interval": "1hour", "from": frm, "to": now}, key)
        for srs in d:
            b = sym2base.get(srs["symbol"])
            if not b: continue
            for h in srs.get("history", []):
                k = (b, h["t"]); oi[k] = oi.get(k, 0.0) + (h.get("c", 0) or 0)
        time.sleep(PACE)
    rows = [(b, datetime.fromtimestamp(t, timezone.utc), v) for (b, t), v in oi.items()]
    with coins.cursor() as c:
        execute_values(c, "INSERT INTO ext_open_interest(base_asset,ts,oi_value) VALUES %s ON CONFLICT DO NOTHING", rows)
    log.info("OI: %d Zeilen", len(rows))

    # --- Funding (Mittel je base/bucket) ---
    fu = {}
    for batch in chunked(allsyms, BATCH):
        d = api(base, "funding-rate-history", {"symbols": ",".join(batch), "interval": "1hour", "from": frm, "to": now}, key)
        for srs in d:
            b = sym2base.get(srs["symbol"])
            if not b: continue
            for h in srs.get("history", []):
                k = (b, h["t"]); a = fu.setdefault(k, [0.0, 0]); a[0] += h.get("c", 0) or 0; a[1] += 1
        time.sleep(PACE)
    rows = [(b, datetime.fromtimestamp(t, timezone.utc), v[0]/v[1]) for (b, t), v in fu.items() if v[1]]
    with coins.cursor() as c:
        execute_values(c, "INSERT INTO ext_funding(base_asset,ts,funding_avg) VALUES %s ON CONFLICT DO NOTHING", rows)
    log.info("Funding: %d Zeilen", len(rows))

    # --- Deribit DVOL (BTC/ETH, gratis, kein Limit) ---
    with coins.cursor() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS ext_dvol(currency TEXT, ts TIMESTAMPTZ, dvol DOUBLE PRECISION,
                     PRIMARY KEY(currency,ts))""")
        c.execute("ALTER TABLE ext_dvol OWNER TO api_reader")
    nowms = now * 1000; frmms = (now - DAYS * 86400) * 1000
    for cur in dv["currencies"]:
        d = api(dv["base_url"], "public/get_volatility_index_data",
                {"currency": cur, "start_timestamp": frmms, "end_timestamp": nowms, "resolution": 3600})
        data = (d.get("result") or {}).get("data", []) if isinstance(d, dict) else []
        rows = [(cur, datetime.fromtimestamp(p[0]/1000, timezone.utc), p[4]) for p in data]
        if rows:
            with coins.cursor() as c:
                execute_values(c, "INSERT INTO ext_dvol(currency,ts,dvol) VALUES %s ON CONFLICT DO NOTHING", rows)
        log.info("DVOL %s: %d Zeilen", cur, len(rows))
    coins.close()
    log.info("Backfill fertig.")

if __name__ == "__main__":
    main()
