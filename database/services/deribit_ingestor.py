#!/usr/bin/env python3
"""Deribit-Options-Ingestor (21.06.2026): BTC/ETH Options-Aggregate (gratis, kein Key).
Pro Waehrung je Pass: Put/Call-OI-Ratio, ATM-IV, Max-Pain, Total-OI, Underlying.
Public API, grosszuegiges Limit. 5min-Takt, beide Server. Quelle settings.deribit."""
import json, time, logging, re
import urllib.request, urllib.parse
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("deribit")
SETTINGS = "/opt/coin/settings.json"
INSTR = re.compile(r'^[A-Z]+-\w+-(\d+)-([CP])$')  # CUR-EXPIRY-STRIKE-C/P

def load():
    s = json.load(open(SETTINGS)); return s, s["deribit"]

def dbc(s):
    d = s["databases"]["coins"]
    cn = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])
    cn.autocommit = True; return cn

def api(cfg, path, params):
    url = f"{cfg['base_url']}/{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=int(cfg["request_timeout_seconds"])) as r:
        return json.loads(r.read().decode())["result"]

def metrics(rows):
    """rows = book_summary je Option. Aggregiert zu PC-Ratio, ATM-IV, Max-Pain, OI."""
    under = None
    calls = []  # (strike, oi)
    puts = []
    ivs = []    # (abs_dist_to_atm, mark_iv)
    call_oi = put_oi = 0.0
    for r in rows:
        m = INSTR.match(r["instrument_name"])
        if not m: continue
        strike = float(m.group(1)); typ = m.group(2)
        oi = float(r.get("open_interest") or 0)
        u = r.get("underlying_price") or r.get("estimated_delivery_price")
        if u: under = float(u)
        if typ == "C": calls.append((strike, oi)); call_oi += oi
        else:          puts.append((strike, oi)); put_oi += oi
        iv = r.get("mark_iv")
        if iv and under: ivs.append((abs(strike - under), float(iv)))
    if under is None or (call_oi + put_oi) == 0: return None
    pc = (put_oi / call_oi) if call_oi > 0 else None
    # ATM-IV: Mittel der IV der 6 strike-naechsten Optionen
    ivs.sort(key=lambda x: x[0]); atm_iv = (sum(v for _, v in ivs[:6]) / min(6, len(ivs))) if ivs else None
    # Max-Pain: Strike, der den Gesamt-Auszahlungswert an Halter minimiert
    strikes = sorted({s for s, _ in calls + puts})
    best_k = None; best_pay = None
    for S in strikes:
        pay = sum(max(0.0, S - k) * oi for k, oi in calls) + sum(max(0.0, k - S) * oi for k, oi in puts)
        if best_pay is None or pay < best_pay: best_pay = pay; best_k = S
    return dict(under=under, total_oi=call_oi + put_oi, call_oi=call_oi, put_oi=put_oi,
                pc=pc, atm_iv=atm_iv, max_pain=best_k)

def main():
    s, cfg = load()
    log.info("Deribit start: %s, %ds-Takt", cfg["currencies"], cfg["interval_seconds"])
    iv = int(cfg["interval_seconds"])
    while True:
        t0 = time.time()
        try:
            coins = dbc(s); ts = datetime.now(timezone.utc); out = []
            for cur in cfg["currencies"]:
                rows = api(cfg, "public/get_book_summary_by_currency", {"currency": cur, "kind": "option"})
                m = metrics(rows)
                if m:
                    out.append((cur, ts, m["under"], m["total_oi"], m["call_oi"], m["put_oi"], m["pc"], m["atm_iv"], m["max_pain"]))
                    log.info("%s: under %.0f PC %.2f ATM-IV %.1f maxpain %.0f OI %.0f",
                             cur, m["under"], m["pc"] or 0, m["atm_iv"] or 0, m["max_pain"] or 0, m["total_oi"])
                time.sleep(1)
            if out:
                with coins.cursor() as c:
                    execute_values(c, """INSERT INTO ext_options(currency,ts,underlying_px,total_oi,call_oi,put_oi,pc_ratio,atm_iv,max_pain)
                        VALUES %s ON CONFLICT DO NOTHING""", out)
            coins.close()
        except Exception as e:
            log.exception("pass failed: %s", e)
        time.sleep(max(5, iv - (time.time() - t0)))

if __name__ == "__main__":
    main()
