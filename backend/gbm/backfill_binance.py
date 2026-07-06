#!/usr/bin/env python3
"""Backfill 2 Jahre OHLCV von Binance Spot fuer die Predictor-Coins (15.06.2026).

Zieht 5m/1h/4h/1d-Klines ab 2024-06-01 in eigene Tabellen hist_{5m,1h,4h,1d}
(coins-DB), Struktur = agg_*-kompatibel (symbol,bucket,ohlcv,trades,taker_buy_base).
bucket = Binance open_time = Intervall-START (gleiche Semantik wie unsere agg_*,
also Leak-Fix bucket+L<=t greift konsistent). Coins ohne Binance-Listing -> skip.
Nur OHLCV (HL-Mikrostruktur gibt es historisch nicht; kommt spaeter als Live-Feature).
nohup-Lauf, Rate-limit-bewusst.
"""
import json, time, urllib.request, urllib.error
import psycopg2
import datetime as dt

SETTINGS = "/opt/coin/settings.json"
START_MS = int(dt.datetime(2024, 6, 1).timestamp() * 1000)
INTERVALS = {'5m': 'hist_5m', '1h': 'hist_1h', '4h': 'hist_4h', '1d': 'hist_1d'}
PAGE = 1000
SLEEP = 0.25

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def db():
    d = json.load(open(SETTINGS))['databases']['coins']
    c = psycopg2.connect(dbname=d['name'], user=d['user'], password=d['password'], host=d['host'], port=d['port'])
    c.autocommit = True
    return c

def top_coins(c):
    cur = c.cursor()
    cur.execute("""SELECT symbol FROM agg_1m WHERE bucket>now()-interval '24 hours'
                   GROUP BY symbol ORDER BY SUM(quote_asset_volume) DESC NULLS LAST LIMIT 50""")
    return [r[0] for r in cur.fetchall()]

def fetch(sym, interval, start_ms):
    out = []
    t = start_ms
    while True:
        u = (f"https://api.binance.com/api/v3/klines?symbol={sym}USDT"
             f"&interval={interval}&startTime={t}&limit={PAGE}")
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(u, timeout=20))
                break
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    return out if out else None   # Symbol existiert nicht
                if e.code == 429:
                    log(f"  429 -> 30s backoff"); time.sleep(30); continue
                time.sleep(3)
            except Exception:
                time.sleep(3)
        else:
            break
        if not r:
            break
        out.extend(r)
        if len(r) < PAGE:
            break
        t = r[-1][0] + 1
        time.sleep(SLEEP)
    return out

def main():
    t0 = time.time()
    c = db()
    # Tabellen hist_* sind vorab als postgres angelegt (owner api_reader).
    coins = top_coins(c)
    log(f"{len(coins)} Coins, Backfill ab {dt.datetime.utcfromtimestamp(START_MS/1000).date()}")
    for sym in coins:
        for interval, tbl in INTERVALS.items():
            rows = fetch(sym, interval, START_MS)
            if not rows:
                log(f"{sym} {interval}: kein Binance-Listing -> skip"); continue
            vals = []
            for k in rows:
                bucket = dt.datetime.fromtimestamp(k[0] / 1000, dt.timezone.utc)
                vals.append((sym, bucket, float(k[1]), float(k[2]), float(k[3]),
                             float(k[4]), float(k[5]), float(k[8]), float(k[9])))
            with c.cursor() as cu:
                cu.executemany(f"""INSERT INTO {tbl}(symbol,bucket,open,high,low,close,volume,trades,taker_buy_base)
                                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                   ON CONFLICT(symbol,bucket) DO NOTHING""", vals)
            log(f"{sym} {interval}: {len(vals)} Kerzen")
    log(f"fertig in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
