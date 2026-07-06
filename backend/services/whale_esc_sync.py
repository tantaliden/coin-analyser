#!/usr/bin/env python3
"""Wal-Eskalations-Sync Test<-Live (Volker 25.06.2026): holt per ssh die Leader-Positionen
(whale_positions auf LIVE, address aus settings) im scan-Takt und erkennt zwei Eskalations-Arten
pro symbol+side, schreibt sie in whale_escalations (TEST app-DB) fuer claude-wallet v4_whale:
  1) CROSSING: pct=position_value/account_value kreuzt von <thr auf >=thr.
  2) ADD (Aufstockung): qty (Token-Menge, preisunabhaengig) waechst seit der letzten Basis um
     >=add_qty_threshold_pct, solange pct>=thr. Basis wird bei jedem Event/Erstsicht neu gesetzt,
     sodass schrittweises Aufstocken akkumuliert (10x +2% => irgendwann +>=15% => ein Event).
State (last_pct, last_qty-Basis je symbol+side) in whale_esc_state. ERSTER Lauf (leerer State) =
nur Baseline, KEINE Events (sonst Flut). Kein Echtgeld-Bezug. Alles aus settings.whale_esc_sync."""
import json, time, subprocess
import psycopg2

SETTINGS = "/opt/coin/settings.json"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def req(d, k, c):
    if k not in d: raise RuntimeError(f"settings.{c}.{k} fehlt — KEINE Defaults")
    return d[k]

def dba(s):
    d = s["databases"]["app"]
    c = psycopg2.connect(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"]); c.autocommit = True; return c

DDL = """
CREATE TABLE IF NOT EXISTS whale_escalations(
  id BIGSERIAL PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
  ts_event TIMESTAMPTZ NOT NULL, pct DOUBLE PRECISION, kind TEXT,
  processed BOOLEAN NOT NULL DEFAULT false, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
ALTER TABLE whale_escalations ADD COLUMN IF NOT EXISTS kind TEXT;
CREATE INDEX IF NOT EXISTS whale_esc_unproc ON whale_escalations(processed, ts_event);
CREATE TABLE IF NOT EXISTS whale_esc_state(
  symbol TEXT NOT NULL, side TEXT NOT NULL, last_pct DOUBLE PRECISION NOT NULL,
  last_qty DOUBLE PRECISION, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(symbol, side));
ALTER TABLE whale_esc_state ADD COLUMN IF NOT EXISTS last_qty DOUBLE PRECISION;
"""

def fetch_leader(host, leader):
    """ssh -> letzter Snapshot der Leader-Positionen. Returns (ts_epoch, {(sym,side):(pct,qty)}) | (None,None).
    Befehl via stdin (bash -s) statt -c, damit kein Shell-Quoting/$$-Problem entsteht."""
    remote = ("sudo -u postgres psql -d analyser_app -At -F'|' <<'SQL'\n"
              "SELECT extract(epoch from ts), symbol, side, position_value/NULLIF(account_value,0), qty\n"
              f"FROM whale_positions WHERE address='{leader}'\n"
              f"AND ts=(SELECT max(ts) FROM whale_positions WHERE address='{leader}');\n"
              "SQL\n")
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "bash -s"],
                           input=remote, capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"ssh exception: {e}"); return None, None
    if r.returncode != 0:
        log(f"ssh/psql rc={r.returncode}: {r.stderr.strip()[:200]}"); return None, None
    rows = [ln for ln in r.stdout.strip().split("\n") if ln]
    if not rows: return None, None
    ts_epoch = None; pos = {}
    for ln in rows:
        p = ln.split("|")
        if len(p) < 5 or p[3] == "" or p[1] == "": continue
        try: ts_epoch = float(p[0]); pct = float(p[3]); qty = float(p[4])
        except ValueError: continue
        pos[(p[1], p[2])] = (pct, qty)
    if ts_epoch is None: return None, None
    return ts_epoch, pos

def main():
    s = json.load(open(SETTINGS)); C = req(s, "whale_esc_sync", "root")
    host = str(req(C, "live_ssh_host", "whale_esc_sync"))
    leader = str(req(C, "leader_address", "whale_esc_sync"))
    scan = int(req(C, "scan_interval_seconds", "whale_esc_sync"))
    thr = float(req(C, "escalation_threshold_pct", "whale_esc_sync")) / 100.0
    add_thr = float(req(C, "add_qty_threshold_pct", "whale_esc_sync")) / 100.0
    log(f"WHALE-ESC-SYNC start: leader {leader[:14]}.. thr>={thr*100:.1f}% add>=+{add_thr*100:.0f}%qty scan{scan}s host {host}")
    with dba(s) as app:
        with app.cursor() as cur: cur.execute(DDL)
    while True:
        t0 = time.time()
        try:
            ts_epoch, pos = fetch_leader(host, leader)
            if pos is None:
                log("Leader-Snapshot leer/Fehler -> skip")
            else:
                with dba(s) as app:
                    with app.cursor() as cur:
                        cur.execute("SELECT symbol, side, last_pct, last_qty FROM whale_esc_state")
                        state = {(r[0], r[1]): (float(r[2]), (float(r[3]) if r[3] is not None else None)) for r in cur.fetchall()}
                    baseline = (len(state) == 0)
                    events = 0
                    with app.cursor() as cur:
                        for (sym, side), (pct, qty) in pos.items():
                            prev = state.get((sym, side))
                            prev_pct = prev[0] if prev else None
                            qty_base = prev[1] if prev else None
                            kind = None
                            if not baseline:
                                if (prev_pct is None or prev_pct < thr) and pct >= thr:
                                    kind = "crossing"
                                elif qty_base and qty_base > 0 and pct >= thr and (qty / qty_base - 1.0) >= add_thr:
                                    kind = "add"
                            if kind:
                                cur.execute("""INSERT INTO whale_escalations(symbol,side,ts_event,pct,kind)
                                               VALUES(%s,%s,to_timestamp(%s),%s,%s)""", (sym, side, ts_epoch, pct, kind))
                                events += 1
                                log(f"ESCALATION [{kind}] {side} {sym} pct={pct*100:.0f}% qty={qty:.4g}")
                            # Basis-qty nur bei Event/Erstsicht neu setzen (sonst akkumuliert schrittweises Aufstocken)
                            new_base = qty if (qty_base is None or kind) else qty_base
                            cur.execute("""INSERT INTO whale_esc_state(symbol,side,last_pct,last_qty,updated_at)
                                           VALUES(%s,%s,%s,%s,now()) ON CONFLICT(symbol,side)
                                           DO UPDATE SET last_pct=EXCLUDED.last_pct, last_qty=EXCLUDED.last_qty, updated_at=now()""",
                                        (sym, side, pct, new_base))
                        # nicht mehr vorhandene Positionen -> 0 (ermoeglicht erneutes Crossing/Add nach Re-Open)
                        for (sym, side) in list(state.keys()):
                            if (sym, side) not in pos:
                                cur.execute("UPDATE whale_esc_state SET last_pct=0, last_qty=0, updated_at=now() WHERE symbol=%s AND side=%s", (sym, side))
                    if baseline: log(f"Baseline gesetzt ({len(pos)} Positionen), keine Events beim ersten Lauf")
                    elif events: log(f"sync: {events} neue Eskalation(en), {len(pos)} Wal-Positionen")
        except Exception as e:
            log(f"loop error: {e}")
        time.sleep(max(1, scan - (time.time() - t0)))

if __name__ == "__main__":
    main()
