"""Predictor Watch Service (v6.4, 2026-05-26, Volker).

Eigenständiger Daemon, der die schnellen DB-getriebenen Loop-Tasks aus dem
predictor.service auslagert: watch_pass_mh, paper_watch, timeout_watch, sync_hl_to_db.

Hintergrund: im predictor.service liefen diese im selben Thread wie observer und
process_due_mh_hindsight. Wenn das Modell-gebundene process_due lange braucht
(viele Hindsight-Einträge oder Stalker-Refreshes blockieren), staut sich der gesamte
Watch-Loop — TP/SL-Hits werden um Minuten verzögert (am 26.05. beobachtet: 9 min
Stau, 111 Paper-Closes auf einen Schlag).

Dieser Service hat KEINEN Modell-Zugriff (watch_pass_mh nutzt mh_model nur als
Parameter, kein interner Zugriff). Damit ist watch komplett vom Modell-Lock entkoppelt
und kann unabhängig mit watch_interval_seconds laufen.

Modell-gebundene Tasks (observer_reeval, process_due_mh_hindsight) BLEIBEN im
predictor.service mit dem mh_model.
"""
import logging
import sys
import time

# predictor_service als Modul importieren (Funktionen wiederverwenden, kein Code-Dupe)
sys.path.insert(0, '/opt/coin/backend')
from services.predictor_service import (
    watch_pass_mh, timeout_watch, sync_hl_to_db,
    load_settings, get_timeout_hours, db_app, db_coins,
)
from services.paper_engine import paper_watch

log = logging.getLogger("predictor.watch")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)


def main():
    s = load_settings()
    cfg = s["predictor"]
    state_path = cfg.get("multi_head", {}).get(
        "state_path", "/opt/coin/database/data/models/predictor_v5_mh.pkl")
    watch_interval = float(cfg["watch_interval_seconds"])

    log.info("predictor-watch start. watch_interval=%ss", watch_interval)

    last_timeout_watch = 0.0
    last_hl_sync = 0.0
    last_settings_reload = time.time()

    while True:
        try:
            now = time.time()

            # Settings live-reload alle 60s (analog Hauptservice)
            if now - last_settings_reload >= 60:
                s = load_settings()
                cfg = s["predictor"]
                last_settings_reload = now

            # 1) watch_pass_mh — TP/SL/Timeout-Check für open_predictions.
            #    mh_model=None: die Funktion nutzt es intern nicht (nur Signatur-Parameter).
            t0 = time.monotonic()
            try:
                c = watch_pass_mh(s, None, state_path)
                if c > 0:
                    log.info("watch-pass: %d closed (%.2fs)", c, time.monotonic() - t0)
            except Exception as e:
                log.exception("watch_pass_mh failed: %s", e)

            # 2) timeout_watch — Trader-Welt, 10s-Tick
            if now - last_timeout_watch >= 10:
                try:
                    n_to = timeout_watch(s)
                    if n_to > 0:
                        log.info("timeout-watch: %d via Timeout geschlossen", n_to)
                except Exception as e:
                    log.exception("timeout-watch failed: %s", e)
                last_timeout_watch = now

            # 3) sync_hl_to_db — Hyperliquid-Realität, 30s-Tick
            if now - last_hl_sync >= 30:
                try:
                    n_sync = sync_hl_to_db(s)
                    if n_sync > 0:
                        log.info("hl-sync: %d Phantom-Positionen reaktiviert", n_sync)
                except Exception as e:
                    log.exception("hl-sync failed: %s", e)
                last_hl_sync = now

            # 4) paper_watch — paper_positions gekoppelt an open_predictions
            t1 = time.monotonic()
            try:
                with db_coins(s) as _pc, db_app(s) as _pa:
                    n_pc = paper_watch(_pa, _pc, cfg, get_timeout_hours(s))
                if n_pc > 0:
                    log.info("paper-watch: %d Paper-Positionen geschlossen (%.2fs)",
                             n_pc, time.monotonic() - t1)
            except Exception as e:
                log.exception("paper-watch failed: %s", e)

        except Exception as e:
            log.exception("watch-loop top-level error: %s", e)
        time.sleep(watch_interval)


if __name__ == "__main__":
    main()
