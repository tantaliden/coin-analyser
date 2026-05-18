"""HL Websocket Live-State Singleton.

Background-Thread connectet zu HL-WS und haelt mids + user-state in-memory aktuell.
Wallet/Predictor-Routes lesen aus diesem Store statt REST-Calls
→ keine 429er, sub-200ms Latenz vom Markt-Update zur API-Antwort.

Subscriptions:
- allMids: alle Mid-Preise (~100-200ms updates)
- webData2(user): user clearinghouseState (assetPositions, marginSummary, openOrders, spotState)
- userFills(user): recent fills

Bei Stale-Daten (>10s ohne Update) → fallback auf REST in den Endpoint-Wrappern.
"""
import logging
import threading
import time
from typing import Optional, Dict, Any, List

from hyperliquid.info import Info

log = logging.getLogger("wallet.hl_ws_state")

STALE_THRESHOLD_S = 10.0


class HLWsState:
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz"):
        self.base_url = base_url
        self._info: Optional[Info] = None
        self._lock = threading.RLock()
        self._mids: Dict[str, float] = {}
        self._mids_ts: float = 0.0
        self._webdata: Dict[str, Any] = {}
        self._webdata_ts: float = 0.0
        self._fills: List[Dict[str, Any]] = []
        self._fills_ts: float = 0.0
        self._user_address: Optional[str] = None
        self._sub_ids: List[int] = []
        self._started = False
        self._started_lock = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def start(self, user_address: str):
        with self._started_lock:
            if self._started and self._user_address == user_address:
                return
            if self._started:
                self._stop_internal()
            self._user_address = user_address
            self._stop_flag.clear()
            self._connect_and_subscribe()
            # Reconnect-Watchdog: prüft alle 30s ob Daten stale sind
            if not self._reconnect_thread or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(
                    target=self._reconnect_loop, daemon=True, name="hl_ws_reconnect")
                self._reconnect_thread.start()

    def _connect_and_subscribe(self):
        try:
            self._info = Info(self.base_url, skip_ws=False)
            time.sleep(0.5)  # WS braucht kurz zum Verbinden
            self._sub_ids = []
            self._sub_ids.append(self._info.subscribe(
                {"type": "allMids"}, self._on_all_mids))
            self._sub_ids.append(self._info.subscribe(
                {"type": "webData2", "user": self._user_address.lower()}, self._on_webdata))
            self._sub_ids.append(self._info.subscribe(
                {"type": "userFills", "user": self._user_address.lower()}, self._on_fills))
            self._started = True
            log.info("HL WS subscribed: allMids + webData2 + userFills for %s",
                     self._user_address)
        except Exception as e:
            log.exception("HL WS connect failed: %s", e)
            self._info = None
            self._started = False

    def _reconnect_loop(self):
        while not self._stop_flag.wait(30.0):
            try:
                # Wenn schon lange keine Mids → Reconnect
                age_mids = time.time() - self._mids_ts if self._mids_ts > 0 else 999
                if age_mids > 60 and self._user_address:
                    log.warning("HL WS stale (mids age %.1fs) — reconnecting", age_mids)
                    self._stop_internal()
                    time.sleep(2)
                    self._connect_and_subscribe()
            except Exception as e:
                log.warning("reconnect loop error: %s", e)

    def _stop_internal(self):
        if self._info:
            try:
                self._info.disconnect_websocket()
            except Exception as e:
                log.warning("disconnect_websocket failed: %s", e)
            self._info = None
        self._sub_ids = []
        self._started = False

    def stop(self):
        self._stop_flag.set()
        self._stop_internal()

    def _on_all_mids(self, ws_msg):
        try:
            data = ws_msg.get("data", {}) if isinstance(ws_msg, dict) else {}
            mids_raw = data.get("mids", {})
            new_mids = {}
            for coin, px in mids_raw.items():
                try:
                    new_mids[coin] = float(px)
                except (ValueError, TypeError):
                    continue
            with self._lock:
                self._mids = new_mids
                self._mids_ts = time.time()
        except Exception as e:
            log.warning("on_all_mids parse failed: %s", e)

    def _on_webdata(self, ws_msg):
        try:
            data = ws_msg.get("data", {}) if isinstance(ws_msg, dict) else {}
            with self._lock:
                self._webdata = data
                self._webdata_ts = time.time()
        except Exception as e:
            log.warning("on_webdata failed: %s", e)

    def _on_fills(self, ws_msg):
        try:
            data = ws_msg.get("data", {}) if isinstance(ws_msg, dict) else {}
            new_fills = data.get("fills", []) if isinstance(data, dict) else []
            if not new_fills:
                return
            with self._lock:
                # webhook sendet inkrementell — append, dedupliziere via tid+time, keep last 1000
                existing_keys = {(f.get("tid"), f.get("time")) for f in self._fills}
                for f in new_fills:
                    if (f.get("tid"), f.get("time")) not in existing_keys:
                        self._fills.append(f)
                # newest first, keep 1000
                self._fills.sort(key=lambda x: x.get("time", 0), reverse=True)
                self._fills = self._fills[:1000]
                self._fills_ts = time.time()
        except Exception as e:
            log.warning("on_fills failed: %s", e)

    # Public Read-API ------------------------------------------------

    def is_mids_fresh(self, max_age_s: float = STALE_THRESHOLD_S) -> bool:
        return self._mids_ts > 0 and (time.time() - self._mids_ts) < max_age_s

    def is_webdata_fresh(self, max_age_s: float = STALE_THRESHOLD_S) -> bool:
        return self._webdata_ts > 0 and (time.time() - self._webdata_ts) < max_age_s

    def is_fills_fresh(self, max_age_s: float = 60.0) -> bool:
        # Fills kommen seltener (nur bei trades) — TTL grosszuegiger
        return self._fills_ts > 0

    def get_mids(self) -> Optional[Dict[str, float]]:
        with self._lock:
            if self.is_mids_fresh():
                return dict(self._mids)
        return None

    def get_webdata(self) -> Optional[Dict[str, Any]]:
        """webData2-Payload. Enthaelt clearinghouseState, openOrders, spotState etc."""
        with self._lock:
            if self.is_webdata_fresh():
                return dict(self._webdata)
        return None

    def get_user_state_view(self) -> Optional[Dict[str, Any]]:
        """Liefert den 'clearinghouseState'-Teil aus webData2 — kompatibel zur
        REST-Antwort von info.user_state()."""
        wd = self.get_webdata()
        if not wd:
            return None
        return wd.get("clearinghouseState")

    def get_spot_state_view(self) -> Optional[Dict[str, Any]]:
        """Liefert spotState aus webData2 — kompatibel zu info.spot_user_state()."""
        wd = self.get_webdata()
        if not wd:
            return None
        return wd.get("spotState")

    def get_open_orders_view(self) -> Optional[List[Dict[str, Any]]]:
        wd = self.get_webdata()
        if not wd:
            return None
        return wd.get("openOrders", [])

    def get_fills(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._fills)


_global_state: Optional[HLWsState] = None
_global_lock = threading.Lock()


def get_ws_state() -> HLWsState:
    """Singleton accessor. user_address wird beim ersten ensure_started() gesetzt."""
    global _global_state
    with _global_lock:
        if _global_state is None:
            _global_state = HLWsState()
        return _global_state


def ensure_started(user_address: str):
    """Startet WS-Subscriber falls noch nicht laufend (idempotent)."""
    s = get_ws_state()
    s.start(user_address)
