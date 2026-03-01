"""
SimClock — Globale Zeitquelle für Scanner (Live + Simulation).
Live-Modus: clock.now() = datetime.now(UTC)
Sim-Modus:  clock.now() = simulierte Zeit, clock.advance() spult vor.
"""

from datetime import datetime, timedelta, timezone


class SimClock:
    def __init__(self, start_time=None):
        self._time = start_time  # None = Echtzeit

    def now(self):
        if self._time is not None:
            return self._time
        return datetime.now(timezone.utc)

    def advance(self, minutes):
        if self._time is not None:
            self._time += timedelta(minutes=minutes)

    def set_time(self, t):
        self._time = t

    @property
    def is_sim(self):
        return self._time is not None


# Globale Instanz — wird von scanner.py / scanner_2h.py importiert
clock = SimClock()
