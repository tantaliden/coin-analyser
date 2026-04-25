"""
RL-Agent Gym Environment V2 — Erweiterter Action-Space mit Hebel.

Action-Space: Discrete(9)
  Ohne Position: 0=skip, 1-4=long (1x/3x/5x/10x), 5-8=short (1x/3x/5x/10x)
  Mit Position:  0=halten, 1-8=schließen

Phase 1 (erste 6 Monate): Nur Lernen, kein Budget.
  Reward = PnL% * Hebel (Agent lernt Richtung + Hebel-Risiko)
Phase 2 (Monate 7-18): $50/Position, max 40 gleichzeitig, kein Startkapital.
  Portfolio startet bei $0, geht ins Minus wenn Trades offen.
  Agent darf IMMER traden (bis max 40 gleichzeitig).
  Wenn Portfolio > $2000 → 1/40 davon > $50, darf er mit mehr handeln.

Skip-Penalty: -1.0
SL: -5% (auf Basis-Position, vor Hebel)
"""
import bisect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta

from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SL_PERCENT = 5.0
SKIP_PENALTY = 1.0
STEP_MINUTES = 5

# Hebel-Mapping für Actions
LEVERAGE_MAP = {
    1: 1, 2: 3, 3: 5, 4: 10,   # Long 1x/3x/5x/10x
    5: 1, 6: 3, 7: 5, 8: 10,   # Short 1x/3x/5x/10x
}

# Phase 2 Budget
TRADE_SIZE = 50.0
MAX_FRACTION = 1 / 40
MAX_CONCURRENT = 40

# Phase 1 Dauer: 6 Monate ab Trainingsstart
PHASE1_MONTHS = 6


class TradingEnvV2(gym.Env):
    """
    Gym Environment V2 mit erweitertem Action-Space (Hebel).

    Args:
        opportunities: list of dicts, sorted by detection_time
            {symbol, detection_time, actual_direction}
        market_data: dict with preloaded numpy arrays
        train_start: datetime — Beginn des Trainings (für Phase-Berechnung)
    """

    metadata = {'render_modes': []}

    def __init__(self, opportunities, market_data, train_start=None):
        super().__init__()
        self.opportunities = opportunities
        self.market_data = market_data
        self.train_start = train_start or datetime(2024, 7, 1)

        # Phase 1 endet nach 6 Monaten
        self.phase2_start = self.train_start + timedelta(days=PHASE1_MONTHS * 30)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.opp_idx = -1
        self._active_entries = []

        # Episode state
        self.symbol = None
        self.in_position = False
        self.position_direction = 0
        self.position_leverage = 1
        self.entry_price = 0.0
        self.entry_time = None
        self.current_time = None
        self.step_count = 0

        # Portfolio (Phase 2): startet bei $0, geht ins Minus
        self.portfolio = 0.0
        self._open_margins = []  # Liste von (opp_idx, margin, leverage)

    def _is_phase2(self):
        """Prüft ob wir in Phase 2 sind (nach 6 Monaten)."""
        if self.current_time is None:
            return False
        return self.current_time >= self.phase2_start

    def _get_trade_size(self):
        """Positionsgröße: $50 oder 1/40 Portfolio (was höher ist).
        Agent darf IMMER traden — Portfolio geht halt ins Minus."""
        if self.portfolio > 2000:
            return self.portfolio * MAX_FRACTION
        return TRADE_SIZE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.opp_idx += 1
        if self.opp_idx >= len(self.opportunities):
            self.opp_idx = 0

        opp = self.opportunities[self.opp_idx]
        self.symbol = opp['symbol']
        self.current_time = opp['detection_time']
        self.in_position = False
        self.position_direction = 0
        self.position_leverage = 1
        self.entry_price = 0.0
        self.entry_time = None
        self.step_count = 0

        # Alte Einträge cleanup
        if self.current_time:
            cutoff = self.current_time - timedelta(hours=48)
            self._active_entries = [
                e for e in self._active_entries if e[0] > cutoff
            ]

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if not self.in_position:
            # === Entry-Entscheidung ===
            n_open = len(self._active_entries)
            if action == 0:
                # Skip: Penalty nur wenn er KÖNNTE (< 40 offen)
                reward = -SKIP_PENALTY if n_open < MAX_CONCURRENT else 0.0
                terminated = True
            elif action in range(1, 9):
                price = self._current_close()
                if price <= 0:
                    reward = 0.0  # Kein Preis = keine Strafe
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
                    reward = 0.0  # Am Limit = keine Strafe
                    terminated = True
                else:
                    # Richtung: 1-4 = long, 5-8 = short
                    self.position_direction = 1 if action <= 4 else -1
                    self.position_leverage = LEVERAGE_MAP[action]
                    self.entry_price = price
                    self.entry_time = self.current_time
                    self.in_position = True
                    self._active_entries.append((self.current_time, self.opp_idx))

                    # Phase 2: Margin vom Portfolio abziehen (geht ins Minus)
                    if self._is_phase2():
                        margin = self._get_trade_size()
                        self.portfolio -= margin
                        self._open_margins.append((self.opp_idx, margin, self.position_leverage))
        else:
            # === Management-Entscheidung ===
            if action != 0:
                # Agent will schließen
                pnl_pct = self._unrealized_pnl()
                reward = pnl_pct * self.position_leverage

                # Phase 2: Margin zurück + PnL auf Portfolio
                if self._is_phase2():
                    margin_entry = [m for m in self._open_margins if m[0] == self.opp_idx]
                    if margin_entry:
                        _, margin, lev = margin_entry[0]
                        pnl_dollar = margin * lev * pnl_pct / 100
                        self.portfolio += margin + pnl_dollar  # Margin zurück + Gewinn/Verlust
                        self._open_margins = [m for m in self._open_margins if m[0] != self.opp_idx]

                self._close()
                terminated = True

        # Zeit voranschreiten
        if not terminated:
            self.current_time += timedelta(minutes=STEP_MINUTES)
            self.step_count += 1

            price = self._current_close()
            if price <= 0:
                pnl_pct = self._unrealized_pnl()
                reward = pnl_pct * self.position_leverage
                self._close()
                terminated = True
            elif self.in_position:
                if self._check_sl():
                    reward = -SL_PERCENT * self.position_leverage
                    self._close()
                    terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # ============================================================
    # Preis & PnL
    # ============================================================

    def _current_close(self):
        data = self.market_data['agg_5m'].get(self.symbol)
        if not data:
            return 0.0
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return 0.0
        return float(data['close'][idx])

    def _unrealized_pnl(self):
        price = self._current_close()
        if self.entry_price <= 0 or price <= 0:
            return 0.0
        if self.position_direction == 1:
            return (price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - price) / self.entry_price * 100

    def _check_sl(self):
        data = self.market_data['agg_5m'].get(self.symbol)
        if not data:
            return False
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['high']):
            return False

        if self.position_direction == 1:
            sl_level = self.entry_price * (1 - SL_PERCENT / 100)
            return float(data['low'][idx]) <= sl_level
        else:
            sl_level = self.entry_price * (1 + SL_PERCENT / 100)
            return float(data['high'][idx]) >= sl_level

    def _close(self):
        self.in_position = False
        self.position_leverage = 1
        self._active_entries = [
            e for e in self._active_entries if e[1] != self.opp_idx
        ]

    # ============================================================
    # Observation
    # ============================================================

    def _get_obs(self):
        if not self.current_time:
            return np.zeros(N_FEATURES, dtype=np.float32)

        candles_5m = self._slice('agg_5m', self.symbol, 144)
        candles_1h = self._slice('agg_1h', self.symbol, 24)
        candles_4h = self._slice('agg_4h', self.symbol, 12)
        candles_1d = self._slice('agg_1d', self.symbol, 14)
        btc_1h = self._slice_ref('btc_1h', 25)
        eth_1h = self._slice_ref('eth_1h', 25)
        km = self._get_km()

        pos_state = None
        if self.in_position and self.entry_time:
            duration = (self.current_time - self.entry_time).total_seconds() / 60
            pos_state = {
                'in_position': True,
                'direction': self.position_direction,
                'unrealized_pnl': self._unrealized_pnl(),
                'duration_min': duration,
            }

        return compute_observation(
            candles_5m, candles_1h, candles_4h, candles_1d,
            btc_1h, eth_1h,
            self.current_time.hour, self.current_time.weekday(),
            kline_metrics=km,
            position_state=pos_state,
            n_open_positions=len(self._active_entries),
        )

    def _slice(self, table, symbol, n_candles):
        data = self.market_data.get(table, {}).get(symbol)
        if not data or 'timestamps' not in data:
            return empty_candles()
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0:
            return empty_candles()
        start = max(0, idx - n_candles + 1)
        end = idx + 1
        return {k: data[k][start:end] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}

    def _slice_ref(self, key, n_candles):
        data = self.market_data.get(key)
        if not data or 'timestamps' not in data:
            return empty_candles()
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0:
            return empty_candles()
        start = max(0, idx - n_candles + 1)
        end = idx + 1
        return {k: data[k][start:end] for k in ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker']}

    def _get_km(self):
        km_data = self.market_data.get('kline_metrics', {}).get(self.symbol)
        if not km_data or 'timestamps' not in km_data:
            return None
        idx = bisect.bisect_right(km_data['timestamps'], self.current_time) - 1
        if idx < 0:
            return None
        result = {}
        for col in ['pct_30m', 'pct_60m', 'pct_90m', 'pct_120m', 'pct_180m', 'pct_240m',
                     'pct_300m', 'pct_360m', 'pct_420m', 'pct_480m', 'pct_540m', 'pct_600m']:
            if col in km_data:
                result[col] = float(km_data[col][idx])
        return result
