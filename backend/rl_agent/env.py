"""
RL-Agent Gym Environment.

Episode = eine Opportunity (Coin mit erwartetem ±5% Move).
Agent entscheidet: long/short/skip, dann hold/close alle 5 Min.

Actions:
  Ohne Position: 0=skip, 1=long, 2=short
  Mit Position:  0=halten, 1=schließen, 2=schließen

Reward:
  Proportional zum tatsächlichen PnL.
  Skip = -0.25 (1/20 von max loss).
  SL bei -5% = harte Sicherheitslinie.
"""
import bisect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import timedelta

from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SL_PERCENT = 5.0
SKIP_PENALTY = 1.0
STEP_MINUTES = 5


class TradingEnv(gym.Env):
    """
    Gym Environment für PPO Trading Agent.

    Args:
        opportunities: list of dicts, sorted by detection_time
            {symbol, detection_time, actual_direction}
        market_data: dict with preloaded numpy arrays
            {
                'agg_5m': {symbol: {timestamps, open, high, low, close, volume, trades, taker}},
                'agg_1h': {symbol: {...}},
                'agg_4h': {symbol: {...}},
                'agg_1d': {symbol: {...}},
                'btc_1h': {timestamps, open, high, low, close, ...},
                'eth_1h': {timestamps, open, high, low, close, ...},
                'kline_metrics': {symbol: {timestamps, pct_30m, ...}},
            }
    """

    metadata = {'render_modes': []}

    def __init__(self, opportunities, market_data):
        super().__init__()
        self.opportunities = opportunities
        self.market_data = market_data

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.opp_idx = -1
        self._active_entries = []  # (entry_time, opp_idx) für n_open tracking

        # Episode state
        self.symbol = None
        self.in_position = False
        self.position_direction = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_time = None
        self.step_count = 0

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
        self.entry_price = 0.0
        self.entry_time = None
        self.step_count = 0

        # Alte Einträge aus active_entries entfernen (älter als 48h)
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
            if action == 0:
                reward = -SKIP_PENALTY
                terminated = True
            elif action in (1, 2):
                price = self._current_close()
                if price <= 0:
                    reward = -SKIP_PENALTY
                    terminated = True
                else:
                    self.position_direction = 1 if action == 1 else -1
                    self.entry_price = price
                    self.entry_time = self.current_time
                    self.in_position = True
                    self._active_entries.append((self.current_time, self.opp_idx))
        else:
            # === Management-Entscheidung ===
            if action in (1, 2):
                # Agent will schließen
                reward = self._unrealized_pnl()
                self._close()
                terminated = True

        # Zeit voranschreiten (wenn Episode weiterläuft)
        if not terminated:
            self.current_time += timedelta(minutes=STEP_MINUTES)
            self.step_count += 1

            # Preis für neuen Zeitpunkt prüfen
            price = self._current_close()
            if price <= 0:
                # Keine Daten mehr → schließen mit aktuellem Stand
                reward = self._unrealized_pnl()
                self._close()
                terminated = True
            elif self.in_position:
                # SL prüfen: High/Low der aktuellen Candle
                if self._check_sl():
                    reward = -SL_PERCENT
                    self._close()
                    terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # ============================================================
    # Preis & PnL
    # ============================================================

    def _current_close(self):
        """Close-Preis der aktuellen 5m-Candle."""
        data = self.market_data['agg_5m'].get(self.symbol)
        if not data:
            return 0.0
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return 0.0
        return float(data['close'][idx])

    def _unrealized_pnl(self):
        """Aktueller unrealisierter PnL in Prozent."""
        price = self._current_close()
        if self.entry_price <= 0 or price <= 0:
            return 0.0
        if self.position_direction == 1:
            return (price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - price) / self.entry_price * 100

    def _check_sl(self):
        """Prüft ob SL in der aktuellen 5m-Candle getriggert wurde (High/Low)."""
        data = self.market_data['agg_5m'].get(self.symbol)
        if not data:
            return False
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['high']):
            return False

        if self.position_direction == 1:
            # Long: SL wenn Low unter Entry - 5%
            sl_level = self.entry_price * (1 - SL_PERCENT / 100)
            return float(data['low'][idx]) <= sl_level
        else:
            # Short: SL wenn High über Entry + 5%
            sl_level = self.entry_price * (1 + SL_PERCENT / 100)
            return float(data['high'][idx]) >= sl_level

    def _close(self):
        """Position schließen."""
        self.in_position = False
        self._active_entries = [
            e for e in self._active_entries if e[1] != self.opp_idx
        ]

    # ============================================================
    # Observation
    # ============================================================

    def _get_obs(self):
        """Observation-Vektor aus aktuellem Marktzustand."""
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
        """Sliced vorgeladene Candle-Daten bis current_time."""
        data = self.market_data.get(table, {}).get(symbol)
        if not data or 'timestamps' not in data:
            return empty_candles()

        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0:
            return empty_candles()

        start = max(0, idx - n_candles + 1)
        end = idx + 1
        return {
            'open': data['open'][start:end],
            'high': data['high'][start:end],
            'low': data['low'][start:end],
            'close': data['close'][start:end],
            'volume': data['volume'][start:end],
            'trades': data['trades'][start:end],
            'taker': data['taker'][start:end],
        }

    def _slice_ref(self, key, n_candles):
        """Sliced BTC/ETH Referenzdaten."""
        data = self.market_data.get(key)
        if not data or 'timestamps' not in data:
            return empty_candles()

        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0:
            return empty_candles()

        start = max(0, idx - n_candles + 1)
        end = idx + 1
        return {
            'open': data['open'][start:end],
            'high': data['high'][start:end],
            'low': data['low'][start:end],
            'close': data['close'][start:end],
            'volume': data['volume'][start:end],
            'trades': data['trades'][start:end],
            'taker': data['taker'][start:end],
        }

    def _get_km(self):
        """kline_metrics für aktuellen Zeitpunkt."""
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
