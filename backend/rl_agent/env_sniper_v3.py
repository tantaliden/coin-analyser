"""
RL-Agent V3.0 — Sniper Environment (basierend auf env_wallet.py)

Observation: Identisch zu env_wallet (59 Features, bewährt, funktioniert).
Reward: Alles unter +5% nach Hebel = Bestrafung. Große Moves belohnt.

Hebel: 5x bis 10x
Max 5 Positionen
SL bei -10% nach Hebel (doppelte Strafe)
24h Timeout (4x Strafe bei Verlust)

Discrete(13): skip(0), long 5x-10x (1-6), short 5x-10x (7-12)
"""
import bisect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta

from rl_agent.features import compute_observation, empty_candles, N_FEATURES as BASE_N_FEATURES

N_FEATURES = BASE_N_FEATURES + 2  # 57 + 2 = 59

ENTRY_STEP_MINUTES = 5
MGMT_STEP_MINUTES = 1

FUNDING_RATE_PER_HOUR = 0.01
TAKER_FEE = 0.035

TIMEOUT_MINUTES = 1440  # 24h
TIMEOUT_LOSS_MULT = 4   # Verlust-Bestrafung bei Timeout

LEVERAGE_MAP = {
    1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10,       # long 5x-10x
    7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10,     # short 5x-10x
}

SL_EFFECTIVE = 10.0  # -10% nach Hebel = SL

MAX_CONCURRENT = 5
START_POINTS = 2000.0
WEEKLY_TARGET_PCT = 15.0
YEARLY_TARGET_POINTS = 50


def _reward_sniper(pnl_pct, leverage):
    """
    Sniper V3 Reward — nach Hebel berechnet.

    effective = pnl_pct * leverage (= das was aufs Konto fließt)

    Verlust (effective < 0):
      1:1 Minus
      SL (-10%): Doppelt

    Gewinn unter 5%: BESTRAFUNG
      0-1%: -0.5 pro % (harte Strafe für Mikro-Exits)
      1-3%: -0.3 pro %
      3-5%: -0.1 pro %

    Gewinn ab 5%: BELOHNUNG
      5-10%:  0.5 pro % (steigend)
      10-15%: 1.0 pro % (1:1)
      15-25%: 1.2 pro %
      25%+:   1.5 pro %
    """
    effective = pnl_pct * leverage

    # === VERLUST ===
    if effective < 0:
        if effective <= -SL_EFFECTIVE:
            return effective * 2  # SL: doppelte Strafe
        return effective  # 1:1 Minus

    # === GEWINN UNTER 5%: BESTRAFUNG ===
    if effective < 1.0:
        return -0.5 * effective  # Härteste Strafe für Mini-Exits
    elif effective < 3.0:
        return -0.3 * effective
    elif effective < 5.0:
        return -0.1 * effective

    # === GEWINN AB 5%: BELOHNUNG ===
    if effective < 10.0:
        return effective * 0.5
    elif effective < 15.0:
        return effective * 1.0  # 1:1
    elif effective < 25.0:
        return effective * 1.2
    else:
        return effective * 1.5


class TradingEnvSniperV3(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, coins, market_data, trade_start, trade_end, train_start=None, mgmt_step_minutes=None):
        super().__init__()
        self.coins = coins
        self.market_data = market_data
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.train_start = train_start or trade_start
        self._mgmt_step = mgmt_step_minutes or MGMT_STEP_MINUTES

        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.global_time = trade_start
        self.coin_idx = 0
        self._active_entries = []

        self.symbol = None
        self.in_position = False
        self.position_direction = 0
        self.position_leverage = 5
        self.entry_price = 0.0
        self.entry_time = None
        self.current_time = None
        self.step_count = 0

        # Punkte-System
        self.total_points = START_POINTS
        self.current_day = ''
        self.day_pnl = 0.0
        self.current_week = ''
        self.week_points = 0.0
        self.week_points_raw = 0.0
        self.week_start_points = START_POINTS
        self.weekly_target_points = 0

        self._last_trade_time = {}
        self.total_wins = 0
        self.total_losses = 0
        self.total_hold_minutes = 0.0
        self.coin_trade_counts = {}

        self.total_longs = 0
        self.total_shorts = 0

        self._episode_id = 0

    def _check_week_rollover(self):
        if self.current_time is None:
            return 0.0
        iso = self.current_time.isocalendar()
        current_week = f"{iso[0]}-W{iso[1]:02d}"
        if self.current_week == current_week:
            return 0.0

        week_reward = 0.0
        if self.current_week:
            pct_change = (self.total_points - self.week_start_points) / max(self.week_start_points, 1) * 100
            if pct_change >= WEEKLY_TARGET_PCT:
                self.weekly_target_points += 1
                week_reward = 10.0
            elif pct_change < 0:
                week_reward = -5.0

        self.current_week = current_week
        self.week_points = 0.0
        self.week_points_raw = 0.0
        self.week_start_points = self.total_points
        return week_reward

    def _check_day_rollover(self):
        if self.current_time is None:
            return
        today = self.current_time.strftime('%Y-%m-%d')
        if self.current_day == today:
            return
        self.current_day = today
        self.day_pnl = 0.0

    def _next_coin(self):
        self.coin_idx += 1
        if self.coin_idx >= len(self.coins):
            self.coin_idx = 0
            self.global_time += timedelta(minutes=ENTRY_STEP_MINUTES)

    def _has_price_data(self, symbol):
        data = self.market_data['agg_5m'].get(symbol)
        if not data or 'timestamps' not in data:
            return False
        idx = bisect.bisect_right(data['timestamps'], self.global_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return False
        return float(data['close'][idx]) > 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        attempts = 0
        while attempts < len(self.coins) * 2:
            if self.global_time >= self.trade_end:
                self.global_time = self.trade_start
                self.coin_idx = 0
            self.symbol = self.coins[self.coin_idx]
            if self._has_price_data(self.symbol):
                break
            self._next_coin()
            attempts += 1

        self.current_time = self.global_time
        self.in_position = False
        self.position_direction = 0
        self.position_leverage = 5
        self.entry_price = 0.0
        self.entry_time = None
        self.step_count = 0
        self._episode_id += 1

        self._check_day_rollover()
        self._check_week_rollover()

        if self.current_time:
            cutoff = self.current_time - timedelta(hours=48)
            self._active_entries = [e for e in self._active_entries if e[0] > cutoff]

        self._next_coin()
        return self._get_obs(), {}

    def _apply_reward(self, pnl_pct):
        """Trade abschließen, Reward berechnen."""
        if pnl_pct > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1
        if self.entry_time and self.current_time:
            self.total_hold_minutes += (self.current_time - self.entry_time).total_seconds() / 60

        if self.position_direction == 1:
            self.total_longs += 1
        else:
            self.total_shorts += 1

        reward = _reward_sniper(pnl_pct, self.position_leverage)

        self.total_points += reward
        self.week_points += reward
        self.week_points_raw += reward
        self.day_pnl += pnl_pct

        self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1
        return reward

    def _apply_timeout_reward(self, pnl_pct):
        """Timeout-Close: bei Verlust x4 Bestrafung."""
        if pnl_pct > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1
        if self.entry_time and self.current_time:
            self.total_hold_minutes += (self.current_time - self.entry_time).total_seconds() / 60

        if self.position_direction == 1:
            self.total_longs += 1
        else:
            self.total_shorts += 1

        effective = pnl_pct * self.position_leverage
        if effective < 0:
            reward = effective * TIMEOUT_LOSS_MULT
        else:
            # Timeout mit Gewinn: normaler Reward
            reward = _reward_sniper(pnl_pct, self.position_leverage)

        self.total_points += reward
        self.week_points += reward
        self.week_points_raw += reward
        self.day_pnl += pnl_pct

        self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1
        return reward

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if not self.in_position:
            # === ENTRY ===
            n_open = len(self._active_entries)
            if action == 0:
                # Skip
                reward = 0.0
                terminated = True
            elif action in range(1, 13):
                price = self._price_5m()
                if price <= 0:
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
                    reward = -1.0
                    terminated = True
                elif self.symbol in [e[2] for e in self._active_entries if len(e) > 2]:
                    terminated = True
                else:
                    self.position_direction = 1 if action <= 6 else -1
                    self.position_leverage = LEVERAGE_MAP[action]
                    self.entry_price = price
                    self.entry_time = self.current_time
                    self.in_position = True
                    self._active_entries.append((self.current_time, self._episode_id, self.symbol))
                    self._last_trade_time[self.symbol] = self.current_time
        else:
            # === MANAGEMENT ===

            # SL Check: -10% nach Hebel
            pnl_pct = self._unrealized_pnl()
            effective = pnl_pct * self.position_leverage
            if effective <= -SL_EFFECTIVE:
                reward = self._apply_reward(pnl_pct)
                self._close()
                terminated = True
            # Timeout Check: 24h
            elif self.entry_time and self.current_time:
                minutes_held = (self.current_time - self.entry_time).total_seconds() / 60
                if minutes_held >= TIMEOUT_MINUTES:
                    reward = self._apply_timeout_reward(pnl_pct)
                    self._close()
                    terminated = True

            if not terminated:
                if action != 0:
                    # Agent will schließen
                    pnl_pct = self._unrealized_pnl()
                    reward = self._apply_reward(pnl_pct)
                    self._close()
                    terminated = True

        if not terminated:
            self.current_time += timedelta(minutes=self._mgmt_step)
            self.step_count += 1
            price = self._current_close()
            if price <= 0:
                pnl_pct = self._unrealized_pnl()
                reward = self._apply_reward(pnl_pct)
                self._close()
                terminated = True

        if terminated:
            week_reward = self._check_week_rollover()
            reward += week_reward

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _current_close(self):
        if self.in_position:
            price = self._price_1m()
            if price > 0:
                return price
        return self._price_5m()

    def _price_1m(self):
        data = self.market_data.get('klines_1m', {}).get(self.symbol)
        if not data or 'timestamps' not in data:
            return 0.0
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return 0.0
        return float(data['close'][idx])

    def _price_5m(self):
        data = self.market_data.get('agg_5m', {}).get(self.symbol)
        if not data:
            return 0.0
        idx = bisect.bisect_right(data['timestamps'], self.current_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return 0.0
        return float(data['close'][idx])

    def _unrealized_pnl(self):
        """PnL in % (VOR Hebel) mit Fees+Funding."""
        price = self._current_close()
        if self.entry_price <= 0 or price <= 0:
            return 0.0
        if self.position_direction == 1:
            raw_pnl = (price - self.entry_price) / self.entry_price * 100
        else:
            raw_pnl = (self.entry_price - price) / self.entry_price * 100
        fees = TAKER_FEE * 2
        hours_held = 0.0
        if self.entry_time and self.current_time:
            hours_held = (self.current_time - self.entry_time).total_seconds() / 3600
        funding = FUNDING_RATE_PER_HOUR * hours_held * self.position_leverage
        return raw_pnl - fees - funding

    def _close(self):
        self.in_position = False
        self.position_leverage = 5
        self._active_entries = [
            e for e in self._active_entries if e[1] != self._episode_id
        ]

    # ============================================================
    # Observation — identisch zu env_wallet.py (bewährt)
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

        base_obs = compute_observation(
            candles_5m, candles_1h, candles_4h, candles_1d,
            btc_1h, eth_1h,
            self.current_time.hour, self.current_time.weekday(),
            kline_metrics=km,
            position_state=pos_state,
            n_open_positions=len(self._active_entries),
        )

        obs = np.zeros(N_FEATURES, dtype=np.float32)
        obs[:57] = base_obs[:57]
        obs[57] = self.weekly_target_points / max(YEARLY_TARGET_POINTS, 1)
        week_progress = (self.total_points - self.week_start_points) / max(self.week_start_points, 1)
        obs[58] = week_progress

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

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
