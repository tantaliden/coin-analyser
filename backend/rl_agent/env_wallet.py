"""
RL-Agent Gym Environment W-V1.0 — Wallet-Agent mit Short-Incentives.

Basierend auf env_v11, erweitert um:
- Wochenziel 15% = 1 Punkt (als Feature sichtbar)
- 59 Features (57 Base + Wochenpunkte + Wochenfortschritt)
- 3h Zwangsschließung
- Verkaufs-Block -0.5% bis +0.5%
- Short-Incentives: Penalty bei Long im fallenden Markt, Counterfactual, 30% Short-Minimum

Action-Space: Discrete(21)
  0 = skip
  1-10 = long 1x-10x
  11-20 = short 1x-10x
  Mit Position: 0=halten, 1-20=schliessen
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
MGMT_STEP_MINUTES_FAST = 5

FUNDING_RATE_PER_HOUR = 0.01
TAKER_FEE = 0.035

TIMEOUT_MINUTES = 180  # 3h

LEVERAGE_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10,
}

# Reward-System von env_v11 (bewährt)
LOSS_PENALTY = {
    1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 1.25,
    6: 1.35, 7: 1.45, 8: 1.55, 9: 1.65, 10: 1.75,
}
WIN_BONUS = {
    1: 1.50, 2: 1.45, 3: 1.40, 4: 1.35, 5: 1.30,
    6: 1.25, 7: 1.20, 8: 1.15, 9: 1.10, 10: 1.15,
}

MAX_CONCURRENT = 15
START_POINTS = 2000.0
WEEKLY_TARGET_PCT = 15.0
YEARLY_TARGET_POINTS = 50
PHASE1_MONTHS = 3

# Short-Incentives
SHORT_MIN_PCT = 0.30  # Min 30% Shorts in den ersten 60 Tagen

LEVERAGE_UNLOCK = [(999, 10)]


def _reward(pnl_pct, leverage):
    """Reward berechnen — identisch zu env_v11."""
    base = pnl_pct * leverage

    if pnl_pct < 0 and pnl_pct >= -0.4:
        lev_penalty = LOSS_PENALTY.get(leverage, 1.1)
        return base * lev_penalty * 1.5
    elif pnl_pct < -0.4:
        lev_penalty = LOSS_PENALTY.get(leverage, 1.1)
        abs_pnl = abs(pnl_pct)
        if abs_pnl < 0.5:
            depth_mult = 1.5
        elif abs_pnl < 1.5:
            depth_mult = 1.25
        elif abs_pnl < 4.0:
            depth_mult = 1.0
        else:
            depth_mult = 1.75
        return base * lev_penalty * depth_mult
    elif base < 0.4:
        return -abs(base)
    else:
        if pnl_pct < 1.0:
            mult = 0.50
        elif pnl_pct < 2.0:
            mult = 1.00
        elif pnl_pct < 3.0:
            mult = 1.10
        elif pnl_pct < 5.0:
            mult = 1.20
        elif pnl_pct < 10.0:
            mult = 1.25
        else:
            mult = 1.60
        bonus = WIN_BONUS.get(leverage, 1.1)
        return base * mult * bonus


class TradingEnvWallet(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, coins, market_data, trade_start, trade_end, train_start=None, mgmt_step_minutes=None, event_times=None):
        super().__init__()
        self.coins = coins
        self.market_data = market_data
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.train_start = train_start or trade_start
        self._mgmt_step = mgmt_step_minutes or MGMT_STEP_MINUTES

        # Event/Replay-Modus: Springt zu vorgegebenen Zeitpunkten
        self._event_times = sorted(event_times) if event_times else None
        self._event_idx = 0
        self._replay_rewards = {}

        self.phase2_start = self.train_start + timedelta(days=PHASE1_MONTHS * 30)
        self._learn_phase_end = self.trade_start + timedelta(days=60)
        self._short_explore_end = self.trade_start + timedelta(days=60)

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.global_time = trade_start
        self.coin_idx = 0
        self._active_entries = []

        self.symbol = None
        self.in_position = False
        self.position_direction = 0
        self.position_leverage = 1
        self.entry_price = 0.0
        self.entry_time = None
        self.current_time = None
        self.step_count = 0

        # Punkte-System (env_v11)
        self.total_points = START_POINTS
        self.current_day = ''
        self.day_pnl = 0.0
        self.losing_streak_days = 0
        self.current_week = ''
        self.week_points = 0.0
        self.week_points_raw = 0.0
        self.prev_week_raw_points = 0
        self.winning_streak_weeks = 0

        # Wochenziel-System
        self.weekly_target_points = 0
        self.week_start_points = START_POINTS

        self._last_trade_time = {}
        self.leverage_stats = {i: {'total_points': 0.0, 'trades': 0, 'wins': 0} for i in range(1, 11)}
        self.total_wins = 0
        self.total_losses = 0
        self.total_hold_minutes = 0.0
        self.coin_trade_counts = {}

        self.win_streak = 0
        self.lose_streak = 0
        self._trade_day = ''
        self._trades_today = 0

        # Short-Tracking
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

        self.prev_week_raw_points = self.week_points_raw
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
        self._trade_day = today
        self._trades_today = 0

    def _get_max_leverage(self):
        return 10

    def _get_effective_max_leverage(self):
        max_lev = self._get_max_leverage()
        if self.week_points < 0:
            max_lev = min(max_lev, 3)
        return max_lev

    def _next_coin(self):
        if self._event_times:
            self._event_idx += 1
            if self._event_idx >= len(self._event_times):
                self._event_idx = 0
            self.global_time = self._event_times[self._event_idx]
            self.current_time = self.global_time
            self.coin_idx = 0
        else:
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

    def _is_market_falling(self):
        """Prüft ob der Markt (BTC) in den letzten 2h gefallen ist."""
        btc_data = self.market_data.get('btc_1h', {})
        if not btc_data or 'close' not in btc_data or len(btc_data['close']) < 3:
            return False
        recent = btc_data['close']
        idx = bisect.bisect_right(btc_data.get('timestamps', []), self.current_time) - 1
        if idx < 2:
            return False
        return float(recent[idx]) < float(recent[idx - 2]) * 0.99  # -1% in 2h

    def _counterfactual_pnl(self, pnl_pct):
        """Was hätte die andere Richtung gebracht?"""
        return -pnl_pct  # Spiegelung

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._event_times:
            # Replay-Modus: Zum aktuellen Event springen
            if self._event_idx >= len(self._event_times):
                self._event_idx = 0
            self.global_time = self._event_times[self._event_idx]
            self.current_time = self.global_time
            self.coin_idx = 0

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
        self.position_leverage = 1
        self.entry_price = 0.0
        self.entry_time = None
        self.step_count = 0
        self._episode_id += 1

        self._check_day_rollover()
        self._check_week_rollover()

        if self.current_time:
            cutoff = self.current_time - timedelta(hours=48)
            self._active_entries = [e for e in self._active_entries if e[0] > cutoff]
            self._last_trade_time = {
                s: t for s, t in self._last_trade_time.items()
                if (self.current_time - t).total_seconds() < 7200
            }

        self._next_coin()
        return self._get_obs(), {}

    def _apply_reward(self, pnl_pct):
        # Win/Loss + Haltedauer
        if pnl_pct > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1
        if self.entry_time and self.current_time:
            self.total_hold_minutes += (self.current_time - self.entry_time).total_seconds() / 60

        # Short-Tracking
        if self.position_direction == 1:
            self.total_longs += 1
        else:
            self.total_shorts += 1

        # Lernphase: Einfaches +/- PnL
        if self.current_time and self.current_time < self._learn_phase_end:
            reward = pnl_pct
            self.total_points += reward
            self.week_points += reward
            self.week_points_raw += reward
            self.day_pnl += pnl_pct
            lev = self.position_leverage
            self.leverage_stats[lev]['total_points'] += reward
            self.leverage_stats[lev]['trades'] += 1
            if pnl_pct > 0:
                self.leverage_stats[lev]['wins'] += 1
            self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1
            self._trades_today += 1
            return reward

        # Volles Reward-System
        reward = _reward(pnl_pct, self.position_leverage)

        # === SHORT-INCENTIVE 1: Penalty bei Long im fallenden Markt ===
        if self.position_direction == 1 and pnl_pct < 0 and self._is_market_falling():
            reward *= 1.5  # 50% härtere Strafe für Long bei fallendem Markt

        # === SHORT-INCENTIVE 2: Counterfactual ===
        if pnl_pct < -1.0:
            counter_pnl = self._counterfactual_pnl(pnl_pct)
            if counter_pnl > 1.0:
                # Die andere Richtung hätte >1% gebracht — Extra-Penalty
                reward -= abs(counter_pnl) * 0.5

        # === SHORT-INCENTIVE 3: Erzwungene Short-Exploration ===
        if self.current_time and self.current_time < self._short_explore_end:
            total_trades = self.total_longs + self.total_shorts
            if total_trades > 20:
                short_pct = self.total_shorts / total_trades
                if short_pct < SHORT_MIN_PCT and self.position_direction == 1:
                    # Zu wenig Shorts — Long-Gewinne reduzieren
                    if reward > 0:
                        reward *= 0.5

        # Streak-System
        if pnl_pct >= 0.5:
            self.win_streak += 1
            self.lose_streak = 0
            if self.win_streak >= 3:
                streak_mult = 1.5 + (self.win_streak - 3) * 0.1
                reward *= streak_mult
        elif pnl_pct < 0.0:
            self.lose_streak += 1
            self.win_streak = 0
            if self.lose_streak >= 3:
                streak_mult = 1.5 + (self.lose_streak - 3) * 0.1
                reward *= streak_mult
        else:
            self.lose_streak = 0
            self.win_streak = 0

        self.total_points += reward
        self.week_points += reward
        self.week_points_raw += reward
        self.day_pnl += pnl_pct

        lev = self.position_leverage
        self.leverage_stats[lev]['total_points'] += reward
        self.leverage_stats[lev]['trades'] += 1
        if pnl_pct > 0:
            self.leverage_stats[lev]['wins'] += 1

        self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1
        self._trades_today += 1

        return reward

    def _get_replay_override(self, action):
        """Replay-Modus: Reward aus echtem Trade-Ergebnis ableiten."""
        if not self._replay_rewards:
            return None
        key = (self.symbol, self.current_time.isoformat()[:16] if self.current_time else '')
        real = self._replay_rewards.get(key)
        if not real:
            return None

        real_dir = real['direction']  # 'long' oder 'short'
        real_reward = real['reward']
        real_pnl = real['pnl_pct']

        if action == 0:
            # Skip wo ein Trade war
            if real_pnl > 0:
                return -abs(real_reward) * 0.3  # verpasste profitable Chance
            return abs(real_reward) * 0.2  # Verlust-Trade vermieden = gut

        agent_dir = 'long' if action <= 10 else 'short'

        if agent_dir == real_dir:
            # Gleiche Richtung wie im echten Trade
            return real_reward  # gute oder schlechte Entscheidung bestätigen/spüren
        else:
            # Andere Richtung als im echten Trade
            if real_pnl > 0:
                return -abs(real_reward) * 0.5  # echter Trade war gut, Agent dreht falsch
            return abs(real_reward) * 0.7  # echter Trade war Verlust, Agent korrigiert = belohnen

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if not self.in_position:
            n_open = len(self._active_entries)
            # Replay-Override: Echten Trade-Reward nutzen
            replay_reward = self._get_replay_override(action)
            if replay_reward is not None:
                reward = replay_reward
                terminated = True
            elif action == 0:
                reward = 0.0
                terminated = True
            elif action in range(1, 21):
                price = self._price_5m()
                if price <= 0:
                    reward = 0.0
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
                    reward = -1.0
                    terminated = True
                elif self.symbol in [e[2] for e in self._active_entries if len(e) > 2]:
                    reward = 0.0
                    terminated = True
                else:
                    requested_leverage = LEVERAGE_MAP[action]
                    max_lev = self._get_effective_max_leverage()
                    if requested_leverage > max_lev:
                        requested_leverage = max_lev

                    self.position_direction = 1 if action <= 10 else -1
                    self.position_leverage = requested_leverage
                    self.entry_price = price
                    self.entry_time = self.current_time
                    self.in_position = True
                    self._active_entries.append((self.current_time, self._episode_id, self.symbol))
                    self._last_trade_time[self.symbol] = self.current_time
        else:
            # Timeout: 3h
            if self.entry_time and self.current_time:
                minutes_held = (self.current_time - self.entry_time).total_seconds() / 60
                if minutes_held >= TIMEOUT_MINUTES:
                    pnl_pct = self._unrealized_pnl()
                    reward = self._apply_reward(pnl_pct)
                    self._close()
                    terminated = True

            if not terminated:
                if action != 0:
                    pnl_pct = self._unrealized_pnl()
                    # Verkauf blockieren bei -0.5% bis +0.5%
                    if -0.5 <= pnl_pct <= 0.5:
                        action = 0
                    else:
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
        """PnL mit Fees+Funding (wie im Training von V11)."""
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
        self.position_leverage = 1
        self._active_entries = [
            e for e in self._active_entries if e[1] != self._episode_id
        ]

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

        # Erweiterte Features (57 + 2 = 59)
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
