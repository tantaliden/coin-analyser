"""
RL-Agent Gym Environment V6 — Autonomous Agent, 1-Min Management.

Wie V5 aber: Positions-Management im 1-Minuten-Takt statt 5 Minuten.
Agent kann schneller auf Preisbewegungen reagieren.

Entry-Scan: alle 5 Min (wie V5)
Management: jede 1 Min (NEU)
Preise in Position: klines_1m (statt agg_5m)
SL-Check: klines_1m high/low (praeziser)

Action-Space: Discrete(21)
  0 = skip (kein Penalty)
  1-10 = long 1x-10x
  11-20 = short 1x-10x
  Mit Position: 0=halten, 1-20=schliessen
"""
import bisect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta

from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SL_PERCENT = 2.5
ENTRY_STEP_MINUTES = 5     # Entry-Scan alle 5 Min
MGMT_STEP_MINUTES = 1      # Management jede 1 Min (default)
MGMT_STEP_MINUTES_FAST = 5 # Management alle 5 Min (fuer schnelleres Training)
MGMT_SWITCH_DATE = None  # Kein Switch
MAX_HOLD_MINUTES = 60

# Hebel-Mapping fuer Actions: 1-10 = long, 11-20 = short
LEVERAGE_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10,
}

# Asymmetrische Verlust-Bestrafung pro Hebel
LOSS_PENALTY = {
    1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 1.25,
    6: 1.35, 7: 1.45, 8: 1.55, 9: 1.65, 10: 1.75,
}

# Bonus fuer Gewinne
WIN_BONUS = {
    1: 1.50, 2: 1.45, 3: 1.40, 4: 1.35, 5: 1.30,
    6: 1.25, 7: 1.20, 8: 1.15, 9: 1.10, 10: 1.15,
}

TRADE_SIZE = 50.0
MAX_FRACTION = 1 / 40
MAX_CONCURRENT = 25
PHASE1_MONTHS = 3

# Alle Hebel von Anfang an freigeschaltet
LEVERAGE_UNLOCK = [
    (999, 10),
]


def _reward(pnl_pct, leverage):
    """Reward berechnen mit mehrstufigem System."""
    base = pnl_pct * leverage

    if pnl_pct < 0 and pnl_pct >= -0.4:
        # Leichtes Minus (0 bis -0.4%): Proportionale Strafe statt Pauschale
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


class TradingEnvV6(gym.Env):
    """
    Gym Environment V6 — Autonomous Agent, 1-Min Management.

    Entry-Scan alle 5 Min (Coin-Rotation), Management jede 1 Min.
    Preise in Position aus klines_1m fuer praezisere Reaktion.
    """

    metadata = {'render_modes': []}

    def __init__(self, coins, market_data, trade_start, trade_end, train_start=None, mgmt_step_minutes=None):
        super().__init__()
        self.coins = coins
        self.market_data = market_data
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.train_start = train_start or trade_start
        self._mgmt_step = mgmt_step_minutes or MGMT_STEP_MINUTES

        self.phase2_start = self.train_start + timedelta(days=PHASE1_MONTHS * 30)

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self.global_time = trade_start
        self.coin_idx = 0

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

        # Portfolio (Phase 2)
        self.portfolio = 0.0
        self._open_margins = []

        # Serien-Tracking
        self.total_points = 0.0
        self.current_day = ''
        self.day_pnl = 0.0
        self.losing_streak_days = 0
        self.current_week = ''
        self.week_points = 0.0
        self.week_points_raw = 0.0
        self.prev_week_raw_points = 0
        self.winning_streak_weeks = 0

        self._last_trade_time = {}
        self.leverage_stats = {i: {'total_points': 0.0, 'trades': 0} for i in range(1, 11)}
        self.coin_trade_counts = {}  # symbol -> trade count

        # Streak-Tracking
        self.win_streak = 0    # Aufeinanderfolgende Trades >= 0.5%
        self.lose_streak = 0   # Aufeinanderfolgende Trades < 0.0%

        # Tages-Trade-Tracking (Mindestens 500 Trades/Tag sonst -500)
        self._trade_day = ''
        self._trades_today = 0

        # Lernphase: Erste 2 Monate einfaches +/- PnL, danach volles System
        self._learn_phase_end = self.trade_start + timedelta(days=60)
        self._daily_multiplier = 1  # Verdoppelt sich bei 5% Tages-PnL in Lernphase
        self._episode_id = 0

    def _get_max_leverage(self):
        if self.current_time is None:
            return 3
        months_elapsed = (self.current_time - self.train_start).days / 30
        for month_limit, max_lev in LEVERAGE_UNLOCK:
            if months_elapsed < month_limit:
                return max_lev
        return 10

    def _get_effective_max_leverage(self):
        max_lev = self._get_max_leverage()
        if self.week_points < 0:
            max_lev = min(max_lev, 3)
        return max_lev

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

    def _check_week_rollover(self):
        if self.current_time is None:
            return
        iso = self.current_time.isocalendar()
        current_week = f"{iso[0]}-W{iso[1]:02d}"
        if self.current_week == current_week:
            return
        self.prev_week_raw_points = self.week_points_raw
        self.current_week = current_week
        self.week_points = 0.0
        self.week_points_raw = 0.0

    def _is_phase2(self):
        if self.current_time is None:
            return False
        return self.current_time >= self.phase2_start

    def _get_trade_size(self):
        if self.portfolio > 2000:
            return self.portfolio * MAX_FRACTION
        return TRADE_SIZE

    def _is_repeat_trade(self, symbol):
        last = self._last_trade_time.get(symbol)
        if last and self.current_time:
            return (self.current_time - last).total_seconds() < 3600
        return False

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
        self.position_leverage = 1
        self.entry_price = 0.0
        self.entry_time = None
        self.step_count = 0
        self._episode_id += 1

        self._check_day_rollover()
        self._check_week_rollover()

        if self.current_time:
            cutoff = self.current_time - timedelta(hours=48)
            self._active_entries = [
                e for e in self._active_entries if e[0] > cutoff
            ]
            self._last_trade_time = {
                s: t for s, t in self._last_trade_time.items()
                if (self.current_time - t).total_seconds() < 7200
            }

        self._next_coin()

        return self._get_obs(), {}

    def _apply_reward(self, pnl_pct):
        # Lernphase (erste 2 Monate): Einfaches +/- PnL
        if self.current_time and self.current_time < self._learn_phase_end:
            reward = pnl_pct
            self.total_points += reward
            self.week_points += reward
            self.week_points_raw += reward
            self.day_pnl += pnl_pct
            lev = self.position_leverage
            self.leverage_stats[lev]['total_points'] += reward
            self.leverage_stats[lev]['trades'] += 1
            self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1
            self._trades_today += 1
            return reward

        # Ab Monat 3: Volles Reward-System
        reward = _reward(pnl_pct, self.position_leverage)

        if hasattr(self, '_is_repeat') and self._is_repeat and reward > 0:
            reward *= 0.8

        # Streak-System: Belohnt Konstanz, bestraft Serien von Verlusten
        if pnl_pct >= 0.5:
            # Win (>= 0.5%): Win-Streak zaehlen, Lose-Streak resetten
            self.win_streak += 1
            self.lose_streak = 0
            if self.win_streak >= 3:
                streak_mult = 1.5 + (self.win_streak - 3) * 0.1
                reward *= streak_mult
        elif pnl_pct < 0.0:
            # Loss (< 0.0%): Lose-Streak zaehlen, Win-Streak resetten
            self.lose_streak += 1
            self.win_streak = 0
            if self.lose_streak >= 3:
                streak_mult = 1.5 + (self.lose_streak - 3) * 0.1
                reward *= streak_mult  # Reward ist negativ, wird staerker negativ
        else:
            # 0.0% - 0.5%: Lose-Streak resetten, Win-Streak resetten
            self.lose_streak = 0
            self.win_streak = 0

        self.total_points += reward
        self.week_points += reward
        self.week_points_raw += reward
        self.day_pnl += pnl_pct

        lev = self.position_leverage
        self.leverage_stats[lev]['total_points'] += reward
        self.leverage_stats[lev]['trades'] += 1

        # Coin-Tracking
        self.coin_trade_counts[self.symbol] = self.coin_trade_counts.get(self.symbol, 0) + 1

        # Tages-Trade-Counter
        self._trades_today += 1

        if self._is_phase2():
            margin_entry = [m for m in self._open_margins if m[0] == self._episode_id]
            if margin_entry:
                _, margin, lev = margin_entry[0]
                pnl_dollar = margin * lev * pnl_pct / 100
                self.portfolio += margin + pnl_dollar
                self._open_margins = [m for m in self._open_margins if m[0] != self._episode_id]

        return reward

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if not self.in_position:
            # === Entry-Entscheidung ===
            n_open = len(self._active_entries)
            if action == 0:
                reward = 0.0
                terminated = True
            elif action in range(1, 21):
                price = self._price_5m()
                if price <= 0:
                    reward = 0.0
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
                    reward = -1.0  # V8: Strafe fuer Kaufversuch bei vollen Slots
                    terminated = True
                elif self.symbol in [e[2] for e in self._active_entries if len(e) > 2]:
                    # Doppel-Coin-Check: gleicher Coin schon offen
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

                    self._is_repeat = self._is_repeat_trade(self.symbol)
                    self._last_trade_time[self.symbol] = self.current_time

                    if self._is_phase2():
                        margin = self._get_trade_size()
                        self.portfolio -= margin
                        self._open_margins.append((self._episode_id, margin, self.position_leverage))
        else:
            # === Management-Entscheidung ===
            if action != 0:
                pnl_pct = self._unrealized_pnl()
                reward = self._apply_reward(pnl_pct)
                self._close()
                terminated = True

        # Zeit voranschreiten
        if not terminated:
            if MGMT_SWITCH_DATE and self.current_time < MGMT_SWITCH_DATE:
                mgmt_step = MGMT_STEP_MINUTES_FAST
            else:
                mgmt_step = self._mgmt_step
            self.current_time += timedelta(minutes=mgmt_step)
            self.step_count += 1

            price = self._current_close()
            if price <= 0:
                pnl_pct = self._unrealized_pnl()
                reward = self._apply_reward(pnl_pct)
                self._close()
                terminated = True
            elif self.in_position:
                # SL Check (1m-Daten)
                if self._check_sl():
                    reward = self._apply_reward(-SL_PERCENT)
                    self._close()
                    terminated = True
                # 60min Timeout
                elif self.entry_time and (self.current_time - self.entry_time).total_seconds() >= MAX_HOLD_MINUTES * 60:
                    pnl_pct = self._unrealized_pnl()
                    reward = self._apply_reward(pnl_pct)
                    self._close()
                    terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # ============================================================
    # Preis & PnL — 1m wenn in Position, sonst 5m
    # ============================================================

    def _current_close(self):
        """Aktueller Preis: klines_1m wenn in Position, sonst agg_5m."""
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
        price = self._current_close()
        if self.entry_price <= 0 or price <= 0:
            return 0.0
        if self.position_direction == 1:
            return (price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - price) / self.entry_price * 100

    def _check_sl(self):
        """SL-Check mit klines_1m (praeziser), Fallback agg_5m."""
        data = self.market_data.get('klines_1m', {}).get(self.symbol)
        if not data or 'timestamps' not in data:
            data = self.market_data.get('agg_5m', {}).get(self.symbol)
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
            e for e in self._active_entries if e[1] != self._episode_id
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
