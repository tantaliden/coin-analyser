"""
RL-Agent Gym Environment V5 — Autonomous Agent.

Der Agent bekommt KEINE externen Predictions/Vorschläge.
Er lebt durch die Zeit, sieht alle HL-Coins und entscheidet selbst:
- Welchen Coin handeln
- Richtung (long/short) + Hebel (1x-10x)
- Wann schließen (oder Zwangsschließung nach 60min)

Kein Skip-Penalty. 60min max Haltezeit. SL -5%.

Action-Space: Discrete(21)
  0 = skip (kein Penalty)
  1-10 = long 1x-10x
  11-20 = short 1x-10x
  Mit Position: 0=halten, 1-20=schließen

Trainingsphase: Marktdaten ab 01.04.25 (Vorlauf), Trading ab 01.07.25-31.12.25.
"""
import bisect
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta

from rl_agent.features import compute_observation, empty_candles, N_FEATURES

SL_PERCENT = 5.0
STEP_MINUTES = 5
MAX_HOLD_MINUTES = 60

# Hebel-Mapping für Actions: 1-10 = long, 11-20 = short
LEVERAGE_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10,
}

# Asymmetrische Verlust-Bestrafung pro Hebel
LOSS_PENALTY = {
    1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 1.25,
    6: 1.35, 7: 1.45, 8: 1.55, 9: 1.65, 10: 1.75,
}

# Bonus für Gewinne
WIN_BONUS = {
    1: 1.50, 2: 1.45, 3: 1.40, 4: 1.35, 5: 1.30,
    6: 1.25, 7: 1.20, 8: 1.15, 9: 1.10, 10: 1.15,
}

TRADE_SIZE = 50.0
MAX_FRACTION = 1 / 40
MAX_CONCURRENT = 40
PHASE1_MONTHS = 3  # 3 Monate (Jul-Sep) ohne Portfolio, dann Portfolio-Management

# Zeitbasierte Hebel-Freischaltung (Monate ab Trainingsstart)
LEVERAGE_UNLOCK = [
    (2, 3),    # Monat 1-2: max 3x
    (4, 5),    # Monat 3-4: max 5x
    (999, 10), # Ab Monat 5+: max 10x
]


def _reward(pnl_pct, leverage):
    """Reward berechnen mit mehrstufigem System."""
    base = pnl_pct * leverage

    if pnl_pct < 0:
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
        # Unter 0.4% gehebelt: Strafe — zu wenig Gewinn, Fees fressen das auf
        return -abs(base)

    else:
        # Ab 0.5%: Gewinn-Stufen + hebelabhängiger Bonus
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


class TradingEnvV5(gym.Env):
    """
    Gym Environment V5 — Autonomous Agent.

    Der Agent wird NICHT durch CNN-Predictions getriggert.
    Er bekommt alle HL-Coins präsentiert und entscheidet selbst.
    Jede Episode = ein Coin zu einem Zeitpunkt.
    Der Agent lebt chronologisch durch den Trainingszeitraum.
    """

    metadata = {'render_modes': []}

    def __init__(self, coins, market_data, trade_start, trade_end, train_start=None):
        """
        Args:
            coins: Liste der handelbaren Symbole (z.B. ['BTCUSDC', 'ETHUSDC', ...])
            market_data: Dict mit agg_5m, agg_1h, agg_4h, agg_1d, btc_1h, eth_1h, kline_metrics
            trade_start: Ab wann gehandelt wird (z.B. 2025-07-01)
            trade_end: Bis wann gehandelt wird (z.B. 2025-12-31)
            train_start: Für Hebel-Freischaltung (= trade_start wenn None)
        """
        super().__init__()
        self.coins = coins
        self.market_data = market_data
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.train_start = train_start or trade_start

        self.phase2_start = self.train_start + timedelta(days=PHASE1_MONTHS * 30)

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        # Zeitsteuerung: Agent durchlebt die Zeit chronologisch
        self.global_time = trade_start
        self.coin_idx = 0  # Welcher Coin gerade präsentiert wird

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

        # Wiederholungs-Tracking (symbol → letzte Trade-Time)
        self._last_trade_time = {}

        # Hebel-Statistiken
        self.leverage_stats = {i: {'total_points': 0.0, 'trades': 0} for i in range(1, 11)}

        # Episode-Counter für interne Verwaltung
        self._episode_id = 0

    def _get_max_leverage(self):
        """Zeitbasierte Hebel-Freischaltung."""
        if self.current_time is None:
            return 3
        months_elapsed = (self.current_time - self.train_start).days / 30
        for month_limit, max_lev in LEVERAGE_UNLOCK:
            if months_elapsed < month_limit:
                return max_lev
        return 10

    def _get_effective_max_leverage(self):
        """Max Hebel unter Berücksichtigung von Drawdown-Modus."""
        max_lev = self._get_max_leverage()
        if self.week_points < 0:
            max_lev = min(max_lev, 3)
        return max_lev

    def _check_day_rollover(self):
        """Tageswechsel prüfen und Verlust-Serie aktualisieren."""
        if self.current_time is None:
            return
        today = self.current_time.strftime('%Y-%m-%d')
        if self.current_day == today:
            return

        if self.current_day:
            if self.day_pnl < 0:
                self.losing_streak_days += 1
                if self.losing_streak_days >= 2:
                    penalty_pct = min(2 ** (self.losing_streak_days - 2), 20) / 100
                    penalty = self.total_points * penalty_pct
                    self.total_points -= penalty
            else:
                self.losing_streak_days = 0

        self.current_day = today
        self.day_pnl = 0.0

    def _check_week_rollover(self):
        """Wochenwechsel: Wochen-Bonus nur wenn Rohpunkte ≥10% über Vorwoche."""
        if self.current_time is None:
            return
        iso = self.current_time.isocalendar()
        current_week = f"{iso[0]}-W{iso[1]:02d}"
        if self.current_week == current_week:
            return

        if self.current_week:
            raw_points = self.week_points_raw
            prev_raw = self.prev_week_raw_points

            # Nach Serienbruch: jede positive Woche startet neue Serie (Schwelle = 0)
            # Innerhalb einer Serie: 10% über Vorwoche nötig
            threshold = prev_raw * 1.10 if prev_raw > 0 and self.winning_streak_weeks > 0 else 0
            if raw_points > 0 and raw_points >= threshold:
                self.winning_streak_weeks += 1
                multiplier = 1.20 + (self.winning_streak_weeks - 1) * 0.05
                bonus_points = raw_points * (multiplier - 1.0)
                self.total_points += bonus_points
            else:
                self.winning_streak_weeks = 0

            self.prev_week_raw_points = raw_points

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
        """Prüft ob gleicher Coin innerhalb der letzten Stunde getradet wurde."""
        last = self._last_trade_time.get(symbol)
        if last and self.current_time:
            return (self.current_time - last).total_seconds() < 3600
        return False

    def _next_coin(self):
        """Nächsten Coin auswählen. Rotiert durch alle Coins, dann Zeit voranschreiten."""
        self.coin_idx += 1
        if self.coin_idx >= len(self.coins):
            self.coin_idx = 0
            # Alle Coins durchlaufen → Zeit 5 Minuten vorspulen
            self.global_time += timedelta(minutes=STEP_MINUTES)

    def _has_price_data(self, symbol):
        """Prüft ob für den Coin Preisdaten zum aktuellen Zeitpunkt existieren."""
        data = self.market_data['agg_5m'].get(symbol)
        if not data or 'timestamps' not in data:
            return False
        idx = bisect.bisect_right(data['timestamps'], self.global_time) - 1
        if idx < 0 or idx >= len(data['close']):
            return False
        return float(data['close'][idx]) > 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Nächsten Coin mit Preisdaten finden
        attempts = 0
        while attempts < len(self.coins) * 2:
            if self.global_time >= self.trade_end:
                # Zeitraum vorbei → von vorne (neuer Trainings-Durchlauf)
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

        # Tag/Wochen-Rollover
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

        # Nächsten Coin vorbereiten (für nächsten reset)
        self._next_coin()

        return self._get_obs(), {}

    def _apply_reward(self, pnl_pct):
        """Reward berechnen, Punkte verbuchen, Statistiken aktualisieren."""
        reward = _reward(pnl_pct, self.position_leverage)

        # Wiederholungs-Penalty
        if hasattr(self, '_is_repeat') and self._is_repeat and reward > 0:
            reward *= 0.8

        self.total_points += reward
        self.week_points += reward
        self.week_points_raw += reward
        self.day_pnl += pnl_pct

        lev = self.position_leverage
        self.leverage_stats[lev]['total_points'] += reward
        self.leverage_stats[lev]['trades'] += 1

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
                # Skip — kein Penalty
                reward = 0.0
                terminated = True
            elif action in range(1, 21):
                price = self._current_close()
                if price <= 0:
                    reward = 0.0
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
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
                    self._active_entries.append((self.current_time, self._episode_id))

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

        # Zeit voranschreiten (wenn noch in Position)
        if not terminated:
            self.current_time += timedelta(minutes=STEP_MINUTES)
            self.step_count += 1

            price = self._current_close()
            if price <= 0:
                pnl_pct = self._unrealized_pnl()
                reward = self._apply_reward(pnl_pct)
                self._close()
                terminated = True
            elif self.in_position:
                # SL Check
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
