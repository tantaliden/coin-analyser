"""
RL-Agent Gym Environment V3 — Alle Hebel (1-10x) + asymmetrische Bestrafung.

Action-Space: Discrete(21)
  0 = skip
  1-10 = long 1x/2x/3x/4x/5x/6x/7x/8x/9x/10x
  11-20 = short 1x/2x/3x/4x/5x/6x/7x/8x/9x/10x
  Mit Position: 0=halten, 1-20=schließen

Reward-System (3 Stufen):
  Verluste: pnl_pct * leverage * LOSS_PENALTY (mild progressiv)
    1x→1.05, 2x→1.10, 3x→1.15, 4x→1.20, 5x→1.25,
    6x→1.35, 7x→1.45, 8x→1.55, 9x→1.65, 10x→1.75

  Gewinne <3.5%: pnl_pct * leverage * EARLY_EXIT_PENALTY (Scalping bestraft)
    0.0-0.5%→0.50, 0.5-1.0%→0.55, 1.0-1.5%→0.60, 1.5-2.0%→0.65,
    2.0-2.5%→0.70, 2.5-3.0%→0.75, 3.0-3.5%→0.80

  Gewinne ≥4%: pnl_pct * leverage * WIN_BONUS (Geduld belohnt)
    1x→1.50, 2x→1.45, 3x→1.40, 4x→1.35, 5x→1.30,
    6x→1.25, 7x→1.20, 8x→1.15, 9x→1.10, 10x→1.15

  Gewinne 3.5-4%: pnl_pct * leverage (normal, kein Modifier)

  Verlust-Tiefe (zusätzlich zu Hebel-Penalty):
    -0.0% bis -0.5%: x1.5 (Panik-Close bestraft, das ist Rauschen)
    -0.5% bis -1.5%: x1.25 (noch zu früh)
    -1.5% bis -4.0%: x1.0 (realistischer Exit, normal)
    ab -4.0%: x1.75 (zu lang gewartet)
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

# Hebel-Mapping für Actions: 1-10 = long, 11-20 = short
LEVERAGE_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,       # Long 1x-10x
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, # Short 1x-10x
}

# Asymmetrische Verlust-Bestrafung pro Hebel
# +0.05 pro Stufe bis 5x, dann +0.1 ab 6x
LOSS_PENALTY = {
    1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 1.25,
    6: 1.35, 7: 1.45, 8: 1.55, 9: 1.65, 10: 1.75,
}

# Bonus für Gewinne ≥4% (Geduld belohnen)
WIN_BONUS = {
    1: 1.50, 2: 1.45, 3: 1.40, 4: 1.35, 5: 1.30,
    6: 1.25, 7: 1.20, 8: 1.15, 9: 1.10, 10: 1.15,
}

# Phase 2 Budget
TRADE_SIZE = 50.0
MAX_FRACTION = 1 / 40
MAX_CONCURRENT = 40

# Phase 1 Dauer: 6 Monate ab Trainingsstart
PHASE1_MONTHS = 6


def _reward(pnl_pct, leverage):
    """Reward berechnen mit mehrstufigem System.

    Verlust: Hebel-Penalty * Verlust-Tiefe-Modifier
    Gewinn <2.5%: Scalping-Penalty (zu früh raus)
    Gewinn 2.5-4%: normal
    Gewinn ≥4%: Bonus (Geduld belohnt)
    """
    base = pnl_pct * leverage

    if pnl_pct < 0:
        # Hebel-Penalty
        lev_penalty = LOSS_PENALTY.get(leverage, 1.1)
        # Verlust-Tiefe-Modifier (auf Basis-PnL, vor Hebel)
        abs_pnl = abs(pnl_pct)
        if abs_pnl < 0.5:
            depth_mult = 1.5     # Panik-Close bei Rauschen
        elif abs_pnl < 1.5:
            depth_mult = 1.25    # Noch zu früh
        elif abs_pnl < 4.0:
            depth_mult = 1.0     # Realistischer Exit
        else:
            depth_mult = 1.75    # Zu lang gewartet
        return base * lev_penalty * depth_mult

    elif base < 0.5:
        # Unter 0.5% gehebelt: Strafe — zu wenig Gewinn, Fees fressen das auf
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


class TradingEnvV3(gym.Env):
    """
    Gym Environment V3 mit asymmetrischer Hebel-Bestrafung.

    Identisch mit V2, nur Reward-Berechnung bei Verlusten anders.
    """

    metadata = {'render_modes': []}

    def __init__(self, opportunities, market_data, train_start=None):
        super().__init__()
        self.opportunities = opportunities
        self.market_data = market_data
        self.train_start = train_start or datetime(2024, 7, 1)

        # Phase 1 endet nach 6 Monaten
        self.phase2_start = self.train_start + timedelta(days=PHASE1_MONTHS * 30)

        self.action_space = spaces.Discrete(21)
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
            elif action in range(1, 21):
                price = self._current_close()
                if price <= 0:
                    reward = 0.0  # Kein Preis = keine Strafe
                    terminated = True
                elif n_open >= MAX_CONCURRENT:
                    reward = 0.0  # Am Limit = keine Strafe
                    terminated = True
                else:
                    # Richtung: 1-10 = long, 11-20 = short
                    self.position_direction = 1 if action <= 10 else -1
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
                reward = _reward(pnl_pct, self.position_leverage)

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
                reward = _reward(pnl_pct, self.position_leverage)
                self._close()
                terminated = True
            elif self.in_position:
                if self._check_sl():
                    reward = _reward(-SL_PERCENT, self.position_leverage)
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
