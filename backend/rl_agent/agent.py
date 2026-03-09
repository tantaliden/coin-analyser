"""
RL-Agent — Policy-Gradient Trading-Agent.
Lernt aus echten Erfahrungen (Observations, Actions, Rewards).
Kein Dummy-Environment, kein Fake-Training.

Training: Reward-gewichtetes Policy-Gradient direkt auf dem Experience Buffer.
- Positive Rewards → Action verstärken (das war gut, mehr davon)
- Negative Rewards → Action abschwächen (das war schlecht, weniger davon)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

MODEL_DIR = Path("/opt/coin/database/data/models")
MODEL_PATH = MODEL_DIR / "rl_agent_policy.pt"
BUFFER_PATH = MODEL_DIR / "rl_agent_buffer.npz"

# Observation: 24 features
# --- CNN Prediction ---
# 0: confidence (0-1)
# 1: direction (0=short, 1=long)
# 2: tp_target_pct / 10 (normalized)
# 3: sl_target_pct / 10 (normalized)
# --- Sentiment ---
# 4: sentiment_score (-1 to 1)
# 5: fear_greed / 100 (0 to 1)
# --- Coin Stats ---
# 6: coin_hit_rate (0-1)
# --- Account State ---
# 7: balance_normalized (balance / 10000)
# 8: open_positions / 10
# 9: unrealized_pnl_normalized (pnl / balance, clamped -1 to 1)
# --- Zeitkontext ---
# 10: hour_of_day / 24 (0-1)
# 11: day_of_week / 7 (0-1)
# --- 1m Klines (letzte 15 Min) ---
# 12: price_return_1m (letzte Minute, %)
# 13: price_return_5m (letzte 5 Min, %)
# 14: price_return_15m (letzte 15 Min, %)
# 15: volatility_15m (range/price der letzten 15 Min)
# 16: volume_trend (aktuelles Vol / Durchschnitt 15 Min)
# 17: taker_ratio_1m (buyer ratio letzte Minute, 0-1)
# 18: taker_ratio_5m (buyer ratio letzte 5 Min, 0-1)
# 19: trades_trend (aktuelle Trades / Durchschnitt 15 Min)
# --- Live-Preis ---
# 20: hl_price_vs_entry (Hyperliquid-Preis vs CNN Entry, %)
# 21: spot_perp_spread (Binance Spot vs HL Perp Differenz, %)
# --- Reserviert ---
# 22: reserved_1 (0)
# 23: reserved_2 (0)
OBS_DIM = 24

# Action: 3 continuous values
# 0: take_trade (-1 to 1, > 0 = take)
# 1: size_factor (0 to 1, mapped to min-max position size)
# 2: leverage_factor (0 to 1, mapped to 1-max_leverage)
ACTION_DIM = 3


class PolicyNetwork(nn.Module):
    """
    Einfaches Policy-Netz: Observation → Action (Mean + Std für Gaussian Policy).
    64-64 MLP, tanh-Aktivierung für bounded output.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Mean-Head: Output für jede Action-Dimension
        self.mean_head = nn.Linear(64, action_dim)
        # Log-Std: lernbar, pro Action-Dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        features = self.shared(obs)
        mean = torch.tanh(self.mean_head(features))  # Bounded -1 to 1
        std = torch.exp(self.log_std.clamp(-2, 0.5))  # Std zwischen ~0.13 und ~1.65
        return mean, std

    def get_action(self, obs, deterministic=False):
        """Gibt Action + Log-Probability zurück."""
        mean, std = self.forward(obs)
        if deterministic:
            return mean, None
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs, actions):
        """Log-Probability für gegebene Actions."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)


class ExperienceBuffer:
    """
    Speichert echte Erfahrungen (obs, action, reward) für Training.
    Persistent auf Disk.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.load()

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        if len(self.observations) > self.max_size:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)

    def size(self):
        return len(self.observations)

    def get_batch(self):
        """Gibt alle Erfahrungen als Tensoren zurück."""
        obs = torch.FloatTensor(np.array(self.observations, dtype=np.float32))
        actions = torch.FloatTensor(np.array(self.actions, dtype=np.float32))
        rewards = torch.FloatTensor(np.array(self.rewards, dtype=np.float32))
        return obs, actions, rewards

    def save(self):
        if self.observations:
            np.savez(
                BUFFER_PATH,
                obs=np.array(self.observations, dtype=np.float32),
                actions=np.array(self.actions, dtype=np.float32),
                rewards=np.array(self.rewards, dtype=np.float32),
            )

    def load(self):
        if BUFFER_PATH.exists():
            try:
                data = np.load(BUFFER_PATH)
                self.observations = list(data["obs"])
                self.actions = list(data["actions"])
                self.rewards = list(data["rewards"])
                print(f"[RL-AGENT] Buffer geladen: {len(self.observations)} Erfahrungen")
            except Exception as e:
                print(f"[RL-AGENT] Buffer laden fehlgeschlagen: {e}")


class RLAgent:
    """
    Policy-Gradient Agent für Trading-Entscheidungen.
    Trainiert direkt auf echten Erfahrungen aus dem Buffer.
    """

    def __init__(self):
        self.buffer = ExperienceBuffer()
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.total_decisions = 0
        self.load_or_create()

    def load_or_create(self):
        """Modell laden oder neu starten."""
        if MODEL_PATH.exists():
            try:
                checkpoint = torch.load(MODEL_PATH, weights_only=True)
                self.policy.load_state_dict(checkpoint["policy"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.total_decisions = checkpoint.get("total_decisions", 0)
                print(f"[RL-AGENT] Modell geladen: {MODEL_PATH}")
                return
            except Exception as e:
                print(f"[RL-AGENT] Modell laden fehlgeschlagen: {e}")

        print("[RL-AGENT] Neues Modell erstellt (64-64 Policy Network)")

    def decide(self, observation: np.ndarray) -> dict:
        """
        Entscheidet bei einer Prediction: traden oder nicht.
        Returns: {take: bool, size_factor: float, leverage_factor: float, raw_action: ndarray}
        """
        obs = np.clip(observation, -1.0, 1.0).astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        self.policy.eval()
        with torch.no_grad():
            # Deterministic wenn genug Erfahrung, sonst explorieren
            deterministic = self.buffer.size() >= 200
            action, _ = self.policy.get_action(obs_tensor, deterministic=deterministic)
            action = action.squeeze(0).numpy()

        self.total_decisions += 1

        take = float(action[0]) > 0
        size_factor = (float(action[1]) + 1) / 2  # Map -1..1 to 0..1
        leverage_factor = (float(action[2]) + 1) / 2  # Map -1..1 to 0..1

        return {
            "take": take,
            "size_factor": np.clip(size_factor, 0.0, 1.0),
            "leverage_factor": np.clip(leverage_factor, 0.0, 1.0),
            "raw_action": action,
        }

    def record_outcome(self, observation: np.ndarray, action: np.ndarray, reward: float):
        """Speichert Ergebnis eines Trades/Skip für späteres Training."""
        self.buffer.add(observation, action, reward)

    def train_on_buffer(self, min_samples: int = 50) -> bool:
        """
        Trainiert die Policy mit echten Erfahrungen aus dem Buffer.

        Methode: Reward-gewichtetes Policy Gradient.
        - Für jede Erfahrung (obs, action, reward):
          - Berechne log_prob der Action unter aktueller Policy
          - Loss = -log_prob * normalized_reward
          - Positive Rewards → Action verstärken
          - Negative Rewards → Action abschwächen

        Returns True wenn Training stattfand.
        """
        if self.buffer.size() < min_samples:
            print(f"[RL-AGENT] Nicht genug Daten für Training ({self.buffer.size()}/{min_samples})")
            return False

        print(f"[RL-AGENT] Training mit {self.buffer.size()} echten Erfahrungen...")

        try:
            obs, actions, rewards = self.buffer.get_batch()

            # Rewards normalisieren (mean=0, std=1) für stabiles Training
            reward_mean = rewards.mean()
            reward_std = rewards.std()
            if reward_std > 1e-8:
                normalized_rewards = (rewards - reward_mean) / reward_std
            else:
                print("[RL-AGENT] Rewards haben keine Varianz, Training übersprungen")
                return False

            self.policy.train()

            # Mehrere Epochen über den ganzen Buffer
            n_epochs = 10
            batch_size = min(64, self.buffer.size())

            for epoch in range(n_epochs):
                # Shuffle
                indices = torch.randperm(len(obs))
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(obs), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_obs = obs[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_rewards = normalized_rewards[batch_idx]

                    # Log-Probability der tatsächlichen Actions
                    log_probs = self.policy.log_prob(batch_obs, batch_actions)

                    # Policy Gradient Loss: -log_prob * reward
                    # Positive reward → minimize -log_prob → maximize log_prob → Action verstärken
                    # Negative reward → minimize +log_prob → Action abschwächen
                    loss = -(log_probs * batch_rewards).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient Clipping für Stabilität
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)

            # Stats loggen
            pos_rewards = (rewards > 0).sum().item()
            neg_rewards = (rewards < 0).sum().item()
            zero_rewards = (rewards == 0).sum().item()
            print(
                f"[RL-AGENT] Training fertig — Loss: {avg_loss:.4f}, "
                f"Rewards: {pos_rewards} pos / {neg_rewards} neg / {zero_rewards} neutral, "
                f"Mean Reward: {reward_mean:.4f}"
            )

            self.save()
            return True

        except Exception as e:
            print(f"[RL-AGENT] Training fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save(self):
        """Modell und Buffer speichern."""
        try:
            torch.save({
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_decisions": self.total_decisions,
            }, MODEL_PATH)
            self.buffer.save()
        except Exception as e:
            print(f"[RL-AGENT] Speichern fehlgeschlagen: {e}")

    def get_stats(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "buffer_size": self.buffer.size(),
            "model_exists": MODEL_PATH.exists(),
            "exploration": "stochastic" if self.buffer.size() < 200 else "deterministic",
        }
