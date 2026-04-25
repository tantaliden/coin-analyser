"""
RL-Agent — PPO Trading Agent.

Wrapper um Stable Baselines3 PPO.
Trainiert auf synthetischen ±5% Moves, deployed auf CNN-Predictions.
"""
from pathlib import Path
from stable_baselines3 import PPO

MODEL_DIR = Path("/opt/coin/database/data/models")
MODEL_PATH = MODEL_DIR / "rl_ppo_trading.zip"


class TradingAgent:

    def __init__(self):
        self.model = None

    def create(self, env):
        """Neues PPO Modell erstellen."""
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs={
                'net_arch': [256, 256],
            },
        )
        print(f"[RL-AGENT] PPO Modell erstellt: 256-256 MLP")

    def train(self, total_timesteps):
        """Training starten/fortsetzen."""
        if self.model is None:
            raise RuntimeError("Kein Modell. Erst create() oder load() aufrufen.")
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        """Action für eine Observation (deterministic)."""
        if self.model is None:
            return 0
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def save(self):
        """Modell speichern."""
        if self.model:
            self.model.save(str(MODEL_PATH))
            print(f"[RL-AGENT] Modell gespeichert: {MODEL_PATH}")

    def load(self, env=None):
        """Modell laden."""
        if MODEL_PATH.exists():
            try:
                self.model = PPO.load(str(MODEL_PATH), env=env)
                print(f"[RL-AGENT] PPO Modell geladen: {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"[RL-AGENT] Laden fehlgeschlagen: {e}")
                return False
        print("[RL-AGENT] Kein Modell vorhanden")
        return False
