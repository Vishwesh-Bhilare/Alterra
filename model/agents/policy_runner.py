"""
Unified stepping interface across plain stable_baselines3 algorithms
(PPO) and recurrent ones (sb3-contrib RecurrentPPO). Recurrent policies
need their LSTM hidden state carried across predict() calls plus an
episode_start flag that must be True only on the very first step of an
episode -- PolicyRunner hides that bookkeeping so callers (live
simulation stepping in PythonBridge, and multi-model comparison runs)
use the same .reset()/.predict(obs) interface regardless of which kind
of model is loaded.
"""
from __future__ import annotations

import numpy as np

from model.agents import model_registry


class PolicyRunner:
    def __init__(self, model, algo_class: str):
        self.model = model
        self.algo_class = algo_class
        self._lstm_states = None
        self._episode_start = True

    def reset(self) -> None:
        self._lstm_states = None
        self._episode_start = True

    def predict(self, obs):
        if self.algo_class == "RecurrentPPO":
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=True,
            )
            self._episode_start = False
            return action
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def load_model(repo_root: str, model_id: str) -> PolicyRunner:
    path = model_registry.resolve_model_path(repo_root, model_id)
    algo_class = model_registry.get_algo_class(repo_root, model_id)

    if algo_class == "RecurrentPPO":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as e:
            raise RuntimeError(
                "sb3-contrib is required to load RecurrentPPO checkpoints. "
                "Install it with `pip install sb3-contrib`."
            ) from e
        model = RecurrentPPO.load(path)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(path)

    return PolicyRunner(model, algo_class)


if __name__ == "__main__":
    import argparse
    from simulation.environment.gym_env import AlterraEnv
    from simulation.utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="Test PolicyRunner on Alterra environment")
    parser.add_argument("--model", default="model/agents/checkpoints/best/best_model.zip", help="Path to checkpoint")
    parser.add_argument("--algo", default="PPO", choices=["PPO", "RecurrentPPO"], help="Algorithm class")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    args = parser.parse_args()

    print(f"[PolicyRunner] Loading {args.algo} model from: {args.model}")
    if args.algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        raw_model = RecurrentPPO.load(args.model)
    else:
        from stable_baselines3 import PPO
        raw_model = PPO.load(args.model)

    runner = PolicyRunner(raw_model, args.algo)
    env = AlterraEnv(load_config(args.config))

    obs, info = env.reset(seed=42)
    runner.reset()

    print(f"[PolicyRunner] Successfully initialized. Running {args.steps} test steps:")
    hits = 0
    total_reward = 0.0
    for step in range(args.steps):
        action = runner.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        hit = bool(info.get("hit", False))
        if hit:
            hits += 1
        if step < 10 or hit:
            print(f"  Step {step:03d} | Band: {info.get('band_idx', 0):03d} | Dwell: {info.get('dwell_slots', 0):02d} | Hit: {hit} | Reward: {reward:+.2f}")
        if terminated or truncated:
            break

    print(f"\n[PolicyRunner] Test run complete! Hits: {hits}/{args.steps}, Total Reward: {total_reward:.2f}")

