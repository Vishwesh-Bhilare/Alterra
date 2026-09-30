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
        if self.algo_class == "Hybrid" and hasattr(self.model, "reset"):
            self.model.reset()

    def _adapt_obs(self, obs):
        if not hasattr(self.model, "observation_space") or not hasattr(self.model.observation_space, "spaces"):
            return obs
        adapted = {}
        for k, space in self.model.observation_space.spaces.items():
            if k not in obs:
                continue
            val = np.asarray(obs[k])
            expected_shape = space.shape
            if val.shape != expected_shape:
                if all(v_dim >= e_dim for v_dim, e_dim in zip(val.shape, expected_shape)):
                    slices = tuple(slice(0, dim) for dim in expected_shape)
                    val = val[slices]
                else:
                    padded = np.zeros(expected_shape, dtype=val.dtype)
                    copy_slices = tuple(slice(0, min(v_dim, e_dim)) for v_dim, e_dim in zip(val.shape, expected_shape))
                    padded[copy_slices] = val[copy_slices]
                    val = padded
            adapted[k] = val
        return adapted

    def predict(self, obs, last_info=None):
        if self.algo_class == "Hybrid":
            return self.model.predict(obs, last_info=last_info, deterministic=True)
        obs = self._adapt_obs(obs)
        if self.algo_class == "RecurrentPPO":
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=True,
            )
            self._episode_start = False
        else:
            action, _ = self.model.predict(obs, deterministic=True)

        # In absolute action mode, convert relative step index (0, 1, 2) to adjacent absolute band
        if isinstance(action, (np.ndarray, list)) and len(action) >= 2:
            try:
                import yaml
                from pathlib import Path
                cfg_path = Path("configs/default_config.yaml")
                if cfg_path.exists():
                    with open(cfg_path, "r") as f:
                        cfg_dict = yaml.safe_load(f)
                    if cfg_dict.get("environment", {}).get("action_mode") == "absolute":
                        dir_idx = int(action[0])
                        if dir_idx in (0, 1, 2) and "receiver" in obs:
                            curr_band = int(round(float(obs["receiver"][0]) * 127))
                            delta = [-1, 0, 1][dir_idx]
                            abs_band = max(0, min(127, curr_band + delta))
                            return np.array([abs_band, action[1]], dtype=np.int64)
            except Exception:
                pass

        return action


def load_model(repo_root: str, model_id: str) -> PolicyRunner:
    path = model_registry.resolve_model_path(repo_root, model_id)
    algo_class = model_registry.get_algo_class(repo_root, model_id)

    if algo_class == "Hybrid":
        from model.hybrid.policy import HybridPolicy
        backbone = "lstm"
        for candidate in ("transformer", "gru", "rnn", "lstm"):
            if candidate in model_id.lower() or candidate in path.lower():
                backbone = candidate
                break
        hopping_mode = "local" if "local" in model_id.lower() else "global"
        model = HybridPolicy(checkpoint_path=path, backbone=backbone, hopping_mode=hopping_mode)
    elif algo_class == "RecurrentPPO":
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

