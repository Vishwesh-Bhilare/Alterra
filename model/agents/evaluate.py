"""
Evaluate a trained PPO scheduler against AlterraEnv, both deterministic and
stochastic action selection, with per-episode reward reported (not just
the mean) so variance is visible.
"""
from __future__ import annotations

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_venv_dir = os.path.join(_repo_root, ".venv")
_venv_python = os.path.join(_venv_dir, "bin", "python")
if os.path.exists(_venv_python) and sys.prefix != _venv_dir:
    os.execv(_venv_python, [_venv_python, "-m", "model.agents.evaluate"] + sys.argv[1:])

import argparse

import numpy as np
from stable_baselines3 import PPO

from model.agents.lstm_policy import PPOLSTMExtractor

__all__ = ["PPOLSTMExtractor"]

from simulation.environment import AlterraEnv
from simulation.metrics import MetricsTracker
from simulation.utils.config_loader import load_config


def run(model, env, episodes, steps_per_episode, deterministic, base_seed):
    per_episode = []
    episode_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=base_seed + ep)
        tracker = MetricsTracker()
        ep_reward = 0.0
        for _ in range(steps_per_episode):
            model_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
            action, _ = model.predict(model_obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.record_step(env.last_dwell_result, reward)
            ep_reward += reward
            if terminated or truncated:
                break
        per_episode.append(tracker.finalize(env))
        episode_rewards.append(ep_reward)
    return per_episode, episode_rewards


def summarize(label, per_episode, episode_rewards):
    def avg(attr):
        vals = [getattr(m, attr) for m in per_episode if getattr(m, attr) is not None]
        return (sum(vals) / len(vals)) if vals else None

    print(f"\n--- {label} ---")
    print(f"Per-episode total reward: {[round(r, 1) for r in episode_rewards]}")
    print(f"  mean={np.mean(episode_rewards):.2f}  std={np.std(episode_rewards):.2f}  "
          f"min={np.min(episode_rewards):.2f}  max={np.max(episode_rewards):.2f}")
    for lbl, attr in [
        ("Pd", "probability_of_detection"),
        ("Pfa", "probability_of_false_alarm"),
        ("Avg intercept rate", "avg_intercept_rate"),
        ("Percent correct", "percent_correct"),
        ("Avg intercept time error (slots)", "avg_intercept_time_error_slots"),
    ]:
        v = avg(attr)
        print(f"{lbl}: {v:.3f}" if v is not None else f"{lbl}: n/a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    base_seed = args.base_seed if args.base_seed is not None else config.rng_seed
    env = AlterraEnv(config)
    model = PPO.load(args.model)

    det_episodes, det_rewards = run(model, env, args.episodes, args.steps_per_episode, True, base_seed)
    summarize("Deterministic", det_episodes, det_rewards)

    sto_episodes, sto_rewards = run(model, env, args.episodes, args.steps_per_episode, False, base_seed)
    summarize("Stochastic", sto_episodes, sto_rewards)


if __name__ == "__main__":
    main()
