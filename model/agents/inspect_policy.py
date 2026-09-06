"""
Runs one deterministic episode and prints the (band, dwell) action chosen
at every step, to check whether the policy is actually reacting to the
observation or just emitting a constant action.
"""
from __future__ import annotations

import argparse
from collections import Counter

from stable_baselines3 import PPO

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    env = AlterraEnv(config)
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=config.rng_seed + ep)
        actions = []
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            actions.append((int(action[0]), int(action[1])))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        counts = Counter(actions)
        print(f"\nEpisode {ep}: {len(actions)} steps, {len(counts)} unique (band,dwell) actions")
        for action, n in counts.most_common(10):
            print(f"  action={action}  count={n}  ({100*n/len(actions):.1f}%)")


if __name__ == "__main__":
    main()
