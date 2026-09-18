"""
Evaluate a trained hybrid doctrine+MaskablePPO checkpoint. Reports the
usual figures of merit plus the doctrine mode distribution (EXPLORE /
INVESTIGATE / TRACK / RELOCATE) -- the actual diagnostic that
distinguishes this scheduler from the earlier pure-PPO runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config


def evaluate(model_path: str, config_path: str, episodes: int) -> None:
    config = load_config(config_path)
    env = AlterraEnv(config, enable_doctrine=True)
    model = MaskablePPO.load(model_path)

    rewards, hit_rates, false_alarm_rates, coverage_rates = [], [], [], []
    mean_jump_distances, hit_counts, episode_lengths = [], [], []
    mode_counts: Counter = Counter()

    for episode in range(episodes):
        obs, _ = env.reset(seed=10_000 + episode)

        total_reward, hits, false_alarms, dwells = 0.0, 0, 0, 0
        jumps = []
        visited = set()

        while True:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            dwells += 1
            hits += int(info["any_hit"])
            false_alarms += int(info["any_false_alarm"])
            jumps.append(float(info["jump_distance"]))
            visited.add(int(info["band"]))
            mode_counts[info["doctrine_mode"]] += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        hit_rates.append(hits / max(dwells, 1))
        false_alarm_rates.append(false_alarms / max(dwells, 1))
        coverage_rates.append(len(visited) / env.num_bands)
        mean_jump_distances.append(float(np.mean(jumps)) if jumps else 0.0)
        hit_counts.append(hits)
        episode_lengths.append(dwells)

    total_decisions = sum(mode_counts.values()) or 1
    print("\n=== Alterra Hybrid Doctrine+MaskablePPO Evaluation ===")
    print(f"Model: {Path(model_path)}")
    print(f"Episodes: {episodes}")
    print(f"Reward mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Hit rate: {np.mean(hit_rates):.4f}")
    print(f"False-alarm rate: {np.mean(false_alarm_rates):.4f}")
    print(f"Band coverage: {np.mean(coverage_rates):.4f}")
    print(f"Mean jump distance: {np.mean(mean_jump_distances):.3f} bands")
    print(f"Mean hits/episode: {np.mean(hit_counts):.3f}")
    print(f"Mean RL steps/episode: {np.mean(episode_lengths):.3f}")
    print("\nDoctrine mode distribution:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode:12s} {count:6d}  ({100*count/total_decisions:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.model, args.config, args.episodes)


if __name__ == "__main__":
    main()
