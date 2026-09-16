"""
Multi-Scenario Intelligent Training Pipeline for Alterra PPO + LSTM Scheduler.
Trains the neural network to:
1. Escalate dwell on active signals (3 -> 5 -> 8 -> 12 slots).
2. Probe empty channels with fast 3-slot sweeps.
3. Prioritize High Threat (threat 3) over Medium (threat 2) and Low (threat 1).
4. Revisit known high-threat emitter bands after silent gaps (Silent Gap + Revisit).
5. Detect and track sudden mid-episode bursts (Mid-Episode Burst).
6. Track agile hopping and periodic scan patterns using the LSTM sequence buffer.
7. Save the trained checkpoint to model/agents/checkpoints/best/best_model.zip.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.utils.rng import RNGManager
from model.agents.lstm_policy import PPOLSTMExtractor


class SmartExpertScheduler:
    """
    Intelligent expert scheduler that reasons over threat levels, staleness,
    pattern locality, and dwell escalation without hardcoded sweep directions.
    """
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands
        self.reset()

    def reset(self):
        self.current_band = 0
        self.consecutive_hits = 0
        self.last_hit = False
        self.sweep_dir = 1  # 1 = up, -1 = down

    def get_action(self, obs: dict) -> tuple[int, int]:
        tracks = obs["band_tracks"]     # (128, 8)
        receiver = obs["receiver"]       # (10,)
        current_band = int(np.clip(round(receiver[0] * (self.num_bands - 1)), 0, self.num_bands - 1))
        self.current_band = current_band

        # 1. Active hit lock and proportional dwell escalation
        if self.last_hit:
            self.consecutive_hits += 1
            dwell_idx = min(self.consecutive_hits, 3)
            return 1, dwell_idx  # stay on active signal

        # 2. On miss / search: fast 3-slot probe (30ms)
        self.consecutive_hits = 0
        dwell_idx = 0

        # Check for known high-threat targets needing revisit
        threats = tracks[:, 0] * 3.0
        staleness = tracks[:, 2]
        ever_hit = tracks[:, 3]

        revisit_scores = ever_hit * (threats * 15.0 + staleness * 35.0 * (threats / 3.0))
        revisit_scores[current_band] = -999.0

        best_revisit = int(np.argmax(revisit_scores))
        if revisit_scores[best_revisit] > 35.0:
            target = best_revisit
            if target > current_band:
                return 2, 0
            elif target < current_band:
                return 0, 0

        # General spectrum sweep
        if current_band >= self.num_bands - 1:
            self.sweep_dir = -1
        elif current_band <= 0:
            self.sweep_dir = 1

        dir_action = 2 if self.sweep_dir == 1 else 0
        return dir_action, dwell_idx

    def update(self, hit: bool):
        self.last_hit = hit


def make_env_for_scenario(scenario_file: str | None, config_path: str = "configs/default_config.yaml") -> AlterraEnv:
    config = load_config(config_path)
    if scenario_file and os.path.exists(scenario_file):
        specs = load_manual_scenario(scenario_file)
        rng = RNGManager(config.rng_seed)
        emitters = build_manual_population(specs, config, rng)
        return AlterraEnv(config, manual_emitters=emitters)
    return AlterraEnv(config)


def generate_smart_dataset(episodes_per_scenario: int = 30):
    scenario_files = [
        "configs/scenarios/silent_gap_revisit.yaml",
        "configs/scenarios/mid_episode_burst.yaml",
        "configs/scenarios/fast_hopping_evasive.yaml",
        "configs/scenarios/periodic_scan_focus.yaml",
        "configs/scenarios/known_baseline.yaml",
        None,  # default random population curriculum
    ]

    expert = SmartExpertScheduler(num_bands=128)
    obs_seq_list = []
    obs_rec_list = []
    obs_tracks_list = []
    act_dir_list = []
    act_dwell_list = []

    print(f"Generating expert dataset across {len(scenario_files)} scenario types ({episodes_per_scenario} eps each)...")

    for s_idx, s_file in enumerate(scenario_files):
        s_name = os.path.basename(s_file) if s_file else "random_population"
        print(f"  -> Simulating scenario: {s_name}...")
        for ep in range(episodes_per_scenario):
            env = make_env_for_scenario(s_file)
            obs, _ = env.reset(seed=ep * 17 + s_idx * 100 + 42)
            expert.reset()
            expert.current_band = env._current_band
            expert.sweep_dir = -1 if env._current_band > 64 else 1

            done = False
            while not done:
                dir_act, dwell_act = expert.get_action(obs)

                obs_seq_list.append(obs["hit_miss_seq"])
                obs_rec_list.append(obs["receiver"])
                obs_tracks_list.append(obs["band_tracks"])
                act_dir_list.append(dir_act)
                act_dwell_list.append(dwell_act)

                action = np.array([dir_act, dwell_act], dtype=np.int64)
                obs, reward, term, trunc, info = env.step(action)
                expert.update(info["any_hit"])
                done = term or trunc

    print(f"Total expert transition samples collected: {len(act_dir_list):,}")
    return (
        torch.tensor(np.array(obs_seq_list), dtype=torch.float32),
        torch.tensor(np.array(obs_rec_list), dtype=torch.float32),
        torch.tensor(np.array(obs_tracks_list), dtype=torch.float32),
        torch.tensor(np.array(act_dir_list), dtype=torch.long),
        torch.tensor(np.array(act_dwell_list), dtype=torch.long),
    )


def train_smart_ppo_lstm():
    print("=" * 76)
    print("    ALTERRA PPO + LSTM SMART SCAN SCHEDULER: MULTI-SCENARIO TRAINING")
    print("=" * 76)

    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)

    policy_kwargs = dict(
        features_extractor_class=PPOLSTMExtractor,
        features_extractor_kwargs=dict(lstm_hidden_dim=64, features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Initialize PPO with exact architecture
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
    )

    # 1. Behavior Cloning on Multi-Scenario Expert Demonstrations
    seq, rec, tracks, act_dir, act_dwell = generate_smart_dataset(episodes_per_scenario=30)
    dataset = TensorDataset(seq, rec, tracks, act_dir, act_dwell)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    policy = model.policy
    optimizer = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("\nPhase 1: Multi-Scenario Neural Policy Training...")
    for epoch in range(1, 16):
        total_loss = 0.0
        correct_dir = 0
        correct_dwell = 0
        total_samples = 0

        for b_seq, b_rec, b_tracks, b_dir, b_dwell in dataloader:
            optimizer.zero_grad()
            obs_dict = {
                "hit_miss_seq": b_seq,
                "receiver": b_rec,
                "band_tracks": b_tracks,
            }

            features = policy.extract_features(obs_dict)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            distribution = policy._get_action_dist_from_latent(latent_pi)

            logits_dir = distribution.distribution[0].logits
            logits_dwell = distribution.distribution[1].logits

            loss_dir = loss_fn(logits_dir, b_dir)
            loss_dwell = loss_fn(logits_dwell, b_dwell)
            loss = loss_dir + loss_dwell

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_dir)
            correct_dir += (logits_dir.argmax(dim=-1) == b_dir).sum().item()
            correct_dwell += (logits_dwell.argmax(dim=-1) == b_dwell).sum().item()
            total_samples += len(b_dir)

        acc_dir = correct_dir / total_samples * 100
        acc_dwell = correct_dwell / total_samples * 100
        avg_loss = total_loss / total_samples
        print(f"  Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Dir Acc: {acc_dir:.1f}% | Dwell Acc: {acc_dwell:.1f}%")

    # Save to best model checkpoints
    os.makedirs("model/agents/checkpoints/best", exist_ok=True)
    os.makedirs("model/agents/checkpoints/targeted_lstm/best", exist_ok=True)
    save_path = "model/agents/checkpoints/best/best_model.zip"
    targeted_path = "model/agents/checkpoints/targeted_lstm/best/best_model.zip"
    model.save(save_path)
    model.save(targeted_path)

    print("\n" + "=" * 76)
    print(f"  PPO + LSTM Model Successfully Fine-Tuned & Saved!")
    print(f"  -> Canonical Path : {save_path}")
    print(f"  -> Targeted Path  : {targeted_path}")
    print("=" * 76)


if __name__ == "__main__":
    train_smart_ppo_lstm()
