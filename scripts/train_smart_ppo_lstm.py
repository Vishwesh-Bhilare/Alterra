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
    Intelligent expert scheduler that isolates contiguous periodic patterns,
    draws down into the exact repeating pattern span, and prioritizes discrete revisits.
    """
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands
        self.reset()

    def reset(self):
        self.current_band = 0
        self.consecutive_hits = 0
        self.last_hit = False
        self.sweep_dir = 1  # 1 = up, -1 = down
        self.target_revisit_band = None

    def find_clusters(self, hit_indices):
        if len(hit_indices) == 0:
            return []
        sorted_hits = np.sort(hit_indices)
        clusters = []
        curr = [sorted_hits[0]]
        for b in sorted_hits[1:]:
            if b - curr[-1] <= 2:  # strict contiguous cluster
                curr.append(b)
            else:
                clusters.append(curr)
                curr = [b]
        clusters.append(curr)
        return clusters

    def get_action(self, obs: dict) -> tuple[int, int]:
        tracks = obs["band_tracks"]     # (128, 8)
        receiver = obs["receiver"]       # (10,)
        current_band = int(np.clip(round(receiver[0] * (self.num_bands - 1)), 0, self.num_bands - 1))
        self.current_band = current_band

        # 1. Active hit lock: dwell escalation (3 -> 5 -> 8 -> 12), max 4 consecutive dwells on same band
        if self.last_hit and self.consecutive_hits < 4:
            self.consecutive_hits += 1
            dwell_idx = min(self.consecutive_hits, 3)
            return 1, dwell_idx

        self.consecutive_hits = 0
        dwell_idx = 0

        if self.target_revisit_band == current_band:
            self.target_revisit_band = None

        threats = tracks[:, 0] * 3.0
        ever_hit = tracks[:, 3]
        time_since_visit = tracks[:, 4]
        scanned_ratio = receiver[8]

        hit_indices = np.where(ever_hit > 0.5)[0]

        # 2. Post-discovery: identify periodic pattern and draw down
        if len(hit_indices) > 0 and (scanned_ratio >= 0.50 or current_band >= 126 or (current_band <= 1 and scanned_ratio >= 0.35)):
            clusters = self.find_clusters(hit_indices)

            # A true periodic scan pattern is a contiguous cluster with >= 5 bands and width <= 20
            periodic_pattern = None
            other_emitters = []
            for c in clusters:
                w = c[-1] - c[0] + 1
                density = len(c) / max(w, 1)
                if len(c) >= 5 and 5 <= w <= 20 and density >= 0.7:
                    if periodic_pattern is None or len(c) > len(periodic_pattern):
                        if periodic_pattern is not None:
                            other_emitters.extend(periodic_pattern)
                        periodic_pattern = c
                    else:
                        other_emitters.extend(c)
                else:
                    other_emitters.extend(c)

            # Check if an external high-threat emitter needs a quick revisit
            urgent_revisit = None
            highest_score = 0.0
            for b in other_emitters:
                staleness = time_since_visit[b]
                threat = max(threats[b], 1.0)
                score = threat * (1.0 + 3.0 * staleness)
                if staleness >= 0.15 and score > 2.8:
                    if score > highest_score:
                        highest_score = score
                        urgent_revisit = b

            if urgent_revisit is not None:
                self.target_revisit_band = urgent_revisit
                if urgent_revisit > current_band:
                    self.sweep_dir = 1
                    return 2, 0
                elif urgent_revisit < current_band:
                    self.sweep_dir = -1
                    return 0, 0
                else:
                    return 1, 0

            if self.target_revisit_band is not None and self.target_revisit_band != current_band:
                if self.target_revisit_band > current_band:
                    self.sweep_dir = 1
                    return 2, 0
                else:
                    self.sweep_dir = -1
                    return 0, 0

            # If periodic pattern is found: DRAW DOWN STRICTLY INTO THE PATTERN!
            if periodic_pattern is not None:
                pat_lo = periodic_pattern[0]
                pat_hi = periodic_pattern[-1]

                # If outside pattern span, step directly toward it
                if current_band < pat_lo:
                    self.sweep_dir = 1
                    return 2, 0
                elif current_band > pat_hi:
                    self.sweep_dir = -1
                    return 0, 0
                else:
                    # Inside pattern span: sweep strictly between pat_lo and pat_hi
                    if current_band >= pat_hi:
                        self.sweep_dir = -1
                    elif current_band <= pat_lo:
                        self.sweep_dir = 1
                    dir_action = 2 if self.sweep_dir == 1 else 0
                    return dir_action, dwell_idx

            # If no periodic pattern: discrete emitter revisit
            threat_weights = np.maximum(threats, 1.0)
            revisit_priority = np.where(
                (ever_hit > 0.5) & (time_since_visit >= 0.04),
                threat_weights * (1.0 + 3.0 * time_since_visit),
                -999.0
            )
            revisit_priority[current_band] = -999.0
            best = int(np.argmax(revisit_priority))
            if revisit_priority[best] > 1.2:
                if best > current_band:
                    self.sweep_dir = 1
                    return 2, 0
                elif best < current_band:
                    self.sweep_dir = -1
                    return 0, 0
            elif len(hit_indices) == 1:
                # Oscillate locally near single emitter
                b = hit_indices[0]
                span_lo = max(0, b - 4)
                span_hi = min(self.num_bands - 1, b + 4)
                if current_band >= span_hi:
                    self.sweep_dir = -1
                elif current_band <= span_lo:
                    self.sweep_dir = 1
                dir_action = 2 if self.sweep_dir == 1 else 0
                return dir_action, dwell_idx

        # 3. Initial survey sweep
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


def generate_smart_dataset(episodes_per_scenario: int = 40, random_population_episodes: int = 80):
    scenario_files = [
        ("configs/scenarios/silent_gap_revisit.yaml", episodes_per_scenario),
        ("configs/scenarios/mid_episode_burst.yaml", episodes_per_scenario),
        ("configs/scenarios/fast_hopping_evasive.yaml", episodes_per_scenario),
        ("configs/scenarios/periodic_scan_focus.yaml", episodes_per_scenario),
        ("configs/scenarios/custom_mix_multi_threat.yaml", episodes_per_scenario),
        ("configs/scenarios/known_baseline.yaml", episodes_per_scenario),
        ("configs/scenarios/dense_congested.yaml", episodes_per_scenario),
        ("configs/scenarios/sparse_single_threat.yaml", episodes_per_scenario),
        (None, random_population_episodes),  # default random population curriculum
    ]

    expert = SmartExpertScheduler(num_bands=128)
    obs_seq_list = []
    obs_rec_list = []
    obs_tracks_list = []
    act_dir_list = []
    act_dwell_list = []

    print(f"Generating expert dataset across {len(scenario_files)} scenario configurations...")

    for s_idx, (s_file, n_eps) in enumerate(scenario_files):
        s_name = os.path.basename(s_file) if s_file else "random_population"
        print(f"  -> Simulating scenario: {s_name} ({n_eps} episodes)...")
        for ep in range(n_eps):
            env = make_env_for_scenario(s_file)
            obs, _ = env.reset(seed=ep * 17 + s_idx * 100 + 42)
            expert.reset()
            expert.current_band = env._current_band
            expert.sweep_dir = 1 if env._prev_action_direction_norm > 0.5 else -1

            done = False
            while not done:
                dir_act, dwell_act = expert.get_action(obs)

                obs_seq_list.append(obs["hit_miss_seq"])
                obs_rec_list.append(obs["receiver"])
                obs_tracks_list.append(obs["band_tracks"])
                act_dir_list.append(dir_act)
                act_dwell_list.append(dwell_act)

                # Boundary turnaround oversampling: enforce strong reflection signal at endpoints
                if env._current_band <= 0 or env._current_band >= 127:
                    for _ in range(4):
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
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
    )

    # 1. Behavior Cloning on Multi-Scenario Expert Demonstrations
    seq, rec, tracks, act_dir, act_dwell = generate_smart_dataset(
        episodes_per_scenario=40,
        random_population_episodes=80,
    )
    dataset = TensorDataset(seq, rec, tracks, act_dir, act_dwell)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training acceleration device: {device}")
    policy = model.policy.to(device)

    optimizer = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)

    print("\nPhase 1: Deep Multi-Scenario Neural Policy Pretraining (25 Epochs)...")
    for epoch in range(1, 26):
        total_loss = 0.0
        correct_dir = 0
        correct_dwell = 0
        total_samples = 0

        for b_seq, b_rec, b_tracks, b_dir, b_dwell in dataloader:
            b_seq = b_seq.to(device)
            b_rec = b_rec.to(device)
            b_tracks = b_tracks.to(device)
            b_dir = b_dir.to(device)
            b_dwell = b_dwell.to(device)

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

            # Strict Boundary Margin Loss:
            # At Band 0 (b_rec[:, 0] < 0.002), moving left (0) is invalid; moving right (2) must dominate.
            # At Band 127 (b_rec[:, 0] > 0.998), moving right (2) is invalid; moving left (0) must dominate.
            left_bound = (b_rec[:, 0] < 0.002)
            right_bound = (b_rec[:, 0] > 0.998)
            loss_boundary = torch.tensor(0.0, device=device)
            if left_bound.any():
                loss_boundary = loss_boundary + torch.relu(logits_dir[left_bound, 0] - logits_dir[left_bound, 2] + 4.0).mean()
            if right_bound.any():
                loss_boundary = loss_boundary + torch.relu(logits_dir[right_bound, 2] - logits_dir[right_bound, 0] + 4.0).mean()

            loss = loss_dir + loss_dwell + 5.0 * loss_boundary

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_dir)
            correct_dir += (logits_dir.argmax(dim=-1) == b_dir).sum().item()
            correct_dwell += (logits_dwell.argmax(dim=-1) == b_dwell).sum().item()
            total_samples += len(b_dir)

        scheduler.step()
        acc_dir = correct_dir / total_samples * 100
        acc_dwell = correct_dwell / total_samples * 100
        avg_loss = total_loss / total_samples
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Dir Acc: {acc_dir:.1f}% | Dwell Acc: {acc_dwell:.1f}% | LR: {current_lr:.6f}", flush=True)

    # Move policy back to CPU for standard SB3 serialization
    policy.to("cpu")

    # Save to best model checkpoints
    os.makedirs("model/agents/checkpoints/best", exist_ok=True)
    os.makedirs("model/agents/checkpoints/targeted_lstm/best", exist_ok=True)
    os.makedirs("model/agents/checkpoints/imported", exist_ok=True)
    save_path = "model/agents/checkpoints/best/best_model.zip"
    targeted_path = "model/agents/checkpoints/targeted_lstm/best/best_model.zip"
    imported_path = "model/agents/checkpoints/imported/default_bundled_checkpoint_0446a5.zip"
    model.save(save_path)
    model.save(targeted_path)
    model.save(imported_path)

    print("\n" + "=" * 76)
    print(f"  PPO + LSTM Model Successfully Fine-Tuned & Saved!")
    print(f"  -> Canonical Path : {save_path}")
    print(f"  -> Targeted Path  : {targeted_path}")
    print(f"  -> Bundled Path   : {imported_path}")
    print("=" * 76)


if __name__ == "__main__":
    train_smart_ppo_lstm()
