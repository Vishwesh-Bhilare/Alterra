"""
Behavior Cloning & PPO Fine-Tuning for Alterra EW Smart Scan Scheduler.
Trains the PPO + LSTM network to execute the single-line continuous sweep with dwell escalation:
1. Tracks hits, misses, scanned bands, unscanned bands, continuous hit length.
2. Sweeps as a continuous single line across the spectrum (direction -1 or +1).
3. On hit: locks onto the exact band (dir = stay) and escalates dwell time (3 -> 5 -> 8 -> 12 slots).
4. On signal end: immediately resumes the single line sweep across remaining bands.
5. Replicates the exact behavior on any new band.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.lstm_policy import PPOLSTMExtractor


class SingleLineRelativeAgent:
    """Continuous single-line sweep with proportional dwell escalation on hits."""
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands
        self.reset()

    def reset(self):
        self.current_band = 0
        self.consecutive_hits = 0
        self.last_hit = False
        self.sweep_dir = 1  # 1 = up, -1 = down

    def get_action(self, current_band: int) -> tuple[int, int]:
        self.current_band = current_band
        if self.last_hit:
            self.consecutive_hits += 1
            # Stay on active signal (dir = 1 -> delta = 0)
            dir_action = 1
            # Proportional dwell escalation:
            # 1st hit -> dwell 1 (5 slots, 50ms)
            # 2nd hit -> dwell 2 (8 slots, 80ms)
            # 3rd+ hit -> dwell 3 (12 slots, 120ms max lock)
            dwell_idx = min(self.consecutive_hits, 3)
            return dir_action, dwell_idx

        # Searching or signal ended: reset consecutive hits, fast 30ms sweep
        self.consecutive_hits = 0
        dwell_idx = 0  # 3 slots (30ms fast sweep)

        # Single continuous line sweep across spectrum
        if self.current_band <= 0:
            self.sweep_dir = 1  # Reverse upward
        elif self.current_band >= self.num_bands - 1:
            self.sweep_dir = -1  # Reverse downward

        # 0 = down (delta -1), 2 = up (delta +1)
        dir_action = 0 if self.sweep_dir == -1 else 2
        return dir_action, dwell_idx

    def update(self, hit: bool):
        self.last_hit = hit


def generate_expert_dataset(n_episodes: int = 120):
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    expert = SingleLineRelativeAgent(num_bands=config.spectrum.num_bands)

    obs_seq_list = []
    obs_rec_list = []
    obs_tracks_list = []
    act_dir_list = []
    act_dwell_list = []

    print(f"Generating single-line expert transitions across {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 11 + 42)
        expert.reset()
        expert.current_band = env._current_band
        expert.sweep_dir = -1 if env._current_band > 64 else 1

        done = False
        while not done:
            dir_act, dwell_act = expert.get_action(env._current_band)

            obs_seq_list.append(obs["hit_miss_seq"])
            obs_rec_list.append(obs["receiver"])
            obs_tracks_list.append(obs["band_tracks"])
            act_dir_list.append(dir_act)
            act_dwell_list.append(dwell_act)

            action = np.array([dir_act, dwell_act], dtype=np.int64)
            obs, reward, term, trunc, info = env.step(action)
            expert.update(info["any_hit"])
            done = term or trunc

    print(f"Generated {len(act_dir_list)} expert transition samples.")
    return (
        torch.tensor(np.array(obs_seq_list), dtype=torch.float32),
        torch.tensor(np.array(obs_rec_list), dtype=torch.float32),
        torch.tensor(np.array(obs_tracks_list), dtype=torch.float32),
        torch.tensor(np.array(act_dir_list), dtype=torch.long),
        torch.tensor(np.array(act_dwell_list), dtype=torch.long),
    )


def train_policy():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)

    policy_kwargs = dict(
        features_extractor_class=PPOLSTMExtractor,
        features_extractor_kwargs=dict(lstm_hidden_dim=64, features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
    )

    seq, rec, tracks, act_dir, act_dwell = generate_expert_dataset(n_episodes=120)
    dataset = TensorDataset(seq, rec, tracks, act_dir, act_dwell)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    policy = model.policy
    optimizer = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("Training PPO + LSTM policy via Behavior Cloning (Single-Line Sweep)...")
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
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Dir Acc: {acc_dir:.1f}% | Dwell Acc: {acc_dwell:.1f}%")

    os.makedirs("model/agents/checkpoints/best", exist_ok=True)
    os.makedirs("model/agents/checkpoints/targeted_lstm/best", exist_ok=True)
    save_path = "model/agents/checkpoints/best/best_model.zip"
    targeted_path = "model/agents/checkpoints/targeted_lstm/best/best_model.zip"
    model.save(save_path)
    model.save(targeted_path)
    print(f"Model successfully saved to:\n  - {save_path}\n  - {targeted_path}")


if __name__ == "__main__":
    train_policy()
