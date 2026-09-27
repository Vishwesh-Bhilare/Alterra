"""
GRU-based feature extractor for PPO Smart Scan Scheduler.
Encodes the rolling sequence of hit/miss dwell events across time
using a Gated Recurrent Unit (GRU) network with reset/update gates
and LayerNorm feature balancing, combined with receiver state and
compressed spectrum band tracks.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PPOGRUExtractor(BaseFeaturesExtractor):
    """
    Gated Recurrent Unit (GRU) Feature Extractor.
    Processes the (seq_len, 5) hit/miss history using an nn.GRU layer,
    extracting the temporal hidden state h_n to guide PPO on where to search next.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        gru_hidden_dim: int = 64,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        seq_space = observation_space.spaces["hit_miss_seq"]
        rec_space = observation_space.spaces["receiver"]
        tracks_space = observation_space.spaces["band_tracks"]

        seq_in_dim = seq_space.shape[1]
        rec_dim = rec_space.shape[0]
        tracks_flat_dim = tracks_space.shape[0] * tracks_space.shape[1]

        # 1. Gated Recurrent Unit (GRU) encoding the hit/miss sequence over time
        self.gru = nn.GRU(
            input_size=seq_in_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden_dim)

        # 2. Band tracks compressor
        self.tracks_compressor = nn.Sequential(
            nn.Linear(tracks_flat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # 3. Fusion layer combining GRU sequence state, receiver state, and compressed spectrum tracks
        fusion_in_dim = gru_hidden_dim + rec_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # observations['hit_miss_seq']: (batch_size, seq_len, 5)
        seq = observations["hit_miss_seq"]
        _, h_n = self.gru(seq)       # h_n: (num_layers, batch_size, gru_hidden_dim)
        gru_features = self.gru_norm(h_n[-1])  # (batch_size, gru_hidden_dim)

        # Receiver features: (batch_size, 8)
        rec = observations["receiver"]

        # Global spectrum band tracks: (batch_size, 128, 7) -> (batch_size, 128*7) -> (batch_size, 128)
        tracks_flat = torch.flatten(observations["band_tracks"], start_dim=1)
        tracks_emb = self.tracks_compressor(tracks_flat)

        # Fuse GRU sequence state + receiver state + spectrum embedding
        combined = torch.cat([gru_features, rec, tracks_emb], dim=1)
        return self.fusion(combined)
