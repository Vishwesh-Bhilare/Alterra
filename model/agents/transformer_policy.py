"""
Transformer-based feature extractor for PPO Smart Scan Scheduler.
Encodes the rolling sequence of hit/miss dwell events across time
using Multi-Head Self-Attention (TransformerEncoder) with learnable
positional encodings and Pre-LayerNorm, combined with receiver
state and compressed spectrum band tracks.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PPOTransformerExtractor(BaseFeaturesExtractor):
    """
    Multi-Head Self-Attention Transformer Feature Extractor.
    Processes the (seq_len, 5) hit/miss history using an nn.TransformerEncoder,
    capturing long-range temporal attention across receiver dwell outcomes to guide
    PPO on where to search next.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        transformer_d_model: int = 64,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        seq_space = observation_space.spaces["hit_miss_seq"]
        rec_space = observation_space.spaces["receiver"]
        tracks_space = observation_space.spaces["band_tracks"]

        seq_len = seq_space.shape[0]     # 16
        seq_in_dim = seq_space.shape[1]  # 5
        rec_dim = rec_space.shape[0]     # 8
        tracks_flat_dim = tracks_space.shape[0] * tracks_space.shape[1]

        # 1. Linear projection from input feature dim (5) to transformer_d_model
        self.input_proj = nn.Linear(seq_in_dim, transformer_d_model)

        # 2. Learnable positional embeddings for chronological dwell sequence
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, transformer_d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Multi-Head Self-Attention Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
            norm=nn.LayerNorm(transformer_d_model),
            enable_nested_tensor=False,
        )

        # 4. Pooling / feature normalization
        self.seq_norm = nn.LayerNorm(transformer_d_model)

        # 5. Global spectrum band tracks compressor
        self.tracks_compressor = nn.Sequential(
            nn.Linear(tracks_flat_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # 6. Fusion layer combining Transformer sequence state, receiver state, and compressed spectrum tracks
        fusion_in_dim = transformer_d_model + rec_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # observations['hit_miss_seq']: (batch_size, seq_len, 5)
        seq = observations["hit_miss_seq"]

        # Linear projection and add positional embeddings
        x = self.input_proj(seq) + self.pos_embedding
        encoded = self.transformer_encoder(x)

        # Extract latest temporal attention token
        transformer_features = self.seq_norm(encoded[:, -1, :])

        # Receiver features: (batch_size, 8)
        rec = observations["receiver"]

        # Global spectrum band tracks: (batch_size, 128, 7) -> (batch_size, 128*7) -> (batch_size, 128)
        tracks_flat = torch.flatten(observations["band_tracks"], start_dim=1)
        tracks_emb = self.tracks_compressor(tracks_flat)

        # Fuse Transformer sequence state + receiver state + spectrum embedding
        combined = torch.cat([transformer_features, rec, tracks_emb], dim=1)
        return self.fusion(combined)
