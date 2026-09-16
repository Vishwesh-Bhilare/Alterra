"""
1D-CNN Spectral Feature Extractor for Alterra EW Smart Scan Scheduler.

Processes the (128, 7) band_tracks matrix using 1D convolutional layers
across the frequency dimension to exploit spectral spatial locality,
frequency-hopping clusters, and radar chirp/sweep gradients.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Spectral1DFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observation spaces:
    - band_tracks: Box(low=0.0, high=1.0, shape=(128, 7))
    - receiver: Box(low=0.0, high=1.0, shape=(8,))

    Extracts:
    1. 1D-CNN over the 128 frequency channels (treating the 7 track features as channels).
    2. Dense MLP over the 8 receiver telemetry features.
    3. Concatenates both representations into a single feature vector for actor/critic heads.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        band_tracks_space = observation_space.spaces["band_tracks"]
        receiver_space = observation_space.spaces["receiver"]

        num_bands, num_track_features = band_tracks_space.shape
        num_receiver_features = receiver_space.shape[0]

        # 1D-CNN over frequency channels
        # Input shape: (Batch, num_track_features, num_bands) -> (B, 7, 128)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_track_features, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 -> 32

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # 32 -> 8

            nn.Flatten(),  # 64 * 8 = 512
            nn.Linear(512, 192),
            nn.ReLU(),
        )

        # MLP for receiver telemetry state (8 dims)
        self.receiver_net = nn.Sequential(
            nn.Linear(num_receiver_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # Final projection: 192 (CNN) + 64 (Receiver) = 256
        self.proj = nn.Sequential(
            nn.Linear(192 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # band_tracks shape: (B, 128, 7) -> permute to (B, 7, 128) for Conv1D
        tracks = observations["band_tracks"].transpose(1, 2)
        cnn_out = self.cnn(tracks)

        # receiver telemetry: (B, 8)
        receiver = observations["receiver"]
        rec_out = self.receiver_net(receiver)

        # Combined fused representation
        combined = torch.cat([cnn_out, rec_out], dim=1)
        return self.proj(combined)
