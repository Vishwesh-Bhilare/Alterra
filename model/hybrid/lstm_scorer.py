"""
Hybrid LSTM Band Utility Scorer for ALTERA Cognitive Scheduler.
Fuses:
  1. Temporal sequence of dwell hit/miss history via nn.LSTM
  2. Global spectrum band tracks via feature compressor
  3. Receiver telemetry and cognitive doctrine mode

Outputs 128 band utility logits with invalid-action masking before Softmax.
Includes optional Value Head for state evaluation and RL/PPO fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from model.hybrid.doctrine import DoctrineMode


class HybridLSTMScorer(nn.Module):
    """
    Hybrid neural decision model:
    Evaluates temporal observation sequences and spectrum tracks to produce
    a utility score/logit for each candidate frequency band (0 to 127).
    """

    def __init__(
        self,
        num_bands: int = 128,
        track_feature_dim: int = 8,
        receiver_feature_dim: int = 10,
        seq_feature_dim: int = 6,
        lstm_hidden_dim: int = 64,
        features_dim: int = 256,
        doctrine_dim: int = 4,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.track_feature_dim = track_feature_dim
        self.receiver_feature_dim = receiver_feature_dim
        self.seq_feature_dim = seq_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.features_dim = features_dim

        # 1. Temporal Sequence Encoder (LSTM)
        # Encodes the rolling (seq_len=16, 6) history of dwell outcomes, power, and hits
        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 2. Spectrum Band Tracks Compressor
        # Compresses 128 bands x 8 features = 1024 dims into a compact 128-dim spectrum embedding
        tracks_flat_dim = num_bands * track_feature_dim
        self.tracks_compressor = nn.Sequential(
            nn.Linear(tracks_flat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 3. Receiver & Doctrine State Encoder
        rec_in_dim = receiver_feature_dim + doctrine_dim
        self.receiver_encoder = nn.Sequential(
            nn.Linear(rec_in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # 4. Feature Fusion Trunk
        fusion_in_dim = lstm_hidden_dim + 128 + 32
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fusion_in_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        # 5. Band Utility Head (128 logits)
        self.band_utility_head = nn.Linear(features_dim, num_bands)

        # 6. Value Head (Critic for state valuation / PPO)
        self.value_head = nn.Linear(features_dim, 1)

    def encode_doctrine_mode(self, mode: DoctrineMode | str, batch_size: int, device: torch.device) -> torch.Tensor:
        """One-hot encodes doctrine mode: [EXPLORE, INVESTIGATE, TRACK, RELOCATE]."""
        mode_val = mode.value if isinstance(mode, DoctrineMode) else mode
        order = ["EXPLORE", "INVESTIGATE", "TRACK", "RELOCATE"]
        idx = order.index(mode_val) if mode_val in order else 0
        one_hot = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        one_hot[:, idx] = 1.0
        return one_hot

    def forward(
        self,
        hit_miss_seq: torch.Tensor,
        band_tracks: torch.Tensor,
        receiver: torch.Tensor,
        doctrine_one_hot: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass producing raw logits and state value.

        Args:
            hit_miss_seq: (batch, seq_len, 6)
            band_tracks: (batch, 128, 8)
            receiver: (batch, 10)
            doctrine_one_hot: (batch, 4)
            lstm_state: Optional (h, c) tuple

        Returns:
            (raw_logits, value, next_lstm_state)
        """
        # Ensure dimensions match expected shape
        if hit_miss_seq.shape[-1] > self.seq_feature_dim:
            hit_miss_seq = hit_miss_seq[:, :, : self.seq_feature_dim]
        if band_tracks.shape[-1] > self.track_feature_dim:
            band_tracks = band_tracks[:, :, : self.track_feature_dim]
        if receiver.shape[-1] > self.receiver_feature_dim:
            receiver = receiver[:, : self.receiver_feature_dim]

        # 1. Sequence encoding
        lstm_out, next_lstm_state = self.lstm(hit_miss_seq, lstm_state)
        # Take hidden state of last step: (batch, lstm_hidden_dim)
        seq_features = lstm_out[:, -1, :]

        # 2. Spectrum tracks compression
        tracks_flat = torch.flatten(band_tracks, start_dim=1)
        tracks_emb = self.tracks_compressor(tracks_flat)

        # 3. Receiver & Doctrine encoding
        rec_combined = torch.cat([receiver, doctrine_one_hot], dim=1)
        rec_emb = self.receiver_encoder(rec_combined)

        # 4. Fusion
        fused = torch.cat([seq_features, tracks_emb, rec_emb], dim=1)
        latent = self.fusion_trunk(fused)

        # 5. Output Heads
        logits = self.band_utility_head(latent)
        value = self.value_head(latent)

        return logits, value, next_lstm_state

    @staticmethod
    def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies candidate mask to logits by assigning large negative value (-1e9) to invalid bands.
        """
        masked_logits = logits.clone()
        # mask is True for valid bands, False for invalid
        masked_logits[~mask] = -1e9
        return masked_logits

    def predict_action(
        self,
        obs: dict[str, np.ndarray],
        mode: DoctrineMode | str,
        mask: np.ndarray,
        deterministic: bool = True,
        device: torch.device | None = None,
    ) -> tuple[int, float, float]:
        """
        Inference interface for a single environment step.

        Returns:
            (selected_band, action_log_prob, state_value)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            seq_t = torch.tensor(obs["hit_miss_seq"], dtype=torch.float32, device=device).unsqueeze(0)
            tracks_t = torch.tensor(obs["band_tracks"], dtype=torch.float32, device=device).unsqueeze(0)
            rec_t = torch.tensor(obs["receiver"], dtype=torch.float32, device=device).unsqueeze(0)
            mode_t = self.encode_doctrine_mode(mode, batch_size=1, device=device)

            logits, value, _ = self.forward(seq_t, tracks_t, rec_t, mode_t)

            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            masked_logits = self.apply_mask(logits, mask_t)

            dist = Categorical(logits=masked_logits)

            if deterministic:
                action = int(torch.argmax(masked_logits, dim=-1).item())
            else:
                action = int(dist.sample().item())

            log_prob = float(dist.log_prob(torch.tensor(action, device=device)).item())
            val = float(value.squeeze().item())

            return action, log_prob, val

    def compute_masked_imitation_loss(
        self,
        hit_miss_seq: torch.Tensor,
        band_tracks: torch.Tensor,
        receiver: torch.Tensor,
        doctrine_one_hot: torch.Tensor,
        masks: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes masked cross-entropy loss against teacher target actions:
          L_imitation = - (1/B) sum_{b=1..B} log pi_theta(a_teacher,b | X_b, M_b)
        """
        logits, _, _ = self.forward(hit_miss_seq, band_tracks, receiver, doctrine_one_hot)
        masked_logits = self.apply_mask(logits, masks)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(masked_logits, teacher_actions)

        # Measure teacher agreement accuracy
        pred_actions = torch.argmax(masked_logits, dim=-1)
        accuracy = (pred_actions == teacher_actions).float().mean()

        return loss, accuracy
