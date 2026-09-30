"""
Modular Sequence Scorers for ALTERA Hybrid Cognitive Scheduler.
Supports 4 distinct temporal sequence backbones:
  1. RNN: nn.RNN with ReLU
  2. LSTM: nn.LSTM
  3. GRU: nn.GRU
  4. Transformer: Multi-Head Self-Attention with Positional Encoding

All backbones share the same:
  - Global spectrum track compressor (128x8 -> 128)
  - Receiver and doctrine mode encoder (14 -> 32)
  - Latent fusion trunk (224 -> 256)
  - Masked band utility logit head (256 -> 128)
  - Value critic head (256 -> 1)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from model.hybrid.doctrine import DoctrineMode


class HybridScorer(nn.Module):
    """
    Unified neural decision model with selectable temporal sequence backbone.
    Backbone options: 'rnn', 'lstm', 'gru', 'transformer'.
    """

    def __init__(
        self,
        backbone: str = "lstm",
        num_bands: int = 128,
        track_feature_dim: int = 8,
        receiver_feature_dim: int = 10,
        seq_feature_dim: int = 6,
        hidden_dim: int = 64,
        features_dim: int = 256,
        doctrine_dim: int = 4,
    ):
        super().__init__()
        self.backbone = backbone.lower()
        self.num_bands = num_bands
        self.track_feature_dim = track_feature_dim
        self.receiver_feature_dim = receiver_feature_dim
        self.seq_feature_dim = seq_feature_dim
        self.hidden_dim = hidden_dim
        self.features_dim = features_dim

        # 1. Temporal Sequence Backbone
        if self.backbone == "rnn":
            self.seq_encoder = nn.RNN(
                input_size=seq_feature_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                nonlinearity="relu",
                batch_first=True,
            )
        elif self.backbone == "lstm":
            self.seq_encoder = nn.LSTM(
                input_size=seq_feature_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.backbone == "gru":
            self.seq_encoder = nn.GRU(
                input_size=seq_feature_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.backbone == "transformer":
            self.seq_proj = nn.Linear(seq_feature_dim, hidden_dim)
            self.pos_emb = nn.Parameter(torch.zeros(1, 32, hidden_dim))
            nn.init.normal_(self.pos_emb, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
            )
            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise ValueError(f"Unknown backbone {backbone!r}. Expected: 'rnn', 'lstm', 'gru', 'transformer'")

        # 2. Spectrum Band Tracks Compressor
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
        fusion_in_dim = hidden_dim + 128 + 32
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

        # 6. Value Critic Head
        self.value_head = nn.Linear(features_dim, 1)

    def encode_doctrine_mode(self, mode: DoctrineMode | str, batch_size: int, device: torch.device) -> torch.Tensor:
        mode_val = mode.value if isinstance(mode, DoctrineMode) else mode
        order = ["EXPLORE", "INVESTIGATE", "TRACK", "RELOCATE"]
        idx = order.index(mode_val) if mode_val in order else 0
        one_hot = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        one_hot[:, idx] = 1.0
        return one_hot

    def _encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if self.backbone == "transformer":
            b, s, _ = seq.shape
            x = self.seq_proj(seq) + self.pos_emb[:, :s, :]
            out = self.seq_encoder(x)
            return out[:, -1, :]
        else:
            out, _ = self.seq_encoder(seq)
            return out[:, -1, :]

    def forward(
        self,
        hit_miss_seq: torch.Tensor,
        band_tracks: torch.Tensor,
        receiver: torch.Tensor,
        doctrine_one_hot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hit_miss_seq.shape[-1] > self.seq_feature_dim:
            hit_miss_seq = hit_miss_seq[:, :, : self.seq_feature_dim]
        if band_tracks.shape[-1] > self.track_feature_dim:
            band_tracks = band_tracks[:, :, : self.track_feature_dim]
        if receiver.shape[-1] > self.receiver_feature_dim:
            receiver = receiver[:, : self.receiver_feature_dim]

        # 1. Temporal sequence features
        seq_features = self._encode_sequence(hit_miss_seq)

        # 2. Spectrum tracks embedding
        tracks_flat = torch.flatten(band_tracks, start_dim=1)
        tracks_emb = self.tracks_compressor(tracks_flat)

        # 3. Receiver & doctrine embedding
        rec_combined = torch.cat([receiver, doctrine_one_hot], dim=1)
        rec_emb = self.receiver_encoder(rec_combined)

        # 4. Latent fusion
        fused = torch.cat([seq_features, tracks_emb, rec_emb], dim=1)
        latent = self.fusion_trunk(fused)

        # 5. Output heads
        logits = self.band_utility_head(latent)
        value = self.value_head(latent)

        return logits, value

    @staticmethod
    def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
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
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            seq_t = torch.tensor(obs["hit_miss_seq"], dtype=torch.float32, device=device).unsqueeze(0)
            tracks_t = torch.tensor(obs["band_tracks"], dtype=torch.float32, device=device).unsqueeze(0)
            rec_t = torch.tensor(obs["receiver"], dtype=torch.float32, device=device).unsqueeze(0)
            mode_t = self.encode_doctrine_mode(mode, batch_size=1, device=device)

            logits, value = self.forward(seq_t, tracks_t, rec_t, mode_t)

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
        logits, _ = self.forward(hit_miss_seq, band_tracks, receiver, doctrine_one_hot)
        masked_logits = self.apply_mask(logits, masks)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(masked_logits, teacher_actions)

        pred_actions = torch.argmax(masked_logits, dim=-1)
        accuracy = (pred_actions == teacher_actions).float().mean()

        return loss, accuracy


# Subclass aliases for explicit typing
class HybridRNNScorer(HybridScorer):
    def __init__(self, **kwargs):
        super().__init__(backbone="rnn", **kwargs)


class HybridLSTMScorer(HybridScorer):
    def __init__(self, **kwargs):
        super().__init__(backbone="lstm", **kwargs)


class HybridGRUScorer(HybridScorer):
    def __init__(self, **kwargs):
        super().__init__(backbone="gru", **kwargs)


class HybridTransformerScorer(HybridScorer):
    def __init__(self, **kwargs):
        super().__init__(backbone="transformer", **kwargs)
