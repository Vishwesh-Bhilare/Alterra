"""
SEDCAM-style PDW encoder: maps each pulse in a window to an embedding such
that pulses from the same emitter cluster together (via a supervised
contrastive loss at training time) and can be separated with DBSCAN at
inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PDWEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        h = self.input_proj(x)
        h = self.transformer(h)
        z = self.output_proj(h)
        return nn.functional.normalize(z, dim=-1)
