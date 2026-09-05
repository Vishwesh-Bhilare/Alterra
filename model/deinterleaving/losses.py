"""Supervised contrastive loss (Khosla et al.), applied per-window."""
from __future__ import annotations

import torch


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    # embeddings: (B, T, D), labels: (B, T)
    losses = []
    for b in range(embeddings.shape[0]):
        z = embeddings[b]
        y = labels[b]
        sim = z @ z.T / temperature
        mask_pos = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        mask_self = torch.eye(len(y), device=z.device)
        mask_pos = mask_pos - mask_self

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask_pos.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            continue
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)[valid] / pos_count[valid]
        losses.append(-mean_log_prob_pos.mean())

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    return torch.stack(losses).mean()
