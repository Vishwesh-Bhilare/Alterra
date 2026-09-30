"""
Supervised Imitation Learning Trainer for ALTERA Hybrid LSTM Scheduler.
Trains the HybridLSTMScorer to imitate the teacher doctrine using Masked Cross-Entropy Loss:
    L_imitation = - (1/B) sum_{b=1..B} log pi_theta(a_teacher,b | X_b, M_b)

Includes:
  - Masked cross-entropy with invalid-band suppression (-1e9)
  - Validation-based early stopping and model checkpointing
  - Cosine annealing learning rate scheduler with AdamW
  - Gradient norm clipping
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.hybrid.dataset import ImitationTrajectoryDataset
from model.hybrid.scorers import HybridScorer


def train_epoch(
    model: HybridLSTMScorer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        seq = batch["hit_miss_seq"].to(device)
        tracks = batch["band_tracks"].to(device)
        rec = batch["receiver"].to(device)
        mode = batch["doctrine_one_hot"].to(device)
        masks = batch["mask"].to(device)
        targets = batch["target_band"].to(device)

        optimizer.zero_grad()

        loss, acc = model.compute_masked_imitation_loss(
            hit_miss_seq=seq,
            band_tracks=tracks,
            receiver=rec,
            doctrine_one_hot=mode,
            masks=masks,
            teacher_actions=targets,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        b_size = len(targets)
        total_loss += float(loss.item()) * b_size
        total_correct += int((acc * b_size).item())
        total_samples += b_size

    mean_loss = total_loss / max(total_samples, 1)
    mean_acc = total_correct / max(total_samples, 1)
    return mean_loss, mean_acc


def evaluate_epoch(
    model: HybridLSTMScorer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["hit_miss_seq"].to(device)
            tracks = batch["band_tracks"].to(device)
            rec = batch["receiver"].to(device)
            mode = batch["doctrine_one_hot"].to(device)
            masks = batch["mask"].to(device)
            targets = batch["target_band"].to(device)

            loss, acc = model.compute_masked_imitation_loss(
                hit_miss_seq=seq,
                band_tracks=tracks,
                receiver=rec,
                doctrine_one_hot=mode,
                masks=masks,
                teacher_actions=targets,
            )

            b_size = len(targets)
            total_loss += float(loss.item()) * b_size
            total_correct += int((acc * b_size).item())
            total_samples += b_size

    mean_loss = total_loss / max(total_samples, 1)
    mean_acc = total_correct / max(total_samples, 1)
    return mean_loss, mean_acc


def main():
    parser = argparse.ArgumentParser(description="Train ALTERA Hybrid Scheduler via Imitation Learning")
    parser.add_argument("--backbone", default="lstm", choices=["rnn", "lstm", "gru", "transformer"], help="Sequence backbone")
    parser.add_argument("--train-data", default="data/imitation/train.npz", help="Path to train .npz")
    parser.add_argument("--val-data", default="data/imitation/val.npz", help="Path to val .npz")
    parser.add_argument("--output-dir", default="model/hybrid/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = out_dir / f"best_hybrid_{args.backbone}.pt"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print(f" ALTERRA HYBRID {args.backbone.upper()} IMITATION LEARNING TRAINING")
    print(f" Target Device: {device} | Max Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Load Datasets
    print(f"[Data] Loading training dataset from {args.train_data}...")
    train_dataset = ImitationTrajectoryDataset(args.train_data)
    val_dataset = ImitationTrajectoryDataset(args.val_data)
    print(f"[Data] Loaded {len(train_dataset)} training steps and {len(val_dataset)} validation steps.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model
    model = HybridScorer(backbone=args.backbone, num_bands=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized HybridScorer [{args.backbone.upper()}] (Trainable Parameters: {total_params:,})")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting Training Loop...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^12} | {'Val Acc':^11} | {'LR':^10} | {'Status'}")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            status = "★ Best Saved"
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc*100:^10.1f}% | {val_loss:^12.4f} | {val_acc*100:^10.1f}% | {current_lr:^10.2e} | Early Stopped")
                print(f"\n[Early Stopping] No improvement in validation loss for {args.patience} epochs.")
                break

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc*100:^10.1f}% | {val_loss:^12.4f} | {val_acc*100:^10.1f}% | {current_lr:^10.2e} | {status}")
        scheduler.step()

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Training Complete in {total_time:.1f}s!")
    print(f"Best Model Checkpoint: {best_checkpoint_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Teacher Agreement Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
