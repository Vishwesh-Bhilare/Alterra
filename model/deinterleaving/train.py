"""
Train the SEDCAM-style PDW deinterleaving encoder on the Turing Synthetic
Radar Dataset (scan or stare split). Run from the alterra/ repo root:

  python -m model.deinterleaving.train --train-dir <path>/train_scan --val-dir <path>/test_scan
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from torch.utils.data import DataLoader

from model.deinterleaving.dataset import PDWWindowDataset
from model.deinterleaving.losses import supervised_contrastive_loss
from model.deinterleaving.model import PDWEncoder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(data_dir: str, cfg: dict, max_files: int, seed: int) -> DataLoader:
    ds = PDWWindowDataset(
        data_dir=data_dir,
        window_size=cfg["window_size"],
        windows_per_file=cfg["windows_per_file"],
        max_files=max_files,
        seed=seed,
    )
    return DataLoader(ds, batch_size=cfg["batch_size"], num_workers=0)


def evaluate(model, loader, device, dbscan_eps, dbscan_min_samples, max_batches=20) -> dict:
    model.eval()
    aris, vms, amis = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            z = model(x.to(device)).cpu().numpy()
            y = y.numpy()
            for b in range(z.shape[0]):
                pred = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(z[b])
                aris.append(adjusted_rand_score(y[b], pred))
                vms.append(v_measure_score(y[b], pred))
                amis.append(adjusted_mutual_info_score(y[b], pred))
    return {
        "ari": float(np.mean(aris)) if aris else None,
        "v_measure": float(np.mean(vms)) if vms else None,
        "ami": float(np.mean(amis)) if amis else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deinterleaving_train.yaml")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out-dir", default="model/deinterleaving/checkpoints")
    parser.add_argument("--max-train-files", type=int, default=20)
    parser.add_argument("--max-val-files", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader = build_dataloader(args.train_dir, cfg, args.max_train_files, seed=0)
    val_loader = build_dataloader(args.val_dir, cfg, args.max_val_files, seed=1)

    model = PDWEncoder(
        in_dim=5,
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            z = model(x)
            loss = supervised_contrastive_loss(z, y, temperature=cfg["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = evaluate(model, val_loader, args.device, cfg["dbscan_eps"], cfg["dbscan_min_samples"])
        print(
            f"epoch {epoch+1}/{cfg['epochs']}  loss={avg_loss:.4f}  "
            f"val_ari={metrics['ari']}  val_v_measure={metrics['v_measure']}  val_ami={metrics['ami']}"
        )
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"encoder_epoch{epoch+1}.pt"))

    print(f"\nDone. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
