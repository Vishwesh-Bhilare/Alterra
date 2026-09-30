"""
PyTorch Dataset Loader for ALTERA Hybrid Scheduler Imitation Learning.
Loads pre-processed .npz trajectory files containing sequences, spectrum tracks,
receiver states, doctrine modes, candidate masks, and teacher target actions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ImitationTrajectoryDataset(Dataset):
    """
    PyTorch Dataset yielding decision step tuples for masked imitation learning:
      (hit_miss_seq, band_tracks, receiver, doctrine_one_hot, mask, target_band, dwell_idx)
    """

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found at: {path}")

        data = np.load(path)
        self.hit_miss_seq = torch.tensor(data["hit_miss_seq"], dtype=torch.float32)
        self.band_tracks = torch.tensor(data["band_tracks"], dtype=torch.float32)
        self.receiver = torch.tensor(data["receiver"], dtype=torch.float32)
        self.mode = torch.tensor(data["mode"], dtype=torch.long)
        self.mask = torch.tensor(data["mask"], dtype=torch.bool)
        self.target_band = torch.tensor(data["target_band"], dtype=torch.long)
        self.dwell_idx = torch.tensor(data["dwell_idx"], dtype=torch.long)

        # One-hot encode mode (4 dims: EXPLORE, INVESTIGATE, TRACK, RELOCATE)
        self.doctrine_one_hot = torch.zeros((len(self.mode), 4), dtype=torch.float32)
        self.doctrine_one_hot.scatter_(1, self.mode.unsqueeze(1), 1.0)

    def __len__(self) -> int:
        return len(self.target_band)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hit_miss_seq": self.hit_miss_seq[idx],
            "band_tracks": self.band_tracks[idx],
            "receiver": self.receiver[idx],
            "doctrine_one_hot": self.doctrine_one_hot[idx],
            "mask": self.mask[idx],
            "target_band": self.target_band[idx],
            "dwell_idx": self.dwell_idx[idx],
        }
