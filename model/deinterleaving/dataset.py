"""
Windowed PDW dataset for the Turing Synthetic Radar Dataset (scan/stare
splits). Randomly samples fixed-length windows of consecutive (ToA-sorted)
PDWs per file, normalizes features, yields (window_features, window_labels)
for contrastive deinterleaving training.
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

TOA_DELTA_SCALE = 1.0 / 0.01     # normalize ~10ms inter-pulse deltas to O(1)
FREQ_SCALE = 1.0 / 18000.0       # receiver covers 0-18 GHz (freq in MHz)
PW_SCALE = 1.0 / 200.0           # pulse width in us
AOA_SCALE = 1.0 / 180.0          # angle of arrival in degrees
AMP_OFFSET_DB = 120.0
AMP_SCALE = 1.0 / 60.0


def _list_h5_files(data_dir: str, max_files: int | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if max_files is not None:
        files = files[:max_files]
    return files


def _normalize(data: np.ndarray) -> np.ndarray:
    # columns: [ToA_us, Freq_MHz, PW_us, AoA_deg, Amp_dB]
    toa_us = data[:, 0]
    toa_delta_s = np.diff(toa_us, prepend=toa_us[0]) / 1e6
    freq = data[:, 1]
    pw = data[:, 2]
    aoa = data[:, 3]
    amp = data[:, 4]

    return np.stack(
        [
            toa_delta_s * TOA_DELTA_SCALE,
            freq * FREQ_SCALE,
            pw * PW_SCALE,
            aoa * AOA_SCALE,
            (amp + AMP_OFFSET_DB) * AMP_SCALE,
        ],
        axis=1,
    ).astype(np.float32)


class PDWWindowDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        window_size: int = 256,
        windows_per_file: int = 50,
        max_files: int | None = None,
        seed: int = 0,
    ):
        self.files = _list_h5_files(data_dir, max_files)
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")
        self.window_size = window_size
        self.windows_per_file = windows_per_file
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for path in self.files:
            with h5py.File(path, "r") as f:
                data = f["data"][:]
                labels = f["labels"][:, 0]
            n = data.shape[0]
            if n <= self.window_size:
                continue
            starts = rng.integers(0, n - self.window_size, size=self.windows_per_file)
            for s in starts:
                window = data[s : s + self.window_size]
                lbl = labels[s : s + self.window_size]
                yield (
                    torch.from_numpy(_normalize(window)),
                    torch.from_numpy(lbl.astype(np.int64)),
                )
