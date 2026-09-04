"""
Waterfall plot: full ground-truth band x time occupancy, with the
receiver's actual dwell path overlaid, so an episode can be eyeballed
instead of only trusted from metrics.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from simulation.environment.spectrum_world import SpectrumWorld


def plot_episode(
    spectrum_world: SpectrumWorld,
    dwell_bands: list[int],
    dwell_starts: list[int],
    dwell_ends: list[int],
    save_path: str,
) -> None:
    truth = spectrum_world.full_truth_matrix()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(truth, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")

    for band, start, end in zip(dwell_bands, dwell_starts, dwell_ends):
        ax.plot([start, end], [band, band], color="red", linewidth=2, alpha=0.8)

    ax.set_xlabel("time slot")
    ax.set_ylabel("band")
    ax.set_title("Ground truth occupancy (grey) vs receiver dwell path (red)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
