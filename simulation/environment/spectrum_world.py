"""
Layer A: ground-truth engine. Compiles all emitters' precomputed schedules
into per-slot band occupancy, handling multiple emitters overlapping in the
same band/slot. Purely deterministic — involves no randomness of its own,
only reads emitters' already-built schedules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.emitters.base_emitter import BaseEmitter, EmitterState


@dataclass(frozen=True)
class BandOccupancy:
    band: int
    emitter_states: list[EmitterState]  # all emitters truly active on this band at this slot

    @property
    def is_occupied(self) -> bool:
        return len(self.emitter_states) > 0

    @property
    def combined_power_dbm(self) -> float:
        """Linear-domain sum of all active emitters' power on this band,
        converted back to dBm — the 'true' aggregate signal a receiver
        dwelling here would see."""
        if not self.emitter_states:
            return float("-inf")
        linear_mw = sum(10 ** (s.power_dbm / 10.0) for s in self.emitter_states)
        return 10.0 * np.log10(linear_mw)

    @property
    def strongest_emitter(self) -> EmitterState | None:
        if not self.emitter_states:
            return None
        return max(self.emitter_states, key=lambda s: s.power_dbm)


class SpectrumWorld:
    """Ground-truth compiler over a fixed set of (already-`reset()`) emitters
    for one episode."""

    def __init__(self, emitters: list[BaseEmitter], num_bands: int, episode_length: int):
        self.emitters = emitters
        self.num_bands = num_bands
        self.episode_length = episode_length

    def band_at(self, t: int, band: int) -> BandOccupancy:
        """Occupancy of a single band at a single slot — the receiver's
        primary query, called on every dwell slot."""
        states = []
        for emitter in self.emitters:
            state = emitter.state_at(t)
            if state.active and state.band == band:
                states.append(state)
        return BandOccupancy(band=band, emitter_states=states)

    def occupancy_at(self, t: int) -> dict[int, BandOccupancy]:
        """All occupied bands at slot t, keyed by band index. Bands with no
        active emitter are omitted — use occupancy_matrix_at for a
        zero-filled view."""
        by_band: dict[int, list[EmitterState]] = {}
        for emitter in self.emitters:
            state = emitter.state_at(t)
            if state.active:
                by_band.setdefault(state.band, []).append(state)
        return {b: BandOccupancy(band=b, emitter_states=s) for b, s in by_band.items()}

    def occupancy_matrix_at(self, t: int) -> np.ndarray:
        """Boolean array of length num_bands — True where >=1 emitter active."""
        matrix = np.zeros(self.num_bands, dtype=bool)
        for band in self.occupancy_at(t):
            matrix[band] = True
        return matrix

    def full_truth_matrix(self) -> np.ndarray:
        """(num_bands, episode_length) boolean ground-truth matrix for the
        whole episode. Expensive — for offline analysis/viz, not the RL
        per-step loop (use band_at there)."""
        matrix = np.zeros((self.num_bands, self.episode_length), dtype=bool)
        for t in range(self.episode_length):
            matrix[:, t] = self.occupancy_matrix_at(t)
        return matrix
