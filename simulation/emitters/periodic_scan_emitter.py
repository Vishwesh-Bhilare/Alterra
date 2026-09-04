from __future__ import annotations

import numpy as np

from .base_emitter import BaseEmitter
from .schedule_utils import power_schedule_from_mask, two_state_markov_mask


class PeriodicScanEmitter(BaseEmitter):
    """
    A scanning emitter (e.g. a mechanically/electronically scanning radar):
    sweeps a contiguous window of `sweep_width` bands, dwelling
    `dwell_slots` on each before advancing, cycling indefinitely. This is
    the archetype the problem statement calls out for optimal interception
    of a periodic scan receiver.
    """

    def __init__(
        self,
        emitter_id: str,
        threat_level: int,
        rng: np.random.Generator,
        num_bands: int,
        sweep_width: int,
        dwell_slots: int,
        duty_cycle: float,
        burst_mean_slots: float,
        power_mean_dbm: float,
        power_jitter_std_db: float,
    ):
        super().__init__(emitter_id, threat_level, rng)
        self.num_bands = num_bands
        self.sweep_width = min(sweep_width, num_bands)
        self.dwell_slots = dwell_slots
        self.duty_cycle = duty_cycle
        self.burst_mean_slots = burst_mean_slots
        self.power_mean_dbm = power_mean_dbm
        self.power_jitter_std_db = power_jitter_std_db

    def _build_schedule(self, episode_length: int) -> None:
        max_start = self.num_bands - self.sweep_width
        sweep_start = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0
        sweep_bands = np.arange(sweep_start, sweep_start + self.sweep_width)

        slot_positions = np.arange(episode_length) // self.dwell_slots
        band_schedule = sweep_bands[slot_positions % self.sweep_width]

        mask = two_state_markov_mask(
            self._rng, episode_length, self.duty_cycle, self.burst_mean_slots
        )

        self._band_schedule = band_schedule.astype(int)
        self._active_schedule = mask
        self._power_schedule = power_schedule_from_mask(
            self._rng, mask, self.power_mean_dbm, self.power_jitter_std_db
        )
