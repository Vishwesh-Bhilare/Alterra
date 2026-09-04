from __future__ import annotations

import numpy as np

from .base_emitter import BaseEmitter
from .schedule_utils import power_schedule_from_mask, two_state_markov_mask


class AgileEmitter(BaseEmitter):
    def __init__(
        self,
        emitter_id: str,
        threat_level: int,
        rng: np.random.Generator,
        pri_s: float,
        pw_s: float,
        pri_jitter_std_s: float,
        doa_deg: float,
        num_bands: int,
        hop_bandset_size: int,
        hop_dwell_slots: int,
        burst_duty_cycle: float,
        burst_mean_slots: float,
        power_mean_dbm: float,
        power_jitter_std_db: float,
    ):
        super().__init__(emitter_id, threat_level, rng, pri_s, pw_s, pri_jitter_std_s, doa_deg)
        self.num_bands = num_bands
        self.hop_bandset_size = min(hop_bandset_size, num_bands)
        self.hop_dwell_slots = hop_dwell_slots
        self.burst_duty_cycle = burst_duty_cycle
        self.burst_mean_slots = burst_mean_slots
        self.power_mean_dbm = power_mean_dbm
        self.power_jitter_std_db = power_jitter_std_db

    def _build_schedule(self, episode_length: int) -> None:
        hop_bands = self._rng.choice(self.num_bands, size=self.hop_bandset_size, replace=False)

        band_schedule = np.empty(episode_length, dtype=int)
        num_hops = int(np.ceil(episode_length / self.hop_dwell_slots))
        hop_choices = self._rng.choice(hop_bands, size=num_hops, replace=True)

        for hop_idx in range(num_hops):
            start = hop_idx * self.hop_dwell_slots
            end = min(start + self.hop_dwell_slots, episode_length)
            band_schedule[start:end] = hop_choices[hop_idx]

        mask = two_state_markov_mask(
            self._rng, episode_length, self.burst_duty_cycle, self.burst_mean_slots
        )

        self._band_schedule = band_schedule
        self._active_schedule = mask
        self._power_schedule = power_schedule_from_mask(
            self._rng, mask, self.power_mean_dbm, self.power_jitter_std_db
        )
