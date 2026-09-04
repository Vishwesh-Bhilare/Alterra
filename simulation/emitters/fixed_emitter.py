from __future__ import annotations

import numpy as np

from .base_emitter import BaseEmitter
from .schedule_utils import power_schedule_from_mask, two_state_markov_mask


class FixedEmitter(BaseEmitter):
    def __init__(
        self,
        emitter_id: str,
        threat_level: int,
        rng: np.random.Generator,
        pri_s: float,
        pw_s: float,
        pri_jitter_std_s: float,
        doa_deg: float,
        band: int,
        duty_cycle: float,
        mean_burst_slots: float,
        power_mean_dbm: float,
        power_jitter_std_db: float,
    ):
        super().__init__(emitter_id, threat_level, rng, pri_s, pw_s, pri_jitter_std_s, doa_deg)
        self.band = band
        self.duty_cycle = duty_cycle
        self.mean_burst_slots = mean_burst_slots
        self.power_mean_dbm = power_mean_dbm
        self.power_jitter_std_db = power_jitter_std_db

    def _build_schedule(self, episode_length: int) -> None:
        mask = two_state_markov_mask(
            self._rng, episode_length, self.duty_cycle, self.mean_burst_slots
        )
        self._band_schedule = np.full(episode_length, self.band, dtype=int)
        self._active_schedule = mask
        self._power_schedule = power_schedule_from_mask(
            self._rng, mask, self.power_mean_dbm, self.power_jitter_std_db
        )
