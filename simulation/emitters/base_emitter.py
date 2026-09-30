from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class EmitterState:
    emitter_id: str
    band: Optional[int]
    active: bool
    power_dbm: float
    threat_level: int


class BaseEmitter(ABC):
    def __init__(
        self,
        emitter_id: str,
        threat_level: int,
        rng: np.random.Generator,
        pri_s: float,
        pw_s: float,
        pri_jitter_std_s: float,
        doa_deg: float,
    ):
        self.emitter_id = emitter_id
        self.threat_level = threat_level
        self._rng = rng

        self.pri_s = pri_s
        self.pw_s = pw_s
        self.pri_jitter_std_s = pri_jitter_std_s
        self.doa_deg = doa_deg

        self._band_schedule: Optional[np.ndarray] = None
        self._active_schedule: Optional[np.ndarray] = None
        self._power_schedule: Optional[np.ndarray] = None
        self._episode_length = 0

    @property
    def kind(self) -> str:
        return type(self).__name__

    @abstractmethod
    def _build_schedule(self, episode_length: int) -> None:
        raise NotImplementedError

    def reset(self, episode_length: int) -> None:
        self._episode_length = episode_length
        self._build_schedule(episode_length)
        assert self._band_schedule is not None
        assert self._active_schedule is not None
        assert self._power_schedule is not None
        assert len(self._band_schedule) == episode_length
        assert len(self._active_schedule) == episode_length
        assert len(self._power_schedule) == episode_length

    def state_at(self, t: int) -> EmitterState:
        if self._active_schedule is None:
            raise RuntimeError(f"{self.emitter_id}: reset() must be called before state_at()")
        if not (0 <= t < self._episode_length):
            raise IndexError(f"{self.emitter_id}: t={t} outside [0, {self._episode_length})")

        active = bool(self._active_schedule[t])
        return EmitterState(
            emitter_id=self.emitter_id,
            band=int(self._band_schedule[t]) if active else None,
            active=active,
            power_dbm=float(self._power_schedule[t]),
            threat_level=self.threat_level,
        )

    def generate_pdws(
        self, slot_duration_s: float, band_center_freq_fn: Callable[[int], float]
    ) -> list:
        """Expand this emitter's own schedule into its true pulse train
        (TOA, PRI, PW, CF, DOA, amplitude). Ground-truth stand-in for the
        raw-IQ + MS-UNet1D detection stage — no raw IQ is synthesized."""
        from simulation.environment.pdw_export import PulseDescriptorWord

        if self._active_schedule is None:
            raise RuntimeError(f"{self.emitter_id}: reset() must be called before generate_pdws()")

        pulses: list[PulseDescriptorWord] = []
        episode_duration_s = self._episode_length * slot_duration_s
        t = 0.0
        while t < episode_duration_s:
            slot_idx = min(int(t // slot_duration_s), self._episode_length - 1)
            state = self.state_at(slot_idx)
            if state.active:
                pw = min(self.pw_s, episode_duration_s - t)
                pulses.append(
                    PulseDescriptorWord(
                        emitter_id=self.emitter_id,
                        toa_s=t,
                        pw_s=pw,
                        pri_s=self.pri_s,
                        cf_hz=band_center_freq_fn(state.band),
                        doa_deg=self.doa_deg,
                        amplitude_dbm=state.power_dbm,
                        true_band=state.band,
                        true_threat_level=self.threat_level,
                    )
                )
            jitter = self._rng.normal(0.0, self.pri_jitter_std_s)
            t += max(self.pri_s + jitter, 1e-6)
        return pulses
