"""
Abstract base for all emitter types.

Design: schedules are precomputed once per episode via reset(), so
state_at(t) is a pure O(1) lookup — callers (spectrum_world, receiver,
metrics) can query it in any order without re-triggering randomness or
getting inconsistent answers for the same t.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EmitterState:
    emitter_id: str
    band: Optional[int]      # None when inactive
    active: bool
    power_dbm: float         # NaN when inactive
    threat_level: int


class BaseEmitter(ABC):
    def __init__(self, emitter_id: str, threat_level: int, rng: np.random.Generator):
        self.emitter_id = emitter_id
        self.threat_level = threat_level
        self._rng = rng

        self._band_schedule: Optional[np.ndarray] = None
        self._active_schedule: Optional[np.ndarray] = None
        self._power_schedule: Optional[np.ndarray] = None
        self._episode_length = 0

    @property
    def kind(self) -> str:
        return type(self).__name__

    @abstractmethod
    def _build_schedule(self, episode_length: int) -> None:
        """Must set self._band_schedule, self._active_schedule,
        self._power_schedule — each a length-`episode_length` array — using
        only self._rng for randomness."""
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
