"""
The scanning receiver: given a band and a dwell duration, samples the
sensor model across each slot of the dwell and summarizes the result. One
dwell = one RL step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.environment.sensor_model import Detection, SensorModel
from simulation.environment.spectrum_world import SpectrumWorld


@dataclass(frozen=True)
class DwellResult:
    band: int
    start_t: int
    end_t: int
    detections: list[Detection]

    @property
    def any_hit(self) -> bool:
        return any(d.hit for d in self.detections)

    @property
    def any_false_alarm(self) -> bool:
        return any(d.false_alarm for d in self.detections)

    @property
    def best_hit(self) -> Detection | None:
        hits = [d for d in self.detections if d.hit]
        if not hits:
            return None
        return max(hits, key=lambda d: d.estimated_snr_db or float("-inf"))

    @property
    def mean_measured_power_dbm(self) -> float:
        if not self.detections:
            return float("-inf")
        return float(np.mean([d.measured_power_dbm for d in self.detections]))


class Receiver:
    def __init__(self, spectrum_world: SpectrumWorld, sensor_model: SensorModel):
        self.spectrum_world = spectrum_world
        self.sensor_model = sensor_model

    def dwell(self, band: int, start_t: int, dwell_slots: int) -> DwellResult:
        end_t = min(start_t + dwell_slots, self.spectrum_world.episode_length)
        detections = [
            self.sensor_model.observe(self.spectrum_world.band_at(t, band), t)
            for t in range(start_t, end_t)
        ]
        return DwellResult(band=band, start_t=start_t, end_t=end_t, detections=detections)
