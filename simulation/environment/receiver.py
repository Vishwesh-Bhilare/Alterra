"""
The scanning receiver: given a band and a dwell duration, samples the
sensor model across each slot of the dwell and summarizes the result. One
dwell = one RL step.

Retune modeling: switching to a band different from the last one dwelled
on costs `config.receiver.retune_time_s` (converted to slots), consumed
*before* the dwell's detections begin -- i.e. it eats into the episode's
slot budget exactly like TraditionalScanDriver's dwells do. No cost is
paid on the very first dwell of an episode.

Frequency window: the receiver's actual observed interval is centered on
the selected band and sized by `config.receiver.instantaneous_bandwidth_hz`
(B_I) -- independent of `config.spectrum.band_bandwidth_hz`, which is only
the spectrum's binning resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.environment.sensor_model import Detection, SensorModel
from simulation.environment.spectrum_world import SpectrumWorld
from simulation.utils.config_loader import AlterraConfig


@dataclass(frozen=True)
class DwellResult:
    band: int
    start_t: int
    end_t: int
    detections: list[Detection]
    retune_slots: int = 0
    center_freq_hz: float = 0.0
    freq_lo_hz: float = 0.0
    freq_hi_hz: float = 0.0

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

    def classification_counts(self) -> dict[str, int]:
        counts = {"hit": 0, "miss": 0, "false_alarm": 0, "correct_reject": 0}
        for d in self.detections:
            counts[d.classification] += 1
        return counts


class Receiver:
    def __init__(self, spectrum_world: SpectrumWorld, sensor_model: SensorModel, config: AlterraConfig):
        self.spectrum_world = spectrum_world
        self.sensor_model = sensor_model
        self.spectrum_config = config.spectrum
        self.receiver_config = config.receiver
        self._retune_slots = max(
            0, round(config.receiver.retune_time_s / config.timing.slot_duration_s)
        )
        self._last_band: int | None = None

    def detection_threshold_dbm(self, band: int) -> float:
        return self.sensor_model.detection_threshold_dbm(band)

    def _freq_window_hz(self, band: int) -> tuple[float, float, float]:
        center = self.spectrum_config.band_center_freq_hz(band)
        half_bw = self.receiver_config.instantaneous_bandwidth_hz / 2.0
        return center, center - half_bw, center + half_bw

    def dwell(self, band: int, start_t: int, dwell_slots: int) -> DwellResult:
        retune_slots = (
            self._retune_slots if (self._last_band is not None and band != self._last_band) else 0
        )
        dwell_start_t = min(start_t + retune_slots, self.spectrum_world.episode_length)
        end_t = min(dwell_start_t + dwell_slots, self.spectrum_world.episode_length)
        detections = [
            self.sensor_model.observe(self.spectrum_world.band_at(t, band), t)
            for t in range(dwell_start_t, end_t)
        ]
        self._last_band = band

        center, freq_lo, freq_hi = self._freq_window_hz(band)
        return DwellResult(
            band=band,
            start_t=dwell_start_t,
            end_t=end_t,
            detections=detections,
            retune_slots=retune_slots,
            center_freq_hz=center,
            freq_lo_hz=freq_lo,
            freq_hi_hz=freq_hi,
        )
