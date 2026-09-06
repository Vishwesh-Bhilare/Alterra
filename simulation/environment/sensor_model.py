"""
Layer B: RF front-end / sensor imperfection model. Converts SpectrumWorld's
ground truth into what a receiver dwelling on a given band would actually
observe. Every dwell now yields a measured_power_dbm reading regardless of
detection outcome -- a real energy detector always reports something
(noise floor if empty, signal+noise if occupied); this is what lets the
RL observation carry per-band signal-strength information instead of
being blind on every miss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.environment.spectrum_world import BandOccupancy
from simulation.utils.config_loader import SensorConfig
from simulation.utils.rng import RNGManager


@dataclass(frozen=True)
class Detection:
    band: int
    t: int
    hit: bool
    false_alarm: bool
    true_occupied: bool
    estimated_snr_db: float | None       # only set when occupied (ground-truth-adjacent)
    measured_power_dbm: float             # always set -- the real observable
    true_emitter_id: str | None
    true_threat_level: int | None


class SensorModel:
    def __init__(self, config: SensorConfig, rng_manager: RNGManager, num_bands: int):
        self.config = config
        self._rng = rng_manager.spawn_named("sensor_model")
        self.noise_floor_dbm = self._rng.uniform(
            config.noise_floor_dbm_range.min,
            config.noise_floor_dbm_range.max,
            size=num_bands,
        )

    def _pd(self, snr_db: float) -> float:
        z = (snr_db - self.config.pd_snr50_db) / self.config.pd_slope_db
        return 1.0 / (1.0 + np.exp(-z))

    def observe(self, band_occupancy: BandOccupancy, t: int) -> Detection:
        band = band_occupancy.band
        noise_floor = float(self.noise_floor_dbm[band])
        reading_noise = self._rng.normal(0.0, self.config.noise_reading_std_db)

        if band_occupancy.is_occupied:
            signal_dbm = band_occupancy.combined_power_dbm
            snr_db = signal_dbm - noise_floor
            pd = self._pd(snr_db)
            hit = bool(self._rng.random() < pd)
            strongest = band_occupancy.strongest_emitter
            return Detection(
                band=band, t=t, hit=hit, false_alarm=False, true_occupied=True,
                estimated_snr_db=snr_db,
                measured_power_dbm=signal_dbm + reading_noise,
                true_emitter_id=strongest.emitter_id if strongest else None,
                true_threat_level=strongest.threat_level if strongest else None,
            )

        false_alarm = bool(self._rng.random() < self.config.pfa_rate)
        return Detection(
            band=band, t=t, hit=False, false_alarm=false_alarm, true_occupied=False,
            estimated_snr_db=None,
            measured_power_dbm=noise_floor + reading_noise,
            true_emitter_id=None, true_threat_level=None,
        )
