"""
Ground-truth PDW export — stand-in for the raw-IQ + MS-UNet1D detection
stage: each emitter's schedule is expanded into its true pulse train, then
all emitters' pulses are merged into one time-sorted mixed stream, the same
interleaved input SEDCAM deinterleaving is meant to sort back apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from simulation.emitters.base_emitter import BaseEmitter
from simulation.utils.config_loader import SpectrumConfig


@dataclass(frozen=True)
class PulseDescriptorWord:
    emitter_id: str  # ground truth — not given to deinterleaving
    toa_s: float
    pw_s: float
    pri_s: float
    cf_hz: float
    doa_deg: float
    amplitude_dbm: float
    true_band: int
    true_threat_level: int

    def to_dict(self) -> dict:
        return asdict(self)


def band_center_hz(spectrum: SpectrumConfig, band: int) -> float:
    return spectrum.band_start_freq_hz + (band + 0.5) * spectrum.band_bandwidth_hz


def generate_episode_pdws(
    emitters: list[BaseEmitter], spectrum: SpectrumConfig, slot_duration_s: float
) -> list[PulseDescriptorWord]:
    all_pdws: list[PulseDescriptorWord] = []
    for emitter in emitters:
        all_pdws.extend(
            emitter.generate_pdws(slot_duration_s, lambda b: band_center_hz(spectrum, b))
        )
    all_pdws.sort(key=lambda p: p.toa_s)
    return all_pdws
