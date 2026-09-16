"""
Adapter for the Turing Synthetic Radar Dataset (TSRD).
Loads pulse trains (.h5 files) and maps their emitter properties into
Alterra's simulation emitter models without hardcoding paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from simulation.emitters.base_emitter import BaseEmitter
from simulation.emitters.fixed_emitter import FixedEmitter
from simulation.emitters.agile_emitter import AgileEmitter
from simulation.utils.config_loader import AlterraConfig


def load_tsrd_file(file_path: str | Path) -> dict:
    """
    Read pulse descriptor words (PDWs) from a TSRD .h5 file.
    PDW columns:
      0: ToA (microseconds)
      1: Centre Frequency (MHz)
      2: Pulse Width (microseconds)
      3: AoA (degrees)
      4: Amplitude (dB)
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required to load TSRD dataset files: pip install h5py") from e

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TSRD file not found at: {path}")

    with h5py.File(path, "r") as f:
        data = np.array(f["data"][:], dtype=np.float32)
        labels = np.array(f["labels"][:], dtype=np.int32).squeeze(-1)

    return {"data": data, "labels": labels}


def build_tsrd_population(
    file_path: str | Path,
    config: AlterraConfig,
    max_emitters: int = 15,
    seed: int = 42,
) -> list[BaseEmitter]:
    """
    Extract unique emitters from a TSRD .h5 file and instantiate
    corresponding Alterra Emitters mapped into the simulation frequency bands.
    """
    pdw_dict = load_tsrd_file(file_path)
    data = pdw_dict["data"]
    labels = pdw_dict["labels"]

    num_bands = config.spectrum.num_bands
    start_freq_mhz = config.spectrum.band_start_freq_hz / 1e6
    bandwidth_mhz = config.spectrum.band_bandwidth_hz / 1e6

    unique_labels = np.unique(labels)
    emitters: list[BaseEmitter] = []
    rng = np.random.default_rng(seed)

    for idx, emitter_id in enumerate(unique_labels[:max_emitters]):
        mask = labels == emitter_id
        emitter_pulses = data[mask]
        if len(emitter_pulses) < 2:
            continue

        freqs_mhz = emitter_pulses[:, 1]
        pws_us = emitter_pulses[:, 2]
        aoas_deg = emitter_pulses[:, 3]
        powers_dbm = emitter_pulses[:, 4]
        toas_us = np.sort(emitter_pulses[:, 0])
        diffs = np.diff(toas_us)
        pri_s = float(np.median(diffs) * 1e-6) if len(diffs) > 0 else 0.001
        pri_s = max(pri_s, 0.0001)

        pw_s = float(np.median(pws_us) * 1e-6)
        doa_deg = float(np.median(aoas_deg)) % 360.0
        mean_freq = float(np.mean(freqs_mhz))
        band = int(np.clip((mean_freq - start_freq_mhz) / max(bandwidth_mhz, 1.0), 0, num_bands - 1))
        power_dbm = float(np.clip(np.mean(powers_dbm), -20.0, 40.0))
        threat_level = int((idx % 3) + 1)

        unique_freqs = np.unique(np.round(freqs_mhz / bandwidth_mhz))
        if len(unique_freqs) > 1:
            emitters.append(
                AgileEmitter(
                    emitter_id=f"tsrd_agile_{emitter_id}",
                    threat_level=threat_level,
                    rng=np.random.default_rng(seed + idx),
                    pri_s=pri_s,
                    pw_s=pw_s,
                    pri_jitter_std_s=2e-5,
                    doa_deg=doa_deg,
                    num_bands=num_bands,
                    hop_bandset_size=min(len(unique_freqs), 8),
                    hop_dwell_slots=40,
                    burst_duty_cycle=0.5,
                    burst_mean_slots=60.0,
                    power_mean_dbm=power_dbm,
                    power_jitter_std_db=2.0,
                )
            )
        else:
            emitters.append(
                FixedEmitter(
                    emitter_id=f"tsrd_fixed_{emitter_id}",
                    threat_level=threat_level,
                    rng=np.random.default_rng(seed + idx),
                    pri_s=pri_s,
                    pw_s=pw_s,
                    pri_jitter_std_s=2e-5,
                    doa_deg=doa_deg,
                    band=band,
                    duty_cycle=0.4,
                    mean_burst_slots=100.0,
                    power_mean_dbm=power_dbm,
                    power_jitter_std_db=1.5,
                )
            )

    return emitters
