from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .agile_emitter import AgileEmitter
from .base_emitter import BaseEmitter
from .fixed_emitter import FixedEmitter
from .periodic_scan_emitter import PeriodicScanEmitter
from .schedule_utils import power_schedule_from_mask, two_state_markov_mask
from simulation.utils.rng import RNGManager
from simulation.utils.config_loader import AlterraConfig


def load_manual_scenario(path: str | Path) -> list[dict]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["emitters"]


def _pulse_params(spec: dict, config: AlterraConfig, rng: np.random.Generator) -> dict:
    pulse_cfg = config.pulse
    return {
        "pri_s": float(spec["pri_s"]) if "pri_s" in spec else pulse_cfg.pri_s_range.sample(rng),
        "pw_s": float(spec["pw_s"]) if "pw_s" in spec else pulse_cfg.pw_s_range.sample(rng),
        "pri_jitter_std_s": float(spec.get("pri_jitter_std_s", pulse_cfg.pri_jitter_std_s)),
        "doa_deg": float(spec["doa_deg"]) if "doa_deg" in spec else pulse_cfg.doa_deg_range.sample(rng),
    }


def build_manual_population(
    specs: list[dict], config: AlterraConfig, rng_manager: RNGManager
) -> list[BaseEmitter]:
    emitters: list[BaseEmitter] = []

    for spec in specs:
        kind = spec["kind"]
        emitter_id = spec["id"]
        threat_level = int(spec["threat_level"])
        rng = rng_manager.spawn_named(emitter_id)
        pulse_rng = rng_manager.spawn_named(f"{emitter_id}_pulse")
        pulse_kwargs = _pulse_params(spec, config, pulse_rng)

        if kind == "fixed":
            emitters.append(
                FixedEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=rng,
                    band=int(spec["band"]),
                    duty_cycle=float(spec["duty_cycle"]),
                    mean_burst_slots=float(spec["mean_burst_slots"]),
                    power_mean_dbm=float(spec["power_dbm"]),
                    power_jitter_std_db=float(spec["power_jitter_std_db"]),
                    **pulse_kwargs,
                )
            )

        elif kind == "agile":
            emitters.append(
                _ManualAgileEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=rng,
                    hop_bands=spec["hop_bands"],
                    hop_dwell_slots=int(spec["hop_dwell_slots"]),
                    burst_duty_cycle=float(spec["burst_duty_cycle"]),
                    burst_mean_slots=float(spec["burst_mean_slots"]),
                    power_mean_dbm=float(spec["power_dbm"]),
                    power_jitter_std_db=float(spec["power_jitter_std_db"]),
                    **pulse_kwargs,
                )
            )

        elif kind == "periodic_scan":
            emitters.append(
                _ManualPeriodicScanEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=rng,
                    sweep_start_band=int(spec["sweep_start_band"]),
                    sweep_width=int(spec["sweep_width"]),
                    dwell_slots=int(spec["dwell_slots"]),
                    duty_cycle=float(spec["duty_cycle"]),
                    burst_mean_slots=float(spec["burst_mean_slots"]),
                    power_mean_dbm=float(spec["power_dbm"]),
                    power_jitter_std_db=float(spec["power_jitter_std_db"]),
                    **pulse_kwargs,
                )
            )

        else:
            raise ValueError(f"Unknown manual emitter kind: {kind}")

    return emitters


class _ManualAgileEmitter(AgileEmitter):
    def __init__(self, hop_bands: list[int], **kwargs):
        kwargs.pop("num_bands", None)
        kwargs.pop("hop_bandset_size", None)
        super().__init__(num_bands=max(hop_bands) + 1, hop_bandset_size=len(hop_bands), **kwargs)
        self._fixed_hop_bands = hop_bands

    def _build_schedule(self, episode_length: int) -> None:
        hop_bands = np.array(self._fixed_hop_bands)
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


class _ManualPeriodicScanEmitter(PeriodicScanEmitter):
    def __init__(self, sweep_start_band: int, **kwargs):
        kwargs.pop("num_bands", None)
        super().__init__(num_bands=sweep_start_band + kwargs["sweep_width"] + 1, **kwargs)
        self._fixed_sweep_start = sweep_start_band

    def _build_schedule(self, episode_length: int) -> None:
        sweep_bands = np.arange(self._fixed_sweep_start, self._fixed_sweep_start + self.sweep_width)
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
