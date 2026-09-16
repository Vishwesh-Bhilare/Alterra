from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from .agile_emitter import AgileEmitter
from .base_emitter import BaseEmitter
from .fixed_emitter import FixedEmitter
from .periodic_scan_emitter import PeriodicScanEmitter
from .schedule_utils import power_schedule_from_mask, two_state_markov_mask, windows_mask
from simulation.utils.rng import RNGManager
from simulation.utils.config_loader import AlterraConfig


def load_manual_scenario(path: str | Path) -> list[dict]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["emitters"]


def load_scenario_overrides(path: str | Path) -> dict:
    """A scenario file may also carry a top-level `config_overrides:`
    section (e.g. cranking sensor.pfa_rate for a high-false-alarm test
    case) alongside its `emitters:` list. Returns {} when absent."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("config_overrides", {}) or {}


def apply_scenario_overrides(config: AlterraConfig, overrides: dict) -> AlterraConfig:
    """Applies a one-level-deep {section: {field: value}} override dict
    (as returned by load_scenario_overrides) on top of `config`, e.g.
    {"sensor": {"pfa_rate": 0.15}}. Returns `config` unchanged if
    `overrides` is empty."""
    if not overrides:
        return config
    section_updates = {}
    for section_name, fields in overrides.items():
        current_section = getattr(config, section_name)
        section_updates[section_name] = dataclasses.replace(current_section, **fields)
    return dataclasses.replace(config, **section_updates)


def _pulse_params(spec: dict, config: AlterraConfig, rng: np.random.Generator) -> dict:
    pulse_cfg = config.pulse
    return {
        "pri_s": float(spec["pri_s"]) if "pri_s" in spec else pulse_cfg.pri_s_range.sample(rng),
        "pw_s": float(spec["pw_s"]) if "pw_s" in spec else pulse_cfg.pw_s_range.sample(rng),
        "pri_jitter_std_s": float(spec.get("pri_jitter_std_s", pulse_cfg.pri_jitter_std_s)),
        "doa_deg": float(spec["doa_deg"]) if "doa_deg" in spec else pulse_cfg.doa_deg_range.sample(rng),
    }


def _active_windows(spec: dict) -> list[tuple[int, int]] | None:
    """Optional `active_windows: [[start, end], ...]` on a manual spec --
    a scripted, guaranteed on/off schedule, overriding the usual
    probabilistic duty_cycle/mean_burst_slots mask. Absent -> None."""
    raw = spec.get("active_windows")
    if raw is None:
        return None
    return [(int(w[0]), int(w[1])) for w in raw]


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
        active_windows = _active_windows(spec)

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
                    active_windows=active_windows,
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
                    active_windows=active_windows,
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
                    active_windows=active_windows,
                    **pulse_kwargs,
                )
            )

        else:
            raise ValueError(f"Unknown manual emitter kind: {kind}")

    return emitters


class _ManualAgileEmitter(AgileEmitter):
    def __init__(self, hop_bands: list[int], active_windows: list[tuple[int, int]] | None = None, **kwargs):
        kwargs.pop("num_bands", None)
        kwargs.pop("hop_bandset_size", None)
        super().__init__(num_bands=max(hop_bands) + 1, hop_bandset_size=len(hop_bands), **kwargs)
        self._fixed_hop_bands = hop_bands
        self._fixed_active_windows = active_windows

    def _build_schedule(self, episode_length: int) -> None:
        hop_bands = np.array(self._fixed_hop_bands)
        band_schedule = np.empty(episode_length, dtype=int)
        num_hops = int(np.ceil(episode_length / self.hop_dwell_slots))
        hop_choices = self._rng.choice(hop_bands, size=num_hops, replace=True)
        for hop_idx in range(num_hops):
            start = hop_idx * self.hop_dwell_slots
            end = min(start + self.hop_dwell_slots, episode_length)
            band_schedule[start:end] = hop_choices[hop_idx]

        if self._fixed_active_windows is not None:
            mask = windows_mask(episode_length, self._fixed_active_windows)
        else:
            mask = two_state_markov_mask(
                self._rng, episode_length, self.burst_duty_cycle, self.burst_mean_slots
            )
        self._band_schedule = band_schedule
        self._active_schedule = mask
        self._power_schedule = power_schedule_from_mask(
            self._rng, mask, self.power_mean_dbm, self.power_jitter_std_db
        )


class _ManualPeriodicScanEmitter(PeriodicScanEmitter):
    def __init__(self, sweep_start_band: int, active_windows: list[tuple[int, int]] | None = None, **kwargs):
        kwargs.pop("num_bands", None)
        super().__init__(num_bands=sweep_start_band + kwargs["sweep_width"] + 1, **kwargs)
        self._fixed_sweep_start = sweep_start_band
        self._fixed_active_windows = active_windows

    def _build_schedule(self, episode_length: int) -> None:
        sweep_bands = np.arange(self._fixed_sweep_start, self._fixed_sweep_start + self.sweep_width)
        slot_positions = np.arange(episode_length) // self.dwell_slots
        band_schedule = sweep_bands[slot_positions % self.sweep_width]

        if self._fixed_active_windows is not None:
            mask = windows_mask(episode_length, self._fixed_active_windows)
        else:
            mask = two_state_markov_mask(
                self._rng, episode_length, self.duty_cycle, self.burst_mean_slots
            )
        self._band_schedule = band_schedule.astype(int)
        self._active_schedule = mask
        self._power_schedule = power_schedule_from_mask(
            self._rng, mask, self.power_mean_dbm, self.power_jitter_std_db
        )


def _archetype_catalog() -> dict[str, dict]:
    """One canonical spec per named archetype (matches the defining
    emitter of the corresponding old scenario preset). `_hop_bandset_size`
    is a builder-only key, popped before being handed to
    build_manual_population.
    """
    return {
        "fixed_low": {
            "kind": "fixed", "threat_level": 1, "duty_cycle": 0.5,
            "mean_burst_slots": 40, "power_dbm": 8.0, "power_jitter_std_db": 1.0,
        },
        "fixed_medium": {
            "kind": "fixed", "threat_level": 2, "duty_cycle": 0.5,
            "mean_burst_slots": 40, "power_dbm": 12.0, "power_jitter_std_db": 1.0,
        },
        "fixed_high": {
            "kind": "fixed", "threat_level": 3, "duty_cycle": 0.5,
            "mean_burst_slots": 40, "power_dbm": 16.0, "power_jitter_std_db": 1.0,
        },
        "mid_episode_burst": {
            "kind": "fixed", "threat_level": 3, "duty_cycle": 0.5,
            "mean_burst_slots": 20, "power_dbm": 15.0, "power_jitter_std_db": 1.5,
            "active_windows": [[700, 1050]],
        },
        "silent_gap_revisit": {
            "kind": "fixed", "threat_level": 3, "duty_cycle": 0.5,
            "mean_burst_slots": 20, "power_dbm": 18.0, "power_jitter_std_db": 1.0,
            "active_windows": [[0, 300], [1700, 1950]],
        },
        "fast_hopper": {
            "kind": "agile", "threat_level": 3, "hop_dwell_slots": 4,
            "burst_duty_cycle": 0.8, "burst_mean_slots": 15,
            "power_dbm": 13.0, "power_jitter_std_db": 2.0,
            "_hop_bandset_size": 5,
        },
        "periodic_scanner": {
            "kind": "periodic_scan", "threat_level": 2, "sweep_width": 14,
            "dwell_slots": 10, "duty_cycle": 0.8, "burst_mean_slots": 20,
            "power_dbm": 12.0, "power_jitter_std_db": 1.5,
        },
    }


def build_custom_population(
    emitter_requests: list[dict],
    boost_false_alarm: bool,
    config: AlterraConfig,
    rng_manager: RNGManager,
) -> tuple[AlterraConfig, list[BaseEmitter]]:
    """Composes a custom population from a list of individual emitter
    requests, each `{"archetype": <name>, "band_lo": int, "band_hi": int}`
    -- one entry per emitter instance, each with its own independent band
    placement range (not shared across a type). Ranges are clamped into
    [0, num_bands-1] and swapped if given reversed.

    Placement per kind, sampled uniformly within [band_lo, band_hi]:
      fixed          -> band
      agile          -> hop_bands (bandset shrunk to fit if the range is
                         narrower than the archetype's usual hop-set size)
      periodic_scan  -> sweep_start_band (clamped so sweep_width still
                         fits within the spectrum)

    `boost_false_alarm` reuses the same config_overrides mechanism as the
    old high_false_alarm.yaml preset (sensor.pfa_rate 0.02 -> 0.15).
    """
    if boost_false_alarm:
        config = apply_scenario_overrides(config, {"sensor": {"pfa_rate": 0.15}})

    catalog = _archetype_catalog()
    num_bands = config.spectrum.num_bands
    specs: list[dict] = []

    for i, request in enumerate(emitter_requests):
        archetype_name = request["archetype"]
        band_rng = rng_manager.spawn_named(f"custom_request_{i}_placement")

        if archetype_name == "random":
            archetype_name = _random_archetype(catalog, band_rng)
        elif archetype_name not in catalog:
            raise ValueError(f"Unknown archetype: {archetype_name}")
        template = catalog[archetype_name]

        band_lo = max(0, min(int(request["band_lo"]), num_bands - 1))
        band_hi = max(0, min(int(request["band_hi"]), num_bands - 1))
        if band_lo > band_hi:
            band_lo, band_hi = band_hi, band_lo

        spec = dict(template)
        instance_id = f"{archetype_name}_{i}"
        spec["id"] = instance_id

        if spec["kind"] == "fixed":
            spec["band"] = int(band_rng.integers(band_lo, band_hi + 1))

        elif spec["kind"] == "agile":
            bandset_size = spec.pop("_hop_bandset_size")
            available = band_hi - band_lo + 1
            actual_size = min(bandset_size, available)
            spec["hop_bands"] = [
                int(b) for b in band_rng.choice(
                    np.arange(band_lo, band_hi + 1), size=actual_size, replace=False
                )
            ]

        elif spec["kind"] == "periodic_scan":
            sweep_width = spec["sweep_width"]
            max_start = min(band_hi, num_bands - sweep_width - 1)
            max_start = max(max_start, 0)
            min_start = min(band_lo, max_start)
            spec["sweep_start_band"] = int(band_rng.integers(min_start, max_start + 1))

        specs.append(spec)

    emitters = build_manual_population(specs, config, rng_manager)
    return config, emitters


def _random_archetype(catalog: dict, rng: np.random.Generator) -> str:
    """Uniformly picks a concrete archetype name for a "random" request
    slot -- deterministic given `rng`'s state, so reproducible per seed
    like everything else in this module."""
    return str(rng.choice(list(catalog.keys())))
