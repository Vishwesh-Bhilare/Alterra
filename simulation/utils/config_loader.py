"""
Loads configs/*.yaml into typed dataclasses. Every distribution/range an
emitter, sensor, or environment component needs at runtime comes from here —
nothing in simulation/ code should hardcode a numeric default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class FloatRange:
    min: float
    max: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.min, self.max))


@dataclass(frozen=True)
class IntRange:
    min: int
    max: int

    def sample(self, rng) -> int:
        return int(rng.integers(self.min, self.max + 1))


@dataclass(frozen=True)
class SpectrumConfig:
    num_bands: int
    band_bandwidth_hz: float
    band_start_freq_hz: float


@dataclass(frozen=True)
class TimingConfig:
    slot_duration_s: float
    episode_length_slots: int


@dataclass(frozen=True)
class ThreatLevelConfig:
    levels: list[int]
    weights: list[float]

    def sample(self, rng) -> int:
        probs = [w / sum(self.weights) for w in self.weights]
        return int(rng.choice(self.levels, p=probs))


@dataclass(frozen=True)
class PopulationConfig:
    total_count_range: IntRange
    class_weights: dict[str, float]
    threat_level: ThreatLevelConfig


@dataclass(frozen=True)
class FixedEmitterConfig:
    band_index_range: Optional[IntRange]
    duty_cycle_range: FloatRange
    mean_burst_slots_range: IntRange
    power_dbm_range: FloatRange
    power_jitter_std_db: float


@dataclass(frozen=True)
class AgileEmitterConfig:
    hop_dwell_slots_range: IntRange
    num_hop_bands_range: IntRange
    burst_duty_cycle_range: FloatRange
    burst_mean_slots_range: IntRange
    power_dbm_range: FloatRange
    power_jitter_std_db: float


@dataclass(frozen=True)
class PeriodicScanEmitterConfig:
    dwell_slots_range: IntRange
    sweep_width_bands_range: IntRange
    duty_cycle_range: FloatRange
    burst_mean_slots_range: IntRange
    power_dbm_range: FloatRange
    power_jitter_std_db: float


@dataclass(frozen=True)
class EmittersConfig:
    population: PopulationConfig
    fixed: FixedEmitterConfig
    agile: AgileEmitterConfig
    periodic_scan: PeriodicScanEmitterConfig


@dataclass(frozen=True)
class SensorConfig:
    noise_floor_dbm_range: FloatRange
    pd_snr50_db: float
    pd_slope_db: float
    pfa_rate: float


@dataclass(frozen=True)
class RewardConfig:
    hit_reward_base: float
    threat_weight_levels: list[int]
    threat_weight_values: list[float]
    idle_cost: float
    false_alarm_penalty: float
    staleness_penalty_coeff: float
    staleness_norm_slots: int


@dataclass(frozen=True)
class EnvironmentConfig:
    dwell_slots_per_action: int
    reward: RewardConfig


@dataclass(frozen=True)
class AlterraConfig:
    rng_seed: int
    spectrum: SpectrumConfig
    timing: TimingConfig
    emitters: EmittersConfig
    sensor: SensorConfig
    environment: EnvironmentConfig


def _int_range(d: Optional[dict]) -> Optional[IntRange]:
    return IntRange(**d) if d is not None else None


def _float_range(d: dict) -> FloatRange:
    return FloatRange(**d)


def load_config(path: str | Path) -> AlterraConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    pop_raw = raw["emitters"]["population"]
    population = PopulationConfig(
        total_count_range=_int_range(pop_raw["total_count_range"]),
        class_weights=pop_raw["class_weights"],
        threat_level=ThreatLevelConfig(**pop_raw["threat_level"]),
    )

    fixed_raw = raw["emitters"]["fixed"]
    fixed = FixedEmitterConfig(
        band_index_range=_int_range(fixed_raw["band_index_range"]),
        duty_cycle_range=_float_range(fixed_raw["duty_cycle_range"]),
        mean_burst_slots_range=_int_range(fixed_raw["mean_burst_slots_range"]),
        power_dbm_range=_float_range(fixed_raw["power_dbm_range"]),
        power_jitter_std_db=float(fixed_raw["power_jitter_std_db"]),
    )

    agile_raw = raw["emitters"]["agile"]
    agile = AgileEmitterConfig(
        hop_dwell_slots_range=_int_range(agile_raw["hop_dwell_slots_range"]),
        num_hop_bands_range=_int_range(agile_raw["num_hop_bands_range"]),
        burst_duty_cycle_range=_float_range(agile_raw["burst_duty_cycle_range"]),
        burst_mean_slots_range=_int_range(agile_raw["burst_mean_slots_range"]),
        power_dbm_range=_float_range(agile_raw["power_dbm_range"]),
        power_jitter_std_db=float(agile_raw["power_jitter_std_db"]),
    )

    ps_raw = raw["emitters"]["periodic_scan"]
    periodic_scan = PeriodicScanEmitterConfig(
        dwell_slots_range=_int_range(ps_raw["dwell_slots_range"]),
        sweep_width_bands_range=_int_range(ps_raw["sweep_width_bands_range"]),
        duty_cycle_range=_float_range(ps_raw["duty_cycle_range"]),
        burst_mean_slots_range=_int_range(ps_raw["burst_mean_slots_range"]),
        power_dbm_range=_float_range(ps_raw["power_dbm_range"]),
        power_jitter_std_db=float(ps_raw["power_jitter_std_db"]),
    )

    sensor_raw = raw["sensor"]
    sensor = SensorConfig(
        noise_floor_dbm_range=_float_range(sensor_raw["noise_floor_dbm_range"]),
        pd_snr50_db=float(sensor_raw["pd_snr50_db"]),
        pd_slope_db=float(sensor_raw["pd_slope_db"]),
        pfa_rate=float(sensor_raw["pfa_rate"]),
    )

    env_raw = raw["environment"]
    reward_raw = env_raw["reward"]
    reward = RewardConfig(
        hit_reward_base=float(reward_raw["hit_reward_base"]),
        threat_weight_levels=list(reward_raw["threat_weight_levels"]),
        threat_weight_values=[float(v) for v in reward_raw["threat_weight_values"]],
        idle_cost=float(reward_raw["idle_cost"]),
        false_alarm_penalty=float(reward_raw["false_alarm_penalty"]),
        staleness_penalty_coeff=float(reward_raw["staleness_penalty_coeff"]),
        staleness_norm_slots=int(reward_raw["staleness_norm_slots"]),
    )
    environment = EnvironmentConfig(
        dwell_slots_per_action=int(env_raw["dwell_slots_per_action"]),
        reward=reward,
    )

    return AlterraConfig(
        rng_seed=int(raw["rng_seed"]),
        spectrum=SpectrumConfig(**raw["spectrum"]),
        timing=TimingConfig(**raw["timing"]),
        emitters=EmittersConfig(
            population=population, fixed=fixed, agile=agile, periodic_scan=periodic_scan
        ),
        sensor=sensor,
        environment=environment,
    )
