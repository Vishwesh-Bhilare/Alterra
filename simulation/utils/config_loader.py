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
    noise_reading_std_db: float
    measured_power_norm_min: float
    measured_power_norm_max: float


@dataclass(frozen=True)
class RewardConfig:
    hit_reward_base: float
    threat_weight_levels: list[int]
    threat_weight_values: list[float]
    idle_cost: float
    false_alarm_penalty: float
    staleness_penalty_coeff: float
    staleness_norm_slots: int
    novelty_bonus: float
    hit_confirm_floor: float


@dataclass(frozen=True)
class EnvironmentConfig:
    dwell_options_slots: list[int]
    reward: RewardConfig


@dataclass(frozen=True)
class ScenarioConfig:
    manual_scenario_path: Optional[str]


@dataclass(frozen=True)
class PulseConfig:
    pri_s_range: FloatRange
    pw_s_range: FloatRange
    pri_jitter_std_s: float
    doa_deg_range: FloatRange


@dataclass(frozen=True)
class AlterraConfig:
    rng_seed: int
    spectrum: SpectrumConfig
    timing: TimingConfig
    emitters: EmittersConfig
    sensor: SensorConfig
    environment: EnvironmentConfig
    scenario: ScenarioConfig
    pulse: PulseConfig


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
        noise_reading_std_db=float(sensor_raw["noise_reading_std_db"]),
        measured_power_norm_min=float(sensor_raw["measured_power_norm_min"]),
        measured_power_norm_max=float(sensor_raw["measured_power_norm_max"]),
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
        novelty_bonus=float(reward_raw["novelty_bonus"]),
        hit_confirm_floor=float(reward_raw["hit_confirm_floor"]),
    )
    environment = EnvironmentConfig(
        dwell_options_slots=[int(v) for v in env_raw["dwell_options_slots"]],
        reward=reward,
    )

    scenario_raw = raw.get("scenario", {})
    scenario = ScenarioConfig(manual_scenario_path=scenario_raw.get("manual_scenario_path"))

    pulse_raw = raw["pulse"]
    pulse = PulseConfig(
        pri_s_range=_float_range(pulse_raw["pri_s_range"]),
        pw_s_range=_float_range(pulse_raw["pw_s_range"]),
        pri_jitter_std_s=float(pulse_raw["pri_jitter_std_s"]),
        doa_deg_range=_float_range(pulse_raw["doa_deg_range"]),
    )

    return AlterraConfig(
        rng_seed=int(raw["rng_seed"]),
        spectrum=SpectrumConfig(
            num_bands=int(raw["spectrum"]["num_bands"]),
            band_bandwidth_hz=float(raw["spectrum"]["band_bandwidth_hz"]),
            band_start_freq_hz=float(raw["spectrum"]["band_start_freq_hz"]),
        ),
        timing=TimingConfig(
            slot_duration_s=float(raw["timing"]["slot_duration_s"]),
            episode_length_slots=int(raw["timing"]["episode_length_slots"]),
        ),
        emitters=EmittersConfig(
            population=population, fixed=fixed, agile=agile, periodic_scan=periodic_scan
        ),
        sensor=sensor,
        environment=environment,
        scenario=scenario,
        pulse=pulse,
    )
