from __future__ import annotations

from simulation.utils.config_loader import AlterraConfig
from simulation.utils.rng import RNGManager

from .agile_emitter import AgileEmitter
from .base_emitter import BaseEmitter
from .fixed_emitter import FixedEmitter
from .periodic_scan_emitter import PeriodicScanEmitter


def _normalized_weights(class_weights: dict[str, float]) -> tuple[list[str], list[float]]:
    kinds = list(class_weights.keys())
    total = sum(class_weights.values())
    probs = [class_weights[k] / total for k in kinds]
    return kinds, probs


def build_population(config: AlterraConfig, rng_manager: RNGManager) -> list[BaseEmitter]:
    pop_cfg = config.emitters.population
    pulse_cfg = config.pulse
    selection_rng = rng_manager.spawn_named("emitter_population_selection")

    count = pop_cfg.total_count_range.sample(selection_rng)
    kinds, probs = _normalized_weights(pop_cfg.class_weights)
    assignments = selection_rng.choice(kinds, size=count, p=probs)

    emitters: list[BaseEmitter] = []
    for i, kind in enumerate(assignments):
        emitter_id = f"{kind}_{i:03d}"
        emitter_rng = rng_manager.spawn_named(emitter_id)
        threat_level = pop_cfg.threat_level.sample(selection_rng)

        pri_s = pulse_cfg.pri_s_range.sample(selection_rng)
        pw_s = pulse_cfg.pw_s_range.sample(selection_rng)
        doa_deg = pulse_cfg.doa_deg_range.sample(selection_rng)
        pri_jitter_std_s = pulse_cfg.pri_jitter_std_s

        if kind == "fixed":
            cfg = config.emitters.fixed
            band = (
                cfg.band_index_range.sample(selection_rng)
                if cfg.band_index_range is not None
                else int(selection_rng.integers(0, config.spectrum.num_bands))
            )
            emitters.append(
                FixedEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=emitter_rng,
                    pri_s=pri_s,
                    pw_s=pw_s,
                    pri_jitter_std_s=pri_jitter_std_s,
                    doa_deg=doa_deg,
                    band=band,
                    duty_cycle=cfg.duty_cycle_range.sample(selection_rng),
                    mean_burst_slots=cfg.mean_burst_slots_range.sample(selection_rng),
                    power_mean_dbm=cfg.power_dbm_range.sample(selection_rng),
                    power_jitter_std_db=cfg.power_jitter_std_db,
                )
            )

        elif kind == "agile":
            cfg = config.emitters.agile
            emitters.append(
                AgileEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=emitter_rng,
                    pri_s=pri_s,
                    pw_s=pw_s,
                    pri_jitter_std_s=pri_jitter_std_s,
                    doa_deg=doa_deg,
                    num_bands=config.spectrum.num_bands,
                    hop_bandset_size=cfg.num_hop_bands_range.sample(selection_rng),
                    hop_dwell_slots=cfg.hop_dwell_slots_range.sample(selection_rng),
                    burst_duty_cycle=cfg.burst_duty_cycle_range.sample(selection_rng),
                    burst_mean_slots=cfg.burst_mean_slots_range.sample(selection_rng),
                    power_mean_dbm=cfg.power_dbm_range.sample(selection_rng),
                    power_jitter_std_db=cfg.power_jitter_std_db,
                )
            )

        elif kind == "periodic_scan":
            cfg = config.emitters.periodic_scan
            emitters.append(
                PeriodicScanEmitter(
                    emitter_id=emitter_id,
                    threat_level=threat_level,
                    rng=emitter_rng,
                    pri_s=pri_s,
                    pw_s=pw_s,
                    pri_jitter_std_s=pri_jitter_std_s,
                    doa_deg=doa_deg,
                    num_bands=config.spectrum.num_bands,
                    sweep_width=cfg.sweep_width_bands_range.sample(selection_rng),
                    dwell_slots=cfg.dwell_slots_range.sample(selection_rng),
                    duty_cycle=cfg.duty_cycle_range.sample(selection_rng),
                    burst_mean_slots=cfg.burst_mean_slots_range.sample(selection_rng),
                    power_mean_dbm=cfg.power_dbm_range.sample(selection_rng),
                    power_jitter_std_db=cfg.power_jitter_std_db,
                )
            )

        else:
            raise ValueError(f"Unknown emitter kind in config class_weights: {kind}")

    return emitters
