from __future__ import annotations

import click
import numpy as np

from simulation.emitters import build_manual_population, build_population, load_manual_scenario
from simulation.environment import AlterraEnv, run_traditional_scan
from simulation.metrics import MetricsTracker
from simulation.utils.config_loader import apply_overrides, load_config
from simulation.utils.rng import RNGManager
from simulation.viz import plot_episode


@click.group()
def cli():
    """Alterra — Smart Scan Strategy for Electronic Warfare."""


@cli.group()
def emitters():
    """Inspect emitter populations."""


@emitters.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--episode-length", type=int, default=None)
def preview(config_path: str, episode_length: int | None):
    config = load_config(config_path)
    ep_len = episode_length or config.timing.episode_length_slots
    rng_manager = RNGManager(config.rng_seed)
    population = build_population(config, rng_manager)
    for e in population:
        e.reset(ep_len)

    click.echo(f"Spectrum: {config.spectrum.num_bands} bands, episode length {ep_len} slots")
    click.echo(f"Population: {len(population)} emitters")
    counts: dict[str, int] = {}
    for e in population:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        click.echo(f"  {kind}: {n}")


@cli.group()
def scenario():
    """Manual scenario tools."""


@scenario.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", required=True)
@click.option("--episode-length", type=int, default=None)
def preview(config_path: str, scenario_path: str, episode_length: int | None):
    config = load_config(config_path)
    ep_len = episode_length or config.timing.episode_length_slots
    rng_manager = RNGManager(config.rng_seed)
    specs = load_manual_scenario(scenario_path)
    population = build_manual_population(specs, config, rng_manager)
    for e in population:
        e.reset(ep_len)

    click.echo(f"Manual scenario: {len(population)} emitters from {scenario_path}")
    for e in population:
        click.echo(f"  {e.emitter_id} ({e.kind}, threat={e.threat_level})")


@cli.group()
def env():
    """Inspect / evaluate the Gymnasium environment."""


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
def check(config_path: str):
    from gymnasium.utils.env_checker import check_env
    config = load_config(config_path)
    e = AlterraEnv(config)
    check_env(e.unwrapped, skip_render_check=True)
    click.echo("AlterraEnv passed gymnasium's check_env().")


def _build_manual_emitters(config, scenario_path: str | None):
    if not scenario_path:
        return None
    rng_manager = RNGManager(config.rng_seed)
    specs = load_manual_scenario(scenario_path)
    return build_manual_population(specs, config, rng_manager)


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--steps", type=int, default=50, show_default=True)
def preview(config_path: str, scenario_path: str | None, steps: int):
    config = load_config(config_path)
    manual_emitters = _build_manual_emitters(config, scenario_path)
    e = AlterraEnv(config, manual_emitters=manual_emitters)
    obs, info = e.reset(seed=config.rng_seed)
    total_reward = 0.0
    hits = 0
    false_alarms = 0

    for step in range(steps):
        action = e.action_space.sample()
        obs, reward, terminated, truncated, info = e.step(action)
        total_reward += reward
        hits += int(info["any_hit"])
        false_alarms += int(info["any_false_alarm"])
        click.echo(
            f"  step={step:3d}  band={info['band']:3d}  dwell={info['dwell_slots']:2d}  "
            f"reward={reward:7.3f}  hit={info['any_hit']}  false_alarm={info['any_false_alarm']}"
        )
        if terminated or truncated:
            obs, info = e.reset()

    click.echo(f"\nTotal reward: {total_reward:.2f}  hits: {hits}/{steps}  false_alarms: {false_alarms}/{steps}")


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--episodes", type=int, default=5, show_default=True)
@click.option("--steps-per-episode", type=int, default=200, show_default=True)
def metrics(config_path: str, scenario_path: str | None, episodes: int, steps_per_episode: int):
    config = load_config(config_path)
    manual_emitters = _build_manual_emitters(config, scenario_path)
    e = AlterraEnv(config, manual_emitters=manual_emitters)

    per_episode = []
    for ep in range(episodes):
        obs, info = e.reset(seed=config.rng_seed + ep)
        tracker = MetricsTracker()
        for _ in range(steps_per_episode):
            action = e.action_space.sample()
            obs, reward, terminated, truncated, info = e.step(action)
            tracker.record_step(e.last_dwell_result, reward)
            if terminated or truncated:
                break
        per_episode.append(tracker.finalize(e))

    def avg(attr: str):
        vals = [getattr(m, attr) for m in per_episode if getattr(m, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    click.echo(f"Averaged over {episodes} episodes ({steps_per_episode} steps each):")
    for label, attr, fmt in [
        ("Pd", "probability_of_detection", ".3f"),
        ("Pfa", "probability_of_false_alarm", ".3f"),
        ("Sensitivity", "sensitivity", ".3f"),
        ("Avg intercept rate", "avg_intercept_rate", ".3f"),
        ("Avg reward", "avg_reward", ".3f"),
        ("Percent correct", "percent_correct", ".3f"),
        ("Avg intercept time error (slots)", "avg_intercept_time_error_slots", ".2f"),
    ]:
        v = avg(attr)
        click.echo(f"{label + ':':<36} {v:{fmt}}" if v is not None else f"{label + ':':<36} n/a")


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--out", "out_path", default="episode_waterfall.png", show_default=True)
def plot(config_path: str, scenario_path: str | None, steps: int, out_path: str):
    config = load_config(config_path)
    manual_emitters = _build_manual_emitters(config, scenario_path)
    e = AlterraEnv(config, manual_emitters=manual_emitters)
    obs, info = e.reset(seed=config.rng_seed)

    dwell_bands, dwell_starts, dwell_ends = [], [], []
    for _ in range(steps):
        action = e.action_space.sample()
        obs, reward, terminated, truncated, info = e.step(action)
        dr = e.last_dwell_result
        dwell_bands.append(dr.band)
        dwell_starts.append(dr.start_t)
        dwell_ends.append(dr.end_t)
        if terminated or truncated:
            break

    plot_episode(e._spectrum_world, dwell_bands, dwell_starts, dwell_ends, out_path)
    click.echo(f"Saved {out_path}")


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--episode-length", type=int, default=None,
              help="Manual override: episode length in slots")
@click.option("--num-emitters", type=int, default=None,
              help="Manual override: exact emitter count for this run (overrides the configured min/max range)")
@click.option("--traditional-mode", type=click.Choice(["sequential", "balanced_random"]), default=None,
              help="Manual override: traditional scanner strategy (default: whatever's in config)")
@click.option("--traditional-dwell-slots", type=int, default=None,
              help="Manual override: fixed dwell time for the traditional scanner")
@click.option("--episodes", type=int, default=5, show_default=True)
@click.option("--steps-per-episode", type=int, default=200, show_default=True)
def compare(
    config_path: str,
    scenario_path: str | None,
    episode_length: int | None,
    num_emitters: int | None,
    traditional_mode: str | None,
    traditional_dwell_slots: int | None,
    episodes: int,
    steps_per_episode: int,
):
    """Compare the traditional (non-adaptive) scanner against a
    random-action baseline over the same seeded episodes -- shows how much
    ground an adaptive scheduler has to make up over legacy fixed scanning.
    """
    config = load_config(config_path)
    config = apply_overrides(
        config,
        episode_length_slots=episode_length,
        num_emitters=num_emitters,
        traditional_scan_mode=traditional_mode,
        traditional_dwell_slots=traditional_dwell_slots,
    )

    manual_emitters = _build_manual_emitters(config, scenario_path)
    e = AlterraEnv(config, manual_emitters=manual_emitters)

    traditional_results, random_results = [], []
    for ep in range(episodes):
        seed = config.rng_seed + ep

        traditional_results.append(run_traditional_scan(e, seed=seed))

        obs, info = e.reset(seed=seed)  # same seed -> identical emitter population
        tracker = MetricsTracker()
        for _ in range(steps_per_episode):
            action = e.action_space.sample()
            obs, reward, terminated, truncated, info = e.step(action)
            tracker.record_step(e.last_dwell_result, reward)
            if terminated or truncated:
                break
        random_results.append(tracker.finalize(e))

    def avg(results, attr):
        vals = [getattr(m, attr) for m in results if getattr(m, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    click.echo(
        f"Averaged over {episodes} episodes, {steps_per_episode} steps each "
        f"(traditional mode: {config.comparison.traditional_scan.mode}, "
        f"num_emitters override: {num_emitters or 'default range'}):\n"
    )
    click.echo(f"{'Metric':<34}{'Traditional':<20}{'Random baseline':<20}")
    for label, attr, fmt in [
        ("Pd", "probability_of_detection", ".3f"),
        ("Pfa", "probability_of_false_alarm", ".3f"),
        ("Avg intercept rate", "avg_intercept_rate", ".3f"),
        ("Percent correct", "percent_correct", ".3f"),
        ("Avg intercept time error (slots)", "avg_intercept_time_error_slots", ".2f"),
    ]:
        t_val = avg(traditional_results, attr)
        r_val = avg(random_results, attr)
        t_str = f"{t_val:{fmt}}" if t_val is not None else "n/a"
        r_str = f"{r_val:{fmt}}" if r_val is not None else "n/a"
        click.echo(f"{label:<34}{t_str:<20}{r_str:<20}")


@cli.group()
def pdw():
    """Ground-truth PDW export (stand-in for MS-UNet1D detection output)."""


@pdw.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--episode-length", type=int, default=None)
@click.option("--out", "out_path", default="pdws.jsonl", show_default=True)
def export(config_path: str, scenario_path: str | None, episode_length: int | None, out_path: str):
    import json
    from simulation.environment import generate_episode_pdws

    config = load_config(config_path)
    ep_len = episode_length or config.timing.episode_length_slots
    manual_emitters = _build_manual_emitters(config, scenario_path)
    population = manual_emitters if manual_emitters is not None else build_population(config, RNGManager(config.rng_seed))

    for e in population:
        e.reset(ep_len)

    pdws = generate_episode_pdws(population, config.spectrum, config.timing.slot_duration_s)

    with open(out_path, "w") as f:
        for p in pdws:
            f.write(json.dumps(p.to_dict()) + "\n")

    click.echo(f"Wrote {len(pdws)} PDWs from {len(population)} emitters to {out_path}")


@pdw.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--episode-length", type=int, default=None)
@click.option("--limit", type=int, default=15, show_default=True)
def preview(config_path: str, scenario_path: str | None, episode_length: int | None, limit: int):
    from simulation.environment import generate_episode_pdws

    config = load_config(config_path)
    ep_len = episode_length or config.timing.episode_length_slots
    manual_emitters = _build_manual_emitters(config, scenario_path)
    population = manual_emitters if manual_emitters is not None else build_population(config, RNGManager(config.rng_seed))

    for e in population:
        e.reset(ep_len)

    pdws = generate_episode_pdws(population, config.spectrum, config.timing.slot_duration_s)
    click.echo(f"{len(pdws)} PDWs total, showing first {limit}:")
    for p in pdws[:limit]:
        click.echo(
            f"  toa={p.toa_s:.6f}s  pw={p.pw_s*1e6:.1f}us  pri={p.pri_s*1e3:.3f}ms  "
            f"cf={p.cf_hz/1e6:.1f}MHz  doa={p.doa_deg:.1f}deg  amp={p.amplitude_dbm:.2f}dBm  "
            f"emitter={p.emitter_id}"
        )


if __name__ == "__main__":
    cli()
