from __future__ import annotations

import click
import numpy as np

from simulation.emitters import build_manual_population, build_population, load_manual_scenario
from simulation.environment import AlterraEnv
from simulation.metrics import MetricsTracker
from simulation.utils.config_loader import load_config
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


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--steps", type=int, default=50, show_default=True)
def preview(config_path: str, scenario_path: str | None, steps: int):
    config = load_config(config_path)
    manual_emitters = None
    if scenario_path:
        rng_manager = RNGManager(config.rng_seed)
        specs = load_manual_scenario(scenario_path)
        manual_emitters = build_manual_population(specs, config, rng_manager)

    e = AlterraEnv(config, manual_emitters=manual_emitters)
    obs, info = e.reset(seed=config.rng_seed)
    rng = np.random.default_rng(0)
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
@click.option("--steps", type=int, default=200, show_default=True)
def metrics(config_path: str, scenario_path: str | None, steps: int):
    config = load_config(config_path)
    manual_emitters = None
    if scenario_path:
        rng_manager = RNGManager(config.rng_seed)
        specs = load_manual_scenario(scenario_path)
        manual_emitters = build_manual_population(specs, config, rng_manager)

    e = AlterraEnv(config, manual_emitters=manual_emitters)
    obs, info = e.reset(seed=config.rng_seed)
    tracker = MetricsTracker()

    for _ in range(steps):
        action = e.action_space.sample()
        obs, reward, terminated, truncated, info = e.step(action)
        tracker.record_step(e.last_dwell_result, reward)
        if terminated or truncated:
            break

    m = tracker.finalize(e)
    click.echo(f"Pd:                  {m.probability_of_detection:.3f}")
    click.echo(f"Pfa:                 {m.probability_of_false_alarm:.3f}")
    click.echo(f"Sensitivity:         {m.sensitivity:.3f}")
    click.echo(f"Avg intercept rate:  {m.avg_intercept_rate:.3f}")
    click.echo(f"Avg reward:          {m.avg_reward:.3f}")
    click.echo(f"Percent correct:     {m.percent_correct:.3f}")
    err = m.avg_intercept_time_error_slots
    click.echo(f"Avg intercept time error (slots): {err:.2f}" if err is not None else "Avg intercept time error (slots): n/a")
    click.echo(f"Steps: {m.steps}")


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--scenario", "scenario_path", default=None)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--out", "out_path", default="episode_waterfall.png", show_default=True)
def plot(config_path: str, scenario_path: str | None, steps: int, out_path: str):
    config = load_config(config_path)
    manual_emitters = None
    if scenario_path:
        rng_manager = RNGManager(config.rng_seed)
        specs = load_manual_scenario(scenario_path)
        manual_emitters = build_manual_population(specs, config, rng_manager)

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


if __name__ == "__main__":
    cli()


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
    rng_manager = RNGManager(config.rng_seed)

    if scenario_path:
        specs = load_manual_scenario(scenario_path)
        population = build_manual_population(specs, config, rng_manager)
    else:
        population = build_population(config, rng_manager)

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
    rng_manager = RNGManager(config.rng_seed)

    if scenario_path:
        specs = load_manual_scenario(scenario_path)
        population = build_manual_population(specs, config, rng_manager)
    else:
        population = build_population(config, rng_manager)

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
