from __future__ import annotations

import click
import numpy as np

from simulation.emitters import build_population
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from simulation.utils.rng import RNGManager


@click.group()
def cli():
    """Alterra — Smart Scan Strategy for Electronic Warfare."""


@cli.group()
def emitters():
    """Inspect emitter populations."""


@emitters.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--episode-length", type=int, default=None, help="Overrides timing.episode_length_slots")
def preview(config_path: str, episode_length: int | None):
    """Build a randomized emitter population and print a summary."""
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

    click.echo("\nSample trace (first emitter, first 20 slots):")
    if population:
        e = population[0]
        click.echo(f"  {e.emitter_id} (threat={e.threat_level})")
        for t in range(min(20, ep_len)):
            s = e.state_at(t)
            band_str = f"band={s.band:>3d}" if s.active else "  idle  "
            power_str = f"{s.power_dbm:6.2f} dBm" if s.active else "        "
            click.echo(f"    t={t:3d}  {band_str}  {power_str}")


@cli.group()
def env():
    """Inspect / sanity-check the Gymnasium environment."""


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
def check(config_path: str):
    """Run Gymnasium's built-in environment checker."""
    from gymnasium.utils.env_checker import check_env

    config = load_config(config_path)
    e = AlterraEnv(config)
    check_env(e.unwrapped, skip_render_check=True)
    click.echo("AlterraEnv passed gymnasium's check_env().")


@env.command()
@click.option("--config", "config_path", default="configs/default_config.yaml", show_default=True)
@click.option("--steps", type=int, default=50, show_default=True)
def preview(config_path: str, steps: int):
    """Run a random-action rollout and print rewards/detections per step."""
    config = load_config(config_path)
    e = AlterraEnv(config)
    obs, info = e.reset(seed=config.rng_seed)

    click.echo(f"action_space={e.action_space}  observation shapes="
               f"{ {k: v.shape for k, v in e.observation_space.spaces.items()} }")

    total_reward = 0.0
    hits = 0
    false_alarms = 0
    rng = np.random.default_rng(0)

    for step in range(steps):
        action = int(rng.integers(0, e.action_space.n))
        obs, reward, terminated, truncated, info = e.step(action)
        total_reward += reward
        hits += int(info["any_hit"])
        false_alarms += int(info["any_false_alarm"])
        click.echo(
            f"  step={step:3d}  band={info['band']:3d}  reward={reward:7.3f}  "
            f"hit={info['any_hit']}  false_alarm={info['any_false_alarm']}"
        )
        if terminated or truncated:
            click.echo("  episode ended, resetting")
            obs, info = e.reset()

    click.echo(f"\nTotal reward: {total_reward:.2f}  hits: {hits}/{steps}  false_alarms: {false_alarms}/{steps}")


if __name__ == "__main__":
    cli()
