"""
Runs any mix of RL policies (via PolicyRunner -- so plain PPO and
sb3-contrib RecurrentPPO/LSTM models are stepped identically) and the two
traditional baselines on identical, reproducible episodes (same seed,
same emitter population per sub-run -- built fresh per sub-run since
emitter objects are stateful) for direct side-by-side comparison of the
figures of merit called out in the problem statement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulation.environment.gym_env import AlterraEnv
from simulation.environment.traditional_scanner import TraditionalScanDriver
from simulation.metrics.rollout_metrics import EpisodeMetrics, MetricsTracker
from simulation.utils.config_loader import AlterraConfig


@dataclass
class ComparisonRow:
    label: str
    metrics: EpisodeMetrics


def _run_traditional(config: AlterraConfig, seed: int, mode: str, manual_emitters) -> EpisodeMetrics:
    env = AlterraEnv(config)
    driver = TraditionalScanDriver(env, mode=mode, seed=seed)
    options = {"manual_emitters": manual_emitters} if manual_emitters is not None else None
    driver.reset(seed=seed, options=options)

    tracker = MetricsTracker()
    truncated = False
    while not truncated:
        dwell_result, truncated = driver.step()
        tracker.record_step(dwell_result, reward=0.0)
    return tracker.finalize(env)


def _run_rl(config: AlterraConfig, seed: int, runner, manual_emitters) -> EpisodeMetrics:
    env = AlterraEnv(config)
    options = {"manual_emitters": manual_emitters} if manual_emitters is not None else None
    obs, _ = env.reset(seed=seed, options=options)
    runner.reset()

    tracker = MetricsTracker()
    truncated = False
    while not truncated:
        action = runner.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        tracker.record_step(env.last_dwell_result, reward)
    return tracker.finalize(env)


def run_comparison(config: AlterraConfig, seed: int, jobs: list[dict[str, Any]]) -> list[ComparisonRow]:
    """Each job dict is one of:
      {"label": str, "kind": "rl", "runner": PolicyRunner, "manual_emitters": list|None}
      {"label": str, "kind": "traditional", "mode": "sequential"|"balanced_random", "manual_emitters": list|None}

    manual_emitters must be an independently-built population per job
    (never the same list object reused across jobs) whenever a custom mix
    is active -- see PythonBridge::runComparison, which builds one fresh
    population per job before calling this, so results stay reproducible
    and independent per job.
    """
    rows = []
    for job in jobs:
        if job["kind"] == "rl":
            metrics = _run_rl(config, seed, job["runner"], job["manual_emitters"])
        elif job["kind"] == "traditional":
            metrics = _run_traditional(config, seed, job["mode"], job["manual_emitters"])
        else:
            raise ValueError(f"Unknown job kind: {job['kind']!r}")
        rows.append(ComparisonRow(job["label"], metrics))
    return rows
