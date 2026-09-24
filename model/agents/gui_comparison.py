"""
GUI-facing orchestration for the "Run Comparison" button: lists available
scenario files, builds fresh manual-emitter populations per job, loads
every registered model, and runs the four-way (Traditional / Heuristic /
Hybrid / Pure-PPO -- or however many models are actually registered)
comparison via simulation.environment.comparison. Kept in Python so
PythonBridge stays a thin pybind wrapper instead of duplicating this
orchestration logic in C++.
"""
from __future__ import annotations

from pathlib import Path

from model.agents import model_registry, policy_runner
from simulation.emitters import build_manual_population, load_manual_scenario
from simulation.environment.comparison import run_comparison
from simulation.utils.rng import RNGManager


def list_scenarios(repo_root: str) -> list[str]:
    scen_dir = Path(repo_root) / "configs" / "scenarios"
    if not scen_dir.exists():
        return []
    return sorted(p.name for p in scen_dir.glob("*.yaml"))


def build_scenario_emitters(config, repo_root: str, scenario_filename: str):
    path = Path(repo_root) / "configs" / "scenarios" / scenario_filename
    specs = load_manual_scenario(str(path))
    rng_manager = RNGManager(config.rng_seed)
    return build_manual_population(specs, config, rng_manager)


def run_full_comparison(repo_root: str, config, seed: int, scenario_filename: str | None = None) -> list[dict]:
    def fresh_emitters():
        if not scenario_filename:
            return None
        return build_scenario_emitters(config, repo_root, scenario_filename)

    jobs = [
        {"label": "Traditional (sequential)", "kind": "traditional", "mode": "sequential", "manual_emitters": fresh_emitters()},
        {"label": "Heuristic (rules only)", "kind": "heuristic", "manual_emitters": fresh_emitters()},
    ]

    for entry in model_registry.list_models(repo_root):
        runner = policy_runner.load_model(repo_root, entry["id"])
        jobs.append({
            "label": entry["label"],
            "kind": "rl",
            "runner": runner,
            "manual_emitters": fresh_emitters(),
            "hybrid": entry["algo_class"] == "MaskablePPO",
        })

    rows = run_comparison(config, seed, jobs)
    return [
        {
            "label": r.label,
            "pd": r.metrics.probability_of_detection,
            "pfa": r.metrics.probability_of_false_alarm,
            "avg_intercept_rate": r.metrics.avg_intercept_rate,
            "percent_correct": r.metrics.percent_correct,
            "avg_reward": r.metrics.avg_reward,
        }
        for r in rows
    ]


def run_selected_comparison(
    repo_root: str,
    config,
    seed: int,
    scenario_filename: str | None,
    model_ids: list[str],
    include_sequential: bool,
    include_balanced_random: bool,
    include_heuristic: bool,
) -> list[dict]:
    """Comparison page's actual entry point -- explicit job selection
    (which registered models, which baselines) instead of
    run_full_comparison's old "everything registered" default. Kept
    separate from run_full_comparison rather than replacing it, since the
    CLI's `alterra env compare` and other non-GUI callers may still want
    the simpler always-everything behavior."""
    from simulation.environment.comparison import run_comparison as _run_comparison

    def fresh_emitters():
        if not scenario_filename:
            return None
        return build_scenario_emitters(config, repo_root, scenario_filename)

    jobs = []
    if include_sequential:
        jobs.append({"label": "Traditional (sequential)", "kind": "traditional", "mode": "sequential", "manual_emitters": fresh_emitters()})
    if include_balanced_random:
        jobs.append({"label": "Traditional (balanced random)", "kind": "traditional", "mode": "balanced_random", "manual_emitters": fresh_emitters()})
    if include_heuristic:
        jobs.append({"label": "Heuristic (rules only)", "kind": "heuristic", "manual_emitters": fresh_emitters()})

    all_models = {m["id"]: m for m in model_registry.list_models(repo_root)}
    for model_id in model_ids:
        entry = all_models.get(model_id)
        if entry is None:
            continue
        runner = policy_runner.load_model(repo_root, model_id)
        jobs.append({
            "label": entry["label"],
            "kind": "rl",
            "runner": runner,
            "manual_emitters": fresh_emitters(),
            "hybrid": entry["algo_class"] == "MaskablePPO",
        })

    rows = _run_comparison(config, seed, jobs)
    return [
        {
            "label": r.label,
            "pd": r.metrics.probability_of_detection,
            "pfa": r.metrics.probability_of_false_alarm,
            "avg_intercept_rate": r.metrics.avg_intercept_rate,
            "percent_correct": r.metrics.percent_correct,
            "avg_reward": r.metrics.avg_reward,
        }
        for r in rows
    ]
