"""
Benchmark script comparing all 5 trained models:
1. PPO
2. PPO + RNN
3. PPO + LSTM
4. PPO + GRU
5. PPO + Transformer
Across standard Electronic Warfare benchmark scenarios.
"""
from __future__ import annotations

import os
import sys
import numpy as np

from model.agents import model_registry, policy_runner
from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.metrics.rollout_metrics import MetricsTracker
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager

MODELS_TO_TEST = [
    ("PPO", "ppo_499c7c"),
    ("PPO + RNN", "ppo_rnn_82507e"),
    ("PPO + LSTM", "ppo_lstm_finetune_c83a23"),
    ("PPO + GRU", "ppo_gru_eefb69"),
    ("PPO + Transformer", "ppo_transformer_12310d"),
]

SCENARIOS = [
    ("Single Emitter (Sparse Threat)", "configs/scenarios/sparse_single_threat.yaml", 500),
    ("Agile Fast-Hopping Emitter", "configs/scenarios/fast_hopping_evasive.yaml", 500),
    ("Periodic Sweeping Radar", "configs/scenarios/periodic_scan_focus.yaml", 500),
    ("Dense Congested Spectrum", "configs/scenarios/dense_congested.yaml", 500),
]

def evaluate_model_on_scenario(runner, scenario_path, config, seed=42, episode_length=500):
    specs = load_manual_scenario(scenario_path)
    rng_mgr = RNGManager(seed)
    emitters = build_manual_population(specs, config, rng_mgr)
    env_config = apply_overrides(config, episode_length_slots=episode_length)
    env = AlterraEnv(env_config, manual_emitters=emitters)

    obs, _ = env.reset(seed=seed)
    runner.reset()
    tracker = MetricsTracker()

    truncated = False
    total_reward = 0.0
    while not truncated:
        action = runner.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        tracker.record_step(env.last_dwell_result, reward)
        total_reward += reward

    metrics = tracker.finalize(env)
    return {
        "Pd": metrics.probability_of_detection if metrics.probability_of_detection is not None else 0.0,
        "Pfa": metrics.probability_of_false_alarm if metrics.probability_of_false_alarm is not None else 0.0,
        "Intercept Rate": metrics.avg_intercept_rate if metrics.avg_intercept_rate is not None else 0.0,
        "Avg Reward": metrics.avg_reward if metrics.avg_reward is not None else total_reward,
        "Percent Correct": metrics.percent_correct if metrics.percent_correct is not None else 0.0,
    }

def evaluate_model_stochastic_multiseed(runner, config, seeds=[42, 101, 777, 1337, 2024], episode_length=500):
    results = []
    env_config = apply_overrides(config, episode_length_slots=episode_length)
    for s in seeds:
        env = AlterraEnv(env_config)
        obs, _ = env.reset(seed=s)
        runner.reset()
        tracker = MetricsTracker()

        truncated = False
        while not truncated:
            action = runner.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.record_step(env.last_dwell_result, reward)

        metrics = tracker.finalize(env)
        results.append({
            "Pd": metrics.probability_of_detection or 0.0,
            "Pfa": metrics.probability_of_false_alarm or 0.0,
            "Intercept Rate": metrics.avg_intercept_rate or 0.0,
            "Avg Reward": metrics.avg_reward or 0.0,
            "Percent Correct": metrics.percent_correct or 0.0,
        })

    avg_res = {
        k: float(np.mean([r[k] for r in results])) for k in results[0]
    }
    return avg_res

def print_table(rows):
    headers = ["Model", "Intercept Rate", "Pd (Detection)", "Pfa (False Alarm)", "Avg Reward", "Accuracy"]
    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(r[h])))
    header_str = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_str = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_str)
    print(sep_str)
    for r in rows:
        print(" | ".join(str(r[h]).ljust(col_widths[h]) for h in headers))

def main():
    repo_root = os.getcwd()
    config = load_config("configs/default_config.yaml")

    print("=" * 80)
    print(" ALTERRA 5-MODEL BENCHMARK: PPO, RNN, LSTM, GRU, TRANSFORMER")
    print("=" * 80)

    # 1. Load runners
    loaded_runners = {}
    for name, mid in MODELS_TO_TEST:
        print(f"Loading {name} (ID: {mid})...")
        loaded_runners[name] = policy_runner.load_model(repo_root, mid)
    print("All 5 models loaded successfully!\n")

    # 2. Evaluate on Scenarios
    for sc_name, sc_file, ep_len in SCENARIOS:
        print("\n" + "-" * 75)
        print(f"Scenario: {sc_name}")
        print("-" * 75)
        rows = []
        for name, _ in MODELS_TO_TEST:
            runner = loaded_runners[name]
            res = evaluate_model_on_scenario(runner, sc_file, config, seed=42, episode_length=ep_len)
            rows.append({
                "Model": name,
                "Intercept Rate": f"{res['Intercept Rate']:.3f}",
                "Pd (Detection)": f"{res['Pd']:.3f}",
                "Pfa (False Alarm)": f"{res['Pfa']:.3f}",
                "Avg Reward": f"{res['Avg Reward']:.2f}",
                "Accuracy": f"{res['Percent Correct']*100:.1f}%",
            })
        print_table(rows)

    # 3. Evaluate Stochastic Multi-Seed (Random Emitters)
    print("\n" + "-" * 75)
    print("Scenario: Stochastic Emitter Mix (Average across 5 Seeds)")
    print("-" * 75)
    stoch_rows = []
    for name, _ in MODELS_TO_TEST:
        runner = loaded_runners[name]
        res = evaluate_model_stochastic_multiseed(runner, config, seeds=[42, 101, 777, 1337, 2024], episode_length=500)
        stoch_rows.append({
            "Model": name,
            "Intercept Rate": f"{res['Intercept Rate']:.3f}",
            "Pd (Detection)": f"{res['Pd']:.3f}",
            "Pfa (False Alarm)": f"{res['Pfa']:.3f}",
            "Avg Reward": f"{res['Avg Reward']:.2f}",
            "Accuracy": f"{res['Percent Correct']*100:.1f}%",
        })
    print_table(stoch_rows)

    print("\n" + "=" * 80)
    print(" Benchmark execution completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
