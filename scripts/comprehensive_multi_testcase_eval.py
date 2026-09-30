"""
Comprehensive 7-Scenario Test Suite for ALTERA Scheduler Models
Evaluates models across all 7 canonical test cases:
  1. Single Sparse Threat (sparse_single_threat.yaml)
  2. Agile Frequency-Hopping (fast_hopping_evasive.yaml)
  3. Periodic Sweeper Radar (periodic_scan_focus.yaml)
  4. Dense Congested Multi-Threat (dense_congested.yaml)
  5. High False Alarm / Noise Stress (high_false_alarm.yaml)
  6. Silent Gap & Dynamic Revisit (silent_gap_revisit.yaml)
  7. TSRD Real-World Pulse Trains (held-out test HDF5)
"""
from __future__ import annotations

import glob
import json
import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import (
    load_manual_scenario,
    load_scenario_overrides,
    apply_scenario_overrides,
    build_manual_population,
)
from simulation.emitters.tsrd_adapter import build_tsrd_population
from simulation.metrics.rollout_metrics import MetricsTracker
from simulation.utils.config_loader import load_config
from simulation.utils.rng import RNGManager

from model.hybrid.heuristic_scheduler import HeuristicScheduler
from model.hybrid.policy import HybridPolicy
from model.hybrid.doctrine import CognitiveDoctrine
from model.agents.policy_runner import load_model


class SequentialScanPolicy:
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands

    def reset(self):
        pass

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        return np.array([1, 0], dtype=np.int64)


class PureRandomPolicy:
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands
        self.doctrine = CognitiveDoctrine(num_bands=num_bands)

    def reset(self):
        self.doctrine.reset()

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        if last_info is not None:
            band = int(last_info.get("band", self.doctrine.current_band))
            self.doctrine.current_band = band
        target_band = int(np.random.randint(0, self.num_bands))
        dir_idx = self.doctrine.select_relative_action(target_band)
        return np.array([dir_idx, 0], dtype=np.int64)


class RandomValidPolicy:
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands
        self.doctrine = CognitiveDoctrine(num_bands=num_bands)

    def reset(self):
        self.doctrine.reset()

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        if last_info is not None:
            hit = bool(last_info.get("hit", False) or last_info.get("any_hit", False))
            band = int(last_info.get("band", self.doctrine.current_band))
            power = float(last_info.get("mean_power_norm", 0.0))
            consec = int(last_info.get("consecutive_hits", 0))
            t = int(last_info.get("t", 0))
            self.doctrine.update_state(band, hit, power, consec, t)

        mask = self.doctrine.generate_action_mask(obs.get("band_tracks"), obs.get("receiver"))
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.num_bands)

        target_band = int(np.random.choice(valid_indices))
        dir_idx = self.doctrine.select_relative_action(target_band)
        return np.array([dir_idx, 0], dtype=np.int64)


class PPOPolicyWrapper:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        return self.runner.predict(obs, last_info=last_info)


def evaluate_policy_on_env(env: AlterraEnv, policy, seed: int = 42, max_steps: int = 150) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset()
    tracker = MetricsTracker()

    info = {"band": env._current_band, "hit": False, "mean_power_norm": 0.0, "consecutive_hits": 0, "t": 0}
    total_reward = 0.0

    for step in range(max_steps):
        if hasattr(policy, "select_action"):
            target_band, dwell_idx, _, _ = policy.select_action(obs, last_info=info)
            dir_idx = policy.doctrine.select_relative_action(target_band)
            action = [dir_idx, dwell_idx]
        elif isinstance(policy, HybridPolicy):
            action = policy.predict(obs, last_info=info, deterministic=True)
        else:
            action = policy.predict(obs, last_info=info)

        obs, reward, term, trunc, info = env.step(action)
        tracker.record_step(env.last_dwell_result, reward)
        total_reward += reward

        if term or trunc:
            break

    m = tracker.finalize(env)
    return {
        "Pd": float(m.probability_of_detection if m.probability_of_detection is not None else 0.0),
        "Pfa": float(m.probability_of_false_alarm if m.probability_of_false_alarm is not None else 0.0),
        "Sensitivity": float(m.sensitivity if m.sensitivity is not None else 0.0),
        "Intercept Rate": float(m.avg_intercept_rate if m.avg_intercept_rate is not None else 0.0),
        "Reward": float(total_reward),
        "Accuracy": float(m.percent_correct if m.percent_correct is not None else 0.0),
        "Time Error": float(m.avg_intercept_time_error_slots) if m.avg_intercept_time_error_slots is not None else -1.0,
    }


def main():
    seed = 42
    np.random.seed(seed)
    base_config = load_config("configs/default_config.yaml")

    models = {
        "Traditional: Sequential": SequentialScanPolicy(num_bands=128),
        "Traditional: Uniform Random": PureRandomPolicy(num_bands=128),
        "B0: Heuristic Teacher": HeuristicScheduler(num_bands=128),
        "B1: Random-Valid": RandomValidPolicy(num_bands=128),
        "RL: PPO Baseline": PPOPolicyWrapper(load_model(".", "ppo_499c7c")),
        "RL: PPO + LSTM": PPOPolicyWrapper(load_model(".", "ppo_lstm_finetune_c83a23")),
        "Hybrid LSTM": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_lstm.pt", backbone="lstm", num_bands=128),
        "Hybrid GRU": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_gru.pt", backbone="gru", num_bands=128),
        "Hybrid RNN": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_rnn.pt", backbone="rnn", num_bands=128),
        "Hybrid Transformer": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_transformer.pt", backbone="transformer", num_bands=128),
    }

    scenarios = [
        ("Case 1: Single Sparse Threat", "configs/scenarios/sparse_single_threat.yaml", "manual"),
        ("Case 2: Agile Frequency-Hopping", "configs/scenarios/fast_hopping_evasive.yaml", "manual"),
        ("Case 3: Periodic Sweeper Radar", "configs/scenarios/periodic_scan_focus.yaml", "manual"),
        ("Case 4: Dense Congested Multi-Threat", "configs/scenarios/dense_congested.yaml", "manual"),
        ("Case 5: High Noise / False Alarm Stress", "configs/scenarios/high_false_alarm.yaml", "manual"),
        ("Case 6: Silent Gap & Dynamic Revisit", "configs/scenarios/silent_gap_revisit.yaml", "manual"),
    ]

    # Find TSRD file for Case 7
    tsrd_files = sorted(glob.glob("*turing*/archive/test/*.h5")) or sorted(glob.glob("*turing*/archive/train/*.h5"))
    if tsrd_files:
        scenarios.append(("Case 7: TSRD Real Radar Pulse Trains", tsrd_files[0], "tsrd"))

    all_case_results = {}

    for case_name, file_path, s_type in scenarios:
        case_results = {}
        for m_name, pol in models.items():
            if s_type == "manual":
                specs = load_manual_scenario(file_path)
                overrides = load_scenario_overrides(file_path)
                cfg = apply_scenario_overrides(base_config, overrides)
                pop = build_manual_population(specs, cfg, RNGManager(seed))
                env = AlterraEnv(cfg, manual_emitters=pop)
            else:
                pop = build_tsrd_population(file_path, base_config, max_emitters=6, seed=seed)
                env = AlterraEnv(base_config, manual_emitters=pop)

            res = evaluate_policy_on_env(env, pol, seed=seed)
            case_results[m_name] = res
        all_case_results[case_name] = case_results

    # Save to JSON
    with open("data/comprehensive_evaluation_results.json", "w") as f:
        json.dump(all_case_results, f, indent=2)

    # Print summary per scenario
    for case_name, case_results in all_case_results.items():
        print("\n" + "=" * 110)
        print(f" {case_name}")
        print("=" * 110)
        print(f"{'Model':<28} | {'Pd':^7} | {'Pfa':^7} | {'Sens':^7} | {'IR':^7} | {'Reward':^10} | {'Acc (%)':^9} | {'TTFI (s)':^8}")
        print("-" * 110)
        for m_name, r in case_results.items():
            ttfi_str = f"{r['Time Error']:.0f}" if r['Time Error'] >= 0 else "Miss"
            print(f"{m_name:<28} | {r['Pd']:^7.3f} | {r['Pfa']:^7.3f} | {r['Sensitivity']:^7.3f} | {r['Intercept Rate']:^7.3f} | {r['Reward']:^+10.2f} | {r['Accuracy']*100:^8.2f}% | {ttfi_str:^8}")


if __name__ == "__main__":
    main()
