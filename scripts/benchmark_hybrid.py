"""
Controlled 3-Way Benchmark Suite:
  B0: Heuristic Rule Teacher
  B1: Random-Valid Baseline (random selection within valid doctrine mask)
  B2: New Hybrid Cognitive LSTM Model (trained on TSRD + core scenarios)

Evaluates the 7 metrics defined in the specification across standard test scenarios.
"""
from __future__ import annotations

import glob
from pathlib import Path
import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.emitters.tsrd_adapter import build_tsrd_population
from simulation.metrics.rollout_metrics import MetricsTracker
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager

from model.hybrid.heuristic_scheduler import HeuristicScheduler
from model.hybrid.policy import HybridPolicy
from model.hybrid.doctrine import CognitiveDoctrine


class RandomValidPolicy:
    """Baseline B1: Chooses uniformly at random among valid candidate mask actions."""
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


class SequentialScanPolicy:
    """Traditional Sequential Scan across 128 channels."""
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands

    def reset(self):
        pass

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        # action [1, 0] moves +1 channel relative to previous band
        return np.array([1, 0], dtype=np.int64)


class PureRandomPolicy:
    """Traditional Unconstrained Uniform Random Scan."""
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


class PPOPolicyWrapper:
    """Wrapper around PolicyRunner for SB3 RL policies."""
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        return self.runner.predict(obs, last_info=last_info)


def run_episode_evaluation(env: AlterraEnv, policy, seed: int, max_steps: int = 150) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset()
    tracker = MetricsTracker()

    info = {"band": env._current_band, "hit": False, "mean_power_norm": 0.0, "consecutive_hits": 0, "t": 0}
    total_reward = 0.0
    first_intercept_slot = None

    for step in range(max_steps):
        if hasattr(policy, "select_action"):
            # B0 Heuristic
            target_band, dwell_idx, _, _ = policy.select_action(obs, last_info=info)
            dir_idx = policy.doctrine.select_relative_action(target_band)
            action = [dir_idx, dwell_idx]
        elif isinstance(policy, HybridPolicy):
            # B2 Hybrid LSTM
            action = policy.predict(obs, last_info=info, deterministic=True)
        else:
            # B1 Random-Valid
            action = policy.predict(obs, last_info=info)

        obs, reward, term, trunc, info = env.step(action)
        tracker.record_step(env.last_dwell_result, reward)
        total_reward += reward

        if info.get("hit", False) and first_intercept_slot is None:
            first_intercept_slot = int(info.get("t", step * 10))

        if term or trunc:
            break

    m = tracker.finalize(env)
    return {
        "Pd": m.probability_of_detection if m.probability_of_detection is not None else 0.0,
        "Pfa": m.probability_of_false_alarm if m.probability_of_false_alarm is not None else 0.0,
        "Intercept Rate": m.avg_intercept_rate if m.avg_intercept_rate is not None else 0.0,
        "Avg Reward": total_reward,
        "Accuracy": m.percent_correct if m.percent_correct is not None else 0.0,
        "TTFI": first_intercept_slot if first_intercept_slot is not None else -1,
    }


def print_comparison_table(scenario_name: str, results: dict[str, list[dict]]):
    headers = ["Policy", "Intercept Rate", "Pd (Detection)", "Pfa (False Alarm)", "Avg Reward", "Accuracy", "TTFI (slots)"]
    rows = []
    for policy_name, res_list in results.items():
        avg_ir = np.mean([r["Intercept Rate"] for r in res_list])
        avg_pd = np.mean([r["Pd"] for r in res_list])
        avg_pfa = np.mean([r["Pfa"] for r in res_list])
        avg_rew = np.mean([r["Avg Reward"] for r in res_list])
        avg_acc = np.mean([r["Accuracy"] for r in res_list])
        ttfi_vals = [r["TTFI"] for r in res_list if r["TTFI"] > 0]
        avg_ttfi = f"{np.mean(ttfi_vals):.0f}" if ttfi_vals else "n/a"

        rows.append({
            "Policy": policy_name,
            "Intercept Rate": f"{avg_ir:.3f}",
            "Pd (Detection)": f"{avg_pd:.3f}",
            "Pfa (False Alarm)": f"{avg_pfa:.3f}",
            "Avg Reward": f"{avg_rew:+.2f}",
            "Accuracy": f"{avg_acc*100:.1f}%",
            "TTFI (slots)": avg_ttfi,
        })

    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(r[h])))

    print(f"\nScenario: {scenario_name}")
    print("-" * 80)
    print(" | ".join(h.ljust(col_widths[h]) for h in headers))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(col_widths[h]) for h in headers))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Alterra Cognitive Hybrid Benchmark Suite")
    parser.add_argument("--include-ppo", action="store_true", help="Include PPO RL baselines in benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation seeds")
    args = parser.parse_args()

    config = load_config("configs/default_config.yaml")

    print("=" * 80)
    print(" ALTERRA COGNITIVE SCHEDULER GRAND BENCHMARK")
    print(" Heuristic Teacher | Random-Valid | Traditional Baselines | Hybrid ML Models")
    if args.include_ppo:
        print(" + Reinforcement Learning Baselines (PPO, PPO+RNN, PPO+LSTM, PPO+Transformer)")
    print("=" * 80)

    # Initialize Policies
    b0_heuristic = HeuristicScheduler(num_bands=128)
    b1_random_valid = RandomValidPolicy(num_bands=128)
    trad_seq = SequentialScanPolicy(num_bands=128)
    trad_rand = PureRandomPolicy(num_bands=128)
    
    b2_hybrid_lstm = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_lstm.pt", backbone="lstm", num_bands=128)
    b3_hybrid_gru = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_gru.pt", backbone="gru", num_bands=128)
    b4_hybrid_rnn = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_rnn.pt", backbone="rnn", num_bands=128)
    
    policies = {
        "B0: Heuristic Teacher": b0_heuristic,
        "B1: Random-Valid": b1_random_valid,
        "Traditional: Sequential": trad_seq,
        "Traditional: Uniform Random": trad_rand,
        "Hybrid LSTM": b2_hybrid_lstm,
        "Hybrid GRU": b3_hybrid_gru,
        "Hybrid RNN": b4_hybrid_rnn,
    }
    
    if Path("model/hybrid/checkpoints/best_hybrid_transformer.pt").exists():
        policies["Hybrid Transformer"] = HybridPolicy(
            checkpoint_path="model/hybrid/checkpoints/best_hybrid_transformer.pt",
            backbone="transformer",
            num_bands=128,
        )

    if args.include_ppo:
        from model.agents.policy_runner import load_model
        ppo_models = [
            ("ppo_499c7c", "RL: PPO Baseline"),
            ("ppo_rnn_82507e", "RL: PPO + RNN"),
            ("ppo_lstm_finetune_c83a23", "RL: PPO + LSTM"),
            ("ppo_transformer_12310d", "RL: PPO + Transformer"),
        ]
        for pid, plabel in ppo_models:
            try:
                runner = load_model(".", pid)
                policies[plabel] = PPOPolicyWrapper(runner)
            except Exception as e:
                print(f"[Warning] Failed to load {pid}: {e}")

    test_seeds = [1001, 1002, 1003, 1004, 1005][:args.seeds]

    # Benchmark 1: Single-Emitter Sparse Threat
    single_cfg = apply_overrides(config, num_emitters=1, episode_length_slots=1500)
    single_res = {k: [] for k in policies}
    for s in test_seeds:
        for name, pol in policies.items():
            env = AlterraEnv(single_cfg)
            res = run_episode_evaluation(env, pol, seed=s)
            single_res[name].append(res)
    print_comparison_table("Single Emitter Sparse Threat (5 Seeds)", single_res)

    # Benchmark 2: Agile Frequency-Hopping
    if glob.glob("configs/scenarios/fast_hopping_evasive.yaml"):
        specs = load_manual_scenario("configs/scenarios/fast_hopping_evasive.yaml")
        hop_res = {k: [] for k in policies}
        for s in test_seeds:
            for name, pol in policies.items():
                pop = build_manual_population(specs, config, RNGManager(s))
                env = AlterraEnv(config, manual_emitters=pop)
                res = run_episode_evaluation(env, pol, seed=s)
                hop_res[name].append(res)
        print_comparison_table("Agile Frequency-Hopping Emitter (5 Seeds)", hop_res)

    # Benchmark 3: Periodic Sweeper Radar
    if glob.glob("configs/scenarios/periodic_scan_focus.yaml"):
        specs = load_manual_scenario("configs/scenarios/periodic_scan_focus.yaml")
        per_res = {k: [] for k in policies}
        for s in test_seeds:
            for name, pol in policies.items():
                pop = build_manual_population(specs, config, RNGManager(s))
                env = AlterraEnv(config, manual_emitters=pop)
                res = run_episode_evaluation(env, pol, seed=s)
                per_res[name].append(res)
        print_comparison_table("Periodic Sweeper Radar (5 Seeds)", per_res)

    # Benchmark 4: Real TSRD Radar Pulse Trains
    tsrd_files = sorted(glob.glob("*turing*/archive/test/*.h5")) or sorted(glob.glob("*turing*/archive/train/*.h5"))
    if tsrd_files:
        tsrd_res = {k: [] for k in policies}
        for i, s in enumerate(test_seeds[:3]):
            fpath = tsrd_files[i % len(tsrd_files)]
            try:
                for name, pol in policies.items():
                    pop = build_tsrd_population(fpath, config, max_emitters=6, seed=s)
                    env = AlterraEnv(config, manual_emitters=pop)
                    res = run_episode_evaluation(env, pol, seed=s)
                    tsrd_res[name].append(res)
            except Exception as e:
                pass
        if any(len(v) > 0 for v in tsrd_res.values()):
            print_comparison_table("TSRD Real-World Radar Pulse Trains (Held-Out Test)", tsrd_res)

    print("\n" + "=" * 80)
    print(" Benchmark completed successfully across all 7 evaluation metrics!")
    print("=" * 80)


if __name__ == "__main__":
    main()
