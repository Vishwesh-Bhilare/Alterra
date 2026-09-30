"""
Single Frequency-Hopping Emitter Evaluation (Seed = 42)
Computes all 7 metrics from the specification across all models:
  1. Probability of Detection (Pd)
  2. Probability of False Alarm (Pfa)
  3. Receiver Sensitivity (low-SNR detection efficiency)
  4. Average Interception Rate
  5. Average Episodic Reward
  6. Decision Accuracy (% Correct)
  7. Average Interception-Time Error (TTFI error in slots)
"""
from __future__ import annotations

import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.metrics.rollout_metrics import MetricsTracker
from simulation.utils.config_loader import load_config
from simulation.utils.rng import RNGManager

from model.hybrid.heuristic_scheduler import HeuristicScheduler
from model.hybrid.policy import HybridPolicy
from model.hybrid.doctrine import CognitiveDoctrine
from model.agents.policy_runner import load_model


class SequentialScanPolicy:
    """Traditional Sequential Scan across 128 channels."""
    def __init__(self, num_bands: int = 128):
        self.num_bands = num_bands

    def reset(self):
        pass

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
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


class RandomValidPolicy:
    """Baseline B1: Random within valid doctrine mask."""
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


def evaluate_model(env: AlterraEnv, policy, seed: int = 42, max_steps: int = 150) -> dict:
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
        "Pd": m.probability_of_detection if m.probability_of_detection is not None else 0.0,
        "Pfa": m.probability_of_false_alarm if m.probability_of_false_alarm is not None else 0.0,
        "Sensitivity": m.sensitivity if m.sensitivity is not None else 0.0,
        "Intercept Rate": m.avg_intercept_rate if m.avg_intercept_rate is not None else 0.0,
        "Total Reward": total_reward,
        "Accuracy": m.percent_correct if m.percent_correct is not None else 0.0,
        "Time Error": m.avg_intercept_time_error_slots if m.avg_intercept_time_error_slots is not None else -1.0,
    }


def main():
    seed = 42
    np.random.seed(seed)
    config = load_config("configs/default_config.yaml")
    specs = load_manual_scenario("configs/scenarios/fast_hopping_evasive.yaml")

    models = {}

    # 1. Traditional Baselines
    models["Traditional: Sequential"] = SequentialScanPolicy(num_bands=128)
    models["Traditional: Uniform Random"] = PureRandomPolicy(num_bands=128)

    # 2. Rule & Mask Baselines
    models["B0: Heuristic Teacher"] = HeuristicScheduler(num_bands=128)
    models["B1: Random-Valid"] = RandomValidPolicy(num_bands=128)

    # 3. Hybrid Models
    models["Hybrid LSTM"] = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_lstm.pt", backbone="lstm", num_bands=128)
    models["Hybrid GRU"] = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_gru.pt", backbone="gru", num_bands=128)
    models["Hybrid RNN"] = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_rnn.pt", backbone="rnn", num_bands=128)
    models["Hybrid Transformer"] = HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_transformer.pt", backbone="transformer", num_bands=128)

    # 4. Pure RL Baselines
    for pid, plabel in [
        ("ppo_499c7c", "RL: PPO Baseline"),
        ("ppo_rnn_82507e", "RL: PPO + RNN"),
        ("ppo_lstm_finetune_c83a23", "RL: PPO + LSTM"),
        ("ppo_transformer_12310d", "RL: PPO + Transformer"),
    ]:
        try:
            runner = load_model(".", pid)
            models[plabel] = PPOPolicyWrapper(runner)
        except Exception as e:
            print(f"Skipping {pid}: {e}")

    results = {}
    for name, pol in models.items():
        pop = build_manual_population(specs, config, RNGManager(seed))
        env = AlterraEnv(config, manual_emitters=pop)
        res = evaluate_model(env, pol, seed=seed)
        results[name] = res

    headers = [
        "Model",
        "P_d (Detection)",
        "P_fa (False Alarm)",
        "Sensitivity",
        "Intercept Rate",
        "Reward",
        "Accuracy (%)",
        "Time Error (slots)",
    ]

    print("\n" + "=" * 115)
    print(f" SINGLE FREQUENCY-HOPPING EMITTER TEST (Seed = {seed}, Emitter: evasive_kilo [Bands: 15, 40, 65, 90, 110])")
    print("=" * 115)

    rows = []
    for name, r in results.items():
        time_err_str = f"{r['Time Error']:.0f}" if r['Time Error'] >= 0 else "Missed"
        rows.append([
            name,
            f"{r['Pd']:.3f}",
            f"{r['Pfa']:.3f}",
            f"{r['Sensitivity']:.3f}",
            f"{r['Intercept Rate']:.3f}",
            f"{r['Total Reward']:+.2f}",
            f"{r['Accuracy']*100:.2f}%",
            time_err_str,
        ])

    col_w = [max(len(row[i]) for row in ([headers] + rows)) for i in range(len(headers))]
    
    header_line = " | ".join(headers[i].ljust(col_w[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * col_w[i] for i in range(len(headers)))
    
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(col_w[i]) for i in range(len(headers))))
    print("=" * 115)


if __name__ == "__main__":
    main()
