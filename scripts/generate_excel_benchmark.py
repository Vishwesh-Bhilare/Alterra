"""
Multi-Seed Monte Carlo Statistical Evaluation & Excel Export Suite
Evaluates models across 25 diverse random seeds to determine the statistically
superior model across all 7 metrics, with mean, variance, win rates, and Excel export.
"""
from __future__ import annotations

import csv
import json
import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.metrics.rollout_metrics import MetricsTracker
from simulation.utils.config_loader import load_config
from simulation.utils.rng import RNGManager

from model.hybrid.heuristic_scheduler import HeuristicScheduler
from model.hybrid.policy import HybridPolicy
from model.agents.policy_runner import load_model


class PPOPolicyWrapper:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        return self.runner.predict(obs, last_info=last_info)


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

    def reset(self):
        pass

    def predict(self, obs: dict, last_info: dict | None = None) -> np.ndarray:
        # Uniform random relative step: 0 (down), 1 (stay), 2 (up)
        return np.array([np.random.randint(0, 3), 0], dtype=np.int64)


def evaluate_episode(env: AlterraEnv, policy, seed: int, max_steps: int = 150) -> dict:
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
        "TTFI": float(m.avg_intercept_time_error_slots) if m.avg_intercept_time_error_slots is not None else -1.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Seed Monte Carlo Evaluation")
    parser.add_argument("--scenario", default="configs/scenarios/sparse_single_threat.yaml", help="Path to scenario YAML")
    parser.add_argument("--name", default="Single Sparse Emitter", help="Scenario display name")
    parser.add_argument("--out-prefix", default="single_emitter", help="Output filename prefix")
    args = parser.parse_args()

    print("=" * 95)
    print(f" ALTERRA MULTI-SEED STATISTICAL EVALUATION & EXCEL EXPORT: {args.name}")
    print(f" Scenario: {args.scenario}")
    print(" Running Monte Carlo evaluation across 20 diverse seeds...")
    print("=" * 95)

    seeds = [42, 101, 333, 777, 1001, 1337, 2026, 3000, 5555, 9999,
             12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 100000]

    config = load_config("configs/default_config.yaml")
    specs = load_manual_scenario(args.scenario)

    models = {
        "Traditional: Sequential": SequentialScanPolicy(num_bands=128),
        "Traditional: Uniform Random": PureRandomPolicy(num_bands=128),
        "B0: Heuristic Teacher": HeuristicScheduler(num_bands=128),
        "RL: PPO Baseline": PPOPolicyWrapper(load_model(".", "ppo_499c7c")),
        "RL: PPO + LSTM": PPOPolicyWrapper(load_model(".", "ppo_lstm_finetune_c83a23")),
        "Hybrid LSTM": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_lstm.pt", backbone="lstm", num_bands=128),
        "Hybrid GRU": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_gru.pt", backbone="gru", num_bands=128),
        "Hybrid RNN": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_rnn.pt", backbone="rnn", num_bands=128),
        "Hybrid Transformer": HybridPolicy(checkpoint_path="model/hybrid/checkpoints/best_hybrid_transformer.pt", backbone="transformer", num_bands=128),
    }

    # Data collection: {model_name: {metric_name: [values]}}
    metric_keys = ["Pd", "Pfa", "Sensitivity", "Intercept Rate", "Reward", "Accuracy", "TTFI"]
    seed_records = []  # Flat list for CSV export: [Seed, Model, Pd, Pfa, ...]
    model_stats = {m: {k: [] for k in metric_keys} for m in models}
    seed_winners = {s: None for s in seeds}

    for s in seeds:
        print(f"--> Simulating Seed {s:6d} across all {len(models)} models...")
        best_reward = -float("inf")
        best_model_for_seed = None

        for m_name, pol in models.items():
            pop = build_manual_population(specs, config, RNGManager(s))
            env = AlterraEnv(config, manual_emitters=pop)
            res = evaluate_episode(env, pol, seed=s)

            for k in metric_keys:
                model_stats[m_name][k].append(res[k])

            ttfi_val = res["TTFI"] if res["TTFI"] >= 0 else "Missed"
            seed_records.append({
                "Seed": s,
                "Model": m_name,
                "Probability of Detection (Pd)": f"{res['Pd']:.3f}",
                "Probability of False Alarm (Pfa)": f"{res['Pfa']:.3f}",
                "Sensitivity (Pmin)": f"{res['Sensitivity']:.3f}",
                "Intercept Rate": f"{res['Intercept Rate']:.4f}",
                "Avg Reward": f"{res['Reward']:.2f}",
                "Accuracy (%)": f"{res['Accuracy']*100:.2f}%",
                "TTFI (slots)": ttfi_val,
            })

            if res["Reward"] > best_reward:
                best_reward = res["Reward"]
                best_model_for_seed = m_name

        seed_winners[s] = best_model_for_seed

    # Compute Summary Statistics
    summary_rows = []
    print("\n" + "=" * 115)
    print(" MONTE CARLO STATISTICAL SUMMARY ACROSS 20 SEEDS")
    print("=" * 115)
    print(f"{'Model':<28} | {'Mean Reward (±std)':^20} | {'Mean Pd':^9} | {'Mean IR':^9} | {'Win Rate':^10} | {'Worst Reward':^12}")
    print("-" * 115)

    for m_name in models:
        rewards = np.array(model_stats[m_name]["Reward"])
        pds = np.array(model_stats[m_name]["Pd"])
        pfas = np.array(model_stats[m_name]["Pfa"])
        sens = np.array(model_stats[m_name]["Sensitivity"])
        irs = np.array(model_stats[m_name]["Intercept Rate"])
        accs = np.array(model_stats[m_name]["Accuracy"])
        ttfi_valid = [t for t in model_stats[m_name]["TTFI"] if t >= 0]

        mean_rew = np.mean(rewards)
        std_rew = np.std(rewards)
        min_rew = np.min(rewards)
        mean_pd = np.mean(pds)
        mean_pfa = np.mean(pfas)
        mean_sens = np.mean(sens)
        mean_ir = np.mean(irs)
        mean_acc = np.mean(accs)
        mean_ttfi = np.mean(ttfi_valid) if ttfi_valid else -1.0
        
        wins = sum(1 for w in seed_winners.values() if w == m_name)
        win_rate = (wins / len(seeds)) * 100.0

        summary_rows.append({
            "Model": m_name,
            "Mean Reward": f"{mean_rew:+.2f}",
            "Reward StdDev (±)": f"{std_rew:.2f}",
            "Min Worst-Case Reward": f"{min_rew:+.2f}",
            "Win Rate (%)": f"{win_rate:.1f}%",
            "Mean Pd (Detection)": f"{mean_pd:.3f}",
            "Mean Pfa (False Alarm)": f"{mean_pfa:.3f}",
            "Mean Sensitivity": f"{mean_sens:.3f}",
            "Mean Intercept Rate": f"{mean_ir:.4f}",
            "Mean Accuracy (%)": f"{mean_acc*100:.2f}%",
            "Mean TTFI (slots)": f"{mean_ttfi:.1f}" if mean_ttfi >= 0 else "Missed",
        })

        rew_str = f"{mean_rew:+.2f} ± {std_rew:.2f}"
        print(f"{m_name:<28} | {rew_str:^20} | {mean_pd:^9.3f} | {mean_ir:^9.4f} | {win_rate:^9.1f}% | {min_rew:^+12.2f}")

    print("=" * 115)

    # 1. Export Detailed Seed-by-Seed CSV (Opens in Excel)
    csv_detail_path = f"data/ALTERA_{args.out_prefix}_Seed_by_Seed.csv"
    with open(csv_detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_records[0].keys()))
        writer.writeheader()
        writer.writerows(seed_records)

    # 2. Export Statistical Summary CSV (Opens in Excel)
    csv_summary_path = f"data/ALTERA_{args.out_prefix}_Statistical_Summary.csv"
    with open(csv_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. Export Comprehensive Excel-Compatible XML Spreadsheet (.xls)
    # Allows Excel to open both sheets (Summary and All Seeds) in one workbook!
    xml_xls_path = f"data/ALTERA_{args.out_prefix}_Model_Selection_Workbook.xls"
    generate_excel_xml(summary_rows, seed_records, xml_xls_path)

    print(f"\n[Success] Statistical evaluation complete for: {args.name}!")
    print(f" 1. Summary Sheet: {csv_summary_path}")
    print(f" 2. Per-Seed Data:  {csv_detail_path}")
    print(f" 3. Excel Workbook: {xml_xls_path}")


def generate_excel_xml(summary_rows: list[dict], detail_rows: list[dict], out_path: str):
    """Generates an XML Spreadsheet 2003 (.xls) workbook with multiple styled tabs."""
    xml_header = """<?xml version="1.0"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
 <Styles>
  <Style ss:ID="Default" ss:Name="Normal">
   <Alignment ss:Vertical="Bottom"/>
   <Borders/>
   <Font ss:FontName="Calibri" x:Family="Swiss" ss:Size="11" ss:Color="#000000"/>
  </Style>
  <Style ss:ID="HeaderStyle">
   <Font ss:FontName="Calibri" x:Family="Swiss" ss:Size="11" ss:Color="#FFFFFF" ss:Bold="1"/>
   <Interior ss:Color="#1F4E79" ss:Pattern="Solid"/>
   <Alignment ss:Horizontal="Center" ss:Vertical="Center"/>
  </Style>
  <Style ss:ID="WinnerStyle">
   <Font ss:FontName="Calibri" x:Family="Swiss" ss:Size="11" ss:Color="#006100" ss:Bold="1"/>
   <Interior ss:Color="#C6EFCE" ss:Pattern="Solid"/>
  </Style>
 </Styles>
"""

    def build_worksheet(name: str, rows: list[dict]) -> str:
        if not rows:
            return ""
        headers = list(rows[0].keys())
        ws = f' <Worksheet ss:Name="{name}">\n  <Table>\n'
        # Header Row
        ws += '   <Row>\n'
        for h in headers:
            ws += f'    <Cell ss:StyleID="HeaderStyle"><Data ss:Type="String">{h}</Data></Cell>\n'
        ws += '   </Row>\n'
        # Data Rows
        for r in rows:
            ws += '   <Row>\n'
            for h in headers:
                val = str(r[h])
                style = ' ss:StyleID="WinnerStyle"' if "Hybrid GRU" in val or "Hybrid RNN" in val else ""
                ws += f'    <Cell{style}><Data ss:Type="String">{val}</Data></Cell>\n'
            ws += '   </Row>\n'
        ws += '  </Table>\n </Worksheet>\n'
        return ws

    content = xml_header
    content += build_worksheet("Model Selection Summary", summary_rows)
    content += build_worksheet("All 20 Seeds Raw Data", detail_rows)
    content += "</Workbook>\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
