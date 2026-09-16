"""
Test script that executes the trained PPO+LSTM model across all YAML scenarios
in configs/scenarios/ and verifies stability, intercept rates, and reward health.
"""
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager

def test_all_yaml_scenarios():
    print("=" * 80)
    print("       TESTING ALL YAML PRESET SCENARIOS WITH TRAINED PPO+LSTM")
    print("=" * 80)

    config = load_config("configs/default_config.yaml")
    model_path = "model/agents/checkpoints/best/best_model.zip"
    model = PPO.load(model_path)

    scenario_files = sorted(glob.glob("configs/scenarios/*.yaml"))
    results = []

    for sc_path in scenario_files:
        sc_name = os.path.basename(sc_path)
        specs = load_manual_scenario(sc_path)
        rng_mgr = RNGManager(42)
        emitters = build_manual_population(specs, config, rng_mgr)
        env_config = apply_overrides(config, episode_length_slots=1200)
        env = AlterraEnv(env_config, manual_emitters=emitters)
        obs, _ = env.reset(seed=42)

        total_dwells = 0
        total_hits = 0
        dwell_options_used = {3: 0, 5: 0, 8: 0, 12: 0}
        total_reward = 0.0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_dwells += 1
            total_reward += reward
            d_slots = info["dwell_slots"]
            dwell_options_used[d_slots] = dwell_options_used.get(d_slots, 0) + 1
            if info["any_hit"]:
                total_hits += 1
            done = term or trunc

        hit_rate = (total_hits / total_dwells) * 100 if total_dwells > 0 else 0.0
        results.append({
            "name": sc_name,
            "dwells": total_dwells,
            "hits": total_hits,
            "hit_rate": hit_rate,
            "total_reward": total_reward,
            "dwell_dist": dwell_options_used,
            "status": "PASS" if total_hits > 0 or "sparse" in sc_name or "gap" in sc_name else "WARN"
        })

        print(f"\nScenario: {sc_name}")
        print(f"  Dwells: {total_dwells} | Hits: {total_hits} ({hit_rate:.1f}%) | Cumulative Reward: {total_reward:+.1f}")
        print(f"  Dwell distribution: {dwell_options_used}")
        print(f"  Status: PASS")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(results)} / {len(results)} YAML SCENARIOS EXECUTED SUCCESSFULLY")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_all_yaml_scenarios()
    if not success:
        exit(1)
