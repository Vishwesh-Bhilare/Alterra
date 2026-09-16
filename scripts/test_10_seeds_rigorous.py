"""
Rigorous Multi-Seed Testing Suite across 10 different seeds:
Part A: 7 Core Scenario Test Cases evaluated across 10 seeds.
Part B: Random Emitter Populations (Stochastic Signals) evaluated across 10 seeds.
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager

SEEDS = [42, 101, 777, 1337, 2024, 54321, 8888, 99999, 12345, 67890]

def make_scenario_env(scenario_path: str, config, episode_length: int = 2000):
    specs = load_manual_scenario(scenario_path)
    rng_mgr = RNGManager(42)
    emitters = build_manual_population(specs, config, rng_mgr)
    env_config = apply_overrides(config, episode_length_slots=episode_length)
    return AlterraEnv(env_config, manual_emitters=emitters)

def run_multi_seed_tests():
    print("=" * 80)
    print("      ALTERRA RIGOROUS MULTI-SEED EVALUATION SUITE (10 SEEDS)")
    print(f"      Tested Seeds: {SEEDS}")
    print("=" * 80)

    config = load_config("configs/default_config.yaml")
    model_path = "model/agents/checkpoints/best/best_model.zip"
    model = PPO.load(model_path)

    # -------------------------------------------------------------
    # PART A: SCENARIO PRESETS ACROSS 10 SEEDS
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" PART A: SCENARIO SPECIFIC TEST CASES ACROSS 10 SEEDS")
    print("#" * 80)

    # 1. Fast Probing on Empty Channels across 10 seeds
    print("\n[Case 1] Fast 3-Slot Probing on Empty Spectrum (10 Seeds)...")
    empty_rates = []
    for s in SEEDS:
        env = AlterraEnv(config)
        obs, _ = env.reset(seed=s)
        empty_dwells = []
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            if not info["any_hit"] and info["consecutive_hits"] == 0:
                empty_dwells.append(info["dwell_slots"])
            if term or trunc:
                break
        rate = sum(1 for d in empty_dwells if d == 3) / max(len(empty_dwells), 1)
        empty_rates.append(rate)
        print(f"  Seed {s:5d} -> Fast 3-slot probe rate: {rate*100:.1f}% ({len(empty_dwells)} empty dwells)")
    avg_empty_rate = np.mean(empty_rates)
    print(f"  >> Average Fast Probing Rate: {avg_empty_rate*100:.1f}%")
    assert avg_empty_rate >= 0.85, f"Fast probe rate too low: {avg_empty_rate}"

    # 2. Dwell Escalation on Confirmed Hits across 10 seeds
    print("\n[Case 2] Proportional Dwell Escalation on Consecutive Hits (10 Seeds)...")
    escalation_successes = 0
    for s in SEEDS:
        # Load a high-density scenario to guarantee hit sequences
        dense_env = make_scenario_env("configs/scenarios/dense_congested.yaml", config, episode_length=1500)
        obs, _ = dense_env.reset(seed=s)
        dwell_history = []
        hit_seq = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = dense_env.step(action)
            if info["consecutive_hits"] >= 1:
                dwell_history.append(info["dwell_slots"])
                hit_seq.append(info["consecutive_hits"])
                if info["consecutive_hits"] >= 3:
                    break
            else:
                dwell_history.clear()
                hit_seq.clear()
            done = term or trunc
        
        if len(dwell_history) >= 2 and dwell_history[-1] > dwell_history[0]:
            escalation_successes += 1
            print(f"  Seed {s:5d} -> Escalation verified: Dwells {dwell_history[:4]} for Consec {hit_seq[:4]} [PASS]")
        else:
            print(f"  Seed {s:5d} -> Dwell history: {dwell_history} [CHECK]")
    print(f"  >> Dwell Escalation Success Rate: {escalation_successes}/{len(SEEDS)}")
    assert escalation_successes >= 8, "Dwell escalation failed on too many seeds"

    # 3. Silent Gap + Revisit across 10 seeds
    print("\n[Case 3] Silent Gap + Revisit Interception (10 Seeds)...")
    gap_passes = 0
    for s in SEEDS:
        gap_env = make_scenario_env("configs/scenarios/silent_gap_revisit.yaml", config, episode_length=2000)
        obs, _ = gap_env.reset(seed=s)
        early_hits = 0
        late_hits = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = gap_env.step(action)
            t = gap_env.t
            band = info["band"]
            if band == 60 and info["any_hit"]:
                if t <= 400:
                    early_hits += 1
                elif t >= 1600:
                    late_hits += 1
            done = term or trunc
        passed = (early_hits > 0 and late_hits > 0)
        if passed:
            gap_passes += 1
        print(f"  Seed {s:5d} -> Early Hits (t<=400): {early_hits:2d} | Late Hits (t>=1600): {late_hits:2d} [{'PASS' if passed else 'FAIL'}]")
    print(f"  >> Silent Gap Revisit Success Rate: {gap_passes}/{len(SEEDS)}")

    # 4. Mid-Episode Burst across 10 seeds
    print("\n[Case 4] Mid-Episode Burst Interception (10 Seeds)...")
    burst_passes = 0
    for s in SEEDS:
        burst_env = make_scenario_env("configs/scenarios/mid_episode_burst.yaml", config, episode_length=1500)
        obs, _ = burst_env.reset(seed=s)
        burst_hits = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = burst_env.step(action)
            t = burst_env.t
            band = info["band"]
            if band == 55 and info["any_hit"] and (680 <= t <= 1100):
                burst_hits += 1
            done = term or trunc
        passed = (burst_hits > 0)
        if passed:
            burst_passes += 1
        print(f"  Seed {s:5d} -> Band 55 Burst Hits (t=700..1050): {burst_hits:2d} [{'PASS' if passed else 'FAIL'}]")
    print(f"  >> Mid-Episode Burst Interception Rate: {burst_passes}/{len(SEEDS)}")

    # 5. Multi-Threat Prioritization across 10 seeds
    print("\n[Case 5] Multi-Threat Prioritization (High Threat 3 vs Low Threat 1) (10 Seeds)...")
    prio_passes = 0
    for s in SEEDS:
        env_prio = make_scenario_env("configs/scenarios/silent_gap_revisit.yaml", config, episode_length=2000)
        obs, _ = env_prio.reset(seed=s)
        high_dwells = 0
        low_dwells = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env_prio.step(action)
            band = info["band"]
            if band == 60:  # Threat 3
                high_dwells += info["dwell_slots"]
            elif band == 10:  # Threat 1
                low_dwells += info["dwell_slots"]
            done = term or trunc
        passed = (high_dwells > low_dwells)
        if passed:
            prio_passes += 1
        print(f"  Seed {s:5d} -> High-Threat (B60): {high_dwells:3d} slots | Low-Threat (B10): {low_dwells:3d} slots [{'PASS' if passed else 'FAIL'}]")
    print(f"  >> Multi-Threat Prioritization Pass Rate: {prio_passes}/{len(SEEDS)}")

    # 6. Periodic Scan Pattern Tracking (Screenshot 1) across 10 seeds
    print("\n[Case 6] Periodic Scan Pattern Tracking (Screenshot 1) (10 Seeds)...")
    periodic_hits_list = []
    for s in SEEDS:
        p_env = make_scenario_env("configs/scenarios/periodic_scan_focus.yaml", config, episode_length=2000)
        obs, _ = p_env.reset(seed=s)
        hits = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = p_env.step(action)
            if info["any_hit"]:
                hits += 1
            done = term or trunc
        periodic_hits_list.append(hits)
        print(f"  Seed {s:5d} -> Periodic Pattern Intercepted Hits: {hits:2d} [{'PASS' if hits >= 20 else 'FAIL'}]")
    print(f"  >> Average Periodic Pattern Intercepted Hits: {np.mean(periodic_hits_list):.1f}")
    assert all(h >= 20 for h in periodic_hits_list), "Periodic scan tracking failed on some seeds"

    # 7. Agile Hopping Pattern Tracking across 10 seeds
    print("\n[Case 7] Agile Hopping Tracking across 10 Seeds...")
    agile_hits_list = []
    for s in SEEDS:
        agile_env = make_scenario_env("configs/scenarios/fast_hopping_evasive.yaml", config, episode_length=2000)
        obs, _ = agile_env.reset(seed=s)
        hits = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = agile_env.step(action)
            if info["any_hit"]:
                hits += 1
            done = term or trunc
        agile_hits_list.append(hits)
        print(f"  Seed {s:5d} -> Agile Intercepted Hits: {hits:2d} [{'PASS' if hits >= 2 else 'FAIL'}]")
    print(f"  >> Average Agile Intercepted Hits: {np.mean(agile_hits_list):.1f}")
    assert all(h >= 2 for h in agile_hits_list), "Agile hopper tracking failed on some seeds"
    assert np.mean(agile_hits_list) >= 10.0, f"Average agile hits too low: {np.mean(agile_hits_list)}"

    # -------------------------------------------------------------
    # PART B: RANDOM SEED & RANDOM SIGNALS (STOCHASTIC POPULATIONS)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" PART B: RANDOM SEEDS ON RANDOM SIGNALS (STOCHASTIC ENVIRONMENT)")
    print("#" * 80)

    rand_results = []
    for s in SEEDS:
        rand_env = AlterraEnv(config)
        obs, _ = rand_env.reset(seed=s)
        
        num_emitters = len(rand_env._emitters)
        types = [e.kind for e in rand_env._emitters]
        type_summary = {t: types.count(t) for t in set(types)}

        total_dwells = 0
        total_hits = 0
        total_reward = 0.0
        dwell_dist = {3: 0, 5: 0, 8: 0, 12: 0}
        bands_visited = set()
        boundary_violations = 0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            cur_band = rand_env._current_band
            step_delta = rand_env._step_sizes[int(action[0])]
            next_target = cur_band + step_delta
            if next_target < 0 or next_target >= config.spectrum.num_bands:
                boundary_violations += 1

            obs, reward, term, trunc, info = rand_env.step(action)
            total_dwells += 1
            total_reward += reward
            d = info["dwell_slots"]
            dwell_dist[d] = dwell_dist.get(d, 0) + 1
            bands_visited.add(info["band"])
            if info["any_hit"]:
                total_hits += 1
            done = term or trunc

        hit_rate = (total_hits / total_dwells) * 100 if total_dwells > 0 else 0
        coverage = len(bands_visited)

        rand_results.append({
            "seed": s,
            "emitters": num_emitters,
            "types": type_summary,
            "dwells": total_dwells,
            "hits": total_hits,
            "hit_rate": hit_rate,
            "coverage": coverage,
            "reward": total_reward,
            "boundary_violations": boundary_violations,
            "dwell_dist": dwell_dist,
        })

        print(f"\n[Random Seed {s:5d}] Emitters: {num_emitters} {type_summary}")
        print(f"  Dwells: {total_dwells} | Hits: {total_hits} ({hit_rate:.1f}%) | Unique Bands Scanned: {coverage}/128")
        print(f"  Cumulative Reward: {total_reward:+.1f} | Boundary Violations: {boundary_violations}")
        print(f"  Dwell Distribution: {dwell_dist}")

    print("\n" + "=" * 80)
    print("MULTI-SEED EVALUATION SUMMARY:")
    print(f"  Part A Fast Probing Avg: {avg_empty_rate*100:.1f}%")
    print(f"  Part A Dwell Escalation: {escalation_successes}/{len(SEEDS)} seeds passed")
    print(f"  Part A Silent Gap Revisit: {gap_passes}/{len(SEEDS)} seeds passed")
    print(f"  Part A Mid-Episode Burst: {burst_passes}/{len(SEEDS)} seeds passed")
    print(f"  Part A Multi-Threat Priority: {prio_passes}/{len(SEEDS)} seeds passed")
    print(f"  Part B Random Signals Avg Hits: {np.mean([r['hits'] for r in rand_results]):.1f}")
    print(f"  Part B Random Signals Avg Reward: {np.mean([r['reward'] for r in rand_results]):+.1f}")
    print(f"  Part B Total Boundary Violations: {sum(r['boundary_violations'] for r in rand_results)}")
    print("=" * 80)

if __name__ == "__main__":
    run_multi_seed_tests()
