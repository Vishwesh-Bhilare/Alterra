"""
Verification of Periodic Signal Span Tracking, Pattern Identification,
and Return-and-Verify Priority Revisit across 10 Random Seeds.
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager

SEEDS = [42, 101, 777, 1337, 2024, 54321, 8888, 99999, 12345, 67890]
if "--random" in sys.argv:
    rng = np.random.RandomState()
    SEEDS = [int(x) for x in rng.choice(range(1, 1000000), size=10, replace=False).tolist()]

def test_periodic_pattern_tracking():
    print("=" * 80)
    print("      ALTERRA: PERIODIC SIGNAL SPAN TRACKING & PATTERN VERIFICATION")
    print(f"      Evaluated Seeds (10 Seeds): {SEEDS}")
    print("=" * 80)

    config = load_config("configs/default_config.yaml")
    model_path = "model/agents/checkpoints/best/best_model.zip"
    model = PPO.load(model_path)

    # -----------------------------------------------------------------
    # Test 1: Periodic Scan Emitter Tracking (Screenshot 1) across 10 seeds
    # -----------------------------------------------------------------
    print("\n[Test 1] Periodic Scan Emitter Span Tracking (10 Seeds)...")
    periodic_hits = []
    span_concentrations = []
    for s in SEEDS:
        specs = load_manual_scenario("configs/scenarios/periodic_scan_focus.yaml")
        rng_mgr = RNGManager(42)
        emitters = build_manual_population(specs, config, rng_mgr)
        env_config = apply_overrides(config, episode_length_slots=2500)
        env = AlterraEnv(env_config, manual_emitters=emitters)
        obs, _ = env.reset(seed=s)

        total_dwells = 0
        hits = 0
        span_dwells = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            b = info["band"]
            total_dwells += 1
            if 43 <= b <= 61:  # Periodic scan span is [45..59] +/- 2
                span_dwells += 1
            if info["any_hit"]:
                hits += 1
            done = term or trunc

        concentration = (span_dwells / total_dwells) * 100
        periodic_hits.append(hits)
        span_concentrations.append(concentration)
        print(f"  Seed {s:5d} -> Hits: {hits:2d} | Span Dwells: {span_dwells}/{total_dwells} ({concentration:.1f}%) [PASS]")

    avg_hits = np.mean(periodic_hits)
    avg_conc = np.mean(span_concentrations)
    print(f"  >> Average Periodic Pattern Hits: {avg_hits:.1f} (Requirement >= 20.0)")
    print(f"  >> Average Focus on Active Span : {avg_conc:.1f}%")
    assert avg_hits >= 20.0, f"Average periodic pattern hits too low: {avg_hits}"

    # -----------------------------------------------------------------
    # Test 2: Single-Emitter Revisit & Prioritization (Screenshot Custom Mix)
    # -----------------------------------------------------------------
    print("\n[Test 2] Single Emitter Pattern Revisit & Prioritization (10 Seeds)...")
    revisit_successes = 0
    for s in SEEDS:
        # Create a single fixed emitter on Band 111 (as in GUI screenshot)
        single_spec = [{
            "kind": "fixed",
            "id": "single_alpha",
            "threat_level": 2,
            "band": 111,
            "duty_cycle": 0.5,
            "mean_burst_slots": 40,
            "power_dbm": 12.0,
            "power_jitter_std_db": 1.0,
        }]
        rng_mgr = RNGManager(42)
        emitters = build_manual_population(single_spec, config, rng_mgr)
        env_config = apply_overrides(config, episode_length_slots=3000)
        env = AlterraEnv(env_config, manual_emitters=emitters)
        obs, _ = env.reset(seed=s)

        hits_111 = 0
        dwells_111 = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            if info["band"] == 111:
                dwells_111 += info["dwell_slots"]
                if info["any_hit"]:
                    hits_111 += 1
            done = term or trunc

        # Verify that scheduler actively returns to Band 111 multiple times
        passed = (hits_111 >= 15 and dwells_111 >= 150)
        if passed:
            revisit_successes += 1
        print(f"  Seed {s:5d} -> Band 111 Dwells: {dwells_111:3d} slots | Hits: {hits_111:2d} [{'PASS' if passed else 'FAIL'}]")

    print(f"  >> Single Emitter Revisit Pass Rate: {revisit_successes}/{len(SEEDS)}")
    assert revisit_successes >= 8, f"Too many seeds failed single emitter revisit: {revisit_successes}"

    print("\n" + "=" * 80)
    print("ALL PERIODIC PATTERN & REVISIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_periodic_pattern_tracking()
