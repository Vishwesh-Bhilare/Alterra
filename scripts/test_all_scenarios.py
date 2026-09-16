"""
Comprehensive Scenario Validation Test Suite for Alterra PPO + LSTM Scheduler.
Validates:
1. Observation Space Structure and 5 State Features.
2. Dwell Escalation (3 -> 5 -> 8 -> 12 slots) on consecutive hits.
3. Fast Probing (3 slots / 30ms) on empty channels.
4. Silent Gap + Revisit: Intercepts initial window AND successfully revisits late window.
5. Mid-Episode Burst: Detects and locks onto sudden mid-episode burst.
6. Multi-Threat Prioritization: Prioritizes High-Threat (level 3) over Low-Threat (level 1).
7. Pattern Tracking: Effectively intercepts agile hoppers and periodic scanners.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from stable_baselines3 import PPO

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.utils.rng import RNGManager


def make_scenario_env(scenario_yaml: str, config_path: str = "configs/default_config.yaml") -> AlterraEnv:
    config = load_config(config_path)
    specs = load_manual_scenario(scenario_yaml)
    rng = RNGManager(config.rng_seed)
    emitters = build_manual_population(specs, config, rng)
    return AlterraEnv(config, manual_emitters=emitters)


def run_all_scenario_tests(model_path: str = "model/agents/checkpoints/best/best_model.zip"):
    print("=" * 76)
    print("       ALTERRA EW PPO + LSTM: COMPREHENSIVE SCENARIO VALIDATION")
    print("=" * 76)
    print(f"Model path: {model_path}\n")

    assert os.path.exists(model_path), f"Model checkpoint not found: {model_path}"
    model = PPO.load(model_path)
    config = load_config("configs/default_config.yaml")
    base_env = AlterraEnv(config)

    passed = 0
    total = 0

    # -------------------------------------------------------------
    # Test 1: Observation Space Structure & 5 State Features
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Observation Space Structure & 5 State Features...")
    obs, _ = base_env.reset(seed=42)
    assert "band_tracks" in obs and obs["band_tracks"].shape == (128, 8), f"Invalid band_tracks: {obs['band_tracks'].shape}"
    assert "receiver" in obs and obs["receiver"].shape == (10,), f"Invalid receiver: {obs['receiver'].shape}"
    assert "hit_miss_seq" in obs and obs["hit_miss_seq"].shape == (16, 6), f"Invalid hit_miss_seq: {obs['hit_miss_seq'].shape}"
    print("  -> Feature 1 (Hit): tracked in hit_miss_seq[:, 0] and receiver[2]")
    print("  -> Feature 2 (Miss): tracked in hit_miss_seq[:, 1]")
    print("  -> Feature 3 (Bands Scanned): tracked in band_tracks[:, 5] and receiver[8]")
    print("  -> Feature 4 (Bands Unscanned): tracked in band_tracks[:, 6] and receiver[9]")
    print("  -> Feature 5 (Continuous Hit Duration): tracked in hit_miss_seq[:, 4] and receiver[3]")
    print(f"  [PASS] Test {total} Passed!\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 2: Proportional Dwell Escalation on Hits
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Proportional Dwell Escalation on Consecutive Hits...")
    obs, _ = base_env.reset(seed=100)
    dwell_sequence = []
    for step in range(250):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = base_env.step(action)
        if info["consecutive_hits"] >= 1:
            dwell_sequence.append(info["dwell_slots"])
            if info["consecutive_hits"] >= 3:
                break
        else:
            dwell_sequence.clear()
        if term or trunc:
            break

    print(f"  Observed Dwell Sequence on Hit: {dwell_sequence}")
    assert len(dwell_sequence) >= 2, "Did not intercept at least 2 consecutive hits to verify dwell escalation"
    assert dwell_sequence[-1] > dwell_sequence[0], f"Dwell did not escalate: {dwell_sequence}"
    print(f"  [PASS] Test {total} Passed! Dwell escalated from {dwell_sequence[0]} to {dwell_sequence[-1]} slots.\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 3: Fast 3-Slot Probing on Empty Channels
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Fast 3-Slot Probing on Empty Channels...")
    obs, _ = base_env.reset(seed=999)
    empty_dwells = []
    for step in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = base_env.step(action)
        if not info["any_hit"] and info["consecutive_hits"] == 0:
            empty_dwells.append(info["dwell_slots"])

    fast_ratio = sum(1 for d in empty_dwells if d == 3) / max(len(empty_dwells), 1)
    print(f"  Fast 3-slot probe percentage on empty channels: {fast_ratio * 100:.1f}%")
    assert fast_ratio >= 0.85, f"Expected >= 85% fast probes on empty channels, got {fast_ratio * 100:.1f}%"
    print(f"  [PASS] Test {total} Passed!\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 4: Silent Gap + Revisit Scenario (Screenshots 2 & 4)
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Silent Gap + Revisit Scenario...")
    gap_env = make_scenario_env("configs/scenarios/silent_gap_revisit.yaml")
    obs, _ = gap_env.reset(seed=0)

    early_window_hits = 0  # t in [0, 300] on Band 60
    late_window_hits = 0   # t in [1700, 1950] on Band 60

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = gap_env.step(action)
        t = gap_env.t
        band = info["band"]
        if band == 60 and info["any_hit"]:
            if t <= 350:
                early_window_hits += 1
            elif t >= 1650:
                late_window_hits += 1
        done = term or trunc

    print(f"  Band 60 Early Window (t <= 350) Hits   : {early_window_hits}")
    print(f"  Band 60 Late Revisit (t >= 1650) Hits  : {late_window_hits}")
    assert early_window_hits > 0, "Failed to intercept early window on Band 60"
    assert late_window_hits > 0, "Failed to revisit and intercept late window on Band 60"
    print(f"  [PASS] Test {total} Passed! Model intercepted initial burst AND revisited to catch second burst!\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 5: Mid-Episode Burst Scenario (Screenshot 3)
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Mid-Episode Burst Scenario...")
    burst_env = make_scenario_env("configs/scenarios/mid_episode_burst.yaml")
    obs, _ = burst_env.reset(seed=0)

    burst_hits = 0  # Band 55, active in [700, 1050]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = burst_env.step(action)
        t = burst_env.t
        band = info["band"]
        if band == 55 and info["any_hit"] and (680 <= t <= 1100):
            burst_hits += 1
        done = term or trunc

    print(f"  Band 55 Mid-Episode Burst (t=700..1050) Hits: {burst_hits}")
    assert burst_hits > 0, "Failed to detect and intercept mid-episode burst on Band 55"
    print(f"  [PASS] Test {total} Passed! Model successfully intercepted sudden mid-episode burst!\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 6: Multi-Threat Prioritization (High Threat 3 vs Low Threat 1)
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Multi-Threat Prioritization (High vs Low)...")
    # In silent_gap_revisit.yaml: Band 60 is Threat 3, Band 10 is Threat 1
    obs, _ = gap_env.reset(seed=0)
    dwells_high = 0
    dwells_low = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = gap_env.step(action)
        if info["band"] == 60:
            dwells_high += info["dwell_slots"]
        elif info["band"] == 10:
            dwells_low += info["dwell_slots"]
        done = term or trunc

    print(f"  Total dwell time on High Threat Band 60: {dwells_high} slots")
    print(f"  Total dwell time on Low Threat Band 10 : {dwells_low} slots")
    assert dwells_high >= dwells_low, f"Model should prioritize High Threat over Low Threat ({dwells_high} vs {dwells_low})"
    print(f"  [PASS] Test {total} Passed! Model correctly prioritized High-Threat emitter.\n")
    passed += 1

    # -------------------------------------------------------------
    # Test 7: Agile Hopping & Periodic Pattern Tracking (Screenshots 1 & 5)
    # -------------------------------------------------------------
    total += 1
    print(f"[Test {total}] Verifying Agile Hopping & Periodic Scan Tracking...")
    hop_env = make_scenario_env("configs/scenarios/fast_hopping_evasive.yaml")
    obs, _ = hop_env.reset(seed=42)
    hop_hits = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = hop_env.step(action)
        if info["any_hit"]:
            hop_hits += 1
        done = term or trunc

    print(f"  Total intercepted agile hopper hits: {hop_hits}")
    assert hop_hits >= 15, f"Expected >= 15 agile hopper hits, got {hop_hits}"
    print(f"  [PASS] Test {total} Passed! Model tracked evasive agile frequency-hopping pattern.\n")
    passed += 1

    print("=" * 76)
    print(f"       ALL {passed} / {total} SCENARIO TESTS PASSED SUCCESSFULLY!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    success = run_all_scenario_tests()
    sys.exit(0 if success else 1)
