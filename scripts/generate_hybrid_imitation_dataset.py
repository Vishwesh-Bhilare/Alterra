"""
Dataset Generator for ALTERA Hybrid Scheduler Imitation Learning.
Generates teacher trajectories from HeuristicScheduler across:
  1. Real TSRD pulse trains (turing_synthetic_radar_data)
  2. Single-emitter sparse threat scenarios
  3. Frequency-hopping agile scenarios
  4. Periodic scanning radar scenarios
  5. Dense congested multi-emitter scenarios

Splits trajectories strictly by scenario/seed (Train 70%, Val 15%, Test 15%)
and saves compressed .npz archives containing observation sequences, masks,
and teacher actions.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.emitters.scenario_builder import load_manual_scenario, build_manual_population
from simulation.emitters.tsrd_adapter import build_tsrd_population
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.utils.rng import RNGManager
from model.hybrid.heuristic_scheduler import HeuristicScheduler
from model.hybrid.doctrine import DoctrineMode


def get_tsrd_files() -> list[str]:
    """Finds all TSRD training .h5 files regardless of exact directory encoding."""
    matches = glob.glob("*turing*/archive/train/*.h5")
    return sorted(matches)


def run_episode(
    env: AlterraEnv,
    scheduler: HeuristicScheduler,
    seed: int,
    max_steps: int = 150,
) -> list[dict]:
    """Runs a single episode and records state-action-mask tuples."""
    obs, _ = env.reset(seed=seed)
    scheduler.reset()

    transitions = []
    info = {"band": env._current_band, "hit": False, "mean_power_norm": 0.0, "consecutive_hits": 0, "t": 0}

    mode_map = {
        DoctrineMode.EXPLORE: 0,
        DoctrineMode.INVESTIGATE: 1,
        DoctrineMode.TRACK: 2,
        DoctrineMode.RELOCATE: 3,
    }

    for _ in range(max_steps):
        target_band, dwell_idx, mode, mask = scheduler.select_action(obs, last_info=info)

        # Store step tuple
        transitions.append({
            "hit_miss_seq": np.array(obs["hit_miss_seq"], dtype=np.float32),
            "band_tracks": np.array(obs["band_tracks"], dtype=np.float32),
            "receiver": np.array(obs["receiver"], dtype=np.float32),
            "mode": int(mode_map[mode]),
            "mask": np.array(mask, dtype=bool),
            "target_band": int(target_band),
            "dwell_idx": int(dwell_idx),
        })

        dir_idx = scheduler.doctrine.select_relative_action(target_band)
        action = [dir_idx, dwell_idx]

        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    return transitions


def generate_dataset(
    output_dir: str = "data/imitation",
    tsrd_count: int = 100,
    scenario_count: int = 50,
    steps_per_episode: int = 180,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/default_config.yaml")
    num_bands = config.spectrum.num_bands
    scheduler = HeuristicScheduler(num_bands=num_bands)

    tsrd_files = get_tsrd_files()
    print(f"[Dataset] Found {len(tsrd_files)} TSRD pulse train files.")

    all_episodes = []

    # Category 1: Real TSRD Radar Pulse Trains (100 files)
    n_tsrd = min(tsrd_count, len(tsrd_files))
    print(f"[Dataset] Generating Category 1: TSRD Radar Pulse Trains ({n_tsrd} files)...")
    for i in range(n_tsrd):
        seed = 1000 + i
        fpath = tsrd_files[i]
        try:
            pop = build_tsrd_population(fpath, config, max_emitters=8, seed=seed)
            env = AlterraEnv(config, manual_emitters=pop)
            trajs = run_episode(env, scheduler, seed=seed, max_steps=steps_per_episode)
            all_episodes.append(trajs)
        except Exception as e:
            print(f"  Warning: error on {fpath}: {e}")

    # Category 2: Single-Emitter Sparse Threat Scenarios (50 episodes)
    print(f"[Dataset] Generating Category 2: Single Emitter Scenarios ({scenario_count} episodes)...")
    single_cfg = apply_overrides(config, num_emitters=1, episode_length_slots=2000)
    for i in range(scenario_count):
        seed = 2000 + i
        env = AlterraEnv(single_cfg)
        trajs = run_episode(env, scheduler, seed=seed, max_steps=steps_per_episode)
        all_episodes.append(trajs)

    # Category 3: Frequency-Hopping Agile Scenarios (50 episodes)
    print(f"[Dataset] Generating Category 3: Fast-Hopping Agile Scenarios ({scenario_count} episodes)...")
    if Path("configs/scenarios/fast_hopping_evasive.yaml").exists():
        specs = load_manual_scenario("configs/scenarios/fast_hopping_evasive.yaml")
        for i in range(scenario_count):
            seed = 3000 + i
            rng = RNGManager(seed)
            pop = build_manual_population(specs, config, rng)
            env = AlterraEnv(config, manual_emitters=pop)
            trajs = run_episode(env, scheduler, seed=seed, max_steps=steps_per_episode)
            all_episodes.append(trajs)

    # Category 4: Periodic Sweeper Radar Scenarios (50 episodes)
    print(f"[Dataset] Generating Category 4: Periodic Scan Sweeper Scenarios ({scenario_count} episodes)...")
    if Path("configs/scenarios/periodic_scan_focus.yaml").exists():
        specs = load_manual_scenario("configs/scenarios/periodic_scan_focus.yaml")
        for i in range(scenario_count):
            seed = 4000 + i
            rng = RNGManager(seed)
            pop = build_manual_population(specs, config, rng)
            env = AlterraEnv(config, manual_emitters=pop)
            trajs = run_episode(env, scheduler, seed=seed, max_steps=steps_per_episode)
            all_episodes.append(trajs)

    # Category 5: Congested Spectrum Multi-Emitter Mix (50 episodes)
    print(f"[Dataset] Generating Category 5: Dense Multi-Threat Mix ({scenario_count} episodes)...")
    for i in range(scenario_count):
        seed = 5000 + i
        env = AlterraEnv(config)
        trajs = run_episode(env, scheduler, seed=seed, max_steps=steps_per_episode)
        all_episodes.append(trajs)

    print(f"\n[Dataset] Total generated episodes: {len(all_episodes)}")

    # Split strictly by episode/seed: 70% Train, 15% Val, 15% Test
    rng_split = np.random.RandomState(42)
    indices = np.arange(len(all_episodes))
    rng_split.shuffle(indices)

    n_train = int(len(all_episodes) * 0.70)
    n_val = int(len(all_episodes) * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {
        "train": [all_episodes[i] for i in train_idx],
        "val": [all_episodes[i] for i in val_idx],
        "test": [all_episodes[i] for i in test_idx],
    }

    for split_name, ep_list in splits.items():
        flat_samples = [step for ep in ep_list for step in ep]
        seqs = np.stack([s["hit_miss_seq"] for s in flat_samples], axis=0)
        tracks = np.stack([s["band_tracks"] for s in flat_samples], axis=0)
        recs = np.stack([s["receiver"] for s in flat_samples], axis=0)
        modes = np.array([s["mode"] for s in flat_samples], dtype=np.int64)
        masks = np.stack([s["mask"] for s in flat_samples], axis=0)
        targets = np.array([s["target_band"] for s in flat_samples], dtype=np.int64)
        dwells = np.array([s["dwell_idx"] for s in flat_samples], dtype=np.int64)

        save_file = out_path / f"{split_name}.npz"
        np.savez_compressed(
            save_file,
            hit_miss_seq=seqs,
            band_tracks=tracks,
            receiver=recs,
            mode=modes,
            mask=masks,
            target_band=targets,
            dwell_idx=dwells,
        )
        print(f"  Saved {split_name.upper()} split: {len(flat_samples)} decision steps across {len(ep_list)} episodes -> {save_file}")

    print("[Dataset] Imitation learning dataset generation successfully completed!")


if __name__ == "__main__":
    generate_dataset()
