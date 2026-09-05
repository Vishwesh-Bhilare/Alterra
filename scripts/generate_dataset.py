#!/usr/bin/env python3
"""
Batch dataset generation: for --episodes episodes, saves ground-truth
occupancy matrix + mixed PDW stream + per-emitter metadata — training data
for MS-UNet1D / SEDCAM on the model side. No RL agent involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.emitters import build_manual_population, build_population, load_manual_scenario
from simulation.environment import SpectrumWorld, generate_episode_pdws
from simulation.utils.config_loader import load_config
from simulation.utils.rng import RNGManager


def main():
    parser = argparse.ArgumentParser(description="Batch-generate Alterra training data.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--scenario", default=None,
                         help="Use one fixed manual scenario for every episode instead of random populations")
    args = parser.parse_args()

    config = load_config(args.config)
    episode_length = args.episode_length or config.timing.episode_length_slots
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_rng = RNGManager(config.rng_seed)
    manifest = []
    manual_specs = load_manual_scenario(args.scenario) if args.scenario else None

    for ep in range(args.episodes):
        episode_seed = int(master_rng.spawn_named(f"dataset_episode_{ep}").integers(0, 2**31 - 1))
        episode_rng_manager = RNGManager(episode_seed)

        if manual_specs is not None:
            population = build_manual_population(manual_specs, config, episode_rng_manager)
        else:
            population = build_population(config, episode_rng_manager)

        for e in population:
            e.reset(episode_length)

        world = SpectrumWorld(population, config.spectrum.num_bands, episode_length)
        truth = world.full_truth_matrix()
        pdws = generate_episode_pdws(population, config.spectrum, config.timing.slot_duration_s)

        episode_dir = out_dir / f"episode_{ep:04d}"
        episode_dir.mkdir(exist_ok=True)

        np.save(episode_dir / "truth_matrix.npy", truth)

        with open(episode_dir / "pdws.jsonl", "w") as f:
            for p in pdws:
                f.write(json.dumps(p.to_dict()) + "\n")

        emitters_meta = [
            {
                "emitter_id": e.emitter_id, "kind": e.kind, "threat_level": e.threat_level,
                "pri_s": e.pri_s, "pw_s": e.pw_s, "doa_deg": e.doa_deg,
            }
            for e in population
        ]
        with open(episode_dir / "emitters.json", "w") as f:
            json.dump(emitters_meta, f, indent=2)

        manifest.append({
            "episode": ep,
            "dir": str(episode_dir),
            "num_emitters": len(population),
            "num_pdws": len(pdws),
            "episode_length_slots": episode_length,
            "num_bands": config.spectrum.num_bands,
        })

        print(f"episode {ep:04d}: {len(population)} emitters, {len(pdws)} pdws -> {episode_dir}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {args.episodes} episodes to {out_dir} (manifest.json)")


if __name__ == "__main__":
    main()
