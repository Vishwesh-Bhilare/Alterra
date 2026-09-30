"""
Phase 2 Deep Fine-Tuning of PPO Smart Scan Scheduler.
Features:
- Linear learning rate decay schedule (1e-4 -> 2e-5).
- Agile frequency-hopper and sparse-to-dense curriculum environment (1-8 emitters).
- Evaluation-driven checkpointing preserving top performance.
- Resumes from best_model.zip.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import IntRange, load_config


def linear_schedule(initial_value: float, final_value: float = 2e-5) -> Callable[[float], float]:
    """Linear learning rate decay schedule for Stable-Baselines3."""
    def _schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return _schedule


def make_curriculum_env(config_path: str):
    def _init():
        config = load_config(config_path)
        # Curriculum: 1 to 8 emitters with high representation of agile frequency-hoppers
        config = dataclasses.replace(
            config,
            emitters=dataclasses.replace(
                config.emitters,
                population=dataclasses.replace(
                    config.emitters.population,
                    total_count_range=IntRange(1, 8),
                    class_weights={"fixed": 0.40, "agile": 0.45, "periodic_scan": 0.15},
                ),
            ),
        )
        return Monitor(AlterraEnv(config))
    return _init


def main():
    parser = argparse.ArgumentParser(description="Phase 2 PPO Deep Fine-Tuning")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--base-model", default="model/agents/checkpoints/best/best_model.zip")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--ent-coef", type=float, default=0.012)
    parser.add_argument("--eval-freq", type=int, default=15_000)
    parser.add_argument("--out-dir", default="model/agents/checkpoints")
    args = parser.parse_args()

    print("=" * 78)
    print("           ALTERRA PPO: PHASE 2 DEEP FINE-TUNING")
    print("=" * 78)
    print(f"Base model       : {args.base_model}")
    print(f"Total timesteps  : {args.timesteps:,}")
    print(f"LR schedule      : {args.lr} -> {args.min_lr} (Linear Decay)")
    print(f"Entropy coef     : {args.ent_coef}")
    print(f"Parallel envs    : {args.n_envs}")
    print("=" * 78)

    best_dir = os.path.join(args.out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
    vec_env = make_vec_env(
        make_curriculum_env(args.config),
        n_envs=args.n_envs,
        vec_env_cls=vec_env_cls,
    )
    eval_env = make_vec_env(
        make_curriculum_env(args.config),
        n_envs=1,
    )

    # Load model and configure lr schedule & entropy coefficient
    schedule_fn = linear_schedule(args.lr, args.min_lr)
    model = PPO.load(
        args.base_model,
        env=vec_env,
        tensorboard_log=os.path.join(args.out_dir, "tb_logs"),
    )
    model.learning_rate = schedule_fn
    model.ent_coef = args.ent_coef

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(args.out_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=6,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(30_000 // args.n_envs, 1),
        save_path=args.out_dir,
        name_prefix="ppo_phase2_ckpt",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,
    )

    final_path = os.path.join(args.out_dir, "ppo_phase2_final.zip")
    model.save(final_path)
    print(f"\n[SUCCESS] Phase 2 fine-tuned model saved to: {final_path}")
    print(f"[SUCCESS] Best evaluated checkpoint saved to: {best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
