"""
Train Alterra PPO Scheduler with 1D-CNN Spectral Feature Extractor.
Saves all checkpoints to model/agents/checkpoints/spectral_cnn/ without modifying existing models.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from model.agents.spectral_extractor import Spectral1DFeaturesExtractor
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import IntRange, load_config


def linear_schedule(initial_value: float, final_value: float = 2e-5) -> Callable[[float], float]:
    def _schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return _schedule


def make_spectral_env(config_path: str):
    def _init():
        config = load_config(config_path)
        # Curriculum: 1 to 8 emitters with agile hoppers, sweepers, and fixed emitters
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
    parser = argparse.ArgumentParser(description="Train 1D-CNN Spectral PPO Model")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=12_000)
    parser.add_argument("--out-dir", default="model/agents/checkpoints/spectral_cnn")
    args = parser.parse_args()

    print("=" * 78)
    print("      ALTERRA: 1D-CNN SPECTRAL FEATURE EXTRACTOR PPO TRAINING")
    print("=" * 78)
    print(f"Architecture    : 1D-CNN Spectral (128x7) + Dense Telemetry (8)")
    print(f"Total timesteps : {args.timesteps:,}")
    print(f"LR Schedule     : {args.lr} -> {args.min_lr}")
    print(f"Entropy coef    : {args.ent_coef}")
    print(f"Parallel envs   : {args.n_envs}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)
    best_dir = os.path.join(args.out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
    vec_env = make_vec_env(
        make_spectral_env(args.config),
        n_envs=args.n_envs,
        vec_env_cls=vec_env_cls,
    )
    eval_env = make_vec_env(
        make_spectral_env(args.config),
        n_envs=1,
    )

    policy_kwargs = dict(
        features_extractor_class=Spectral1DFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    schedule_fn = linear_schedule(args.lr, args.min_lr)

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=schedule_fn,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(args.out_dir, "tb_logs"),
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(args.out_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=6,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(24_000 // args.n_envs, 1),
        save_path=args.out_dir,
        name_prefix="spectral_ppo_ckpt",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    final_path = os.path.join(args.out_dir, "spectral_ppo_final.zip")
    model.save(final_path)
    print(f"\n[SUCCESS] Final 1D-CNN Spectral model saved to: {final_path}")
    print(f"[SUCCESS] Best evaluated checkpoint saved to: {best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
