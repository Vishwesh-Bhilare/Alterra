"""
Train the hybrid doctrine+MaskablePPO scheduler: AlterraEnv's
action_masks() (simulation/environment/doctrine.py) constrains which
bands are legal each step per the EXPLORE/INVESTIGATE/TRACK/RELOCATE
doctrine ported from heuristic_scheduler.py; MaskablePPO chooses the best
band among whatever the doctrine currently allows. Dwell stays fully
rule-based -- see doctrine.py's docstring for the one documented caveat.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config


def mask_fn(env: AlterraEnv):
    return env.action_masks()


def build_env(config_path: str, seed: int | None = None) -> Monitor:
    config = load_config(config_path)
    env = AlterraEnv(config, enable_doctrine=True)
    if seed is not None:
        env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)
    return Monitor(env)


def make_env(config_path: str, seed: int):
    def _init():
        torch.set_num_threads(1)
        return build_env(config_path, seed)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Alterra hybrid doctrine+MaskablePPO scheduler.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--n-envs", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-name", default="hybrid_maskable_v1")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", default="model/agents/checkpoints")
    parser.add_argument("--tb-log-dir", default="model/agents/tb_logs")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    args = parser.parse_args()

    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.n_envs > 1:
        env = SubprocVecEnv(
            [make_env(args.config, seed=10_000 + i) for i in range(args.n_envs)],
            start_method="forkserver" if os.name == "posix" else None,
        )
    else:
        env = DummyVecEnv([lambda: build_env(args.config, seed=10_000)])

    eval_env = DummyVecEnv([lambda: build_env(args.config, seed=100_000)])

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir = checkpoint_dir / "best"
    best_dir.mkdir(exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="hybrid_ckpt",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(checkpoint_dir / "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        ),
    ]

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tb_log_dir,
        device=resolved_device,
        verbose=1,
    )

    print(f"[hybrid] n_envs={args.n_envs} device={resolved_device}")
    print("[hybrid] doctrine controls band legality + dwell; MaskablePPO picks band within mask")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, tb_log_name=args.run_name)
    finally:
        env.close()
        eval_env.close()

    final_path = checkpoint_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n[hybrid] Training complete: {final_path}")
    print(f"[hybrid] Best checkpoint: {best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
