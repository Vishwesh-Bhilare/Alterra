"""
Train the PPO Smart Scan Scheduler against AlterraEnv, with periodic
checkpointing and eval-driven best-model saving. Uses SubprocVecEnv so
each of --n-envs training environments runs in its own process/core. Run
from alterra/ repo root:

  python -m model.agents.train_ppo --timesteps 2000000 --ent-coef 0.01 --n-envs 8
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.rnn_policy import PPORNNExtractor


def make_env(config_path: str):
    def _init():
        config = load_config(config_path)
        return Monitor(AlterraEnv(config))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--out", default="model/agents/checkpoints/best/best_model.zip")
    parser.add_argument("--tensorboard-log", default="model/agents/tb_logs")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
    vec_env = make_vec_env(make_env(args.config), n_envs=args.n_envs, vec_env_cls=vec_env_cls)
    eval_env = make_vec_env(make_env(args.config), n_envs=1)  # single env — no need to subprocess

    ckpt_dir = os.path.dirname(args.out) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    best_dir = os.path.join(ckpt_dir, "best") if not ckpt_dir.endswith("best") else ckpt_dir
    os.makedirs(best_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=PPORNNExtractor,
        features_extractor_kwargs=dict(rnn_hidden_dim=args.rnn_hidden_dim, features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    if args.resume_from:
        model = PPO.load(args.resume_from, env=vec_env, tensorboard_log=args.tensorboard_log)
        model.ent_coef = args.ent_coef
        print(f"Resumed from {args.resume_from}, ent_coef set to {args.ent_coef}")
    else:
        model = PPO(
            "MultiInputPolicy", vec_env, verbose=1,
            tensorboard_log=args.tensorboard_log, ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_scheduler_ckpt",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(ckpt_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=args.resume_from is None,
    )

    model.save(args.out)
    print(f"Saved final PPO scheduler to {args.out}")
    print(f"Best model (by eval reward) saved under {best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
