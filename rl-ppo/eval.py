#!/usr/bin/env python3
# eval.py (Last update 03/12/2025)

import argparse
import os
from datetime import datetime
import itertools
import torch
from gymnasium.wrappers import RecordVideo

from env.wrappers import make_atari_env
from agents.ppo import PPOAgent, PPOConfig
from utils.eval_utils import evaluate


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _build_cfg_from_ckpt(ckpt: dict, device: str) -> PPOConfig:
    """
    Support both layouts:
      - New: ckpt["cfg"] contains PPOConfig-like fields
      - Old: flat keys like "obs_shape", "n_actions", "gamma", ...
    """
    cfg_dict = ckpt.get("cfg")
    if cfg_dict is None:
        # Fallback: read from top-level keys with PPO defaults
        obs_shape = tuple(ckpt["obs_shape"])
        n_actions = int(ckpt["n_actions"])
        cfg_dict = {
            "obs_shape": obs_shape,
            "n_actions": n_actions,
            "gamma": float(ckpt.get("gamma", 0.99)),
            "gae_lambda": float(ckpt.get("gae_lambda", 0.95)),
            "clip_range": float(ckpt.get("clip_range", 0.1)),
            "ent_coef": float(ckpt.get("ent_coef", 0.01)),
            "vf_coef": float(ckpt.get("vf_coef", 0.5)),
            "max_grad_norm": float(ckpt.get("max_grad_norm", 0.5)),
            "lr": float(ckpt.get("lr", 2.5e-4)),
        }
    cfg = PPOConfig(**cfg_dict)
    cfg.device = device
    return cfg


def make_eval_env_with_video(
    env_id: str,
    out_dir: str,
    seed: int,
    run_name: str | None = None,
    every_ep: bool = True,
    full_action_space: bool = False,
    sticky_action_prob: float | None = None,
    name_suffix: str = "",
):
    """Build Atari env for evaluation with optional video recording."""
    if run_name is None:
        env_tag = env_id.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{env_tag}_seed{seed}_{timestamp}"

    video_dir = os.path.join(out_dir, "videos", run_name)
    os.makedirs(video_dir, exist_ok=True)

    env = make_atari_env(
        env_id,
        training=False,
        render_mode="rgb_array",
        full_action_space=full_action_space,
        sticky_action_prob=sticky_action_prob,
        channel_first=True,
    )

    safe_env_id = env_id.replace("/", "_").replace("-", "_")
    prefix = f"{safe_env_id}_seed{seed}{('_' + name_suffix) if name_suffix else ''}"

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=(lambda ep: True) if every_ep else (lambda ep: ep % 10 == 0),
        name_prefix=prefix,
    )
    return env


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent")
    parser.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="runs/eval")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--record_all", action="store_true", help="Record all episodes (default: only first)")
    parser.add_argument("--minimal_actions", action="store_true", default=True, help="Use minimal action set")
    parser.add_argument("--non_sticky", action="store_true", help="Disable sticky actions for eval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = _device()

    # Build run name from checkpoint if missing
    if args.run_name is None:
        ckpt_name = os.path.basename(args.checkpoint).replace(".pt", "")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{ckpt_name}_{timestamp}"

    print(f"📦 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Robust cfg reconstruction
    cfg = _build_cfg_from_ckpt(ckpt, device=device)

    # Build agent & load weights
    agent = PPOAgent(cfg)
    agent.load(args.checkpoint)

    # Meta for pretty prints
    meta_step = int(ckpt.get("global_step", 0))
    meta_updates = int(ckpt.get("updates", 0))
    print(f"🤖 Agent loaded (step={meta_step:,}, updates={meta_updates:,})")

    # Action-space alignment
    expected_n_actions = cfg.n_actions
    full_action_space = not args.minimal_actions
    sticky_prob = 0.0 if args.non_sticky else None

    tmp = make_atari_env(args.env_id, training=False, full_action_space=full_action_space,channel_first=True,)
    try:
        assert tmp.action_space.n == expected_n_actions, (
            f"Action-space mismatch: env={tmp.action_space.n} vs agent={expected_n_actions}. "
            f"Hint: try toggling --minimal_actions."
        )
    finally:
        tmp.close()

    # Unique video names per episode instance
    ep_counter = itertools.count()

    def env_fn():
        ep_idx = next(ep_counter)
        return make_eval_env_with_video(
            args.env_id,
            args.out_dir,
            seed=args.seed,
            run_name=args.run_name,
            every_ep=args.record_all,
            full_action_space=full_action_space,
            sticky_action_prob=sticky_prob,
            name_suffix=f"ep{ep_idx:03d}",
        )

    print(f"🎮 Evaluating {args.env_id} for {args.episodes} episodes...")
    print(f"📹 Videos → {os.path.join(args.out_dir, 'videos', args.run_name)}")

    results = evaluate(
        agent=agent,
        env_fn=env_fn,
        num_episodes=args.episodes,
        epsilon_eval=None,  # PPO path in evaluate will call deterministic=True
        seed=args.seed,
    )

    print(
        f"\n✅ Evaluation complete!\n"
        f"  Mean Return: {results['eval_return_mean']:.2f} ± {results['eval_return_std']:.2f}\n"
        f"  Min/Max: [{results['eval_return_min']:.2f}, {results['eval_return_max']:.2f}]\n"
        f"  Mean Episode Length: {results['eval_len_mean']:.1f}\n"
        f"🎥 Videos saved in: {os.path.join(args.out_dir, 'videos', args.run_name)}"
    )


if __name__ == "__main__":
    main()
