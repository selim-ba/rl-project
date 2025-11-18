#!/usr/bin/env python3
# eval.py
import argparse
import os
from datetime import datetime
import itertools
import torch
from gymnasium.wrappers import RecordVideo

from env.wrappers import make_atari_env
from agents.dqn import DQNAgent, DQNConfig
from utils.eval_utils import evaluate


def make_eval_env_with_video(
    env_id: str,
    out_dir: str,
    seed: int,
    run_name: str | None = None,
    every_ep: bool = True,
    full_action_space: bool | None = None,
    sticky_action_prob: float | None = None,
    name_suffix: str = "",                  # <— NEW: ensures unique filenames
):
    """Build Atari env for evaluation with video recording."""
    # Unique run name
    if run_name is None:
        env_tag = env_id.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{env_tag}_seed{seed}_{timestamp}"

    # one folder per run; multiple videos inside it
    video_dir = os.path.join(out_dir, "videos", run_name)
    os.makedirs(video_dir, exist_ok=True)

    # Use the requested action space (do not hard-code False)
    env = make_atari_env(
        env_id,
        training=False,
        render_mode="rgb_array",
        full_action_space=bool(full_action_space),
        sticky_action_prob=sticky_action_prob,
    )

    # Record all episodes (or every 10th) for THIS env instance
    # name_prefix gets a unique suffix so files don't overwrite
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
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="runs/eval")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--record_all", action="store_true",help="Record all episodes (default: only first)")
    parser.add_argument("--epsilon_eval", type=float, default=0.001)
    parser.add_argument("--minimal_actions", action="store_true", default=True)
    parser.add_argument("--non_sticky", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Run name based on checkpoint if missing
    if args.run_name is None:
        ckpt_name = os.path.basename(args.checkpoint).replace(".pt", "")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{ckpt_name}_{timestamp}"

    print(f"📦 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg_dict = checkpoint.get("cfg", {})
    cfg = DQNConfig(**cfg_dict); cfg.device = device

    agent = DQNAgent(cfg)
    agent.load(args.checkpoint)
    print(f"🤖 Agent loaded (step={agent.step_count:,}, updates={agent.update_count:,})")

    # action-space alignment
    expected_n_actions = cfg.n_actions
    #full_action_space = True if expected_n_actions >= 10 else False
    full_action_space = not args.minimal_actions
    sticky_prob = 0.0 if args.non_sticky else None


    tmp = make_atari_env(args.env_id, training=False, full_action_space=full_action_space)
    try:
        assert tmp.action_space.n == expected_n_actions, (
            f"Action-space mismatch: env={tmp.action_space.n} vs agent={expected_n_actions}. "
            f"Hint: Re-evaluate with: --minimal_actions  or  without it."
        )
    finally:
        tmp.close()

    # --- Ensure each env instance (if evaluate creates many) has a unique video name ---
    ep_counter = itertools.count()   # 0,1,2,...

    def env_fn():
        ep_idx = next(ep_counter)
        # record_all=True -> record episode 0 for this env (unique name), so you still get 1 file per eval episode
        return make_eval_env_with_video(
            args.env_id,
            args.out_dir,
            seed=args.seed,
            run_name=args.run_name,
            every_ep=args.record_all,
            full_action_space=full_action_space,
            sticky_action_prob=sticky_prob,
            name_suffix=f"ep{ep_idx:03d}",   # <— UNIQUE per env instance
        )

    print(f"🎮 Evaluating {args.env_id} for {args.episodes} episodes...")
    print(f"📹 Videos will be saved to: {os.path.join(args.out_dir, 'videos', args.run_name)}")

    results = evaluate(
        agent=agent,
        env_fn=env_fn,
        num_episodes=args.episodes,
        epsilon_eval=args.epsilon_eval,
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
