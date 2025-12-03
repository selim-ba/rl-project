#!/usr/bin/env python3
# train.py (Last update 03/12/2025)

import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from env.wrappers import make_atari_env
from utils.config import load_yaml, _get_nested
from utils.eval_utils import evaluate
from utils.logger import CSVLogger
from utils.seed import set_seed
from utils.training_summary import create_training_summary, update_training_summary

from agents.ppo import PPOAgent, PPOConfig
from memory.rollout_buffer import RolloutBuffer


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Atari")

    # Config file
    p.add_argument("--config", type=str, default="configs/ppo_breakout.yaml",
                   help="Path to YAML config file")

    # CLI overrides (optional)
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--n_envs", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def build_run_dir(runs_dir: str, env_id: str, seed: int, resume_path: str | None = None) -> Path:
    """Create timestamped run directory OR reuse existing one if resuming"""
    if resume_path:
        ckpt_path = Path(resume_path)
        run_dir = ckpt_path.parent
        print(f"📂 Resuming in existing directory: {run_dir}")
        return run_dir
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_env = env_id.replace("/", "_").replace("-", "_")
        run_dir = Path(runs_dir) / f"ppo-{safe_env}-seed{seed}-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created new directory: {run_dir}")
        return run_dir


def _device() -> str:
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def make_vec_env(env_id: str, n_envs: int, seed: int, clip_rewards: bool = True,
                  full_action_space: bool = False, sticky_action_prob: float | None = None) -> gym.vector.VectorEnv:
    def make_one(rank: int):
        def thunk():
            env_seed = None if seed is None else seed + rank
            env = make_atari_env(
                env_id,
                training=clip_rewards,
                render_mode=None,
                channel_first=True,  # HWC uint8 frames
                seed=env_seed,
                full_action_space=full_action_space,
                sticky_action_prob=sticky_action_prob,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1000)
            return env
        return thunk

    return SyncVectorEnv([make_one(i) for i in range(n_envs)])


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear annealing schedule"""
    def schedule_fn(progress: float) -> float:
        """Progress goes from 0.0 to 1.0"""
        return final_value + (initial_value - final_value) * (1 - progress)
    return schedule_fn


def main():
    args = parse_args()

    # Load YAML config
    cfg_yaml = load_yaml(args.config)

    # Merge YAML + CLI (CLI takes precedence)
    env_id = args.env_id or _get_nested(cfg_yaml, "env_id", default="ALE/Breakout-v5")
    seed = args.seed if args.seed is not None else _get_nested(cfg_yaml, "seed", default=0)

    # Training schedule
    total_steps = args.total_steps or _get_nested(cfg_yaml, "train", "total_steps", default=10_000_000)
    n_envs = args.n_envs or _get_nested(cfg_yaml, "train", "n_envs", default=8)
    n_steps = args.n_steps or _get_nested(cfg_yaml, "train", "n_steps", default=128)

    # Logging / eval cadence
    eval_interval = args.eval_interval or _get_nested(cfg_yaml, "train", "eval_interval", default=250_000)
    eval_episodes = args.eval_episodes or _get_nested(cfg_yaml, "train", "eval_episodes", default=30)
    save_interval = args.save_interval or _get_nested(cfg_yaml, "train", "save_interval", default=250_000)

    # PPO hyperparams
    agent_cfg_yaml = _get_nested(cfg_yaml, "agent", default={})
    gamma = float(_get_nested(agent_cfg_yaml, "gamma", default=0.99))
    gae_lambda = float(_get_nested(agent_cfg_yaml, "gae_lambda", default=0.95))
    clip_range = float(_get_nested(agent_cfg_yaml, "clip_range", default=0.1))
    ent_coef = float(_get_nested(agent_cfg_yaml, "ent_coef", default=0.01))
    vf_coef = float(_get_nested(agent_cfg_yaml, "vf_coef", default=0.5))
    max_grad_norm = float(_get_nested(agent_cfg_yaml, "max_grad_norm", default=0.5))
    lr = float(_get_nested(agent_cfg_yaml, "lr", default=2.5e-4))
    n_epochs = int(_get_nested(agent_cfg_yaml, "n_epochs", default=3))  # FIXED: default now 3 (paper value)
    batch_size = int(_get_nested(agent_cfg_yaml, "batch_size", default=256))
    clip_rewards = bool(_get_nested(agent_cfg_yaml, "clip_rewards", default=True))
    full_action_space = bool(_get_nested(agent_cfg_yaml, "full_action_space", default=False))
    sticky_action_prob = _get_nested(agent_cfg_yaml, "sticky_action_prob", default=None)
    
    # Annealing flags (from paper)
    anneal_lr = bool(_get_nested(agent_cfg_yaml, "anneal_lr", default=True))
    anneal_clip = bool(_get_nested(agent_cfg_yaml, "anneal_clip", default=True))

    # Seed + device
    set_seed(seed)
    device = _device()

    # Build a single env to probe shapes via wrappers
    probe_env = make_atari_env(env_id, training=clip_rewards, full_action_space=full_action_space,channel_first=True,)
    probe_obs, _ = probe_env.reset(seed=seed)
    obs_shape = probe_obs.shape  # HWC
    n_actions = probe_env.action_space.n
    probe_env.close()
    print(f"🎮 Action space: {n_actions} actions")
    print(f"👁️  Observation shape: {obs_shape}")

    # Create run directory (reuse if resuming)
    run_dir = build_run_dir(args.runs_dir, env_id, seed, args.resume)
    start_time = datetime.now()

    # Save config for traceability (only if new run)
    if not args.resume:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_yaml, f, sort_keys=False)

    # Setup logging (APPEND mode if resuming)
    logger = CSVLogger(str(run_dir), append=args.resume is not None)

    def log(row: dict):
        row = dict(row)
        row["env_id"] = env_id
        logger.log(row)

    # Initialize with all columns (only if new run)
    if not args.resume:
        log({
            "step": 0, "updates": 0,
            "policy_loss": None, "value_loss": None, "entropy": None,
            "approx_kl": None, "clipfrac": None, "fps": None,
            "episode": 0, "episode_return": None, "episode_length": None,
            "eval_return_mean": None, "eval_return_std": None,
            "eval_return_min": None, "eval_return_max": None, "eval_len_mean": None,
            "learning_rate": None, "clip_range": None
        })

    # Create vectorized environments
    envs = make_vec_env(env_id, n_envs=n_envs, seed=seed, clip_rewards=clip_rewards,
                        full_action_space=full_action_space, sticky_action_prob=sticky_action_prob)
    obs, info = envs.reset(seed=seed)
    assert obs.dtype == np.uint8, "Expect uint8 frames from wrappers (HWC)"

    # Create agent + buffer
    ppo_cfg = PPOConfig(
        obs_shape=obs.shape[1:],
        n_actions=n_actions,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        device=device,
    )
    agent = PPOAgent(ppo_cfg)

    # Resume if requested
    global_step = 0
    update_count = 0
    last_checkpoint_step = 0  # Track last checkpoint to avoid duplicates
    last_eval_step = 0  # Track last evaluation to avoid duplicates
    
    if args.resume:
        meta = agent.load(args.resume, device=device)
        global_step = int(meta.get("global_step", 0))
        update_count = int(meta.get("updates", 0))
        # Initialize tracking from resume point
        last_checkpoint_step = global_step
        last_eval_step = global_step
        print(f"↩️  Resumed from {args.resume} (step={global_step:,}, updates={update_count:,})")

    rb = RolloutBuffer(n_steps=n_steps, n_envs=n_envs, obs_shape=obs.shape[1:], dtype=torch.uint8)

    # Create schedules
    lr_schedule = linear_schedule(lr, 0.0) if anneal_lr else lambda p: lr
    clip_schedule = linear_schedule(clip_range, 0.0) if anneal_clip else lambda p: clip_range

    # Per-env episodic trackers
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    # Training loop
    episode_idx = 0
    t0 = time.time()
    last_report_t = t0
    last_report_step = global_step

    print(f"\n🎮 Training PPO on {env_id}")
    print(f"📂 Run directory: {run_dir}")
    print(f"🎯 Total steps: {total_steps:,}")
    print(f"🔄 Starting from step: {global_step:,}")
    print(f"📊 LR annealing: {anneal_lr}, Clip annealing: {anneal_clip}")
    print(f"🔧 n_epochs: {n_epochs}, batch_size: {batch_size}\n")

    while global_step < total_steps:
        # Collect a rollout of length n_steps
        rb.reset()
        for t in range(n_steps):
            with torch.no_grad():
                actions, logprobs, values = agent.act(obs)  # vectorized
            next_obs, rewards, terms, truncs, infos = envs.step(actions)

            # Accumulate per-env stats & log episode endings
            ep_returns += rewards
            ep_lengths += 1
            dones = (terms | truncs)
            if np.any(dones):
                done_indices = np.where(dones)[0]
                for idx in done_indices:
                    episode_idx += 1
                    ep_ret = float(ep_returns[idx])
                    ep_len = int(ep_lengths[idx])
                    log({
                        "step": global_step + n_envs,
                        "episode": episode_idx,
                        "episode_return": ep_ret,
                        "episode_length": ep_len,
                    })
                    # Print every 10 episodes to reduce log spam
                    if episode_idx % 10 == 0:
                        print(f"Episode {episode_idx} | Step {global_step + n_envs:,} | Return: {ep_ret:.1f} | Length: {ep_len}")
                    ep_returns[idx] = 0.0
                    ep_lengths[idx] = 0

            rb.add(obs=obs, actions=actions, rewards=rewards, dones=dones,
                   values=values, logprobs=logprobs)

            obs = next_obs
            global_step += n_envs

            # Periodic FPS logging + heartbeat
            if global_step % 1000 == 0:
                now = time.time()
                dt = now - last_report_t
                dsteps = global_step - last_report_step
                fps = dsteps / dt if dt > 0 else 0.0
                last_report_t = now
                last_report_step = global_step
                log({"step": global_step, "fps": fps})
            
            if global_step % 50_000 == 0:
                now = time.time()
                elapsed = now - t0
                progress = global_step / total_steps
                eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                print(f"… step {global_step:,}/{total_steps:,} ({progress*100:.1f}%) | ETA: {eta/3600:.1f}h")

        # Bootstrap & compute GAE
        with torch.no_grad():
            _, _, last_values = agent.act(obs, value_only=True)
        rb.compute_returns_advantages(last_values=last_values, gamma=gamma, gae_lambda=gae_lambda)

        # Calculate training progress (0.0 to 1.0)
        progress = global_step / total_steps

        # Update learning rate
        current_lr = lr_schedule(progress)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = current_lr

        # Update clip range
        current_clip = clip_schedule(progress)

        # PPO update (pass current_clip as parameter)
        approx_kl, clipfrac, pg_loss, v_loss, ent = agent.update(
            rb, n_epochs=n_epochs, batch_size=batch_size, clip_range=current_clip
        )
        update_count += 1
        
        log({
            "step": global_step,
            "updates": update_count,
            "approx_kl": float(approx_kl),
            "clipfrac": float(clipfrac),
            "policy_loss": float(pg_loss),
            "value_loss": float(v_loss),
            "entropy": float(ent),
            "learning_rate": current_lr,
            "clip_range": current_clip,
        })
        
        # Compact update line
        if update_count % 10 == 0:
            print(f"upd {update_count:4d} @ {global_step:8,} | "
                  f"pg {pg_loss:.3f} v {v_loss:.3f} ent {ent:.3f} "
                  f"kl {approx_kl:.4f} clip {clipfrac:.2f} | "
                  f"lr {current_lr:.2e} clip_ε {current_clip:.3f}")

        # FIXED: Periodic evaluation (uses threshold check to avoid duplicates)
        if global_step >= last_eval_step + eval_interval:
            eval_step = global_step
            print(f"\n📊 Evaluating at step {eval_step:,}...")
            eval_stats = evaluate(
                env_fn=lambda: make_atari_env(env_id, training=False, full_action_space=full_action_space,channel_first=True,),
                agent=agent,
                num_episodes=eval_episodes,
                seed=seed,
            )
            log({"step": eval_step, **eval_stats})
            print(f"   Mean return: {eval_stats['eval_return_mean']:.2f} ± {eval_stats['eval_return_std']:.2f}")
            print(f"   Min/Max: [{eval_stats['eval_return_min']:.1f}, {eval_stats['eval_return_max']:.1f}]\n")
            last_eval_step = eval_step

        # FIXED: Periodic checkpoint (uses threshold check to avoid duplicates)
        if global_step >= last_checkpoint_step + save_interval:
            checkpoint_step = global_step
            ckpt_path = run_dir / f"ckpt_{checkpoint_step}.pt"
            agent.save(
                path=str(ckpt_path),
                extra={
                    "cfg": {
                        "obs_shape": tuple(obs.shape[1:]),
                        "n_actions": int(n_actions),
                        "gamma": float(gamma),
                        "gae_lambda": float(gae_lambda),
                        "clip_range": float(clip_range),
                        "ent_coef": float(ent_coef),
                        "vf_coef": float(vf_coef),
                        "max_grad_norm": float(max_grad_norm),
                        "lr": float(lr),
                    },
                    "env_id": env_id,
                    "global_step": global_step,
                    "updates": update_count,
                    "seed": seed,
                    "n_envs": n_envs,
                    "n_steps": n_steps,
                },
            )
            print(f"💾 Checkpoint saved: {ckpt_path}")
            last_checkpoint_step = checkpoint_step

    # Final evaluation
    print(f"\n📊 Running final evaluation...")
    final_eval_stats = evaluate(
        env_fn=lambda: make_atari_env(env_id, training=False, full_action_space=full_action_space,channel_first=True,),
        agent=agent,
        num_episodes=eval_episodes,
        seed=seed,
    )
    log({"step": global_step, **final_eval_stats})
    print(f"   Final mean return: {final_eval_stats['eval_return_mean']:.2f} ± {final_eval_stats['eval_return_std']:.2f}")

    # Final save
    agent.save(
        str(run_dir / "final.pt"),
        extra={
            "cfg": {
                "obs_shape": tuple(obs.shape[1:]),
                "n_actions": int(n_actions),
                "gamma": float(gamma),
                "gae_lambda": float(gae_lambda),
                "clip_range": float(clip_range),
                "ent_coef": float(ent_coef),
                "vf_coef": float(vf_coef),
                "max_grad_norm": float(max_grad_norm),
                "lr": float(lr),
            },
            "env_id": env_id,
            "global_step": global_step,
            "updates": update_count,
            "seed": seed,
            "n_envs": n_envs,
            "n_steps": n_steps,
        },
    )
    
    envs.close()
    logger.close()

    # Create training summary
    end_time = datetime.now()
    
    # Ensure all config values are JSON/YAML serializable
    agent_config = {
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": float(clip_range),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "max_grad_norm": float(max_grad_norm),
        "lr": float(lr),
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size),
        "anneal_lr": bool(anneal_lr),
        "anneal_clip": bool(anneal_clip),
    }
    
    # Convert numpy types in final_eval_stats to native Python types
    final_stats_clean = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in final_eval_stats.items()
    }
    
    create_training_summary(
        run_dir=run_dir,
        config=cfg_yaml,
        agent_config=agent_config,
        start_time=start_time,
        end_time=end_time,
        final_stats=final_stats_clean,
    )

    print("\n✅ Training complete!")
    print(f"📂 Results saved to: {run_dir}")
    print(f"⏱️  Total time: {(end_time - start_time).total_seconds() / 3600:.2f} hours")


if __name__ == "__main__":
    main()
