#!/usr/bin/env python3
# train.py
import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

import torch

from env.wrappers import make_atari_env
from agents.dqn import DQNAgent
from utils.config import load_yaml, yaml_to_dqn_config, _get_nested
from utils.eval_utils import evaluate
from utils.logger import CSVLogger
from utils.seed import set_seed
from utils.save_run_info import save_run_info


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on Atari")
    
    # Config file
    p.add_argument("--config", type=str, default="configs/dqn_breakout.yaml",
                   help="Path to YAML config file")
    
    # CLI overrides (optional)
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--resume", type=str, default=None)
    
    return p.parse_args()


def build_run_dir(runs_dir: str, env_id: str, seed: int, resume_path: str = None) -> Path:
    """Create timestamped run directory OR reuse existing one if resuming"""
    if resume_path:
        # Extract run directory from checkpoint path
        # e.g., "runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/ckpt_1000000.pt"
        #    -> "runs/dqn-ALE_Breakout_v5-seed42-20241112-143022"
        ckpt_path = Path(resume_path)
        run_dir = ckpt_path.parent
        print(f"📂 Resuming in existing directory: {run_dir}")
        return run_dir
    else:
        # Create new timestamped directory
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_env = env_id.replace("/", "_").replace("-", "_")
        run_dir = Path(runs_dir) / f"dqn-{safe_env}-seed{seed}-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created new directory: {run_dir}")
        return run_dir


def main():
    args = parse_args()
    
    # Load YAML config
    cfg_yaml = load_yaml(args.config)
    
    # Merge YAML + CLI (CLI takes precedence)
    env_id = args.env_id or _get_nested(cfg_yaml, "env_id", default="ALE/Breakout-v5")
    seed = args.seed if args.seed is not None else _get_nested(cfg_yaml, "seed", default=0)
    total_steps = args.total_steps or _get_nested(cfg_yaml, "train", "total_steps", default=10_000_000)
    eval_interval = args.eval_interval or _get_nested(cfg_yaml, "train", "eval_interval", default=250_000)
    eval_episodes = args.eval_episodes or _get_nested(cfg_yaml, "train", "eval_episodes", default=30)
    save_interval = args.save_interval or _get_nested(cfg_yaml, "train", "save_interval", default=250_000)
    
    # Set global seed
    set_seed(seed)
    
    # Create environment to get observation shape
    env = make_atari_env(env_id, training=True, full_action_space=False)
    obs, info = env.reset(seed=seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print("Action space n =", n_actions)
    env.action_space.seed(seed)
    
    # Build DQN config from YAML
    cfg = yaml_to_dqn_config(cfg_yaml, obs_shape, n_actions)
    
    # Create run directory (reuse if resuming)
    run_dir = build_run_dir(args.runs_dir, env_id, seed, args.resume)
    
    # Save config and run info for traceability (only if new run)
    if not args.resume:
        # Save YAML config
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_yaml, f, sort_keys=False)
        
        # Save comprehensive run information
        save_run_info(run_dir, cfg_yaml, args, cfg)
    
    # Setup logging (APPEND mode if resuming)
    logger = CSVLogger(str(run_dir), append=args.resume is not None)
    def log(row: dict):
        row = dict(row)
        row["env_id"] = env_id
        logger.log(row)

    # Initialize with all columns (only if new run)
    if not args.resume:
        log({
            "step": 0, "updates": 0, "loss": None, "q_max": None, "epsilon": None, "fps": None,
            "episode": 0, "episode_return": None, "episode_length": None,
            "eval_return_mean": None, "eval_return_std": None,
            "eval_return_min": None, "eval_return_max": None, "eval_len_mean": None
        })
    
    # Create agent
    agent = DQNAgent(cfg)
    if args.resume:
        agent.load(args.resume)
        print(f"↩️  Resumed from {args.resume} (step={agent.step_count:,}, updates={agent.update_count:,})")

    anneal = getattr(agent.cfg, "eps_anneal_steps",
            getattr(agent.cfg, "eps_anneal_frames", "MISSING"))
    print("ε schedule:",
        "start=", agent.cfg.eps_start,
        "end=", agent.cfg.eps_end,
        "anneal=", anneal,
        "current=", agent._epsilon())

    
    # Training loop
    episode_idx = 0
    ep_return = 0.0
    ep_len = 0
    
    t0 = time.time()
    last_report_t = t0
    last_report_step = agent.step_count
    
    print(f"🎮 Training DQN on {env_id}")
    print(f"📂 Run directory: {run_dir}")
    print(f"🎯 Total steps: {total_steps:,}")
    print(f"🔄 Starting from step: {agent.step_count:,}")
    
    while agent.step_count < total_steps:
        # Select action
        action = agent.act(obs, eval_mode=False)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        done_for_td = terminated
        ep_return += float(reward)
        ep_len += 1
        
        # Store transition
        agent.observe(obs, action, reward, next_obs, done_for_td)
        
        # Update agent
        update_out = agent.update()
        
        # Log update metrics immediately when available
        if update_out:
            log({
                "step": agent.step_count,
                "updates": update_out["updates"],
                "loss": update_out["loss"],
                "q_max": update_out["q_max"],
                "epsilon": update_out["epsilon"],
            })

        # Separate periodic FPS logging
        if agent.step_count % 1_000 == 0:
            now = time.time()
            dt = now - last_report_t
            dsteps = agent.step_count - last_report_step
            fps = dsteps / dt if dt > 0 else 0.0
            last_report_t = now
            last_report_step = agent.step_count
            
            log({
                "step": agent.step_count,
                "fps": fps,
                "epsilon": agent._epsilon()
            })

        
        # Episode end
        if done:
            episode_idx += 1
            log({
                "step": agent.step_count,
                "episode": episode_idx,
                "episode_return": ep_return,
                "episode_length": ep_len
            })
            
            if episode_idx % 10 == 0:
                print(f"Episode {episode_idx} | Step {agent.step_count:,} | Return: {ep_return:.1f} | Length: {ep_len}")
            
            obs, info = env.reset()
            ep_return = 0.0
            ep_len = 0
        else:
            obs = next_obs
        
        # Periodic evaluation
        if agent.step_count % eval_interval == 0 or agent.step_count == total_steps:
            print(f"\n📊 Evaluating at step {agent.step_count:,}...")
            eval_stats = evaluate(
                env_fn=lambda: make_atari_env(env_id, training=False, full_action_space=False),
                agent=agent,
                num_episodes=eval_episodes,
                epsilon_eval=cfg.eval_epsilon,  # If issues: Use 0.0 for greedy evaluation
                seed=seed
            )
            log({"step": agent.step_count, **eval_stats})
            print(f"   Mean return: {eval_stats['eval_return_mean']:.2f} ± {eval_stats['eval_return_std']:.2f}\n")
        
        # Periodic checkpoint
        if agent.step_count % save_interval == 0 or agent.step_count == total_steps:
            ckpt_path = run_dir / f"ckpt_{agent.step_count}.pt"
            agent.save(str(ckpt_path))
            print(f"💾 Checkpoint saved: {ckpt_path}")
    
    # Final save
    agent.save(str(run_dir / "final.pt"))
    env.close()
    logger.close()
    
    print(f"\n✅ Training complete!")
    print(f"📂 Results saved to: {run_dir}")


if __name__ == "__main__":
    main()