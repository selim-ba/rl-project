# utils/eval_utils.py (last update 03/12/2025)

from __future__ import annotations
import numpy as np


def evaluate(agent, env_fn, num_episodes, epsilon_eval=None, seed=None, **kwargs):
    """
    Evaluate agent over multiple episodes with deterministic policy.
    
    Works with both DQN and PPO agents through standardized interface:
    - DQN: agent.act(obs, eval_mode=True, epsilon=0.05)
    - PPO: agent.act(obs, deterministic=True)
    
    Args:
        agent: Agent with act() method (supports both DQN and PPO)
        env_fn: Callable that returns a new environment
        num_episodes: Number of episodes to evaluate
        seed: Random seed for reproducibility
        epsilon_eval: Epsilon for DQN evaluation (ignored for PPO, kept for compatibility)
    
    Returns:
        Dictionary with evaluation statistics:
        - eval_return_mean: Mean episode return
        - eval_return_std: Standard deviation of returns
        - eval_return_min: Minimum return
        - eval_return_max: Maximum return
        - eval_len_mean: Mean episode length
    """
    returns = []
    lengths = []
    
    for i in range(num_episodes):
        env = env_fn()
        obs, info = env.reset(seed=None if seed is None else seed + i)
        
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        
        while not (done or truncated):
            # Detect agent type and call appropriate interface
            # Try PPO interface first (deterministic=True)
            try:
                action = agent.act(obs, deterministic=True)
            except TypeError:
                # Fall back to DQN interface (eval_mode=True, epsilon=small)
                action = agent.act(obs, eval_mode=True, epsilon=0.05)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
        
        env.close()
        returns.append(episode_return)
        lengths.append(episode_length)
    
    # Convert to numpy for statistics
    returns_arr = np.asarray(returns, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.float32)
    
    return {
        "eval_return_mean": float(np.mean(returns_arr)) if len(returns_arr) else 0.0,
        "eval_return_std": float(np.std(returns_arr)) if len(returns_arr) else 0.0,
        "eval_return_min": float(np.min(returns_arr)) if len(returns_arr) else 0.0,
        "eval_return_max": float(np.max(returns_arr)) if len(returns_arr) else 0.0,
        "eval_len_mean": float(np.mean(lengths_arr)) if len(lengths_arr) else 0.0,
    }
