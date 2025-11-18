# utils/eval_utils.py
from __future__ import annotations
import numpy as np


def evaluate(agent, env_fn, num_episodes: int = 10, epsilon_eval: float = 0.05, seed: int = None) -> dict:
    """Evaluate agent over multiple episodes
    
    Args:
        agent: Agent with act() method
        env_fn: Callable that returns a new environment
        num_episodes: Number of episodes to evaluate
        epsilon_eval: Epsilon for evaluation (small fixed value)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation statistics
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
            action = agent.act(obs, eval_mode=True, epsilon=epsilon_eval)
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