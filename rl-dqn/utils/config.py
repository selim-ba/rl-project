# utils/config.py
from typing import Any, Dict
import yaml
from agents.dqn import DQNConfig


def load_yaml(path: str) -> dict:
    """Load YAML configuration file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_nested(d: dict, *keys, default=None) -> Any:
    """Safely access nested dictionary keys
    
    Example: _get_nested(cfg, 'agent', 'lr', default=1e-4)
    """
    x = d
    for k in keys:
        if x is None or not isinstance(x, dict):
            return default
        x = x.get(k)
        if x is None:
            return default
    return x


def yaml_to_dqn_config(yaml_cfg: dict, obs_shape: tuple, n_actions: int) -> DQNConfig:
    """Convert YAML config to DQNConfig dataclass
    
    Args:
        yaml_cfg: Loaded YAML dictionary
        obs_shape: Observation shape from environment
        n_actions: Number of actions from environment
    
    Returns:
        DQNConfig instance
    """
    return DQNConfig(
        # From environment
        obs_shape=obs_shape,
        n_actions=n_actions,
        
        # Replay buffer
        replay_capacity=_get_nested(yaml_cfg, "agent", "replay_capacity", default=1_000_000),
        replay_warmup=_get_nested(yaml_cfg, "agent", "replay_warmup", default=50_000),
        batch_size=_get_nested(yaml_cfg, "agent", "batch_size", default=32),
        
        # Training
        gamma=_get_nested(yaml_cfg, "agent", "gamma", default=0.99),
        lr=_get_nested(yaml_cfg, "agent", "lr", default=2.5e-4),
        rmsprop_alpha=_get_nested(yaml_cfg, "agent", "rmsprop_alpha", default=0.95),
        rmsprop_eps=_get_nested(yaml_cfg, "agent", "rmsprop_eps", default=0.01),
        optimize_every=_get_nested(yaml_cfg, "agent", "optimize_every", default=4),
        target_update_interval=_get_nested(yaml_cfg, "agent", "target_update_interval", default=10_000),
        huber_delta=_get_nested(yaml_cfg, "agent", "huber_delta", default=1.0),
        max_grad_norm=_get_nested(yaml_cfg, "agent", "max_grad_norm", default=None),
        
        # Exploration
        eps_start=_get_nested(yaml_cfg, "exploration", "eps_start", default=1.0),
        eps_end=_get_nested(yaml_cfg, "exploration", "eps_end", default=0.1),
        eps_anneal_frames=_get_nested(yaml_cfg, "exploration", "eps_anneal_frames", default=250_000),
        eval_epsilon=_get_nested(yaml_cfg, "eval", "epsilon", default=0.05),
    )