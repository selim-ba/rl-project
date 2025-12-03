# utils/config.py (last update 03/12/2025)

from typing import Any, Dict, Tuple
import yaml

try:
    from agents.ppo import PPOConfig  # the PPO dataclass we added
except Exception:
    PPOConfig = None  # type: ignore


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_nested(d: dict, *keys, default=None) -> Any:
    """Safely access nested dictionary keys.

    Example:
        _get_nested(cfg, "agent", "lr", default=2.5e-4)
    """
    x = d
    for k in keys:
        if x is None or not isinstance(x, dict):
            return default
        x = x.get(k)
        if x is None:
            return default
    return x




# ----------------------------
# PPO helper
# ----------------------------
def yaml_to_ppo_config(yaml_cfg: dict, obs_shape: Tuple[int, int, int], n_actions: int):
    """Convert YAML config to PPOConfig.

    Only fields that exist on PPOConfig are mapped here. Scheduling fields
    like total_steps, n_envs, n_steps belong to the training loop and are
    intentionally *not* included in PPOConfig.
    """
    if PPOConfig is None:
        raise ImportError("PPOConfig not available. Is agents/pop.py present?")

    agent = _get_nested(yaml_cfg, "agent", default={})

    return PPOConfig(
        # From environment
        obs_shape=obs_shape,
        n_actions=n_actions,

        # PPO algorithm hyperparameters
        gamma=float(_get_nested(agent, "gamma", default=0.99)),
        gae_lambda=float(_get_nested(agent, "gae_lambda", default=0.95)),
        clip_range=float(_get_nested(agent, "clip_range", default=0.1)),
        ent_coef=float(_get_nested(agent, "ent_coef", default=0.01)),
        vf_coef=float(_get_nested(agent, "vf_coef", default=0.5)),
        max_grad_norm=float(_get_nested(agent, "max_grad_norm", default=0.5)),
        lr=float(_get_nested(agent, "lr", default=2.5e-4)),
        device="cuda"  # the train script will still place the model on the detected device
    )
