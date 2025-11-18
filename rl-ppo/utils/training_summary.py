# utils/training_summary.py
"""Generate comprehensive training summary reports"""
from __future__ import annotations
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import platform
import torch


def get_system_info() -> Dict[str, Any]:
    """Gather system and environment information"""
    info = {
        "platform": str(platform.platform()),
        "python_version": str(platform.python_version()),
        "pytorch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["gpu_count"] = int(torch.cuda.device_count())
    
    return info


def create_training_summary(
    run_dir: str | Path,
    config: Dict[str, Any],
    agent_config: Dict[str, Any],
    start_time: datetime,
    end_time: datetime = None,
    final_stats: Dict[str, Any] = None,
) -> None:
    """
    Create a comprehensive training summary file in YAML and JSON formats.
    
    Args:
        run_dir: Directory to save the summary
        config: Full training configuration
        agent_config: Agent-specific configuration
        start_time: Training start timestamp
        end_time: Training end timestamp (optional, for final summary)
        final_stats: Final evaluation statistics (optional)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to ensure all values are serializable
    def make_serializable(obj):
        """Convert numpy types and other non-serializable objects to native Python types"""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj
    
    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "training_started": start_time.isoformat(),
        },
        "system_info": get_system_info(),
        "environment": {
            "env_id": str(config.get("env_id", "Unknown")),
            "seed": int(config.get("seed", 0)) if config.get("seed") is not None else None,
        },
        "training_config": {
            "total_steps": int(config.get("train", {}).get("total_steps", 0)),
            "n_envs": int(config.get("train", {}).get("n_envs", 0)),
            "n_steps": int(config.get("train", {}).get("n_steps", 0)),
            "eval_interval": int(config.get("train", {}).get("eval_interval", 0)),
            "eval_episodes": int(config.get("train", {}).get("eval_episodes", 0)),
            "save_interval": int(config.get("train", {}).get("save_interval", 0)),
        },
        "agent_config": make_serializable(agent_config),
        "preprocessing": {
            "clip_rewards": bool(config.get("agent", {}).get("clip_rewards", True)),
            "full_action_space": bool(config.get("agent", {}).get("full_action_space", False)),
            "sticky_action_prob": config.get("agent", {}).get("sticky_action_prob"),
            "frame_stack": 4,
            "frame_skip": 4,
            "observation_shape": "84x84 grayscale",
        },
    }
    
    # Add end-of-training info if available
    if end_time:
        duration = (end_time - start_time).total_seconds()
        summary["metadata"]["training_ended"] = end_time.isoformat()
        summary["metadata"]["total_duration_seconds"] = float(duration)
        summary["metadata"]["total_duration_hours"] = float(duration / 3600)
    
    if final_stats:
        summary["final_evaluation"] = make_serializable(final_stats)
    
    # Save as both YAML (human-readable) and JSON (machine-readable)
    yaml_path = run_dir / "training_summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, indent=2)
    
    json_path = run_dir / "training_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Training summary saved:")
    print(f"   - {yaml_path}")
    print(f"   - {json_path}")


def update_training_summary(
    run_dir: str | Path,
    updates: Dict[str, Any],
) -> None:
    """Update existing training summary with new information"""
    run_dir = Path(run_dir)
    yaml_path = run_dir / "training_summary.yaml"
    json_path = run_dir / "training_summary.json"
    
    # Load existing summary
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            summary = yaml.safe_load(f)
    else:
        summary = {}
    
    # Deep merge updates
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    
    deep_update(summary, updates)
    
    # Save updated summary
    with open(yaml_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, indent=2)
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)