# utils/save_run_info.py
"""
To save comprehensive run information including:
- Training configuration (from YAML)
- System information (hardware, library versions)
- Command-line arguments
- Hyperparameters
"""
from __future__ import annotations
import sys
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import torch


def get_system_info() -> Dict[str, str]:
    """Gather system and library information"""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "Unknown"
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        info["gpu_count"] = str(torch.cuda.device_count())
    
    try:
        import gymnasium
        info["gymnasium_version"] = gymnasium.__version__
    except:
        info["gymnasium_version"] = "Not installed"
    
    try:
        import ale_py
        info["ale_py_version"] = ale_py.__version__
    except:
        info["ale_py_version"] = "Not installed"
    
    return info


def format_config_section(title: str, data: Dict[str, Any], indent: int = 0) -> str:
    """Format a configuration section for display"""
    lines = [" " * indent + f"[{title}]"]
    indent_str = " " * (indent + 2)
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append("")
            lines.extend(format_config_section(key, value, indent + 2).split("\n"))
        else:
            lines.append(f"{indent_str}{key}: {value}")
    
    return "\n".join(lines)


def save_run_info(
    run_dir: Path,
    config_yaml: Dict[str, Any],
    args: Any,
    agent_config: Any = None,
) -> None:
    """
    Save comprehensive run information to run_info.txt
    
    Args:
        run_dir: Directory where run_info.txt will be saved
        config_yaml: Loaded YAML configuration dictionary
        args: Parsed command-line arguments
        agent_config: DQNConfig instance (optional)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_dir / "run_info.txt"
    
    # Get system information
    sys_info = get_system_info()
    
    # Build output
    lines = []
    lines.append("=" * 80)
    lines.append("DQN TRAINING RUN INFORMATION")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run directory: {run_dir}")
    lines.append("")
    
    # Command line
    lines.append("-" * 80)
    lines.append("COMMAND LINE")
    lines.append("-" * 80)
    lines.append(f"Command: {' '.join(sys.argv)}")
    lines.append("")
    lines.append("Arguments:")
    for key, value in vars(args).items():
        lines.append(f"  --{key}: {value}")
    lines.append("")
    
    # YAML Configuration
    lines.append("-" * 80)
    lines.append("YAML CONFIGURATION")
    lines.append("-" * 80)
    if "env_id" in config_yaml:
        lines.append(f"Environment: {config_yaml['env_id']}")
    if "seed" in config_yaml:
        lines.append(f"Seed: {config_yaml['seed']}")
    lines.append("")
    
    # Training settings
    if "train" in config_yaml:
        lines.append(format_config_section("Training Settings", config_yaml["train"]))
        lines.append("")
    
    # Agent settings
    if "agent" in config_yaml:
        lines.append(format_config_section("Agent Hyperparameters", config_yaml["agent"]))
        lines.append("")
    
    # Exploration settings
    if "exploration" in config_yaml:
        lines.append(format_config_section("Exploration Schedule", config_yaml["exploration"]))
        lines.append("")
    
    # Evaluation settings
    if "eval" in config_yaml:
        lines.append(format_config_section("Evaluation Settings", config_yaml["eval"]))
        lines.append("")
    
    # Agent config (if provided)
    if agent_config is not None:
        lines.append("-" * 80)
        lines.append("AGENT CONFIGURATION (DQNConfig)")
        lines.append("-" * 80)
        lines.append(f"Observation shape: {agent_config.obs_shape}")
        lines.append(f"Number of actions: {agent_config.n_actions}")
        lines.append(f"Device: {agent_config.device}")
        lines.append("")
        lines.append("Replay Buffer:")
        lines.append(f"  Capacity: {agent_config.replay_capacity:,}")
        lines.append(f"  Warmup: {agent_config.replay_warmup:,}")
        lines.append(f"  Batch size: {agent_config.batch_size}")
        lines.append("")
        lines.append("Training:")
        lines.append(f"  Gamma: {agent_config.gamma}")
        lines.append(f"  Learning rate: {agent_config.lr}")
        lines.append(f"  RMSprop alpha: {agent_config.rmsprop_alpha}")
        lines.append(f"  RMSprop epsilon: {agent_config.rmsprop_eps}")
        lines.append(f"  Optimize every: {agent_config.optimize_every}")
        lines.append(f"  Target update interval: {agent_config.target_update_interval:,}")
        lines.append(f"  Huber delta: {agent_config.huber_delta}")
        lines.append(f"  Max grad norm: {agent_config.max_grad_norm}")
        lines.append("")
        lines.append("Exploration:")
        lines.append(f"  Epsilon start: {agent_config.eps_start}")
        lines.append(f"  Epsilon end: {agent_config.eps_end}")
        lines.append(f"  Epsilon anneal frames: {agent_config.eps_anneal_frames:,}")
        lines.append(f"  Eval epsilon: {agent_config.eval_epsilon}")
        lines.append("")
    
    # System information
    lines.append("-" * 80)
    lines.append("SYSTEM INFORMATION")
    lines.append("-" * 80)
    for key, value in sys_info.items():
        formatted_key = key.replace("_", " ").title()
        lines.append(f"{formatted_key}: {value}")
    lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("END OF RUN INFORMATION")
    lines.append("=" * 80)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"📄 Run information saved to: {output_path}")