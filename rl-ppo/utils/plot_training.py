# utils/plot_training.py - PPO-FOCUSED VERSION
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

plt.style.use("seaborn-v0_8-whitegrid")


def detect_algorithm(df: pd.DataFrame) -> str:
    """Detect whether logs are from DQN or PPO"""
    dqn_cols = {"q_max", "epsilon"}
    ppo_cols = {"policy_loss", "clipfrac", "entropy", "approx_kl"}
    
    if any(col in df.columns for col in ppo_cols):
        return "PPO"
    elif any(col in df.columns for col in dqn_cols):
        return "DQN"
    return "Unknown"


def exponential_moving_average(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Exponential moving average with smoothing factor alpha"""
    return series.ewm(alpha=alpha, adjust=False).mean()


def _maybe_set_atari_ylim(env_id: Optional[str], metric: str):
    """Set appropriate y-axis limits for known Atari games"""
    if not env_id:
        return
    
    if "Pong" in env_id and metric == "episode_return":
        plt.ylim(-22, 22)
    elif "Breakout" in env_id and metric == "episode_return":
        plt.ylim(0, 500)


def plot_metric_with_smoothing(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    out_dir: str,
    env_id: Optional[str] = None,
    ema_alphas: List[float] = [0.1, 0.02],
    show_raw: bool = True,
):
    """Plot a metric with multiple smoothing options"""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"⚠️  Skipping {filename}: missing columns {x_col} or {y_col}")
        return
    
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        print(f"⚠️  Skipping {filename}: no data after dropping NaN")
        return
    
    sub = sub.sort_values(x_col).groupby(x_col, as_index=False)[y_col].mean()
    
    xs = sub[x_col].to_numpy()
    ys_raw = sub[y_col].to_numpy()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_raw and len(xs) > 1:
        ax.plot(xs, ys_raw, lw=0.5, alpha=0.12, color="0.5", label="Raw", zorder=1)
    
    colors = ["#FF8C00", "#DC143C", "#8B0000"]
    for i, alpha in enumerate(ema_alphas):
        ema = exponential_moving_average(sub[y_col], alpha=alpha)
        ax.plot(xs, ema, lw=2.0 + i*0.5, color=colors[i], 
                label=f"EMA (α={alpha})", zorder=2+i, alpha=0.9)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(frameon=True, framealpha=0.95, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # _maybe_set_atari_ylim(env_id, y_col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {filename}")


def plot_fps(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Plot training throughput (FPS) over time"""
    if "step" not in df.columns or "fps" not in df.columns:
        return
    
    sub = df[["step", "fps"]].dropna()
    if sub.empty:
        return
    
    sub = sub.sort_values("step")
    
    plt.figure(figsize=(8, 4))
    plt.plot(sub["step"], sub["fps"], lw=1.2, alpha=0.4, color="#2ca02c", label="Raw")
    
    ema = exponential_moving_average(sub["fps"], alpha=0.05)
    plt.plot(sub["step"], ema, lw=2.5, color="#006400", label="EMA (α=0.05)")
    
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Frames per Second (FPS)", fontsize=12)
    
    suffix = f" ({env_id})" if env_id else ""
    plt.title(f"Training Throughput{suffix}", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved fps.png")


def plot_evaluation_performance(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Plot evaluation performance with confidence bands"""
    if {"step", "eval_return_mean"} > set(df.columns):
        return
    
    cols = ["step", "eval_return_mean"]
    if "eval_return_std" in df.columns:
        cols.append("eval_return_std")
    elif {"eval_return_min", "eval_return_max"} <= set(df.columns):
        cols += ["eval_return_min", "eval_return_max"]
    
    sub = df.dropna(subset=["step", "eval_return_mean"])[cols]
    if sub.empty:
        return
    
    sub = sub.sort_values("step").groupby("step", as_index=False).mean()
    
    steps = sub["step"].to_numpy()
    mean = sub["eval_return_mean"].to_numpy()
    
    std = None
    if "eval_return_std" in sub.columns:
        std = sub["eval_return_std"].to_numpy()
    elif {"eval_return_min", "eval_return_max"} <= set(sub.columns):
        std = (sub["eval_return_max"] - sub["eval_return_min"]).to_numpy() / 2.0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    color = "#1f77b4"
    
    ax.plot(steps, mean, lw=2.5, color=color, label="Mean eval return", marker='o', 
            markersize=4, alpha=0.9)
    
    if std is not None:
        lower = mean - std
        upper = mean + std
        ax.fill_between(steps, lower, upper, alpha=0.25, color=color, 
                        linewidth=0, label="±1 std")
    
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Evaluation Return", fontsize=12)
    
    suffix = f" ({env_id})" if env_id else ""
    ax.set_title(f"Evaluation Performance{suffix}", fontsize=13, pad=10)
    
    # _maybe_set_atari_ylim(env_id, "episode_return")
    
    ax.legend(frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_return.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved eval_return.png")


def plot_ppo_losses(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Plot PPO-specific losses (policy, value, entropy)"""
    has_policy = "policy_loss" in df.columns
    has_value = "value_loss" in df.columns
    has_entropy = "entropy" in df.columns
    
    if not any([has_policy, has_value, has_entropy]):
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    suffix = f" ({env_id})" if env_id else ""
    
    # Policy loss
    if has_policy:
        ax = axes[0]
        sub = df[["updates", "policy_loss"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ax.plot(sub["updates"], sub["policy_loss"], lw=0.8, alpha=0.3, color="0.5")
            ema = exponential_moving_average(sub["policy_loss"], alpha=0.1)
            ax.plot(sub["updates"], ema, lw=2.5, color="#DC143C", label="EMA (α=0.1)")
            ax.set_xlabel("Updates", fontsize=11)
            ax.set_ylabel("Policy Loss", fontsize=11)
            ax.set_title("Policy Loss", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No policy_loss data", ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Value loss
    if has_value:
        ax = axes[1]
        sub = df[["updates", "value_loss"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ax.plot(sub["updates"], sub["value_loss"], lw=0.8, alpha=0.3, color="0.5")
            ema = exponential_moving_average(sub["value_loss"], alpha=0.1)
            ax.plot(sub["updates"], ema, lw=2.5, color="#FF8C00", label="EMA (α=0.1)")
            ax.set_xlabel("Updates", fontsize=11)
            ax.set_ylabel("Value Loss", fontsize=11)
            ax.set_title("Value Loss", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No value_loss data", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    
    # Entropy
    if has_entropy:
        ax = axes[2]
        sub = df[["updates", "entropy"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ax.plot(sub["updates"], sub["entropy"], lw=0.8, alpha=0.3, color="0.5")
            ema = exponential_moving_average(sub["entropy"], alpha=0.1)
            ax.plot(sub["updates"], ema, lw=2.5, color="#9467bd", label="EMA (α=0.1)")
            ax.set_xlabel("Updates", fontsize=11)
            ax.set_ylabel("Entropy", fontsize=11)
            ax.set_title("Policy Entropy", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No entropy data", ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ppo_losses.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved ppo_losses.png")


def plot_ppo_diagnostics(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Plot PPO diagnostic metrics (KL divergence, clip fraction)"""
    has_kl = "approx_kl" in df.columns
    has_clip = "clipfrac" in df.columns
    
    if not any([has_kl, has_clip]):
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    suffix = f" ({env_id})" if env_id else ""
    
    # Approx KL
    if has_kl:
        ax = axes[0]
        sub = df[["updates", "approx_kl"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ax.plot(sub["updates"], sub["approx_kl"], lw=0.8, alpha=0.3, color="0.5")
            ema = exponential_moving_average(sub["approx_kl"], alpha=0.1)
            ax.plot(sub["updates"], ema, lw=2.5, color="#2ca02c", label="EMA (α=0.1)")
            ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label="Target KL=0.01")
            ax.set_xlabel("Updates", fontsize=11)
            ax.set_ylabel("Approx KL Divergence", fontsize=11)
            ax.set_title("KL Divergence (Policy Change)", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No approx_kl data", ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Clip fraction
    if has_clip:
        ax = axes[1]
        sub = df[["updates", "clipfrac"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ax.plot(sub["updates"], sub["clipfrac"], lw=0.8, alpha=0.3, color="0.5")
            ema = exponential_moving_average(sub["clipfrac"], alpha=0.1)
            ax.plot(sub["updates"], ema, lw=2.5, color="#8B0000", label="EMA (α=0.1)")
            ax.set_xlabel("Updates", fontsize=11)
            ax.set_ylabel("Clip Fraction", fontsize=11)
            ax.set_title("Fraction of Clipped Updates", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No clipfrac data", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ppo_diagnostics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved ppo_diagnostics.png")


def create_ppo_dashboard(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Create a 6-panel dashboard for PPO training"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f"PPO Training Dashboard: {env_id or 'PPO'}", fontsize=16, fontweight='bold')
    
    # Top-left: Episode returns
    ax = fig.add_subplot(gs[0, 0])
    if {"step", "episode_return"} <= set(df.columns):
        sub = df[["step", "episode_return"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("step").groupby("step", as_index=False).mean()
            ema = exponential_moving_average(sub["episode_return"], alpha=0.02)
            ax.plot(sub["step"], sub["episode_return"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["step"], ema, lw=2.5, color="#DC143C")
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Episode Return", fontsize=10)
            ax.set_title("Training Returns", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Top-right: Evaluation
    ax = fig.add_subplot(gs[0, 1])
    if {"step", "eval_return_mean"} <= set(df.columns):
        sub = df[["step", "eval_return_mean"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("step").groupby("step", as_index=False).mean()
            ax.plot(sub["step"], sub["eval_return_mean"], lw=2.5, 
                   color="#1f77b4", marker='o', markersize=3)
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Eval Return", fontsize=10)
            ax.set_title("Evaluation Performance", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Middle-left: Policy loss
    ax = fig.add_subplot(gs[1, 0])
    if {"updates", "policy_loss"} <= set(df.columns):
        sub = df[["updates", "policy_loss"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["policy_loss"], alpha=0.1)
            ax.plot(sub["updates"], sub["policy_loss"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#DC143C")
            ax.set_xlabel("Updates", fontsize=10)
            ax.set_ylabel("Policy Loss", fontsize=10)
            ax.set_title("Policy Loss", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Middle-right: Value loss
    ax = fig.add_subplot(gs[1, 1])
    if {"updates", "value_loss"} <= set(df.columns):
        sub = df[["updates", "value_loss"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["value_loss"], alpha=0.1)
            ax.plot(sub["updates"], sub["value_loss"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#FF8C00")
            ax.set_xlabel("Updates", fontsize=10)
            ax.set_ylabel("Value Loss", fontsize=10)
            ax.set_title("Value Loss", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Bottom-left: Entropy
    ax = fig.add_subplot(gs[2, 0])
    if {"updates", "entropy"} <= set(df.columns):
        sub = df[["updates", "entropy"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["entropy"], alpha=0.1)
            ax.plot(sub["updates"], sub["entropy"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#9467bd")
            ax.set_xlabel("Updates", fontsize=10)
            ax.set_ylabel("Entropy", fontsize=10)
            ax.set_title("Policy Entropy", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Bottom-right: KL divergence
    ax = fig.add_subplot(gs[2, 1])
    if {"updates", "approx_kl"} <= set(df.columns):
        sub = df[["updates", "approx_kl"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["approx_kl"], alpha=0.1)
            ax.plot(sub["updates"], sub["approx_kl"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#2ca02c")
            ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.set_xlabel("Updates", fontsize=10)
            ax.set_ylabel("Approx KL", fontsize=10)
            ax.set_title("KL Divergence", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved dashboard.png")


def plot_training(csv_path: str, out_dir: Optional[str] = None, ema_alpha: float = 0.02):
    """Main plotting function - generates all training visualizations"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    out_dir = out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    print(f"\n📊 Loaded {len(df):,} rows from {csv_path}")
    
    env_id = df["env_id"].iloc[0] if "env_id" in df.columns and len(df) else None
    if env_id:
        print(f"Environment: {env_id}")
    
    algo = detect_algorithm(df)
    print(f"Detected algorithm: {algo}")
    
    print("\nGenerating plots...")
    print("-" * 50)
    
    # Training curves
    plot_metric_with_smoothing(
        df, "step", "episode_return",
        xlabel="Environment Steps",
        ylabel="Episode Return",
        title=f"Training Returns ({env_id or algo})",
        filename="train_return.png",
        out_dir=out_dir,
        env_id=env_id,
        ema_alphas=[0.1, ema_alpha],
        show_raw=True
    )
    
    plot_metric_with_smoothing(
        df, "step", "episode_length",
        xlabel="Environment Steps",
        ylabel="Episode Length",
        title=f"Episode Length ({env_id or algo})",
        filename="episode_length.png",
        out_dir=out_dir,
        env_id=env_id,
        ema_alphas=[0.05],
        show_raw=True
    )
    
    # Evaluation
    plot_evaluation_performance(df, out_dir, env_id)
    
    # Algorithm-specific plots
    if algo == "PPO":
        plot_ppo_losses(df, out_dir, env_id)
        plot_ppo_diagnostics(df, out_dir, env_id)
        create_ppo_dashboard(df, out_dir, env_id)
    elif algo == "DQN":
        # Keep DQN plots if needed
        plot_metric_with_smoothing(
            df, "updates", "loss",
            xlabel="Gradient Updates",
            ylabel="TD Loss",
            title=f"Training Loss ({env_id or 'DQN'})",
            filename="loss.png",
            out_dir=out_dir,
            env_id=env_id,
            ema_alphas=[0.1, 0.05],
            show_raw=True
        )
        
        plot_metric_with_smoothing(
            df, "updates", "q_max",
            xlabel="Gradient Updates",
            ylabel="Max Q-value",
            title=f"Q-value Progression ({env_id or 'DQN'})",
            filename="q_max.png",
            out_dir=out_dir,
            env_id=env_id,
            ema_alphas=[0.1, 0.05],
            show_raw=True
        )
    
    # FPS (common to both)
    plot_fps(df, out_dir, env_id)
    
    print("-" * 50)
    print(f"✅ All plots saved to: {out_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <metrics.csv> [ema_alpha]")
        print("  ema_alpha: smoothing factor (default: 0.02, lower = more smooth)")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    ema_alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    
    plot_training(csv_path, ema_alpha=ema_alpha)