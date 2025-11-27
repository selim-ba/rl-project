# utils/plot_training.py  (last update : 27/11/2025)

from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


plt.style.use("seaborn-v0_8-whitegrid")

def exponential_moving_average(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Exponential moving average with smoothing factor alpha.
    
    Lower alpha = more smoothing (0.01-0.05 recommended for noisy RL data)
    Higher alpha = less smoothing (closer to raw data)
    
    Args:
        series: Data series to smooth
        alpha: Smoothing factor (0 < alpha < 1)
    
    Returns:
        Smoothed series
    """
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
    """
    Plot a metric with multiple smoothing options.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis (e.g., "step")
        y_col: Column name for y-axis (e.g., "episode_return")
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        filename: Output filename
        out_dir: Output directory
        env_id: Environment ID for automatic y-axis scaling
        ema_alphas: List of EMA smoothing factors
        show_raw: Whether to show raw data
    """
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
    
    # Multiple EMA smoothing levels
    colors = ["#FF8C00", "#DC143C", "#8B0000"]  # Orange, Crimson, Dark red
    for i, alpha in enumerate(ema_alphas):
        ema = exponential_moving_average(sub[y_col], alpha=alpha)
        ax.plot(xs, ema, lw=2.0 + i*0.5, color=colors[i], 
                label=f"EMA (α={alpha})", zorder=2+i, alpha=0.9)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(frameon=True, framealpha=0.95, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    _maybe_set_atari_ylim(env_id, y_col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {filename}")


def plot_epsilon_schedule(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Plot epsilon exploration schedule to verify decay"""
    if "step" not in df.columns or "epsilon" not in df.columns:
        return
    
    sub = df[["step", "epsilon"]].dropna()
    if sub.empty:
        return
    
    sub = sub.sort_values("step").groupby("step", as_index=False).mean()
    
    plt.figure(figsize=(8, 4))
    plt.plot(sub["step"], sub["epsilon"], lw=2, color="#9467bd", alpha=0.9)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, 
                linewidth=1.5, label="Target ε=0.1")
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Epsilon (ε)", fontsize=12)
    
    suffix = f" ({env_id})" if env_id else ""
    plt.title(f"Exploration Schedule{suffix}", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epsilon_schedule.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved epsilon_schedule.png")


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
    
    # Smooth with EMA
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
    """
    Plot evaluation performance with confidence bands (Nature DQN style).
    No smoothing needed - each point is already aggregated over multiple episodes.
    """
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
    
    # Group by step and average (in case of duplicate evaluations)
    sub = sub.sort_values("step").groupby("step", as_index=False).mean()
    
    steps = sub["step"].to_numpy()
    mean = sub["eval_return_mean"].to_numpy()
    
    # Get standard deviation or approximate from min/max
    std = None
    if "eval_return_std" in sub.columns:
        std = sub["eval_return_std"].to_numpy()
    elif {"eval_return_min", "eval_return_max"} <= set(sub.columns):
        # Approximate std as half the range
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
    
    _maybe_set_atari_ylim(env_id, "episode_return")
    
    ax.legend(frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_return.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved eval_return.png")


def create_dashboard(df: pd.DataFrame, out_dir: str, env_id: Optional[str] = None):
    """Create a 4-panel dashboard summarizing training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Dashboard: {env_id or 'DQN'}", fontsize=16, fontweight='bold')
    
    suffix = f" ({env_id})" if env_id else ""
    
    # Top-left: Episode returns vs steps
    ax = axes[0, 0]
    if {"step", "episode_return"} <= set(df.columns):
        sub = df[["step", "episode_return"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("step").groupby("step", as_index=False).mean()
            ema = exponential_moving_average(sub["episode_return"], alpha=0.02)
            ax.plot(sub["step"], sub["episode_return"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["step"], ema, lw=2.5, color="#DC143C")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Episode Return")
            ax.set_title("Training Returns")
            ax.grid(True, alpha=0.3)
    
    # Top-right: Evaluation performance
    ax = axes[0, 1]
    if {"step", "eval_return_mean"} <= set(df.columns):
        sub = df[["step", "eval_return_mean"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("step").groupby("step", as_index=False).mean()
            ax.plot(sub["step"], sub["eval_return_mean"], lw=2.5, 
                   color="#1f77b4", marker='o', markersize=4)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Eval Return (mean)")
            ax.set_title("Evaluation Performance")
            ax.grid(True, alpha=0.3)
    
    # Bottom-left: Q-values
    ax = axes[1, 0]
    if {"updates", "q_max"} <= set(df.columns):
        sub = df[["updates", "q_max"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["q_max"], alpha=0.05)
            ax.plot(sub["updates"], sub["q_max"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#FF8C00")
            ax.set_xlabel("Updates")
            ax.set_ylabel("Max Q-value")
            ax.set_title("Q-value Progression")
            ax.grid(True, alpha=0.3)
    
    # Bottom-right: Loss
    ax = axes[1, 1]
    if {"updates", "loss"} <= set(df.columns):
        sub = df[["updates", "loss"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("updates").groupby("updates", as_index=False).mean()
            ema = exponential_moving_average(sub["loss"], alpha=0.05)
            ax.plot(sub["updates"], sub["loss"], lw=0.5, alpha=0.15, color="0.5")
            ax.plot(sub["updates"], ema, lw=2.5, color="#9467bd")
            ax.set_xlabel("Updates")
            ax.set_ylabel("TD Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved dashboard.png")


def plot_training(csv_path: str, out_dir: Optional[str] = None, 
                 ema_alpha: float = 0.02):
    """
    Main plotting function - generates all training visualizations.
    
    Args:
        csv_path: Path to metrics.csv file
        out_dir: Output directory (defaults to same dir as CSV)
        ema_alpha: Alpha for exponential moving average (lower = more smoothing)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    out_dir = out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    print(f"\n📊 Loaded {len(df):,} rows from {csv_path}")
    
    env_id = df["env_id"].iloc[0] if "env_id" in df.columns and len(df) else None
    if env_id:
        print(f"Environment: {env_id}")
    
    print("\nGenerating plots...")
    print("-" * 50)
    
    # Training curves 
    plot_metric_with_smoothing(
        df, "step", "episode_return",
        xlabel="Environment Steps",
        ylabel="Episode Return",
        title=f"Training Returns vs Steps ({env_id or 'DQN'})",
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
        title=f"Episode Length Over Time ({env_id or 'DQN'})",
        filename="episode_length.png",
        out_dir=out_dir,
        env_id=env_id,
        ema_alphas=[0.05],
        show_raw=True
    )
    
    # Learning curves
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
    
    plot_evaluation_performance(df, out_dir, env_id)
    plot_epsilon_schedule(df, out_dir, env_id)
    plot_fps(df, out_dir, env_id)
    create_dashboard(df, out_dir, env_id)
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
