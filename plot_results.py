"""
Plot comparison results from training logs and comparison runs.

Usage:
    python plot_results.py
"""

import json
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window=50):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_curves():
    """Plot all training curves."""
    modes = {
        "rl": ("#3498db", "DQN"),
        "improved_rl": ("#e67e22", "DQN + Congestion"),
        "ppo": ("#2ecc71", "PPO"),
        "improved_ppo": ("#e74c3c", "PPO + Congestion"),
        "mappo": ("#9b59b6", "MAPPO (CTDE)"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Curves: DQN vs PPO × Base vs Congestion-Aware",
                 fontsize=16, fontweight='bold')

    for mode, (color, label) in modes.items():
        path = f"checkpoints/{mode}_training_log.json"
        if not os.path.exists(path):
            print(f"  ⚠ {path} not found, skipping.")
            continue

        with open(path) as f:
            logs = json.load(f)

        episodes = [l["episode"] for l in logs]
        rewards = [l["avg_reward"] for l in logs]
        completions = [l["avg_completion"] for l in logs]
        collisions = [l["avg_collisions"] for l in logs]

        axes[0, 0].plot(smooth(rewards), color=color, label=label, linewidth=1.5, alpha=0.9)
        axes[0, 1].plot(smooth(completions), color=color, label=label, linewidth=1.5, alpha=0.9)
        axes[1, 0].plot(smooth(collisions), color=color, label=label, linewidth=1.5, alpha=0.9)

        # Entropy or epsilon (different for DQN/PPO)
        if "entropy" in logs[0]:
            vals = [l.get("entropy", 0) for l in logs]
            axes[1, 1].plot(smooth(vals), color=color, label=f"{label} entropy",
                           linewidth=1.5, alpha=0.9)
        elif "epsilon" in logs[0]:
            vals = [l.get("epsilon", 0) for l in logs]
            axes[1, 1].plot(vals, color=color, label=f"{label} ε",
                           linewidth=1.5, alpha=0.9)

    for ax in axes.flat:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0, 0].set_title("Average Reward (↑ better)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 1].set_title("Task Completion Rate (↑ better)")
    axes[0, 1].set_xlabel("Episode")
    axes[1, 0].set_title("Average Collisions (↓ better)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_title("Exploration (ε / Entropy)")
    axes[1, 1].set_xlabel("Episode")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=200, bbox_inches='tight')
    print("  ✅ Saved: results/training_curves.png")
    plt.close()


def plot_comparison_bars():
    """Plot bar chart comparison."""
    path = "results/comparison.json"
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found. Run 'python main.py --mode compare' first.")
        return

    with open(path) as f:
        results = json.load(f)

    modes = list(results.keys())
    colors = {
        "bfs": "#95a5a6",
        "rl": "#3498db",
        "improved_rl": "#e67e22",
        "ppo": "#2ecc71",
        "improved_ppo": "#e74c3c",
        "mappo": "#9b59b6",
    }
    labels = {
        "bfs": "BFS\n(Baseline)",
        "rl": "DQN",
        "improved_rl": "DQN +\nCongestion",
        "ppo": "PPO",
        "improved_ppo": "PPO +\nCongestion",
        "mappo": "MAPPO\n(CTDE)",
    }

    active_modes = [m for m in modes if results[m]]
    bar_colors = [colors.get(m, "#999") for m in active_modes]
    bar_labels = [labels.get(m, m) for m in active_modes]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Algorithm Comparison: BFS vs DQN vs PPO",
                 fontsize=16, fontweight='bold')

    # Steps
    vals = [np.mean([r["steps"] for r in results[m]]) for m in active_modes]
    errs = [np.std([r["steps"] for r in results[m]]) for m in active_modes]
    axes[0].bar(bar_labels, vals, color=bar_colors, yerr=errs, capsize=5)
    axes[0].set_title("Avg Steps (↓ better)")
    axes[0].set_ylabel("Steps")

    # Collisions
    vals = [np.mean([r["collisions"] for r in results[m]]) for m in active_modes]
    errs = [np.std([r["collisions"] for r in results[m]]) for m in active_modes]
    axes[1].bar(bar_labels, vals, color=bar_colors, yerr=errs, capsize=5)
    axes[1].set_title("Avg Collisions (↓ better)")
    axes[1].set_ylabel("Collisions")

    # Completion
    vals = [np.mean([r["completion"] for r in results[m]]) for m in active_modes]
    errs = [np.std([r["completion"] for r in results[m]]) for m in active_modes]
    axes[2].bar(bar_labels, vals, color=bar_colors, yerr=errs, capsize=5)
    axes[2].set_title("Task Completion (↑ better)")
    axes[2].set_ylabel("Completion %")
    axes[2].set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig("results/comparison_bars.png", dpi=200, bbox_inches='tight')
    print("  ✅ Saved: results/comparison_bars.png")
    plt.close()


if __name__ == "__main__":
    print("  📊 Generating plots...")
    plot_training_curves()
    plot_comparison_bars()
    print("  Done!")
