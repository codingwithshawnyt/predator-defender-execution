"""Figure 1: Lillo-Macri smoke replicate — IS convergence and reward curves.

Reads ``artifacts/logs/smoke_replicate/results.json`` produced by
``pde.training.smoke_replicate`` and renders:
  - Panel A: episode IS for both liquidators (rolling mean)
  - Panel B: episode cumulative reward for both liquidators
  - Panel C: IS scatter (liq_0 vs liq_1) last 500 episodes

Saves to ``artifacts/figures/fig01_smoke_replicate.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(
    results_path: str | Path = "artifacts/logs/smoke_replicate/results.json",
    save_dir: str | Path = "artifacts/figures",
) -> None:
    results_path = Path(results_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    is0 = np.array(data["episode_is_liq_0"])
    is1 = np.array(data["episode_is_liq_1"])
    r0 = np.array(data["episode_rewards_liq_0"])
    r1 = np.array(data["episode_rewards_liq_1"])
    n_ep = len(is0)
    window = max(1, n_ep // 50)

    def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: IS convergence
    ax = axes[0]
    rm0 = rolling_mean(is0, window)
    rm1 = rolling_mean(is1, window)
    x0 = np.arange(len(rm0)) + window
    x1 = np.arange(len(rm1)) + window
    ax.plot(x0, rm0, label="liq_0", alpha=0.8)
    ax.plot(x1, rm1, label="liq_1", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Implementation Shortfall")
    ax.set_title("A: IS Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Reward convergence
    ax = axes[1]
    rr0 = rolling_mean(r0, window)
    rr1 = rolling_mean(r1, window)
    xr0 = np.arange(len(rr0)) + window
    xr1 = np.arange(len(rr1)) + window
    ax.plot(xr0, rr0, label="liq_0", alpha=0.8)
    ax.plot(xr1, rr1, label="liq_1", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("B: Reward Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: IS scatter last 500
    ax = axes[2]
    tail = min(500, n_ep)
    ax.scatter(is0[-tail:], is1[-tail:], alpha=0.3, s=10)
    lims = [min(is0[-tail:].min(), is1[-tail:].min()), max(is0[-tail:].max(), is1[-tail:].max())]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Symmetric")
    ax.set_xlabel("IS liq_0")
    ax.set_ylabel("IS liq_1")
    ax.set_title("C: IS Scatter (last 500)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Lillo-Macri Smoke Replicate", fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = save_dir / "fig01_smoke_replicate.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    plot()
