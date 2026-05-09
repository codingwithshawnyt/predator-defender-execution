"""PPO trainer for defender and predator policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    total_timesteps: int = 1_000_000
    n_parallel_envs: int = 16
    seed: int = 0


def train_defender(config: PPOConfig) -> None:
    """Phase-2 stub."""
    raise NotImplementedError("Phase-2 deliverable.")


def train_adversarial(
    config: PPOConfig, n_predators: int = 1, alternation_rounds: int = 10
) -> None:
    """Phase-3 stub."""
    raise NotImplementedError("Phase-3 deliverable.")
