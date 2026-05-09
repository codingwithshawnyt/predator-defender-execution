"""Build the empirical payoff matrix over discrete strategy archetypes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayoffMatrixConfig:
    n_rollouts_per_cell: int = 100
    seed: int = 0


def build_payoff_matrix(config: PayoffMatrixConfig) -> None:
    """Phase-3 deliverable."""
    raise NotImplementedError("Phase-3 deliverable.")
