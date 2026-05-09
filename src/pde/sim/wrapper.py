"""Thin wrapper around PyMarketSim for the predator-defender setup. Phase-1 stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single simulator episode."""

    horizon_seconds: float = 1800.0
    control_interval_seconds: float = 1.0
    n_background_traders: int = 100
    fundamental_volatility: float = 1.0
    seed: int = 0


class MarketEnv:
    """Defender + K-predator environment built on PyMarketSim. Phase-1 stub."""

    def __init__(self, config: SimConfig, n_predators: int = 1) -> None:
        self.config = config
        self.n_predators = n_predators
        self._sim: Any | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("Phase-1 stub.")

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        raise NotImplementedError("Phase-1 stub.")
