"""Rule-based baseline policies for defender and predator roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pde.agents.base import Agent


@dataclass
class TWAPDefender(Agent):
    """Equal-share-per-step liquidation. Phase-1 stub."""

    total_quantity: float
    n_steps: int

    def act(self, observation: Any) -> Any:
        raise NotImplementedError("Phase-1 stub.")


@dataclass
class AlmgrenChrissDefender(Agent):
    """Closed-form Almgren-Chriss optimal liquidation under no-predator assumption."""

    total_quantity: float
    n_steps: int
    risk_aversion: float
    permanent_impact: float
    temporary_impact: float
    volatility: float

    def act(self, observation: Any) -> Any:
        raise NotImplementedError("Phase-1 stub.")


@dataclass
class NoOpPredator(Agent):
    """Predator that never trades — no-predator oracle baseline."""

    def act(self, observation: Any) -> Any:
        return None
