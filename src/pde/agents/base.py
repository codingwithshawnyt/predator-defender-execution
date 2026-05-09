"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Abstract agent. Both defender and predator implement this."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Select an action given the current observation."""

    def reset(self) -> None:
        """Optional per-episode reset hook."""
        return None

    def observe(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Optional learning hook for online / RL agents."""
        return None
