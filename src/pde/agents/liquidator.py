"""LiquidatorAgent: PyMarketSim agent that sells a parent order via RL-controlled volume.

This agent implements the marketsim.agent.agent.Agent ABC so it can be
registered with a Market.  The RL wrapper calls
``take_action(volume=...)`` to inject the RL-chosen volume as a
cross-spread sell order.

Design mirrors Lillo & Macri (2024) two-liquidator setup:
  - Each liquidator starts with inventory q_0 and must sell to zero.
  - Action = volume to sell at this step (continuous, bounded by
    remaining inventory).
  - Order placed at best_bid (aggressive, crosses spread).
"""

from __future__ import annotations

from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import SELL
from marketsim.fourheap.order import Order
from marketsim.market.market import Market


class LiquidatorAgent(Agent):
    """A liquidating agent that sells a fixed parent order.

    Parameters
    ----------
    agent_id : int
        Unique agent identifier.
    market : Market
        The market this agent trades in.
    q_0 : float
        Initial inventory (number of shares to liquidate).
    order_size : float
        Maximum volume per single order (unused if *volume* is passed
        to ``take_action``; kept for ABC compatibility).
    """

    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_0: float = 100.0,
        order_size: float = 100.0,
    ) -> None:
        self.agent_id = agent_id
        self.market = market
        self.q_0 = q_0
        self.order_size = order_size
        self.position: float = 0.0
        self.cash: float = 0.0
        self._prev_cash: float = 0.0
        self._next_order_id: int = agent_id * 1_000_000

    # ------------------------------------------------------------------
    # Agent ABC
    # ------------------------------------------------------------------

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, volume: float | None = None, **_kwargs) -> list[Order]:
        """Place a sell order for *volume* shares at the best bid.

        If *volume* is None the agent sells a default slice
        (``order_size`` or remaining inventory, whichever is smaller).
        """
        remaining = self.remaining_inventory
        if volume is None:
            volume = min(self.order_size, remaining)
        volume = max(0.0, min(volume, remaining))
        if volume <= 0:
            return []

        best_bid = self.market.order_book.get_best_bid()
        import math

        if math.isinf(best_bid):
            fund = self.market.get_fundamental_value()
            best_bid = fund - 1.0

        t = self.market.get_time()
        order = Order(
            price=best_bid,
            quantity=volume,
            agent_id=self.agent_id,
            time=t,
            order_type=SELL,
            order_id=self._next_order_id,
        )
        self._next_order_id += 1
        return [order]

    def update_position(self, quantity: float, cash: float) -> None:
        """Called by the market wrapper after a match.

        *quantity* is signed (+1 for buy, -1 for sell) so for a
        seller position decreases (goes negative).
        *cash* is the cash change (positive = received cash).
        """
        self.position += quantity
        self.cash += cash

    def get_pos_value(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.position = 0.0
        self.cash = 0.0
        self._prev_cash = 0.0

    # ------------------------------------------------------------------
    # Liquidator-specific helpers
    # ------------------------------------------------------------------

    @property
    def remaining_inventory(self) -> float:
        """Shares still to sell.  Positive until fully liquidated."""
        sold = -self.position
        return max(0.0, self.q_0 - sold)

    @property
    def is_done(self) -> bool:
        return self.remaining_inventory <= 0

    def __str__(self) -> str:
        return f"Liq{self.agent_id}"
