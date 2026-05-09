"""Per-episode metrics: implementation shortfall, regret, profit."""

from __future__ import annotations

import numpy as np


def implementation_shortfall(
    arrival_price: float,
    executed_prices: np.ndarray,
    executed_quantities: np.ndarray,
) -> float:
    """Standard implementation shortfall vs. arrival mid-price.

    IS = sum_t (arrival_price - executed_price_t) * executed_qty_t
    """
    if len(executed_prices) != len(executed_quantities):
        raise ValueError("prices and quantities must align")
    return float(np.sum((arrival_price - executed_prices) * executed_quantities))


def regret_vs_baseline(is_under_predator: float, is_no_predator: float) -> float:
    """Defender regret relative to no-predator baseline (positive = predation cost)."""
    return is_under_predator - is_no_predator
