"""The three orthogonal regime axes used in the headline analysis.

- Volatility regime: low / mid / high (3 levels)
- Spread regime: tight / wide (2 levels)
- Defender-size regime: small / medium / large Q (3 levels)

Total: 18 cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VolatilityRegime(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class SpreadRegime(Enum):
    TIGHT = "tight"
    WIDE = "wide"


class SizeRegime(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class Regime:
    volatility: VolatilityRegime
    spread: SpreadRegime
    size: SizeRegime

    @property
    def cell_id(self) -> str:
        return f"{self.volatility.value}_{self.spread.value}_{self.size.value}"


def all_regimes() -> list[Regime]:
    """Cartesian product of all regime levels — 18 cells."""
    return [Regime(v, s, sz) for v in VolatilityRegime for s in SpreadRegime for sz in SizeRegime]
