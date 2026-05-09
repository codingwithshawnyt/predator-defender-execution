"""Smoke tests: imports, basic data structures, no GPU required."""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    import pde

    assert pde.__version__ == "0.1.0"


def test_subpackage_imports() -> None:
    from pde import agents, analysis, egta, eval, figures, regimes, sim, training

    assert agents is not None
    assert analysis is not None
    assert egta is not None
    assert eval is not None
    assert figures is not None
    assert regimes is not None
    assert sim is not None
    assert training is not None


def test_regime_grid_size() -> None:
    from pde.regimes.axes import all_regimes

    regimes = all_regimes()
    # 3 volatility * 2 spread * 3 size = 18 cells
    assert len(regimes) == 18
    assert len({r.cell_id for r in regimes}) == 18


def test_implementation_shortfall_signature() -> None:
    import numpy as np

    from pde.eval.metrics import implementation_shortfall

    arrival = 100.0
    prices = np.array([99.5, 99.0, 98.5])
    qtys = np.array([10.0, 10.0, 10.0])
    is_value = implementation_shortfall(arrival, prices, qtys)
    assert is_value > 0
    expected = (100.0 - 99.5) * 10 + (100.0 - 99.0) * 10 + (100.0 - 98.5) * 10
    assert is_value == pytest.approx(expected)


def test_implementation_shortfall_validates_lengths() -> None:
    import numpy as np

    from pde.eval.metrics import implementation_shortfall

    with pytest.raises(ValueError):
        implementation_shortfall(100.0, np.array([99.0]), np.array([10.0, 10.0]))
