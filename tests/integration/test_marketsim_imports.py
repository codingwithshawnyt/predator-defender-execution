"""Integration smoke test: PyMarketSim is installed and importable."""

from __future__ import annotations


def test_marketsim_importable() -> None:
    """If this fails, the marketsim git dependency didn't install correctly.

    See REPO_SETUP.md Step 19 fallback procedure.
    """
    import marketsim

    assert marketsim is not None
