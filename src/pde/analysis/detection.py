"""Detection signature analysis: what does the defender leak about its inventory?

Three measurements:
1. Lag-h cross-correlation between defender's inventory and aggregate OFI
2. Power spectral density of defender's order arrivals
3. Mutual information between observable LOB statistics and hidden inventory

Phase-4 deliverable.
"""

from __future__ import annotations
