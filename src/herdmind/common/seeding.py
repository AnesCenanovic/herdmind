"""Seeding: the single source of randomness for a run.

Every stochastic choice takes a ``numpy.random.Generator`` built here, so a
run is reproducible from its seed alone. No global RNG state is touched.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return the run's RNG; pass it down, never re-create it."""
    return np.random.default_rng(seed)
