"""
Jaya algorithm.

Parameter-free: every solution moves towards the best and away from the
worst, ``x' = x + r1 (best - |x|) - r2 (worst - |x|)``, replacing greedily.

References
----------
Rao, R. V. (2016). Jaya: A simple and new optimization algorithm for solving
constrained and unconstrained optimization problems. *International Journal
of Industrial Engineering Computations*, 7(1), 19–34.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class JayaOptimiser(PopulationOptimiser):
    """Jaya algorithm (no algorithm-specific parameters)."""

    def _step(self, problem, state, rng):
        n, d = state.X.shape
        best = state.X[np.argmin(state.F)]
        worst = state.X[np.argmax(state.F)]
        absX = np.abs(state.X)
        cand = (
            state.X
            + rng.random((n, d)) * (best - absX)
            - rng.random((n, d)) * (worst - absX)
        )
        cand = problem.clip(cand)
        self._greedy_replace(state, cand, problem.evaluate(cand))
