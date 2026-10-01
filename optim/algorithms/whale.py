"""
Whale Optimization Algorithm (WOA).

Mimics humpback whales' bubble-net feeding.  Each whale either encircles the
best solution, searches around a random whale (exploration, when |A| >= 1),
or spirals towards the best solution along a logarithmic spiral.

References
----------
Mirjalili, S. & Lewis, A. (2016). The Whale Optimization Algorithm.
*Advances in Engineering Software*, 95, 51–67.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class WhaleOptimiser(PopulationOptimiser):
    """Whale Optimization Algorithm for continuous problems.

    Parameters
    ----------
    spiral_b : float
        Shape constant of the logarithmic spiral.  Default 1.0.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
    """

    def __init__(self, spiral_b: float = 1.0, population_size=None, **kwargs) -> None:
        self.spiral_b = spiral_b
        super().__init__(population_size=population_size, **kwargs)

    def _step(self, problem, state, rng):
        a = 2.0 * (1.0 - state.progress)
        a2 = -1.0 - state.progress
        best = problem.best_x
        n, d = state.X.shape
        new = np.empty_like(state.X)
        for i in range(n):
            x = state.X[i]
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            if rng.random() < 0.5:
                target = best if abs(A) < 1 else state.X[rng.integers(n)]
                new[i] = target - A * np.abs(C * target - x)
            else:
                l = (a2 - 1) * rng.random() + 1
                new[i] = (
                    np.abs(best - x) * np.exp(self.spiral_b * l) * np.cos(2 * np.pi * l)
                    + best
                )
        state.X = problem.clip(new)
        state.F = problem.evaluate(state.X)
