"""
Sine Cosine Algorithm (SCA).

Solutions oscillate around the best solution using sine and cosine
functions; the amplitude ``r1`` decreases linearly so the search moves from
exploration to exploitation.

References
----------
Mirjalili, S. (2016). SCA: A Sine Cosine Algorithm for solving optimization
problems. *Knowledge-Based Systems*, 96, 120–133.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class SineCosineOptimiser(PopulationOptimiser):
    """Sine Cosine Algorithm for continuous problems.

    Parameters
    ----------
    a : float
        Initial amplitude of the oscillation.  Default 2.0.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
    """

    def __init__(self, a: float = 2.0, population_size=None, **kwargs) -> None:
        self.a = a
        super().__init__(population_size=population_size, **kwargs)

    def _step(self, problem, state, rng):
        n, d = state.X.shape
        best = problem.best_x
        r1 = self.a * (1.0 - state.progress)
        r2 = 2 * np.pi * rng.random((n, d))
        r3 = 2 * rng.random((n, d))
        wave = np.where(rng.random((n, d)) < 0.5, np.sin(r2), np.cos(r2))
        state.X = problem.clip(state.X + r1 * wave * np.abs(r3 * best - state.X))
        state.F = problem.evaluate(state.X)
