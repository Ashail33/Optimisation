"""
Grey Wolf Optimizer (GWO).

The pack is led by the three best wolves (alpha, beta, delta).  Every wolf
moves to the average of three positions, each obtained by "encircling" one
leader.  The coefficient ``a`` decreases linearly from 2 to 0 over the run,
shifting the pack from exploration (|A| > 1) to exploitation (|A| < 1).

References
----------
Mirjalili, S., Mirjalili, S. M. & Lewis, A. (2014). Grey Wolf Optimizer.
*Advances in Engineering Software*, 69, 46–61.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class GreyWolfOptimiser(PopulationOptimiser):
    """Grey Wolf Optimizer for continuous problems.

    Parameters are the budget / seed options of
    :class:`~optim.population.PopulationOptimiser`.  ``a`` decays with the
    fraction of the budget used, so set ``max_iterations`` or
    ``max_evaluations`` to the budget you actually intend to spend.
    """

    def _min_population(self) -> int:
        return 3

    def _initialise(self, problem, state, rng):
        order = np.argsort(state.F)[:3]
        state.leaders = state.X[order].copy()
        state.leader_f = state.F[order].copy()

    def _step(self, problem, state, rng):
        a = 2.0 * (1.0 - state.progress)
        state.action = {"type": "gwo/encircle", "a": a}
        n, d = state.X.shape
        new = np.zeros_like(state.X)
        for leader in state.leaders:
            A = 2 * a * rng.random((n, d)) - a
            C = 2 * rng.random((n, d))
            new += leader - A * np.abs(C * leader - state.X)
        state.X = problem.clip(new / 3.0)
        state.F = problem.evaluate(state.X)

        pool_X = np.vstack([state.leaders, state.X])
        pool_F = np.concatenate([state.leader_f, state.F])
        order = np.argsort(pool_F, kind="stable")[:3]
        state.leaders = pool_X[order].copy()
        state.leader_f = pool_F[order].copy()
