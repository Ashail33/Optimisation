"""
Teaching–Learning-Based Optimization (TLBO).

A parameter-free algorithm in two phases: in the *teacher* phase learners
move towards the best solution and away from the class mean; in the
*learner* phase each learner moves towards a better random peer (or away
from a worse one).  Both phases replace greedily.

References
----------
Rao, R. V., Savsani, V. J. & Vakharia, D. P. (2011). Teaching–learning-based
optimization: A novel method for constrained mechanical design optimization
problems. *Computer-Aided Design*, 43(3), 303–315.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class TLBOOptimiser(PopulationOptimiser):
    """Teaching–Learning-Based Optimization (no algorithm-specific parameters).

    Each iteration uses two evaluations per learner.
    """

    def _step(self, problem, state, rng):
        n, d = state.X.shape

        teacher = state.X[np.argmin(state.F)]
        tf = rng.integers(1, 3, size=(n, 1))
        cand = state.X + rng.random((n, d)) * (teacher - tf * state.X.mean(axis=0))
        cand = problem.clip(cand)
        self._greedy_replace(state, cand, problem.evaluate(cand))
        if problem.exhausted:
            return

        peers = self._random_others(n, rng)[:, 0]
        towards = np.where(
            (state.F < state.F[peers])[:, None],
            state.X - state.X[peers],
            state.X[peers] - state.X,
        )
        cand = problem.clip(state.X + rng.random((n, d)) * towards)
        self._greedy_replace(state, cand, problem.evaluate(cand))
