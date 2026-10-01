"""
Bat Algorithm (BA).

Bats fly with frequency-tuned velocities towards the best solution; with a
probability governed by their pulse rate they instead take a local random
walk around the best.  A new solution is accepted if it improves and a
loudness test passes; accepted bats get quieter and pulse faster.

References
----------
Yang, X.-S. (2010). A new metaheuristic bat-inspired algorithm.
*Nature Inspired Cooperative Strategies for Optimization*, 65–74.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class BatOptimiser(PopulationOptimiser):
    """Bat Algorithm for continuous problems.

    Parameters
    ----------
    f_min, f_max : float
        Frequency range.  Defaults 0 and 2.
    loudness : float
        Initial loudness ``A``.  Default 1.0.
    pulse_rate : float
        Final pulse emission rate ``r0``.  Default 0.5.
    alpha, gamma : float
        Loudness decay and pulse-rate growth constants.  Default 0.9 each.
    local_scale : float
        Size of the local random walk as a fraction of each range.
        Default 0.01.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
    """

    def __init__(
        self,
        f_min: float = 0.0,
        f_max: float = 2.0,
        loudness: float = 1.0,
        pulse_rate: float = 0.5,
        alpha: float = 0.9,
        gamma: float = 0.9,
        local_scale: float = 0.01,
        population_size=None,
        **kwargs,
    ) -> None:
        self.f_min = f_min
        self.f_max = f_max
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.local_scale = local_scale
        super().__init__(population_size=population_size, **kwargs)

    def _initialise(self, problem, state, rng):
        n = len(state.X)
        state.V = np.zeros_like(state.X)
        state.A = np.full(n, float(self.loudness))
        state.r = np.zeros(n)

    def _step(self, problem, state, rng):
        n, d = state.X.shape
        best = problem.best_x
        freq = self.f_min + (self.f_max - self.f_min) * rng.random((n, 1))
        state.V = state.V + (state.X - best) * freq
        cand = state.X + state.V

        local = rng.random(n) > state.r
        if np.any(local):
            walk = rng.uniform(-1, 1, (local.sum(), d)) * state.A.mean()
            cand[local] = best + self.local_scale * walk * problem.span
        cand = problem.clip(cand)
        f = problem.evaluate(cand)

        accept = (f <= state.F) & (rng.random(n) < state.A)
        state.X[accept] = cand[accept]
        state.F[accept] = f[accept]
        state.A[accept] *= self.alpha
        state.r[accept] = self.pulse_rate * (1 - np.exp(-self.gamma * (state.iteration + 1)))
