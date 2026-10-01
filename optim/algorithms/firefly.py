"""
Firefly Algorithm (FA).

Each firefly moves towards every brighter (better) firefly with an
attractiveness ``beta0 * exp(-gamma r^2)`` that fades with distance, plus a
small random walk whose size ``alpha`` shrinks over time.  Distances are
measured in the bounds-normalised space so ``gamma`` is scale-free.

References
----------
Yang, X.-S. (2009). Firefly algorithms for multimodal optimization.
*Stochastic Algorithms: Foundations and Applications*, LNCS 5792, 169–178.
"""

from __future__ import annotations

import numpy as np

from ..population import PopulationOptimiser


class FireflyOptimiser(PopulationOptimiser):
    """Firefly Algorithm for continuous problems.

    Parameters
    ----------
    beta0 : float
        Attractiveness at distance zero.  Default 1.0.
    gamma : float
        Light absorption coefficient (in normalised space).  Default 1.0.
    alpha : float
        Initial random-walk scale, as a fraction of each variable's range.
        Default 0.2.
    alpha_decay : float
        ``alpha`` is multiplied by this each iteration.  Default 0.97.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
    """

    default_population_size = 25

    def __init__(
        self,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        alpha_decay: float = 0.97,
        population_size=None,
        **kwargs,
    ) -> None:
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        super().__init__(population_size=population_size, **kwargs)

    def _initialise(self, problem, state, rng):
        state.alpha = self.alpha

    def _step(self, problem, state, rng):
        span = np.where(problem.span > 0, problem.span, 1.0)
        U = (state.X - problem.lo) / span
        n, d = U.shape
        new = U.copy()
        for i in range(n):
            for j in np.flatnonzero(state.F < state.F[i]):
                diff = U[j] - new[i]
                new[i] += self.beta0 * np.exp(-self.gamma * diff @ diff) * diff
            new[i] += state.alpha * (rng.random(d) - 0.5)
        state.X = problem.clip(problem.lo + new * span)
        state.F = problem.evaluate(state.X)
        state.alpha *= self.alpha_decay
