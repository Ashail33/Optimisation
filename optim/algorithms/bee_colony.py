"""
Artificial Bee Colony (ABC).

Employed bees perturb one coordinate of their food source towards/away from
a random neighbour; onlooker bees do the same but pick sources in proportion
to their quality; scouts replace sources that have not improved for
``limit`` trials with random new ones.

References
----------
Karaboga, D. & Basturk, B. (2007). A powerful and efficient algorithm for
numerical function optimization: artificial bee colony (ABC) algorithm.
*Journal of Global Optimization*, 39, 459–471.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..population import PopulationOptimiser


class ArtificialBeeColonyOptimiser(PopulationOptimiser):
    """Artificial Bee Colony for continuous problems.

    Parameters
    ----------
    limit : int, optional
        Trials without improvement before a source is abandoned.  Default
        ``population_size * dimension // 2``.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
        ``population_size`` is the number of food sources.
    """

    def __init__(self, limit: Optional[int] = None, population_size=None, **kwargs) -> None:
        self.limit = limit
        super().__init__(population_size=population_size, **kwargs)

    def _min_population(self) -> int:
        return 2

    def _initialise(self, problem, state, rng):
        state.trials = np.zeros(len(state.X), dtype=int)
        state.limit = self.limit or max(1, len(state.X) * problem.dim // 2)

    def _neighbour_phase(self, problem, state, rng, sources):
        n, d = state.X.shape
        others = rng.integers(0, n - 1, size=len(sources))
        others[others >= sources] += 1
        dims = rng.integers(0, d, size=len(sources))
        cand = state.X[sources].copy()
        phi = rng.uniform(-1, 1, size=len(sources))
        rows = np.arange(len(sources))
        cand[rows, dims] += phi * (cand[rows, dims] - state.X[others, dims])
        cand = problem.clip(cand)
        f = problem.evaluate(cand)
        for k, i in enumerate(sources):
            if f[k] < state.F[i]:
                state.X[i], state.F[i] = cand[k], f[k]
                state.trials[i] = 0
            else:
                state.trials[i] += 1

    def _step(self, problem, state, rng):
        n = len(state.X)
        self._neighbour_phase(problem, state, rng, np.arange(n))

        F = state.F
        quality = np.where(F >= 0, 1.0 / (1.0 + F), 1.0 + np.abs(F))
        quality = np.where(np.isfinite(quality), quality, 0.0)
        probs = quality / quality.sum() if quality.sum() > 0 else np.full(n, 1.0 / n)
        self._neighbour_phase(problem, state, rng, rng.choice(n, size=n, p=probs))

        scouts = np.flatnonzero(state.trials > state.limit)
        if len(scouts):
            state.X[scouts] = problem.random(len(scouts), rng)
            state.F[scouts] = problem.evaluate(state.X[scouts])
            state.trials[scouts] = 0
