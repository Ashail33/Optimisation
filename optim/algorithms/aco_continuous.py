"""
Ant Colony Optimisation for continuous domains (ACO_R).

A solution archive plays the role of the pheromone table.  Each ant picks an
archive member with a rank-based probability (Gaussian weights controlled by
``q``) and samples every variable from a Gaussian centred on it, with a
spread proportional to the average distance to the rest of the archive.
The archive keeps the best ``k`` solutions found.

References
----------
Socha, K. & Dorigo, M. (2008). Ant colony optimization for continuous
domains. *European Journal of Operational Research*, 185(3), 1155–1173.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..population import PopulationOptimiser


class ACOROptimiser(PopulationOptimiser):
    """ACO_R — ant colony optimisation for continuous problems.

    Parameters
    ----------
    n_ants : int, optional
        Solutions sampled per iteration.  Default: archive size.
    q : float
        Locality of the search: small values favour the best archive
        members.  Default 0.2.
    xi : float
        Spread of the sampling Gaussians (like an evaporation rate).
        Default 0.85.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
        ``population_size`` is the archive size ``k``.
    """

    def __init__(
        self,
        n_ants: Optional[int] = None,
        q: float = 0.2,
        xi: float = 0.85,
        population_size=None,
        **kwargs,
    ) -> None:
        self.n_ants = n_ants
        self.q = q
        self.xi = xi
        super().__init__(population_size=population_size, **kwargs)

    def _initialise(self, problem, state, rng):
        k = len(state.X)
        ranks = np.arange(k)
        w = np.exp(-(ranks ** 2) / (2 * (self.q * k) ** 2)) / (self.q * k * math.sqrt(2 * math.pi))
        state.weights = w / w.sum()
        order = np.argsort(state.F, kind="stable")
        state.X, state.F = state.X[order], state.F[order]

    def _step(self, problem, state, rng):
        k, d = state.X.shape
        m = self.n_ants or k
        chosen = rng.choice(k, size=m, p=state.weights)
        centres = state.X[chosen]
        spread = self.xi * np.abs(state.X[None, :, :] - centres[:, None, :]).sum(axis=1) / (k - 1)
        new = problem.clip(rng.normal(centres, spread))
        f = problem.evaluate(new)

        pool_X = np.vstack([state.X, new])
        pool_F = np.concatenate([state.F, f])
        keep = np.argsort(pool_F, kind="stable")[:k]
        state.X, state.F = pool_X[keep], pool_F[keep]
