"""
Harmony Search (HS).

A harmony memory of good solutions is used to *improvise* new ones: each
variable is copied from memory (probability ``HMCR``), optionally pitch-
adjusted (probability ``PAR``), or else drawn at random.  New harmonies
replace the worst in memory when better.  The pitch bandwidth shrinks
exponentially over the run (as in Mahdavi et al.'s improved HS).

References
----------
Geem, Z. W., Kim, J. H. & Loganathan, G. V. (2001). A new heuristic
optimization algorithm: harmony search. *Simulation*, 76(2), 60–68.
"""

from __future__ import annotations

import math

import numpy as np

from ..population import PopulationOptimiser


class HarmonySearchOptimiser(PopulationOptimiser):
    """Harmony Search for continuous problems.

    Parameters
    ----------
    hmcr : float
        Harmony memory considering rate.  Default 0.9.
    par : float
        Pitch adjusting rate.  Default 0.3.
    bandwidth : (float, float)
        Start and end pitch bandwidth as fractions of each range.
        Default ``(0.1, 0.001)``.
    improvisations : int, optional
        New harmonies per iteration.  Default: the memory size.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
        ``population_size`` is the harmony memory size.
    """

    default_population_size = 20

    def __init__(
        self,
        hmcr: float = 0.9,
        par: float = 0.3,
        bandwidth=(0.1, 0.001),
        improvisations=None,
        population_size=None,
        **kwargs,
    ) -> None:
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.improvisations = improvisations
        super().__init__(population_size=population_size, **kwargs)

    def _step(self, problem, state, rng):
        n, d = state.X.shape
        m = self.improvisations or n
        bw_hi, bw_lo = self.bandwidth
        bw = bw_hi * math.exp(math.log(bw_lo / bw_hi) * state.progress)

        memory = state.X[rng.integers(0, n, size=(m, d)), np.arange(d)]
        use_memory = rng.random((m, d)) < self.hmcr
        adjust = use_memory & (rng.random((m, d)) < self.par)
        new = np.where(use_memory, memory, problem.random(m, rng))
        new = new + adjust * rng.uniform(-1, 1, (m, d)) * bw * problem.span
        new = problem.clip(new)
        f = problem.evaluate(new)

        pool_X = np.vstack([state.X, new])
        pool_F = np.concatenate([state.F, f])
        keep = np.argsort(pool_F, kind="stable")[:n]
        state.X, state.F = pool_X[keep], pool_F[keep]
