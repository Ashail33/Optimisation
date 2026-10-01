"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

A (mu/mu_w, lambda)-CMA-ES following Hansen's tutorial.  The search runs in
the unit hypercube (each variable rescaled by its bounds) so a single step
size suits variables of very different scales; samples are clipped to the
box before evaluation.

References
----------
Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
arXiv:1604.00772.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..population import PopulationOptimiser


class CMAESOptimiser(PopulationOptimiser):
    """CMA-ES for continuous problems.

    Parameters
    ----------
    sigma0 : float
        Initial step size as a fraction of each variable's range.
        Default 0.3.
    population_size : int, optional
        Offspring per generation (lambda).  Default ``4 + 3 ln(d)``.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.

    Notes
    -----
    The search starts from the best member of the initial population, so
    warm starts (``initial_solution`` / ``initial_solutions``) set the
    starting mean.
    """

    def __init__(
        self,
        sigma0: float = 0.3,
        population_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        self.sigma0 = sigma0
        super().__init__(population_size=population_size, **kwargs)

    def _min_population(self) -> int:
        return 4

    def _population_size(self, dim: int) -> int:
        return self.population_size or 4 + int(3 * math.log(max(dim, 1)))

    def _initialise(self, problem, state, rng):
        n = problem.dim
        lam = len(state.X)
        mu = lam // 2
        w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        cs = (mueff + 2) / (n + mueff + 5)
        state.cma = dict(
            lam=lam,
            mu=mu,
            w=w,
            mueff=mueff,
            cc=(4 + mueff / n) / (n + 4 + 2 * mueff / n),
            cs=cs,
            c1=2 / ((n + 1.3) ** 2 + mueff),
            cmu=min(
                1 - 2 / ((n + 1.3) ** 2 + mueff),
                2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff),
            ),
            damps=1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs,
            chiN=math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)),
        )
        span = np.where(problem.span > 0, problem.span, 1.0)
        state.span = span
        state.mean = (state.X[np.argmin(state.F)] - problem.lo) / span
        state.sigma = self.sigma0
        state.pc = np.zeros(n)
        state.ps = np.zeros(n)
        state.C = np.eye(n)
        state.B = np.eye(n)
        state.D = np.ones(n)
        state.generation = 0

    def _state_summary(self, state):
        return {"sigma": float(state.sigma), "axis_ratio": float(state.D.max() / state.D.min())}

    def _step(self, problem, state, rng):
        p = state.cma
        n = problem.dim
        lam, mu, w, mueff = p["lam"], p["mu"], p["w"], p["mueff"]
        cc, cs, c1, cmu = p["cc"], p["cs"], p["c1"], p["cmu"]

        Z = rng.standard_normal((lam, n))
        Y = Z @ (state.B * state.D).T
        U = np.clip(state.mean + state.sigma * Y, 0.0, 1.0)
        X = problem.lo + U * state.span
        F = problem.evaluate(X)
        state.X, state.F = X, F

        order = np.argsort(F, kind="stable")[:mu]
        old_mean = state.mean
        state.mean = w @ U[order]
        y_w = (state.mean - old_mean) / state.sigma

        inv_sqrt_C = state.B @ np.diag(1.0 / state.D) @ state.B.T
        state.ps = (1 - cs) * state.ps + math.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ y_w)
        state.generation += 1
        ps_norm = np.linalg.norm(state.ps)
        hsig = (
            ps_norm / math.sqrt(1 - (1 - cs) ** (2 * state.generation)) / p["chiN"]
            < 1.4 + 2 / (n + 1)
        )
        state.pc = (1 - cc) * state.pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w

        art = (U[order] - old_mean) / state.sigma
        state.C = (
            (1 - c1 - cmu) * state.C
            + c1 * (np.outer(state.pc, state.pc) + (1 - hsig) * cc * (2 - cc) * state.C)
            + cmu * (art.T * w) @ art
        )
        state.sigma *= math.exp((cs / p["damps"]) * (ps_norm / p["chiN"] - 1))
        state.sigma = min(state.sigma, 1e3)

        state.C = np.triu(state.C) + np.triu(state.C, 1).T
        D2, state.B = np.linalg.eigh(state.C)
        state.D = np.sqrt(np.maximum(D2, 1e-20))
