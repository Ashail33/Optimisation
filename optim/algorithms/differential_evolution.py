"""
Differential Evolution (DE).

Each generation, every individual ``x_i`` is challenged by a trial vector
built from a *mutant* (a base vector plus a scaled difference of two other
individuals) mixed with ``x_i`` by binomial crossover.  The trial replaces
``x_i`` when it is at least as good.

Supported mutation strategies:

``'rand/1'``            ``v = x_r1 + F (x_r2 - x_r3)``
``'best/1'``            ``v = x_best + F (x_r1 - x_r2)``
``'current-to-best/1'`` ``v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)``
``'rand/2'``            ``v = x_r1 + F (x_r2 - x_r3) + F (x_r4 - x_r5)``

References
----------
Storn, R. & Price, K. (1997). Differential evolution — a simple and efficient
heuristic for global optimization over continuous spaces. *Journal of Global
Optimization*, 11, 341–359.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from ..population import PopulationOptimiser

_STRATEGIES = ("rand/1", "best/1", "current-to-best/1", "rand/2")


class DifferentialEvolutionOptimiser(PopulationOptimiser):
    """Differential Evolution for continuous problems.

    Parameters
    ----------
    strategy : str
        One of ``'rand/1'`` (default), ``'best/1'``, ``'current-to-best/1'``,
        ``'rand/2'``.  Crossover is always binomial.
    F : float or (float, float)
        Differential weight.  A tuple ``(low, high)`` enables *dither*: a new
        weight is drawn uniformly from the range every generation.
        Default ``(0.5, 1.0)``.
    CR : float
        Crossover probability.  Default 0.9.
    **kwargs
        Budget / seed options from :class:`~optim.population.PopulationOptimiser`.
    """

    default_population_size = 40

    def __init__(
        self,
        strategy: str = "rand/1",
        F: Union[float, Tuple[float, float]] = (0.5, 1.0),
        CR: float = 0.9,
        population_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        self.strategy = strategy
        self.F = F
        self.CR = CR
        super().__init__(population_size=population_size, **kwargs)

    def _min_population(self) -> int:
        return 6

    def _step(self, problem, state, rng):
        X, F = state.X, state.F
        n, d = X.shape
        weight = rng.uniform(*self.F) if isinstance(self.F, tuple) else self.F
        r = self._random_others(n, rng, k=5 if self.strategy == "rand/2" else 3)
        best = X[np.argmin(F)]

        if self.strategy == "rand/1":
            mutant = X[r[:, 0]] + weight * (X[r[:, 1]] - X[r[:, 2]])
        elif self.strategy == "best/1":
            mutant = best + weight * (X[r[:, 0]] - X[r[:, 1]])
        elif self.strategy == "current-to-best/1":
            mutant = X + weight * (best - X) + weight * (X[r[:, 0]] - X[r[:, 1]])
        else:  # rand/2
            mutant = (
                X[r[:, 0]]
                + weight * (X[r[:, 1]] - X[r[:, 2]])
                + weight * (X[r[:, 3]] - X[r[:, 4]])
            )

        cross = rng.random((n, d)) < self.CR
        cross[np.arange(n), rng.integers(0, d, size=n)] = True
        trial = problem.clip(np.where(cross, mutant, X))
        self._greedy_replace(state, trial, problem.evaluate(trial))
