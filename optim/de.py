"""
Differential Evolution (DE) optimiser for continuous decision spaces.

Implements the classic DE/rand/1/bin variant (Storn & Price, 1997).  At every
generation, for every parent x_i in the population we build a *trial* vector u
by mutating three other randomly chosen members and crossing the result back
with x_i:

    v = x_a + F * (x_b - x_c)         # mutation, a, b, c distinct, ≠ i
    u_j = v_j  if rand() < CR or j == j_rand  else x_{i,j}    # binomial crossover
    x_i ← u    if f(u) ≤ f(x_i)                              # greedy selection

References
----------
Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces.
*Journal of Global Optimization*, 11(4), 341–359.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .base import BaseOptimiser, OptimisationResult


class DifferentialEvolutionOptimiser(BaseOptimiser):
    """Differential Evolution (DE/rand/1/bin) for continuous decision spaces.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.  Must be ≥ 4.  Default: 30.
    F : float
        Differential weight (mutation scale), typically in [0, 2].
        Default: 0.5.
    CR : float
        Crossover probability, in [0, 1].  Default: 0.9.
    max_generations : int
        Hard upper limit on generations.  Default: 1000.
    max_no_improve : int or None
        Stop after this many generations without improving the global best.
        ``None`` disables this stopping rule.  Default: 100.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import DifferentialEvolutionOptimiser
    >>> de = DifferentialEvolutionOptimiser(population_size=20,
    ...                                     max_no_improve=50, seed=0)
    >>> result = de.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                      bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    >>> result.best_value < 1.0
    True
    """

    def __init__(
        self,
        population_size: int = 30,
        F: float = 0.5,
        CR: float = 0.9,
        max_generations: int = 1000,
        max_no_improve: Optional[int] = 100,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4 for DE/rand/1/bin")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.max_generations = max_generations
        self.max_no_improve = max_no_improve
        self.seed = seed

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run Differential Evolution.

        Parameters
        ----------
        objective_fn : callable
            Scalar objective.
        bounds : list of (min, max)
            Required.  One tuple per decision variable.
        maximise : bool
            Maximise instead of minimise.  Default: ``False``.
        initial_solutions : list of lists, optional
            Seed the population with these solutions.  Remaining slots are
            initialised uniformly at random within ``bounds``.

        Returns
        -------
        OptimisationResult
        """
        if bounds is None:
            raise ValueError("bounds must be provided for DifferentialEvolutionOptimiser")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)
        D = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        N = self.population_size

        # Initialise population
        X = np.zeros((N, D))
        start = 0
        if initial_solutions:
            for k, s in enumerate(initial_solutions[:N]):
                X[k] = np.clip(s, lo, hi)
            start = min(len(initial_solutions), N)
        for k in range(start, N):
            X[k] = lo + np.random.random(D) * (hi - lo)

        fitness = np.array([obj(list(x)) for x in X])
        n_eval = N

        best_idx = int(np.argmin(fitness))
        best_solution = X[best_idx].copy()
        best_value = float(fitness[best_idx])
        history: List[float] = [best_value]

        no_improve = 0

        for _ in range(self.max_generations):
            for i in range(N):
                # Pick three distinct indices a, b, c, all different from i
                pool = [k for k in range(N) if k != i]
                a, b, c = random.sample(pool, 3)

                # Mutation: v = x_a + F * (x_b - x_c), clipped to bounds
                v = X[a] + self.F * (X[b] - X[c])
                v = np.clip(v, lo, hi)

                # Binomial crossover with at least one inherited gene
                j_rand = random.randrange(D)
                mask = np.random.random(D) < self.CR
                mask[j_rand] = True
                u = np.where(mask, v, X[i])

                # Greedy selection
                fu = obj(list(u))
                n_eval += 1
                if fu <= fitness[i]:
                    X[i] = u
                    fitness[i] = fu
                    if fu < best_value:
                        best_value = float(fu)
                        best_solution = u.copy()

            history.append(best_value)
            if history[-1] < history[-2]:
                no_improve = 0
            else:
                no_improve += 1
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

        reported_best = -best_value if maximise else best_value
        reported_history = [-v if maximise else v for v in history]

        return OptimisationResult(
            best_solution=list(best_solution),
            best_value=reported_best,
            history=reported_history,
            n_evaluations=n_eval,
        )
