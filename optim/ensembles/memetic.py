"""
Memetic (hybrid global + local) ensemble.

Alternates a global, population-based search with local refinement of the
elite solutions it finds (Lamarckian learning: refined solutions go back
into the population).  This is a *low-level relay* hybrid — unlike the
``'chain'`` strategy of :class:`~optim.EnsembleOptimiser`, which hands over
once, the two algorithms interleave for ``n_generations`` rounds.

References
----------
Moscato, P. (1989). On evolution, search, optimization, genetic algorithms
and martial arts: Towards memetic algorithms. Caltech C3P Report 826.

Neri, F. & Cotta, C. (2012). Memetic algorithms and memetic computing
optimization: A literature review. *Swarm and Evolutionary Computation*, 2,
1–14.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult
from ..ensemble import EnsembleResult
from ._common import Budget, CountingObjective, SolutionPool, derive_seed, run_optimiser, split_budget


class MemeticOptimiser(BaseOptimiser):
    """Interleave a global optimiser with local refinement of elites.

    Parameters
    ----------
    global_optimiser : BaseOptimiser
        Explorer, ideally population-based (DE, PSO, GA, GWO, ...).
    local_optimiser : BaseOptimiser
        Refiner that accepts ``initial_solution`` (Local Search, SA, Tabu
        Search, CMA-ES with a small ``sigma0`` ...).
    n_generations : int
        Global/local rounds.  Default 10.
    n_refine : int
        Elite solutions refined per round.  Default 3.
    global_evaluations, local_evaluations : int, optional
        Budgets per global run and per local refinement.  Default: half of
        ``max_evaluations`` each, split over rounds (and elites).
    pool_size : int
        Size of the population carried between rounds.  Default 30.
    max_evaluations : int, optional
        Total evaluation budget.
    seed : int, optional
        Seeds the ensemble; each run gets a derived seed.
    """

    def __init__(
        self,
        global_optimiser: BaseOptimiser,
        local_optimiser: BaseOptimiser,
        n_generations: int = 10,
        n_refine: int = 3,
        global_evaluations: Optional[int] = None,
        local_evaluations: Optional[int] = None,
        pool_size: int = 30,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_refine < 1:
            raise ValueError("n_refine must be at least 1")
        self.global_optimiser = global_optimiser
        self.local_optimiser = local_optimiser
        self.n_generations = n_generations
        self.n_refine = n_refine
        self.global_evaluations = global_evaluations
        self.local_evaluations = local_evaluations
        self.pool_size = pool_size
        self.max_evaluations = max_evaluations
        self.seed = seed

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[Sequence[Any]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Run the memetic algorithm.

        ``max_evaluations`` overrides the constructor's total budget; extra
        ``kwargs`` go to both optimisers.
        """
        rng = np.random.default_rng(self.seed)
        objective = CountingObjective(objective_fn, maximise)
        total = self.max_evaluations if max_evaluations is None else max_evaluations
        limit = Budget(total, objective)
        pool = SolutionPool(self.pool_size)
        half = None if total is None else total // 2
        g_budget = self.global_evaluations or split_budget(half, self.n_generations)
        l_budget = self.local_evaluations or split_budget(half, self.n_generations * self.n_refine)

        run_results: List[OptimisationResult] = []
        history: List[float] = []
        for _ in range(self.n_generations):
            if limit.spent:
                break
            seeds = pool.solutions or (list(initial_solutions) if initial_solutions else None)
            result = run_optimiser(
                self.global_optimiser, objective, bounds,
                population=seeds,
                max_evaluations=limit.cap(g_budget),
                seed=derive_seed(rng),
                extra_kwargs=kwargs,
            )
            pool.add_result(result)
            run_results.append(result)

            elites, _ = pool.top(self.n_refine)
            for elite in elites:
                if limit.spent:
                    break
                refined = run_optimiser(
                    self.local_optimiser, objective, bounds,
                    population=[elite],
                    max_evaluations=limit.cap(l_budget),
                    seed=derive_seed(rng),
                    extra_kwargs=kwargs,
                )
                pool.add([refined.best_solution], [refined.best_value])
                run_results.append(refined)
            history.append(pool.best_value)

        sign = objective.sign
        return EnsembleResult(
            best_solution=pool.best_solution,
            best_value=sign * pool.best_value,
            history=[sign * h for h in history],
            n_evaluations=objective.n_evaluations,
            population=pool.solutions,
            population_values=[sign * v for v in pool.values],
            run_results=run_results,
        )
