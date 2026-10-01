"""
Island-model ensemble (cooperative, heterogeneous).

Each optimiser runs on its own *island* — its own sub-population — for a
short epoch.  Between epochs the best solutions *migrate* between islands
according to a topology, so a breakthrough by one algorithm seeds the
others while islands still keep diverse populations.

Topologies
----------
``'ring'``             island ``i`` receives from island ``i - 1``.
``'fully_connected'``  every island receives the best migrants of all others.
``'star'``             island 0 is a hub: it receives the best migrants of
                       all islands and sends its own to every island.
``'random'``           each island receives from one random other island.

References
----------
Whitley, D., Rana, S. & Heckendorn, R. B. (1999). The island model genetic
algorithm: On separability, population size and convergence. *Journal of
Computing and Information Technology*, 7(1), 33–47.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult
from ..ensemble import EnsembleResult
from ._common import CountingObjective, SolutionPool, derive_seed, run_optimiser, split_budget

_TOPOLOGIES = ("ring", "fully_connected", "star", "random")


class IslandModelOptimiser(BaseOptimiser):
    """Run optimisers on separate islands with periodic migration.

    Parameters
    ----------
    optimisers : list of BaseOptimiser
        One optimiser per island (repeat an optimiser to get several islands
        of the same algorithm).
    n_epochs : int
        Number of run-then-migrate cycles.  Default 10.
    epoch_evaluations : int, optional
        Evaluation budget for each island per epoch (passed to optimisers
        that accept ``max_evaluations``).  Default: ``max_evaluations``
        split evenly over epochs and islands, else each optimiser's own
        stopping criteria.
    migration_size : int
        Solutions sent along each migration edge.  Default 2.
    topology : {'ring', 'fully_connected', 'star', 'random'}
        Default ``'ring'``.
    island_size : int
        Solutions each island remembers between epochs.  Default 30.
    max_evaluations : int, optional
        Total budget; no new epoch starts once it is spent.
    seed : int, optional
        Seeds the ensemble; each island run gets a derived seed.
    """

    def __init__(
        self,
        optimisers: Sequence[BaseOptimiser],
        n_epochs: int = 10,
        epoch_evaluations: Optional[int] = None,
        migration_size: int = 2,
        topology: str = "ring",
        island_size: int = 30,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if len(optimisers) < 2:
            raise ValueError("an island model needs at least two optimisers")
        if topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {_TOPOLOGIES}")
        self.optimisers = list(optimisers)
        self.n_epochs = n_epochs
        self.epoch_evaluations = epoch_evaluations
        self.migration_size = migration_size
        self.topology = topology
        self.island_size = island_size
        self.max_evaluations = max_evaluations
        self.seed = seed

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[Sequence[Any]] = None,
        optimiser_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Run the island model.

        ``initial_solutions`` seed every island; ``optimiser_kwargs[i]`` is
        forwarded to island ``i``'s ``optimise`` call.
        """
        rng = np.random.default_rng(self.seed)
        objective = CountingObjective(objective_fn, maximise)
        n_islands = len(self.optimisers)
        islands = [SolutionPool(self.island_size) for _ in range(n_islands)]
        budget = self.epoch_evaluations or split_budget(
            self.max_evaluations, self.n_epochs * n_islands
        )

        run_results: List[OptimisationResult] = []
        history: List[float] = []
        for epoch in range(self.n_epochs):
            for i, (opt, pool) in enumerate(zip(self.optimisers, islands)):
                if self._spent(objective):
                    break
                seeds = pool.solutions or (list(initial_solutions) if initial_solutions else None)
                result = run_optimiser(
                    opt,
                    objective,
                    bounds,
                    population=seeds,
                    max_evaluations=self._cap(budget, objective),
                    seed=derive_seed(rng),
                    extra_kwargs=optimiser_kwargs[i] if optimiser_kwargs else None,
                )
                pool.add_result(result)
                run_results.append(result)
            history.append(min(p.best_value for p in islands if len(p)))
            if self._spent(objective):
                break
            if epoch < self.n_epochs - 1:
                self._migrate(islands, rng)

        best_pool = min((p for p in islands if len(p)), key=lambda p: p.best_value)
        sign = objective.sign
        return EnsembleResult(
            best_solution=best_pool.best_solution,
            best_value=sign * best_pool.best_value,
            history=[sign * h for h in history],
            n_evaluations=objective.n_evaluations,
            population=best_pool.solutions,
            population_values=[sign * v for v in best_pool.values],
            run_results=run_results,
            info={
                "island_best_values": [
                    sign * p.best_value if len(p) else None for p in islands
                ],
                "epochs_completed": len(history),
            },
        )

    # ------------------------------------------------------------------
    def _spent(self, objective: CountingObjective) -> bool:
        return self.max_evaluations is not None and objective.n_evaluations >= self.max_evaluations

    def _cap(self, budget: Optional[int], objective: CountingObjective) -> Optional[int]:
        if self.max_evaluations is None:
            return budget
        remaining = self.max_evaluations - objective.n_evaluations
        return remaining if budget is None else min(budget, remaining)

    def _sources(self, i: int, n: int, rng: np.random.Generator) -> List[int]:
        if self.topology == "ring":
            return [(i - 1) % n]
        if self.topology == "fully_connected":
            return [j for j in range(n) if j != i]
        if self.topology == "star":
            return [j for j in range(1, n)] if i == 0 else [0]
        j = int(rng.integers(n - 1))
        return [j + 1 if j >= i else j]

    def _migrate(self, islands: List[SolutionPool], rng: np.random.Generator) -> None:
        snapshot = [p.top(self.migration_size) for p in islands]
        n = len(islands)
        for i, pool in enumerate(islands):
            migrants: List[Any] = []
            values: List[float] = []
            for j in self._sources(i, n, rng):
                migrants.extend(snapshot[j][0])
                values.extend(snapshot[j][1])
            order = np.argsort(values, kind="stable")[: self.migration_size]
            pool.add([migrants[k] for k in order], [values[k] for k in order])
