"""Helpers shared by the ensemble strategies."""

from __future__ import annotations

import inspect
import math
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..base import BaseOptimiser, OptimisationResult


class CountingObjective:
    """Minimising wrapper around the user's objective that counts calls."""

    def __init__(self, objective_fn: Callable, maximise: bool) -> None:
        self.objective_fn = objective_fn
        self.sign = -1.0 if maximise else 1.0
        self.n_evaluations = 0

    def __call__(self, solution: Any) -> float:
        self.n_evaluations += 1
        return self.sign * self.objective_fn(solution)


def _accepted_kwargs(optimiser: BaseOptimiser) -> set:
    return set(inspect.signature(optimiser.optimise).parameters)


def run_optimiser(
    optimiser: BaseOptimiser,
    objective: Callable,
    bounds: Optional[Sequence],
    *,
    population: Optional[List[Any]] = None,
    max_evaluations: Optional[int] = None,
    seed: Optional[int] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> OptimisationResult:
    """Run one constituent optimiser, minimising ``objective``.

    Warm-start solutions and an evaluation budget are passed only when the
    optimiser's ``optimise`` signature accepts them
    (``initial_solutions`` > ``initial_solution``; ``max_evaluations``), so
    any :class:`BaseOptimiser` can take part — optimisers without budget
    control simply run to their own stopping criteria.

    When ``seed`` is given and the optimiser has a ``seed`` attribute, a
    shallow copy re-seeded with ``seed`` is run, so repeated calls explore
    differently while the ensemble as a whole stays reproducible.
    """
    params = _accepted_kwargs(optimiser)
    kwargs = dict(extra_kwargs or {})
    if population:
        if "initial_solutions" in params:
            kwargs.setdefault("initial_solutions", [list(p) for p in population])
        elif "initial_solution" in params:
            kwargs.setdefault("initial_solution", list(population[0]))
    if max_evaluations is not None and "max_evaluations" in params:
        kwargs.setdefault("max_evaluations", max(1, int(max_evaluations)))
    if seed is not None and hasattr(optimiser, "seed"):
        optimiser = copy(optimiser)
        optimiser.seed = seed
    return optimiser.optimise(objective, bounds, maximise=False, **kwargs)


class SolutionPool:
    """A best-first archive of at most ``size`` distinct (solution, value) pairs
    (values are in minimisation orientation)."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("pool size must be at least 1")
        self.size = size
        self.solutions: List[Any] = []
        self.values: List[float] = []

    def __len__(self) -> int:
        return len(self.solutions)

    @staticmethod
    def _key(solution: Any) -> tuple:
        return tuple(np.round(np.asarray(solution, dtype=float), 12).tolist())

    def add(self, solutions: Sequence[Any], values: Sequence[float]) -> None:
        seen = {self._key(s) for s in self.solutions}
        for s, v in zip(solutions, values):
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                continue
            k = self._key(s)
            if k in seen:
                continue
            seen.add(k)
            self.solutions.append(list(s))
            self.values.append(v)
        order = sorted(range(len(self.values)), key=self.values.__getitem__)[: self.size]
        self.solutions = [self.solutions[i] for i in order]
        self.values = [self.values[i] for i in order]

    def add_result(self, result: OptimisationResult) -> None:
        if result.population is not None and result.population_values is not None:
            self.add(result.population, result.population_values)
        self.add([result.best_solution], [result.best_value])

    def top(self, k: int) -> tuple:
        return self.solutions[:k], self.values[:k]

    @property
    def best_solution(self) -> Any:
        return self.solutions[0]

    @property
    def best_value(self) -> float:
        return self.values[0]


def derive_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def split_budget(total: Optional[int], parts: int) -> Optional[int]:
    if total is None:
        return None
    return max(1, total // max(1, parts))
