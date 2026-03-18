"""
Genetic Algorithm optimiser supporting real-valued, binary, and permutation
encodings.

Highlights
----------
* Pluggable crossover / mutation operators — use the built-in defaults or
  supply your own callables.
* Real-valued encoding: arithmetic (blend) crossover + Gaussian mutation.
* Binary encoding: single-point crossover + bit-flip mutation.
* Permutation encoding: order crossover (OX) + swap mutation (TSP-style).
* Steady-state replacement: the best *elite_size* solutions survive each
  generation.
* Early-stopping via ``max_no_improve`` consecutive generations without
  improvement.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import BaseOptimiser, OptimisationResult


# ---------------------------------------------------------------------------
# Default crossover operators
# ---------------------------------------------------------------------------

def _crossover_real(
    parent1: List[float],
    parent2: List[float],
    alpha: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Arithmetic (blend) crossover for real-valued encodings."""
    child1 = [alpha * a + (1 - alpha) * b for a, b in zip(parent1, parent2)]
    child2 = [(1 - alpha) * a + alpha * b for a, b in zip(parent1, parent2)]
    return child1, child2


def _crossover_binary(
    parent1: List[int],
    parent2: List[int],
) -> Tuple[List[int], List[int]]:
    """Single-point crossover for binary encodings."""
    n = len(parent1)
    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def _crossover_permutation(
    parent1: List[int],
    parent2: List[int],
) -> Tuple[List[int], List[int]]:
    """Order crossover (OX) for permutation encodings."""
    n = len(parent1)
    if n < 3:
        return parent1[:], parent2[:]
    pts = sorted(random.sample(range(n), 2))
    p1, p2 = pts[0], pts[1] + 1

    def _ox(p: List[int], q: List[int]) -> List[int]:
        segment = p[p1:p2]
        child = [None] * n
        child[p1:p2] = segment
        fill = [x for x in q if x not in segment]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child  # type: ignore[return-value]

    return _ox(parent1, parent2), _ox(parent2, parent1)


# ---------------------------------------------------------------------------
# Default mutation operators
# ---------------------------------------------------------------------------

def _mutate_real(
    solution: List[float],
    bounds: List[Tuple[float, float]],
    sigma_scale: float = 0.1,
) -> List[float]:
    """Gaussian mutation clamped to bounds."""
    child = solution[:]
    i = random.randrange(len(child))
    lo, hi = bounds[i]
    sigma = sigma_scale * (hi - lo)
    child[i] = min(hi, max(lo, child[i] + random.gauss(0, sigma)))
    return child


def _mutate_binary(solution: List[int]) -> List[int]:
    """Flip one randomly chosen bit."""
    child = solution[:]
    i = random.randrange(len(child))
    child[i] = 1 - child[i]
    return child


def _mutate_permutation(solution: List[int]) -> List[int]:
    """Swap two randomly chosen positions."""
    child = solution[:]
    i, j = random.sample(range(len(child)), 2)
    child[i], child[j] = child[j], child[i]
    return child


# ---------------------------------------------------------------------------
# Population initialisation helpers
# ---------------------------------------------------------------------------

def _init_real(
    population_size: int,
    bounds: List[Tuple[float, float]],
) -> List[List[float]]:
    return [
        [random.uniform(lo, hi) for lo, hi in bounds]
        for _ in range(population_size)
    ]


def _init_binary(
    population_size: int,
    n_genes: int,
) -> List[List[int]]:
    return [
        [random.randint(0, 1) for _ in range(n_genes)]
        for _ in range(population_size)
    ]


def _init_permutation(
    population_size: int,
    n_genes: int,
) -> List[List[int]]:
    return [
        random.sample(range(n_genes), n_genes)
        for _ in range(population_size)
    ]


# ---------------------------------------------------------------------------
# Genetic Algorithm optimiser
# ---------------------------------------------------------------------------

class GeneticOptimiser(BaseOptimiser):
    """Genetic Algorithm optimiser.

    Parameters
    ----------
    population_size : int
        Number of individuals in each generation.  Default: 50.
    elite_size : int
        How many top solutions are kept each generation (steady-state
        replacement).  Default: 10.
    n_parents : int
        Number of parents selected per reproduction step.  Must be even.
        Default: 6.
    max_no_improve : int
        Stop after this many generations without improvement.  Default: 100.
    max_generations : int or None
        Hard upper limit on the number of generations.  ``None`` means no
        hard limit (use ``max_no_improve`` only).  Default: 1000.
    encoding : {'real', 'binary', 'permutation'}
        Solution representation.  Default: ``'real'``.
    crossover_fn : callable, optional
        Custom crossover operator ``(parent1, parent2) -> (child1, child2)``.
        When ``None`` the built-in default for the chosen *encoding* is used.
    mutation_fn : callable, optional
        Custom mutation operator ``(solution) -> mutated_solution``.
        When ``None`` the built-in default for the chosen *encoding* is used.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import GeneticOptimiser
    >>> opt = GeneticOptimiser(population_size=40, max_no_improve=50)
    >>> result = opt.optimise(lambda x: sum(v**2 for v in x),
    ...                       bounds=[(-5, 5)] * 3)
    >>> result.best_value  # close to 0
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 10,
        n_parents: int = 6,
        max_no_improve: int = 100,
        max_generations: Optional[int] = 1000,
        encoding: str = "real",
        crossover_fn: Optional[Callable] = None,
        mutation_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        if encoding not in ("real", "binary", "permutation"):
            raise ValueError("encoding must be 'real', 'binary', or 'permutation'")
        if n_parents < 2 or n_parents % 2 != 0:
            raise ValueError("n_parents must be a positive even integer")
        if elite_size < 1:
            raise ValueError("elite_size must be at least 1")

        self.population_size = population_size
        self.elite_size = elite_size
        self.n_parents = n_parents
        self.max_no_improve = max_no_improve
        self.max_generations = max_generations
        self.encoding = encoding
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
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
        n_genes: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run the genetic algorithm.

        Parameters
        ----------
        objective_fn : callable
            Objective function to minimise (or maximise if *maximise=True*).
        bounds : list of (min, max) tuples
            Required for ``encoding='real'``.
            For ``encoding='binary'`` the length sets *n_genes* if *n_genes*
            is not provided explicitly.
            Ignored for ``encoding='permutation'``.
        maximise : bool
            If ``True``, maximise the objective.  Default: ``False``.
        n_genes : int, optional
            Number of genes.  Required for ``encoding='binary'`` or
            ``encoding='permutation'`` when *bounds* is not provided.

        Returns
        -------
        OptimisationResult
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)
        population = self._initialise(bounds, n_genes)
        crossover, mutate = self._get_operators(bounds)

        n_eval = 0
        fitness = [obj(ind) for ind in population]
        n_eval += len(population)

        best_idx = int(np.argmin(fitness))
        best_solution = deepcopy(population[best_idx])
        best_value = fitness[best_idx]
        history: List[float] = [best_value]

        no_improve = 0
        generation = 0

        while no_improve < self.max_no_improve:
            if self.max_generations is not None and generation >= self.max_generations:
                break

            parents = self._select_parents(population, fitness)
            children: List = []
            for k in range(0, len(parents) - 1, 2):
                c1, c2 = crossover(parents[k], parents[k + 1])
                children.extend([c1, c2])

            # Mutate one randomly chosen child
            if children:
                mi = random.randrange(len(children))
                children[mi] = mutate(children[mi])

            # Evaluate children
            child_fitness = [obj(c) for c in children]
            n_eval += len(children)

            # Replacement: keep best elite_size from combined pool
            combined = list(zip(population + children, fitness + child_fitness))
            combined.sort(key=lambda t: t[1])
            population = [ind for ind, _ in combined[: self.population_size]]
            fitness = [f for _, f in combined[: self.population_size]]

            gen_best = fitness[0]
            history.append(gen_best)

            if gen_best < best_value:
                best_value = gen_best
                best_solution = deepcopy(population[0])
                no_improve = 0
            else:
                no_improve += 1

            generation += 1

        # Un-negate value if we were maximising
        reported_best = -best_value if maximise else best_value
        reported_history = [-v if maximise else v for v in history]

        return OptimisationResult(
            best_solution=best_solution,
            best_value=reported_best,
            history=reported_history,
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialise(
        self,
        bounds: Optional[List[Tuple[float, float]]],
        n_genes: Optional[int],
    ) -> List:
        if self.encoding == "real":
            if bounds is None:
                raise ValueError("bounds must be provided for encoding='real'")
            return _init_real(self.population_size, bounds)
        if self.encoding == "binary":
            ng = n_genes if n_genes is not None else (len(bounds) if bounds else None)
            if ng is None:
                raise ValueError(
                    "n_genes or bounds must be provided for encoding='binary'"
                )
            return _init_binary(self.population_size, ng)
        # permutation
        if n_genes is None:
            raise ValueError("n_genes must be provided for encoding='permutation'")
        return _init_permutation(self.population_size, n_genes)

    def _get_operators(
        self, bounds: Optional[List[Tuple[float, float]]]
    ) -> Tuple[Callable, Callable]:
        if self.encoding == "real":
            crossover = self.crossover_fn or _crossover_real
            mutate = self.mutation_fn or (
                lambda sol: _mutate_real(sol, bounds)  # type: ignore[arg-type]
            )
        elif self.encoding == "binary":
            crossover = self.crossover_fn or _crossover_binary
            mutate = self.mutation_fn or _mutate_binary
        else:
            crossover = self.crossover_fn or _crossover_permutation
            mutate = self.mutation_fn or _mutate_permutation
        return crossover, mutate

    def _select_parents(
        self,
        population: List,
        fitness: List[float],
    ) -> List:
        """Fitness-proportionate (roulette-wheel) parent selection."""
        n = min(self.n_parents, len(population))
        if n % 2 != 0:
            n -= 1

        # Convert minimisation fitness to selection weights
        inv_fitness = [1.0 / (f + 1e-12) for f in fitness]
        total = sum(inv_fitness)
        weights = [v / total for v in inv_fitness]

        indices = np.random.choice(
            len(population), size=n, replace=False, p=weights
        )
        return [population[i] for i in indices]
