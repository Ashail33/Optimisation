"""
Tabu Search optimiser.

Tabu Search (Glover, 1986) is a deterministic local search that escapes local
optima by maintaining a *tabu list* of recently visited solutions (or moves)
that are temporarily forbidden.  At every iteration the best non-tabu
neighbour is selected — even if it worsens the objective — provided no
*aspiration criterion* (a tabu move that improves the global best) overrides
the ban.

This implementation supports both real-valued and permutation encodings:

* **real**         — neighbourhood = step ±``step_size`` per dimension; the
                     tabu list stores recently visited *positions* (rounded
                     to ``precision`` decimals to avoid float drift).
* **permutation**  — neighbourhood = all 2-opt swaps; the tabu list stores
                     the swapped index pair.

A user-supplied ``neighbourhood_fn`` overrides both.

References
----------
Glover, F. (1986). Future paths for integer programming and links to
artificial intelligence.  *Computers & Operations Research*, 13(5), 533–549.
"""

from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Deque, List, Optional, Tuple

from .base import BaseOptimiser, OptimisationResult


# ---------------------------------------------------------------------------
# Default neighbourhood generators
# ---------------------------------------------------------------------------

def _neighbourhood_real(
    solution: List[float],
    step_size: float,
    bounds: Optional[List[Tuple[float, float]]],
) -> List[Tuple[List[float], Tuple]]:
    """Step neighbourhood for a real-valued solution.

    Returns a list of ``(neighbour, move_key)`` pairs.  ``move_key`` is the
    rounded position tuple stored on the tabu list.
    """
    neighbours: List[Tuple[List[float], Tuple]] = []
    for i in range(len(solution)):
        for delta in (-step_size, step_size):
            nb = solution[:]
            nb[i] = nb[i] + delta
            if bounds is not None:
                lo, hi = bounds[i]
                if not (lo <= nb[i] <= hi):
                    continue
            neighbours.append((nb, tuple(round(v, 6) for v in nb)))
    return neighbours


def _neighbourhood_permutation(
    solution: List[int],
) -> List[Tuple[List[int], Tuple[int, int]]]:
    """2-opt swap neighbourhood for a permutation."""
    neighbours: List[Tuple[List[int], Tuple[int, int]]] = []
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            nb = solution[:]
            nb[i], nb[j] = nb[j], nb[i]
            neighbours.append((nb, (i, j)))
    return neighbours


class TabuSearchOptimiser(BaseOptimiser):
    """Tabu Search optimiser.

    Parameters
    ----------
    encoding : {'real', 'permutation'}
        Solution representation.  Default: ``'real'``.
    step_size : float
        Step used by the real-valued neighbourhood.  Default: ``0.1``.
    tabu_tenure : int
        Number of iterations a move stays on the tabu list.  Default: ``10``.
    max_iterations : int
        Hard upper limit on iterations.  Default: ``500``.
    max_no_improve : int or None
        Stop after this many iterations without improving the global best.
        ``None`` disables this stopping rule.  Default: ``100``.
    neighbourhood_fn : callable, optional
        Custom neighbourhood ``(solution) -> list[(neighbour, move_key)]``.
        If you only have a list of neighbours, return ``(nb, tuple(nb))``
        pairs so the tabu key is simply the neighbour itself.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    *Aspiration criterion*: a tabu move is allowed if it would improve the
    global best solution.

    Examples
    --------
    >>> from optim import TabuSearchOptimiser
    >>> ts = TabuSearchOptimiser(step_size=0.1, tabu_tenure=8, seed=0)
    >>> result = ts.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                      bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    ...                      initial_solution=[3.0, -2.0])
    >>> result.best_value < 9.0
    True
    """

    def __init__(
        self,
        encoding: str = "real",
        step_size: float = 0.1,
        tabu_tenure: int = 10,
        max_iterations: int = 500,
        max_no_improve: Optional[int] = 100,
        neighbourhood_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        if encoding not in ("real", "permutation"):
            raise ValueError("encoding must be 'real' or 'permutation'")
        if tabu_tenure < 0:
            raise ValueError("tabu_tenure must be non-negative")
        self.encoding = encoding
        self.step_size = step_size
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.neighbourhood_fn = neighbourhood_fn
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
        initial_solution: Optional[Any] = None,
        n_genes: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run Tabu Search.

        Parameters
        ----------
        objective_fn : callable
            Scalar objective.
        bounds : list of (min, max), optional
            Required for ``encoding='real'`` when no ``initial_solution`` is
            given.
        maximise : bool
            Maximise instead of minimise.  Default: ``False``.
        initial_solution : list, optional
            Starting solution.  Generated randomly if omitted.
        n_genes : int, optional
            Required for ``encoding='permutation'`` when no
            ``initial_solution`` is given.

        Returns
        -------
        OptimisationResult
        """
        if self.seed is not None:
            random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)
        x = self._resolve_initial(initial_solution, bounds, n_genes)
        z = obj(x)
        n_eval = 1

        best_solution = deepcopy(x)
        best_value = z
        history: List[float] = [best_value]

        tabu: Deque[Tuple] = deque(maxlen=max(self.tabu_tenure, 1))
        no_improve = 0

        for _ in range(self.max_iterations):
            neighbours = self._get_neighbours(x, bounds)
            if not neighbours:
                break

            # Pick the best non-tabu neighbour, or any tabu neighbour that
            # satisfies the aspiration criterion.
            best_nb: Optional[Tuple[List, Tuple, float]] = None
            for nb, move_key in neighbours:
                f = obj(nb)
                n_eval += 1
                is_tabu = move_key in tabu
                aspirate = is_tabu and f < best_value
                if is_tabu and not aspirate:
                    continue
                if best_nb is None or f < best_nb[2]:
                    best_nb = (nb, move_key, f)

            if best_nb is None:
                # All neighbours tabu and none aspirated → stop.
                break

            x, move_key, z = best_nb
            tabu.append(move_key)

            if z < best_value:
                best_value = z
                best_solution = deepcopy(x)
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_value)

            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

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

    def _resolve_initial(
        self,
        initial_solution: Optional[Any],
        bounds: Optional[List[Tuple[float, float]]],
        n_genes: Optional[int],
    ) -> Any:
        if initial_solution is not None:
            return list(initial_solution)
        if self.encoding == "real":
            if bounds is None:
                raise ValueError(
                    "bounds or initial_solution required for encoding='real'"
                )
            return [random.uniform(lo, hi) for lo, hi in bounds]
        # permutation
        if n_genes is None:
            raise ValueError(
                "n_genes or initial_solution required for encoding='permutation'"
            )
        return random.sample(range(n_genes), n_genes)

    def _get_neighbours(
        self,
        solution: Any,
        bounds: Optional[List[Tuple[float, float]]],
    ) -> List[Tuple[Any, Tuple]]:
        if self.neighbourhood_fn is not None:
            return list(self.neighbourhood_fn(solution))
        if self.encoding == "permutation":
            return _neighbourhood_permutation(solution)
        return _neighbourhood_real(solution, self.step_size, bounds)
