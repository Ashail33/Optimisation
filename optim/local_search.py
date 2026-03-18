"""
Local Search optimiser supporting both real-valued (continuous) and binary
decision spaces.

Two neighbourhood structures are provided as defaults:

* **Real-valued** — *step neighbourhood*: for each dimension add or subtract
  *step_size* to get 2 × n_dims neighbours (best-improvement strategy).
* **Binary** — *bit-flip neighbourhood*: flip each bit in turn (n_dims
  neighbours, best-improvement strategy).

A custom neighbourhood function can be passed via the ``neighbourhood_fn``
parameter, which gives full flexibility for any encoding.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

from .base import BaseOptimiser, OptimisationResult


# ---------------------------------------------------------------------------
# Default neighbourhood generators
# ---------------------------------------------------------------------------

def _neighbourhood_real(
    solution: List[float],
    step_size: float,
    bounds: Optional[List[Tuple[float, float]]],
) -> List[List[float]]:
    """Generate step-neighbours for a real-valued solution."""
    neighbours: List[List[float]] = []
    for i in range(len(solution)):
        for delta in (-step_size, step_size):
            neighbour = solution[:]
            neighbour[i] = neighbour[i] + delta
            if bounds is not None:
                lo, hi = bounds[i]
                if not (lo <= neighbour[i] <= hi):
                    continue
            neighbours.append(neighbour)
    return neighbours


def _neighbourhood_binary(solution: List[int]) -> List[List[int]]:
    """Generate all single-bit-flip neighbours for a binary solution."""
    neighbours: List[List[int]] = []
    for i in range(len(solution)):
        neighbour = solution[:]
        neighbour[i] = 1 - neighbour[i]
        neighbours.append(neighbour)
    return neighbours


# ---------------------------------------------------------------------------
# Local Search optimiser
# ---------------------------------------------------------------------------

class LocalSearchOptimiser(BaseOptimiser):
    """Best-improvement local search.

    Parameters
    ----------
    encoding : {'real', 'binary'}
        Solution representation.  Default: ``'real'``.
    step_size : float
        Step size used for the real-valued neighbourhood.  Default: 0.01.
    max_no_improve : int or None
        Stop after this many consecutive moves with no improvement.  ``None``
        means run until a local optimum is reached (no improving neighbour).
        Default: ``None``.
    max_iterations : int or None
        Hard upper limit on the number of local search steps.  ``None`` means
        no hard limit.  Default: 10000.
    neighbourhood_fn : callable, optional
        Custom neighbourhood generator ``(solution) -> list[solution]``.
        When provided, *encoding* and *step_size* are ignored.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import LocalSearchOptimiser
    >>> ls = LocalSearchOptimiser(step_size=0.1)
    >>> result = ls.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                      bounds=[(-5, 5), (-5, 5)],
    ...                      initial_solution=[3.0, -2.0])
    >>> result.best_value  # close to 0
    """

    def __init__(
        self,
        encoding: str = "real",
        step_size: float = 0.01,
        max_no_improve: Optional[int] = None,
        max_iterations: Optional[int] = 10_000,
        neighbourhood_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        if encoding not in ("real", "binary"):
            raise ValueError("encoding must be 'real' or 'binary'")
        self.encoding = encoding
        self.step_size = step_size
        self.max_no_improve = max_no_improve
        self.max_iterations = max_iterations
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
        constraints_fn: Optional[Callable[[Any], bool]] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run the local search.

        Parameters
        ----------
        objective_fn : callable
            Function to minimise (or maximise if *maximise=True*).
        bounds : list of (min, max) tuples, optional
            Used to generate feasible initial solutions for real-valued
            encodings and to clamp the step neighbourhood.
        maximise : bool
            If ``True``, maximise the objective.  Default: ``False``.
        initial_solution : list, optional
            Starting solution.  If ``None`` a random solution is generated.
        constraints_fn : callable, optional
            ``constraints_fn(solution) -> bool`` returning ``True`` when a
            solution is feasible.  Infeasible neighbours are skipped.

        Returns
        -------
        OptimisationResult
        """
        import random as _random
        if self.seed is not None:
            _random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)

        solution = self._resolve_initial(initial_solution, bounds)
        current_val = obj(solution)
        n_eval = 1

        best_solution = deepcopy(solution)
        best_value = current_val
        history = [best_value]

        no_improve = 0
        iteration = 0

        while True:
            if self.max_iterations is not None and iteration >= self.max_iterations:
                break
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

            neighbours = self._get_neighbours(solution, bounds)

            # Filter by feasibility
            if constraints_fn is not None:
                neighbours = [n for n in neighbours if constraints_fn(n)]

            if not neighbours:
                break  # No feasible neighbours → local optimum

            # Evaluate all neighbours and pick the best
            neighbour_vals = [obj(n) for n in neighbours]
            n_eval += len(neighbour_vals)

            best_nb_idx = int(min(range(len(neighbour_vals)), key=lambda k: neighbour_vals[k]))
            best_nb_val = neighbour_vals[best_nb_idx]

            if best_nb_val < current_val:
                solution = neighbours[best_nb_idx]
                current_val = best_nb_val
                if current_val < best_value:
                    best_value = current_val
                    best_solution = deepcopy(solution)
                no_improve = 0
            else:
                no_improve += 1
                if self.max_no_improve is None:
                    break  # No improving move → strict local optimum

            history.append(best_value)
            iteration += 1

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
    ) -> Any:
        import random as _random

        if initial_solution is not None:
            return list(initial_solution)
        if self.encoding == "real":
            if bounds is None:
                raise ValueError(
                    "Either initial_solution or bounds must be provided for "
                    "encoding='real'"
                )
            return [_random.uniform(lo, hi) for lo, hi in bounds]
        # binary — default to all zeros, size inferred from bounds if available
        if bounds is not None:
            return [0] * len(bounds)
        raise ValueError(
            "Either initial_solution or bounds must be provided for "
            "encoding='binary'"
        )

    def _get_neighbours(
        self,
        solution: Any,
        bounds: Optional[List[Tuple[float, float]]],
    ) -> List:
        if self.neighbourhood_fn is not None:
            return self.neighbourhood_fn(solution)
        if self.encoding == "binary":
            return _neighbourhood_binary(solution)
        return _neighbourhood_real(solution, self.step_size, bounds)
