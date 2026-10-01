"""
Tabu Search (TS).

At every iteration the search samples a neighbourhood of the current
solution and moves to the best neighbour that is not *tabu* — even if it is
worse than the current solution — which lets it climb out of local optima.
Recently used moves are held in a short-term memory (the tabu list) for
``tabu_tenure`` iterations.  The *aspiration criterion* overrides the tabu
status of a move that yields a new best-ever solution.

Neighbourhoods by encoding:

``'real'``         Gaussian perturbations (``step_size`` × range per variable).
                   Points within ``tabu_radius`` (normalised Chebyshev
                   distance) of recently visited solutions are tabu.
``'binary'``       Single bit flips; the flipped index is made tabu.
``'permutation'``  Swaps of two positions; the swapped pair is made tabu.

References
----------
Glover, F. (1989). Tabu Search — Part I. *ORSA Journal on Computing*, 1(3),
190–206.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult


class TabuSearchOptimiser(BaseOptimiser):
    """Tabu Search for real, binary and permutation encodings.

    Parameters
    ----------
    encoding : {'real', 'binary', 'permutation'}
        Solution representation.  Default ``'real'``.
    n_neighbours : int
        Neighbours sampled per iteration.  Default 30.
    tabu_tenure : int
        How many iterations a move stays tabu.  Default 10.
    step_size : float
        (``'real'`` only) standard deviation of a move as a fraction of each
        variable's range.  Default 0.1.
    tabu_radius : float, optional
        (``'real'`` only) normalised distance under which a candidate counts
        as revisiting a tabu solution.  Default ``step_size / 2``.
    max_iterations : int or None
        Hard iteration limit.  Default 500.
    max_no_improve : int or None
        Stop after this many iterations without a new best.  Default 100.
    max_evaluations : int or None
        Hard evaluation budget.
    seed : int, optional
        Seed for the private random generator.
    """

    def __init__(
        self,
        encoding: str = "real",
        n_neighbours: int = 30,
        tabu_tenure: int = 10,
        step_size: float = 0.1,
        tabu_radius: Optional[float] = None,
        max_iterations: Optional[int] = 500,
        max_no_improve: Optional[int] = 100,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if encoding not in ("real", "binary", "permutation"):
            raise ValueError("encoding must be 'real', 'binary', or 'permutation'")
        if max_iterations is None and max_evaluations is None and max_no_improve is None:
            raise ValueError("set at least one stopping criterion")
        self.encoding = encoding
        self.n_neighbours = n_neighbours
        self.tabu_tenure = tabu_tenure
        self.step_size = step_size
        self.tabu_radius = tabu_radius
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.max_evaluations = max_evaluations
        self.seed = seed

    # ------------------------------------------------------------------
    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        n_genes: Optional[int] = None,
        initial_solution: Optional[Sequence] = None,
        initial_solutions: Optional[Sequence[Sequence]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run Tabu Search.

        Parameters
        ----------
        objective_fn : callable
            Function of a solution list returning a scalar.
        bounds : list of (min, max), optional
            Required for ``'real'``; sets the length for ``'binary'``.
        maximise : bool
            Maximise instead of minimise.
        n_genes : int, optional
            Length of binary / permutation solutions.
        initial_solution, initial_solutions : optional
            Starting point (the first of ``initial_solutions`` is used when
            ``initial_solution`` is not given).
        max_evaluations : int, optional
            Override the evaluation budget for this call.
        """
        rng = np.random.default_rng(self.seed)
        sign = -1.0 if maximise else 1.0
        budget = self.max_evaluations if max_evaluations is None else max_evaluations
        n_eval = 0

        def evaluate(sol) -> float:
            nonlocal n_eval
            n_eval += 1
            f = sign * float(objective_fn(list(sol)))
            return math.inf if math.isnan(f) else f

        def exhausted() -> bool:
            return budget is not None and n_eval >= budget

        if initial_solution is None and initial_solutions:
            initial_solution = initial_solutions[0]
        current = self._initial(bounds, n_genes, initial_solution, rng)
        current_f = evaluate(current)
        best, best_f = current.copy(), current_f
        history = [best_f]

        ctx = {}
        if self.encoding == "real":
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            ctx = dict(
                lo=lo,
                hi=hi,
                span=np.where(hi > lo, hi - lo, 1.0),
                radius=self.tabu_radius if self.tabu_radius is not None else self.step_size / 2,
            )

        tabu: deque = deque(maxlen=self.tabu_tenure)
        iteration = no_improve = 0
        while not exhausted():
            if self.max_iterations is not None and iteration >= self.max_iterations:
                break
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

            chosen = None  # (f, candidate, attribute)
            fallback = None
            for cand, attr in self._neighbours(current, rng, ctx):
                if exhausted():
                    break
                f = evaluate(cand)
                if fallback is None or f < fallback[0]:
                    fallback = (f, cand, attr)
                if self._is_tabu(attr, tabu, ctx) and not f < best_f:
                    continue
                if chosen is None or f < chosen[0]:
                    chosen = (f, cand, attr)
            chosen = chosen or fallback
            if chosen is None:
                break

            current_f, current, attr = chosen
            tabu.append(attr)
            iteration += 1
            if current_f < best_f:
                best, best_f = current.copy(), current_f
                no_improve = 0
            else:
                no_improve += 1
            history.append(best_f)

        if self.encoding == "real":
            best_solution = [float(v) for v in best]
        else:
            best_solution = [int(v) for v in best]
        return OptimisationResult(
            best_solution=best_solution,
            best_value=sign * best_f,
            history=[sign * h for h in history],
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    def _initial(self, bounds, n_genes, initial, rng) -> np.ndarray:
        if self.encoding == "real":
            if bounds is None:
                raise ValueError("bounds must be provided for encoding='real'")
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            if initial is not None:
                return np.clip(np.asarray(initial, dtype=float), lo, hi)
            return lo + rng.random(len(lo)) * (hi - lo)

        if initial is not None:
            return np.asarray(initial, dtype=int).copy()
        n = n_genes or (len(bounds) if bounds is not None else None)
        if n is None:
            raise ValueError("n_genes (or bounds) must be provided")
        if self.encoding == "binary":
            return rng.integers(0, 2, size=n)
        return rng.permutation(n)

    def _neighbours(self, current, rng, ctx):
        n = len(current)
        if self.encoding == "real":
            noise = rng.normal(0.0, self.step_size, size=(self.n_neighbours, n)) * ctx["span"]
            for cand in np.clip(current + noise, ctx["lo"], ctx["hi"]):
                yield cand, cand
        elif self.encoding == "binary":
            for j in rng.choice(n, size=min(self.n_neighbours, n), replace=False):
                cand = current.copy()
                cand[j] = 1 - cand[j]
                yield cand, int(j)
        else:
            for _ in range(self.n_neighbours):
                i, j = rng.choice(n, size=2, replace=False)
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]
                yield cand, frozenset((int(current[i]), int(current[j])))

    def _is_tabu(self, attr, tabu, ctx) -> bool:
        if self.encoding != "real":
            return attr in tabu
        span, radius = ctx["span"], ctx["radius"]
        return any(np.max(np.abs(attr - t) / span) < radius for t in tabu)
