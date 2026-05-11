"""
Random Search optimiser — uniform-random sampling of the search space.

Random Search is the canonical baseline for any metaheuristic study: if a
sophisticated algorithm cannot beat it, the algorithm or its tuning is
suspect.  It is also useful as a warm-start generator inside an
:class:`EnsembleOptimiser` pipeline.

Two encodings are supported:

* **real**         — each variable sampled uniformly from its ``bounds`` range.
* **binary**       — each gene sampled from ``{0, 1}``.

A custom ``sample_fn`` overrides both.

References
----------
Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter
Optimization.  *Journal of Machine Learning Research*, 13, 281–305.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

from .base import BaseOptimiser, OptimisationResult


class RandomSearchOptimiser(BaseOptimiser):
    """Uniform-random search baseline.

    Parameters
    ----------
    encoding : {'real', 'binary'}
        Solution representation.  Default: ``'real'``.
    max_evaluations : int
        Total number of random samples drawn.  Default: 1000.
    sample_fn : callable, optional
        Custom sampler ``() -> solution``.  When provided, ``encoding`` and
        ``bounds`` are ignored.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import RandomSearchOptimiser
    >>> rs = RandomSearchOptimiser(max_evaluations=500, seed=0)
    >>> result = rs.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                      bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    >>> result.n_evaluations
    500
    """

    def __init__(
        self,
        encoding: str = "real",
        max_evaluations: int = 1000,
        sample_fn: Optional[Callable[[], Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if encoding not in ("real", "binary"):
            raise ValueError("encoding must be 'real' or 'binary'")
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")
        self.encoding = encoding
        self.max_evaluations = max_evaluations
        self.sample_fn = sample_fn
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
        """Run Random Search.

        Parameters
        ----------
        objective_fn : callable
            Scalar objective.
        bounds : list of (min, max), optional
            Required for ``encoding='real'`` when no ``sample_fn`` is given.
            For ``encoding='binary'`` the length sets ``n_genes`` if not given.
        maximise : bool
            Maximise instead of minimise.  Default: ``False``.
        n_genes : int, optional
            Number of bits for ``encoding='binary'``.

        Returns
        -------
        OptimisationResult
        """
        if self.seed is not None:
            random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)
        sampler = self._resolve_sampler(bounds, n_genes)

        best_solution: Any = None
        best_value = float("inf")
        history: List[float] = []

        for _ in range(self.max_evaluations):
            x = sampler()
            f = obj(x)
            if f < best_value:
                best_value = f
                best_solution = deepcopy(x)
            history.append(best_value)

        reported_best = -best_value if maximise else best_value
        reported_history = [-v if maximise else v for v in history]

        return OptimisationResult(
            best_solution=best_solution,
            best_value=reported_best,
            history=reported_history,
            n_evaluations=self.max_evaluations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_sampler(
        self,
        bounds: Optional[List[Tuple[float, float]]],
        n_genes: Optional[int],
    ) -> Callable[[], Any]:
        if self.sample_fn is not None:
            return self.sample_fn
        if self.encoding == "real":
            if bounds is None:
                raise ValueError(
                    "bounds must be provided for encoding='real' "
                    "(or supply a custom sample_fn)"
                )
            local_bounds = list(bounds)
            return lambda: [random.uniform(lo, hi) for lo, hi in local_bounds]
        # binary
        ng = n_genes if n_genes is not None else (len(bounds) if bounds else None)
        if ng is None:
            raise ValueError(
                "n_genes or bounds must be provided for encoding='binary'"
            )
        return lambda: [random.randint(0, 1) for _ in range(ng)]
