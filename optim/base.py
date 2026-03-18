"""
Base classes for the generalised optimisation library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class OptimisationResult:
    """Container for the result returned by every optimiser.

    Attributes
    ----------
    best_solution : Any
        The best solution found (representation depends on the optimiser).
    best_value : float
        The objective-function value of ``best_solution``.
    history : list of float
        Best objective value recorded at each iteration.
    n_evaluations : int
        Total number of objective-function evaluations performed.
    """

    best_solution: Any
    best_value: float
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OptimisationResult("
            f"best_value={self.best_value:.6g}, "
            f"n_evaluations={self.n_evaluations})"
        )


class BaseOptimiser(ABC):
    """Abstract base class that every optimiser must implement.

    All optimisers in this library **minimise** the objective function by
    default.  Pass ``maximise=True`` to ``optimise`` to maximise instead.
    """

    @abstractmethod
    def optimise(
        self,
        objective_fn: Callable[[Any], float],
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run the optimiser and return the best solution found.

        Parameters
        ----------
        objective_fn : callable
            Function to optimise.  Must accept a single solution argument and
            return a scalar value.
        bounds : list of (min, max) tuples, optional
            Bounds for each decision variable.  Required by most optimisers.
        maximise : bool
            If ``True``, maximise ``objective_fn``; otherwise minimise it.
            Defaults to ``False``.
        **kwargs
            Additional algorithm-specific arguments (see each subclass).

        Returns
        -------
        OptimisationResult
        """

    # ------------------------------------------------------------------
    # Helpers shared by all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_objective(
        objective_fn: Callable[[Any], float], maximise: bool
    ) -> Callable[[Any], float]:
        """Return a wrapped objective that always *minimises*.

        When *maximise* is ``True`` the wrapper negates the value so that the
        internal minimisation logic still applies correctly.
        """
        if maximise:
            return lambda sol: -objective_fn(sol)
        return objective_fn
