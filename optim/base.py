"""
Base classes for the generalised optimisation library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Step:
    """One step of an iterative optimiser (the state-action layer).

    A trajectory of these is a ``(state, action, cost, delta_cost)`` sequence
    that downstream models (value / policy models, learned hyper-heuristics)
    can learn from.  Only ``step_idx`` and ``cost`` are required.

    Fields
    ------
    step_idx : int
        Iteration counter within the run (starts at 1; 0 is the initial
        population).
    cost : float
        Best-so-far objective in *minimisation* orientation (``-profit``
        when maximising), so lower is always better.
    delta_cost : float, optional
        Change in ``cost`` caused by this step (``<= 0``); a reward signal.
    accepted : bool
        Whether the step improved the best-so-far solution.
    action : dict
        What the algorithm did, e.g. ``{"type": "de/rand/1", "F": 0.73}``.
    state : dict
        Compact numeric summary of the search state after the step
        (best / mean value, population diversity, evaluations used, plus
        algorithm-specific quantities such as CMA-ES's step size).
    """

    step_idx: int
    cost: float
    delta_cost: Optional[float] = None
    accepted: bool = True
    action: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


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
    population : list, optional
        Final population, sorted best-first, for population-based optimisers
        that expose it (``None`` otherwise).  Ensembles use it to hand a
        whole population from one algorithm to the next.
    population_values : list of float, optional
        Objective values matching ``population`` (same orientation as
        ``best_value``).
    trajectory : list of Step, optional
        Per-iteration state-action record, when the optimiser was asked to
        record one (``optimise(..., record_trajectory=True)``).
    """

    best_solution: Any
    best_value: float
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    population: Optional[List[Any]] = None
    population_values: Optional[List[float]] = None
    trajectory: Optional[List[Step]] = None

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
