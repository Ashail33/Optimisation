"""
Shared machinery for population-based metaheuristics on continuous,
box-bounded search spaces.

Every algorithm in :mod:`optim.algorithms` that works on real-valued vectors
subclasses :class:`PopulationOptimiser`.  The base class owns everything that
is *not* specific to the algorithm:

* seeding a :class:`numpy.random.Generator` from ``seed``,
* building the initial population (optionally warm-started),
* clipping to bounds and counting evaluations,
* enforcing the stopping criteria (iterations, evaluations, stagnation,
  user callback),
* tracking the best-ever solution and the convergence history,
* converting back to the user's orientation when ``maximise=True``.

A subclass implements two hooks:

``_initialise(problem, state, rng)``
    Called once after the initial population ``state.X`` has been evaluated
    into ``state.F``.  Set up any algorithm-specific state here (velocities,
    covariance matrices, trial counters, ...).

``_step(problem, state, rng)``
    Perform one iteration (generation).  Read and update ``state.X`` /
    ``state.F``; evaluate new candidates with ``problem.evaluate``.

``state`` is a :class:`types.SimpleNamespace` with at least ``X`` (``(n, d)``
array), ``F`` (``(n,)`` array), ``iteration`` and ``progress`` (fraction of
the run budget used, in ``[0, 1]`` — handy for parameters that decay over
the run, e.g. GWO's ``a`` or WOA's spiral coefficient).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseOptimiser, OptimisationResult, Step


class Problem:
    """A bounded, budgeted, always-minimising view of an objective function.

    Parameters
    ----------
    objective_fn : callable
        Objective to *minimise*.  Receives a plain Python ``list``.
    bounds : sequence of (min, max)
        Box bounds for every decision variable.
    max_evaluations : int, optional
        Evaluation budget.  Once spent, :meth:`evaluate` returns ``inf`` for
        any further candidate without calling the objective.
    """

    def __init__(
        self,
        objective_fn: Callable[[List[float]], float],
        bounds: Sequence[Tuple[float, float]],
        max_evaluations: Optional[int] = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        if np.any(self.hi < self.lo):
            raise ValueError("every bound must satisfy min <= max")
        self.dim = len(self.lo)
        self.span = self.hi - self.lo
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f = math.inf

    # ------------------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        """``True`` once the evaluation budget has been spent."""
        return (
            self.max_evaluations is not None
            and self.n_evaluations >= self.max_evaluations
        )

    def clip(self, X: np.ndarray) -> np.ndarray:
        """Clip ``X`` (any shape broadcastable to ``(..., dim)``) to bounds."""
        return np.clip(X, self.lo, self.hi)

    def random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``n`` points uniformly inside the bounds."""
        return self.lo + rng.random((n, self.dim)) * self.span

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``X`` (2-D) and return a 1-D array of values.

        Rows evaluated after the budget is exhausted, and rows whose objective
        returns NaN, are given ``inf`` so they never win a comparison.
        """
        X = np.atleast_2d(X)
        out = np.full(len(X), math.inf)
        for i, x in enumerate(X):
            if self.exhausted:
                break
            f = float(self.objective_fn(x.tolist()))
            self.n_evaluations += 1
            if math.isnan(f):
                f = math.inf
            out[i] = f
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
        return out

    def evaluate_one(self, x: np.ndarray) -> float:
        """Evaluate a single 1-D candidate."""
        return float(self.evaluate(np.asarray(x)[None, :])[0])


class PopulationOptimiser(BaseOptimiser):
    """Base class for continuous population-based metaheuristics.

    Parameters
    ----------
    population_size : int, optional
        Number of candidate solutions kept per iteration.  ``None`` lets the
        algorithm choose a size from the problem dimension.
    max_iterations : int or None
        Hard limit on the number of iterations (generations).  Default 1000.
    max_evaluations : int or None
        Hard limit on objective evaluations.  At least one of
        ``max_iterations`` / ``max_evaluations`` must be set.
    max_no_improve : int or None
        Stop after this many consecutive iterations without the best value
        improving by more than ``tol``.  ``None`` disables the check.
    tol : float
        Minimum improvement counted by ``max_no_improve``.  Default 0.
    seed : int, optional
        Seed for the algorithm's private random generator.
    """

    #: Population size used when ``population_size`` is ``None``.
    default_population_size: int = 30

    def __init__(
        self,
        population_size: Optional[int] = None,
        max_iterations: Optional[int] = 1000,
        max_evaluations: Optional[int] = None,
        max_no_improve: Optional[int] = None,
        tol: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if population_size is not None and population_size < self._min_population():
            raise ValueError(
                f"population_size must be at least {self._min_population()} "
                f"for {type(self).__name__}"
            )
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.max_no_improve = max_no_improve
        self.tol = tol
        self.seed = seed

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _min_population(self) -> int:
        return 2

    def _population_size(self, dim: int) -> int:
        return self.population_size or self.default_population_size

    def _initialise(
        self, problem: Problem, state: SimpleNamespace, rng: np.random.Generator
    ) -> None:
        """Set up algorithm-specific state (optional)."""

    def _step(
        self, problem: Problem, state: SimpleNamespace, rng: np.random.Generator
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[Sequence[Sequence[float]]] = None,
        initial_solution: Optional[Sequence[float]] = None,
        max_evaluations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        callback: Optional[Callable[[int, List[float], float], Any]] = None,
        record_trajectory: bool = False,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run the algorithm.

        Parameters
        ----------
        objective_fn : callable
            Function of a list of floats returning a scalar.
        bounds : list of (min, max) tuples
            Box bounds for each variable.  Required.
        maximise : bool
            Maximise instead of minimise.  Default ``False``.
        initial_solutions : list of lists, optional
            Warm-start the population with these solutions (extra slots are
            filled at random; surplus solutions are ignored).
        initial_solution : list, optional
            A single warm-start solution (convenience for ``'chain'``
            ensembles).  Added in front of ``initial_solutions``.
        max_evaluations, max_iterations : int, optional
            Override the constructor's budget for this call only.
        callback : callable, optional
            ``callback(iteration, best_solution, best_value)`` called after
            every iteration; return ``True`` to stop early.
        record_trajectory : bool
            Record a :class:`~optim.base.Step` per iteration in
            ``result.trajectory``.  Default ``False``.

        Returns
        -------
        OptimisationResult
            Includes the final ``population`` (best first).
        """
        if bounds is None:
            raise ValueError(f"bounds must be provided for {type(self).__name__}")

        max_iter = self.max_iterations if max_iterations is None else max_iterations
        max_evals = self.max_evaluations if max_evaluations is None else max_evaluations
        if max_iter is None and max_evals is None:
            raise ValueError("set max_iterations and/or max_evaluations")
        if max_evals is not None and max_evals < 1:
            raise ValueError("max_evaluations must be at least 1")

        rng = np.random.default_rng(self.seed)
        sign = -1.0 if maximise else 1.0
        problem = Problem(lambda x: sign * objective_fn(x), bounds, max_evals)

        seeds: List[Sequence[float]] = []
        if initial_solution is not None:
            seeds.append(initial_solution)
        if initial_solutions is not None:
            seeds.extend(initial_solutions)

        n = self._population_size(problem.dim)
        X = problem.random(n, rng)
        for k, s in enumerate(seeds[:n]):
            X[k] = problem.clip(np.asarray(s, dtype=float))

        state = SimpleNamespace(
            X=X,
            F=problem.evaluate(X),
            iteration=0,
            progress=0.0,
            max_iterations=max_iter,
        )
        self._initialise(problem, state, rng)

        history = [problem.best_f]
        trajectory = (
            [self._make_step(0, problem, state, None, {"type": "initialise"})]
            if record_trajectory else None
        )
        no_improve = 0
        while not problem.exhausted:
            if max_iter is not None and state.iteration >= max_iter:
                break
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

            state.progress = self._progress(state.iteration, problem, max_iter, max_evals)
            previous = problem.best_f
            state.action = {"type": self.action_name}
            self._step(problem, state, rng)
            state.iteration += 1
            history.append(problem.best_f)
            if trajectory is not None:
                trajectory.append(
                    self._make_step(state.iteration, problem, state, previous, state.action)
                )

            if previous - problem.best_f > self.tol:
                no_improve = 0
            else:
                no_improve += 1

            if callback is not None and callback(
                state.iteration, problem.best_x.tolist(), sign * problem.best_f
            ):
                break

        order = np.argsort(state.F, kind="stable")
        return OptimisationResult(
            best_solution=problem.best_x.tolist(),
            best_value=sign * problem.best_f,
            history=[sign * h for h in history],
            n_evaluations=problem.n_evaluations,
            population=state.X[order].tolist(),
            population_values=(sign * state.F[order]).tolist(),
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------
    # State-action layer
    # ------------------------------------------------------------------

    @property
    def action_name(self) -> str:
        """Default ``action["type"]`` recorded for each iteration."""
        return type(self).__name__.replace("Optimiser", "").lower()

    def _state_summary(self, state: SimpleNamespace) -> dict:
        """Algorithm-specific additions to the recorded state (override)."""
        return {}

    def _make_step(self, idx, problem, state, previous, action) -> Step:
        finite = state.F[np.isfinite(state.F)]
        span = np.where(problem.span > 0, problem.span, 1.0)
        summary = {
            "best": float(problem.best_f),
            "mean": float(finite.mean()) if len(finite) else None,
            "diversity": float(np.mean(np.std(state.X / span, axis=0))),
            "evaluations": int(problem.n_evaluations),
            "progress": float(state.progress),
        }
        summary.update(self._state_summary(state))
        delta = None if previous is None else float(problem.best_f - previous)
        return Step(
            step_idx=idx,
            cost=float(problem.best_f),
            delta_cost=delta,
            accepted=delta is not None and delta < 0,
            action=dict(action),
            state=summary,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _progress(
        iteration: int,
        problem: Problem,
        max_iter: Optional[int],
        max_evals: Optional[int],
    ) -> float:
        """Fraction of the budget consumed so far (whichever runs out first)."""
        fractions = []
        if max_iter:
            fractions.append(iteration / max_iter)
        if max_evals:
            fractions.append(problem.n_evaluations / max_evals)
        return min(1.0, max(fractions)) if fractions else 0.0

    @staticmethod
    def _greedy_replace(state: SimpleNamespace, X_new: np.ndarray, F_new: np.ndarray) -> np.ndarray:
        """Keep each new candidate only where it is no worse than the incumbent.

        Returns the boolean mask of replaced rows.
        """
        better = F_new <= state.F
        state.X[better] = X_new[better]
        state.F[better] = F_new[better]
        return better

    @staticmethod
    def _random_others(
        n: int, rng: np.random.Generator, k: int = 1
    ) -> np.ndarray:
        """For each row ``i`` draw ``k`` distinct indices from ``range(n)``
        that are all different from ``i``.  Returns an ``(n, k)`` array."""
        out = np.empty((n, k), dtype=int)
        for i in range(n):
            choices = rng.choice(n - 1, size=k, replace=False)
            choices[choices >= i] += 1
            out[i] = choices
        return out
