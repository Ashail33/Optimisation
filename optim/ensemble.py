"""
Ensemble optimiser — combine multiple optimisers to improve solution quality.

Three combination strategies are provided:

``'best'``
    Run every optimiser independently on the same problem and return the
    best result found across all of them.  This is the safest default and
    works well for difficult landscapes where different algorithms explore
    different regions.

``'chain'``
    Run the optimisers sequentially, warm-starting each one with the best
    solution from the previous.  Useful when a fast global search (e.g. PSO)
    feeds a precise local refinement (e.g. Local Search or SA).

``'random_restart'``
    Run the *same* optimiser multiple times with random initialisations (the
    list must contain exactly one optimiser element, but it may also be a
    list of identical objects).  Return the best result across all runs.

In all cases, every individual run is captured and accessible via the
``run_results`` attribute of the returned :class:`EnsembleResult`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseOptimiser, OptimisationResult


@dataclass
class EnsembleResult(OptimisationResult):
    """Extended result that also stores per-optimiser run results.

    Attributes
    ----------
    run_results : list of OptimisationResult
        Individual results from each optimiser (or restart).
    """

    run_results: List[OptimisationResult] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EnsembleResult("
            f"best_value={self.best_value:.6g}, "
            f"n_runs={len(self.run_results)}, "
            f"n_evaluations={self.n_evaluations})"
        )


class EnsembleOptimiser(BaseOptimiser):
    """Combine multiple optimisers into an ensemble.

    Parameters
    ----------
    optimisers : list of BaseOptimiser
        The constituent optimisers.  For ``strategy='random_restart'`` this
        should be either a list with one entry (the optimiser to restart) or a
        list of identical optimiser instances.
    strategy : {'best', 'chain', 'random_restart'}
        How the optimisers are combined.  Default: ``'best'``.
    n_restarts : int
        Number of restarts for ``strategy='random_restart'``.  Ignored for
        other strategies.  Default: ``5``.

    Examples
    --------
    **Best strategy** — run GA and PSO, take the best::

        from optim import GeneticOptimiser, PSOOptimiser, EnsembleOptimiser

        ga  = GeneticOptimiser(population_size=30, max_no_improve=50)
        pso = PSOOptimiser(n_particles=20, max_no_improve=100)

        ens = EnsembleOptimiser([ga, pso], strategy='best')
        result = ens.optimise(lambda x: x[0]**2 + x[1]**2,
                              bounds=[(-5, 5), (-5, 5)])

    **Chain strategy** — PSO for global exploration, SA for local refinement::

        from optim import PSOOptimiser, SimulatedAnnealingOptimiser, EnsembleOptimiser

        pso = PSOOptimiser(n_particles=20, max_no_improve=50)
        sa  = SimulatedAnnealingOptimiser(initial_temp=100, max_epochs=2000)

        ens = EnsembleOptimiser([pso, sa], strategy='chain')
        result = ens.optimise(lambda x: x[0]**2 + x[1]**2,
                              bounds=[(-5, 5), (-5, 5)])

    **Random restart** — run SA five times, keep the best::

        from optim import SimulatedAnnealingOptimiser, EnsembleOptimiser

        sa  = SimulatedAnnealingOptimiser(initial_temp=1000, max_epochs=3000)
        ens = EnsembleOptimiser([sa], strategy='random_restart', n_restarts=5)
        result = ens.optimise(lambda x: x[0]**2 + x[1]**2,
                              bounds=[(-5, 5), (-5, 5)])
    """

    def __init__(
        self,
        optimisers: List[BaseOptimiser],
        strategy: str = "best",
        n_restarts: int = 5,
    ) -> None:
        if not optimisers:
            raise ValueError("optimisers must be a non-empty list")
        if strategy not in ("best", "chain", "random_restart"):
            raise ValueError(
                "strategy must be one of: 'best', 'chain', 'random_restart'"
            )
        self.optimisers = optimisers
        self.strategy = strategy
        self.n_restarts = n_restarts

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        optimiser_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Run the ensemble.

        Parameters
        ----------
        objective_fn : callable
            Function to optimise.
        bounds : list of (min, max) tuples, optional
            Passed to each constituent optimiser.
        maximise : bool
            If ``True``, maximise the objective.  Default: ``False``.
        optimiser_kwargs : list of dicts, optional
            Per-optimiser keyword arguments.  The i-th dict is passed as
            ``**kwargs`` to the i-th optimiser's ``optimise`` call.  If not
            provided, each optimiser is called with no extra kwargs (beyond
            *objective_fn*, *bounds*, and *maximise*).

        Returns
        -------
        EnsembleResult
        """
        if self.strategy == "best":
            return self._run_best(objective_fn, bounds, maximise, optimiser_kwargs)
        if self.strategy == "chain":
            return self._run_chain(objective_fn, bounds, maximise, optimiser_kwargs)
        return self._run_random_restart(objective_fn, bounds, maximise, optimiser_kwargs)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _run_best(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]],
        maximise: bool,
        opt_kwargs: Optional[List[Dict[str, Any]]],
    ) -> EnsembleResult:
        """Run all optimisers independently; return the best result."""
        run_results: List[OptimisationResult] = []
        for idx, opt in enumerate(self.optimisers):
            kw = (opt_kwargs[idx] if opt_kwargs and idx < len(opt_kwargs) else {})
            result = opt.optimise(
                objective_fn, bounds, maximise=maximise, **kw
            )
            run_results.append(result)

        best = self._pick_best(run_results, maximise)
        return EnsembleResult(
            best_solution=best.best_solution,
            best_value=best.best_value,
            history=best.history,
            n_evaluations=sum(r.n_evaluations for r in run_results),
            run_results=run_results,
        )

    def _run_chain(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]],
        maximise: bool,
        opt_kwargs: Optional[List[Dict[str, Any]]],
    ) -> EnsembleResult:
        """Run optimisers sequentially, feeding the best solution forward."""
        run_results: List[OptimisationResult] = []
        warm_solution: Optional[Any] = None
        total_evals = 0

        for idx, opt in enumerate(self.optimisers):
            kw = dict(opt_kwargs[idx] if opt_kwargs and idx < len(opt_kwargs) else {})
            # Inject warm-start if the optimiser supports it
            if warm_solution is not None and "initial_solution" not in kw:
                kw["initial_solution"] = deepcopy(warm_solution)
            result = opt.optimise(
                objective_fn, bounds, maximise=maximise, **kw
            )
            run_results.append(result)
            warm_solution = result.best_solution
            total_evals += result.n_evaluations

        best = run_results[-1]
        return EnsembleResult(
            best_solution=best.best_solution,
            best_value=best.best_value,
            history=best.history,
            n_evaluations=total_evals,
            run_results=run_results,
        )

    def _run_random_restart(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]],
        maximise: bool,
        opt_kwargs: Optional[List[Dict[str, Any]]],
    ) -> EnsembleResult:
        """Restart the first optimiser *n_restarts* times."""
        opt = self.optimisers[0]
        kw = dict(opt_kwargs[0] if opt_kwargs else {})
        run_results: List[OptimisationResult] = []

        for _ in range(self.n_restarts):
            result = opt.optimise(
                objective_fn, bounds, maximise=maximise, **kw
            )
            run_results.append(result)

        best = self._pick_best(run_results, maximise)
        return EnsembleResult(
            best_solution=best.best_solution,
            best_value=best.best_value,
            history=best.history,
            n_evaluations=sum(r.n_evaluations for r in run_results),
            run_results=run_results,
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best(
        results: List[OptimisationResult], maximise: bool
    ) -> OptimisationResult:
        """Return the result with the best (min or max) value."""
        if maximise:
            return max(results, key=lambda r: r.best_value)
        return min(results, key=lambda r: r.best_value)
