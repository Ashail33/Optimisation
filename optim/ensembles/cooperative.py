"""
Cooperative co-evolution ensemble (variable decomposition).

Large problems are split into groups of variables.  Each group is optimised
in turn as a small sub-problem, with the remaining variables fixed at the
current best solution (the *context vector*); an improvement on a group is
written back into the context.  Several optimisers can share the work —
group ``g`` is handled by ``optimisers[g % len(optimisers)]`` — and with
random grouping the decomposition is reshuffled every cycle so interacting
variables eventually land in the same group.

References
----------
Potter, M. A. & De Jong, K. A. (1994). A cooperative coevolutionary approach
to function optimization. *PPSN III*, LNCS 866, 249–257.

Yang, Z., Tang, K. & Yao, X. (2008). Large scale evolutionary optimization
using cooperative coevolution. *Information Sciences*, 178(15), 2985–2999.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult
from ..ensemble import EnsembleResult
from ._common import Budget, CountingObjective, derive_seed, run_optimiser, split_budget


class CooperativeCoevolutionOptimiser(BaseOptimiser):
    """Decompose continuous variables into groups and optimise them in turn.

    Parameters
    ----------
    optimisers : BaseOptimiser or list of BaseOptimiser
        Optimiser(s) used for the sub-problems (assigned round-robin).
    n_groups : int, optional
        Number of variable groups.  Default ``min(dimension, 4)``.
    grouping : {'random', 'sequential'}
        ``'random'`` reshuffles variables into groups every cycle;
        ``'sequential'`` uses fixed contiguous blocks.  Default ``'random'``.
    n_cycles : int
        Passes over all groups.  Default 10.
    group_evaluations : int, optional
        Evaluation budget per sub-problem.  Default: ``max_evaluations``
        split over cycles and groups.
    max_evaluations : int, optional
        Total evaluation budget.
    seed : int, optional
        Seeds grouping and the derived constituent seeds.
    """

    def __init__(
        self,
        optimisers,
        n_groups: Optional[int] = None,
        grouping: str = "random",
        n_cycles: int = 10,
        group_evaluations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(optimisers, BaseOptimiser):
            optimisers = [optimisers]
        if not optimisers:
            raise ValueError("optimisers must be non-empty")
        if grouping not in ("random", "sequential"):
            raise ValueError("grouping must be 'random' or 'sequential'")
        self.optimisers = list(optimisers)
        self.n_groups = n_groups
        self.grouping = grouping
        self.n_cycles = n_cycles
        self.group_evaluations = group_evaluations
        self.max_evaluations = max_evaluations
        self.seed = seed

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solution: Optional[Sequence[float]] = None,
        initial_solutions: Optional[Sequence[Sequence[float]]] = None,
        optimiser_kwargs: Optional[List[Dict[str, Any]]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Run cooperative co-evolution (continuous ``bounds`` required).

        ``max_evaluations`` overrides the constructor's total budget.
        """
        if bounds is None:
            raise ValueError("bounds must be provided")
        rng = np.random.default_rng(self.seed)
        objective = CountingObjective(objective_fn, maximise)
        total = self.max_evaluations if max_evaluations is None else max_evaluations
        limit = Budget(total, objective)
        dim = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_groups = max(1, min(dim, self.n_groups or min(dim, 4)))
        budget = self.group_evaluations or split_budget(
            total, self.n_cycles * n_groups
        )

        if initial_solution is None and initial_solutions:
            initial_solution = initial_solutions[0]
        if initial_solution is not None:
            context = np.clip(np.asarray(initial_solution, dtype=float), lo, hi)
        else:
            context = lo + rng.random(dim) * (hi - lo)
        context_f = objective(context.tolist())
        history = [context_f]
        run_results: List[OptimisationResult] = []

        groups = np.array_split(np.arange(dim), n_groups)
        for _cycle in range(self.n_cycles):
            if self.grouping == "random":
                groups = np.array_split(rng.permutation(dim), n_groups)
            for g, idx in enumerate(groups):
                if limit.spent:
                    break
                frozen = context.copy()

                def sub_objective(y, idx=idx, frozen=frozen):
                    x = frozen.copy()
                    x[idx] = y
                    return objective(x.tolist())

                k = g % len(self.optimisers)
                result = run_optimiser(
                    self.optimisers[k],
                    sub_objective,
                    [tuple(bounds[j]) for j in idx],
                    population=[context[idx].tolist()],
                    max_evaluations=limit.cap(budget),
                    seed=derive_seed(rng),
                    extra_kwargs=optimiser_kwargs[k] if optimiser_kwargs else None,
                )
                run_results.append(result)
                if result.best_value < context_f:
                    context[idx] = result.best_solution
                    context_f = result.best_value
            history.append(context_f)

        sign = objective.sign
        return EnsembleResult(
            best_solution=context.tolist(),
            best_value=sign * context_f,
            history=[sign * h for h in history],
            n_evaluations=objective.n_evaluations,
            run_results=run_results,
            info={"n_groups": n_groups, "grouping": self.grouping},
        )
