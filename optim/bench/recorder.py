"""
The evaluation recorder — the benchmark's single source of truth.

Every optimiser (and every ensemble member) sees the problem only through a
:class:`Recorder`.  It

* enforces the evaluation budget **exactly**, for every optimiser, by
  raising :class:`BudgetExhausted` on the first call past the budget (so
  optimisers without budget control cannot overspend);
* tracks the incumbent (best observed solution) in the problem's own sense;
* records the anytime curve ``(evaluations, true value of incumbent)`` at
  every improvement — using the noise-free objective on noisy problems.

Because the recorder, not the optimiser, defines what was found and when,
results are comparable across very different algorithms.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

from ..problems import ProblemInstance


class BudgetExhausted(Exception):
    """Raised when an optimiser asks for more evaluations than allowed."""


class Recorder:
    """Minimising, budget-enforcing view of a problem.

    Parameters
    ----------
    problem : ProblemInstance
        The problem as the optimiser solves it (possibly a random-key view).
    budget : int
        Maximum number of objective evaluations.
    """

    def __init__(self, problem: ProblemInstance, budget: int) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.sign = -1.0 if problem.maximise else 1.0
        self.n_evaluations = 0
        self.best_observed = math.inf  # minimisation orientation
        self.best_x: Optional[List[Any]] = None
        self.best_true: Optional[float] = None  # problem orientation
        self.curve: List[Tuple[int, float]] = []

    def __call__(self, x: Sequence) -> float:
        if self.n_evaluations >= self.budget:
            raise BudgetExhausted(self.budget)
        self.n_evaluations += 1
        value = float(self.problem.objective(x))
        m = self.sign * value
        if math.isnan(m):
            m = math.inf
        if m < self.best_observed:
            self.best_observed = m
            self.best_x = list(x)
            true = value if not self.problem.noisy else float(self.problem.true_objective(x))
            self.best_true = true
            self.curve.append((self.n_evaluations, true))
        return m
