"""
Common benchmark problems — the optimisation analogue of clustbench's
datasets.

Problems are organised in three categories, each a registry of *families*
that build reproducible instances (see :func:`make_problem`):

* **continuous** — classic test functions with shift / rotation / noise
  transforms (sphere, ellipsoid, bent cigar, step, zakharov, rosenbrock,
  rastrigin, ackley, griewank, levy, styblinski-tang, schwefel);
* **combinatorial** — tsp, knapsack, onemax, trap, nk, maxcut;
* **profit** — one family per profit-function type: linear
  (``production_planning``), concave (``marketing_budget``), quadratic
  (``pricing``), risk-adjusted (``portfolio``), stochastic (``newsvendor``)
  and discontinuous (``fixed_charge``).

>>> from optim.problems import make_problem, PROBLEMS
>>> p = make_problem("pricing", dim=5, instance=1)
>>> p.sense, round(p.true_objective(p.optimum_solution), 6) == round(p.optimum, 6)
('max', True)
"""

from . import combinatorial, continuous, profit  # noqa: F401  (registration)
from .base import (
    ENCODINGS,
    PROBLEMS,
    TAGS,
    ProblemFamily,
    ProblemInstance,
    make_problem,
    register_problem,
)
from .encodings import as_random_key, decode

__all__ = [
    "ENCODINGS",
    "PROBLEMS",
    "TAGS",
    "ProblemFamily",
    "ProblemInstance",
    "as_random_key",
    "decode",
    "make_problem",
    "register_problem",
]
