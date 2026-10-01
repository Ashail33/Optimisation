"""
optim.bench — a benchmark harness for optimisers and ensembles.

Modelled on clustbench: a YAML config is expanded into a grid of tasks
(problem instance × optimiser × repeat), each task runs through an
evaluation :class:`~optim.bench.recorder.Recorder` that enforces the budget
exactly and records the anytime curve, results are persisted as tidy tables
in a run directory, and :mod:`~optim.bench.analysis` turns them into
scale-free metrics, ranks and statistics that feed a static dashboard.

Requires the ``bench`` extra: ``pip install -e ".[bench]"``.
"""

from .analysis import TARGETS, average_ranks, derive, friedman, summarise
from .config import BenchmarkConfig, OptimiserSpec, ProblemSpec, Task, load_config
from .recorder import BudgetExhausted, Recorder
from .runner import run_benchmark, run_task

__all__ = [
    "BenchmarkConfig",
    "BudgetExhausted",
    "OptimiserSpec",
    "ProblemSpec",
    "Recorder",
    "TARGETS",
    "Task",
    "average_ranks",
    "derive",
    "friedman",
    "load_config",
    "run_benchmark",
    "run_task",
    "summarise",
]
