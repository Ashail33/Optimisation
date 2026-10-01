"""
Standard benchmark functions and a small harness for comparing optimisers.

Every function takes a list/array ``x`` and returns a float to *minimise*;
all have a global minimum value of 0 except where noted in
:data:`BENCHMARKS`.

>>> from optim import DifferentialEvolutionOptimiser, GreyWolfOptimiser
>>> from optim.benchmarks import compare, format_table
>>> table = compare(
...     {"DE": DifferentialEvolutionOptimiser(max_iterations=None),
...      "GWO": GreyWolfOptimiser(max_iterations=None)},
...     problems=["sphere", "rastrigin"], dim=10, n_runs=3,
...     max_evaluations=10_000, seed=0)              # doctest: +SKIP
>>> print(format_table(table))                       # doctest: +SKIP
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .base import BaseOptimiser


def sphere(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rosenbrock(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x) -> float:
    x = np.asarray(x, dtype=float)
    d = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
        + 20
        + math.e
    )


def griewank(x) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))))


def schwefel(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(418.9828872724338 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x) -> float:
    x = np.asarray(x, dtype=float)
    w = 1 + (x - 1) / 4
    return float(
        np.sin(np.pi * w[0]) ** 2
        + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    )


def zakharov(x) -> float:
    x = np.asarray(x, dtype=float)
    s = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return float(np.sum(x ** 2) + s ** 2 + s ** 4)


def styblinski_tang(x) -> float:
    """Minimum ``-39.16599 * d`` at ``x_i = -2.903534``."""
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x))


@dataclass(frozen=True)
class Benchmark:
    """A benchmark function with its usual search box and known optimum."""

    fn: Callable
    bound: Tuple[float, float]
    optimum_per_dim: float = 0.0
    multimodal: bool = True

    def bounds(self, dim: int) -> List[Tuple[float, float]]:
        return [self.bound] * dim

    def optimum(self, dim: int) -> float:
        return self.optimum_per_dim * dim


#: Registry of benchmarks keyed by name.
BENCHMARKS: Dict[str, Benchmark] = {
    "sphere": Benchmark(sphere, (-5.12, 5.12), multimodal=False),
    "rosenbrock": Benchmark(rosenbrock, (-5.0, 10.0), multimodal=False),
    "rastrigin": Benchmark(rastrigin, (-5.12, 5.12)),
    "ackley": Benchmark(ackley, (-32.768, 32.768)),
    "griewank": Benchmark(griewank, (-600.0, 600.0)),
    "schwefel": Benchmark(schwefel, (-500.0, 500.0)),
    "levy": Benchmark(levy, (-10.0, 10.0)),
    "zakharov": Benchmark(zakharov, (-5.0, 10.0), multimodal=False),
    "styblinski_tang": Benchmark(styblinski_tang, (-5.0, 5.0), -39.16599),
}


def compare(
    optimisers: Mapping[str, BaseOptimiser],
    problems: Optional[Iterable[str]] = None,
    dim: int = 10,
    n_runs: int = 5,
    max_evaluations: Optional[int] = 10_000,
    seed: Optional[int] = 0,
) -> List[Dict[str, object]]:
    """Run every optimiser on every benchmark ``n_runs`` times.

    ``max_evaluations`` is passed to ``optimise`` for optimisers that accept
    it.  Each run gets a different derived seed (applied to optimisers that
    have a ``seed`` attribute).  Returns one row per (problem, optimiser)
    with the mean / std / best error (value minus known optimum), mean
    evaluations and mean wall time.
    """
    from .ensembles._common import run_optimiser

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for pname in problems or BENCHMARKS:
        bench = BENCHMARKS[pname]
        bounds = bench.bounds(dim)
        for oname, opt in optimisers.items():
            errors, evals, times = [], [], []
            for _ in range(n_runs):
                start = time.perf_counter()
                res = run_optimiser(
                    opt, bench.fn, bounds,
                    max_evaluations=max_evaluations,
                    seed=int(rng.integers(0, 2 ** 31 - 1)),
                )
                times.append(time.perf_counter() - start)
                errors.append(res.best_value - bench.optimum(dim))
                evals.append(res.n_evaluations)
            rows.append(
                dict(
                    problem=pname,
                    optimiser=oname,
                    mean_error=float(np.mean(errors)),
                    std_error=float(np.std(errors)),
                    best_error=float(np.min(errors)),
                    mean_evaluations=float(np.mean(evals)),
                    mean_seconds=float(np.mean(times)),
                )
            )
    return rows


def format_table(rows: Sequence[Mapping[str, object]]) -> str:
    """Render :func:`compare` output as a fixed-width text table."""
    header = f"{'problem':<16}{'optimiser':<28}{'mean err':>12}{'std':>12}{'best':>12}{'evals':>9}{'sec':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['problem']:<16}{r['optimiser']:<28}"
            f"{r['mean_error']:>12.4g}{r['std_error']:>12.4g}{r['best_error']:>12.4g}"
            f"{r['mean_evaluations']:>9.0f}{r['mean_seconds']:>8.2f}"
        )
    return "\n".join(lines)
