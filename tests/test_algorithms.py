"""Tests for the metaheuristic catalogue in :mod:`optim.algorithms`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from optim import (
    OPTIMISERS,
    ACOROptimiser,
    AntColonyOptimiser,
    ArtificialBeeColonyOptimiser,
    BatOptimiser,
    CMAESOptimiser,
    CuckooSearchOptimiser,
    DifferentialEvolutionOptimiser,
    FireflyOptimiser,
    GreyWolfOptimiser,
    HarmonySearchOptimiser,
    JayaOptimiser,
    OptimisationResult,
    PopulationOptimiser,
    SineCosineOptimiser,
    TabuSearchOptimiser,
    TLBOOptimiser,
    WhaleOptimiser,
)
from optim.benchmarks import BENCHMARKS, compare, format_table, rastrigin, sphere

BOUNDS_5D = [(-5.0, 5.0)] * 5

# (class, sphere tolerance at 5-D with 6000 evaluations)
CONTINUOUS = [
    (DifferentialEvolutionOptimiser, 1e-2),
    (CMAESOptimiser, 1e-6),
    (GreyWolfOptimiser, 1e-6),
    (WhaleOptimiser, 1e-6),
    (FireflyOptimiser, 1e-2),
    (CuckooSearchOptimiser, 1e-1),
    (ArtificialBeeColonyOptimiser, 1e-2),
    (BatOptimiser, 1.0),
    (HarmonySearchOptimiser, 1e-1),
    (TLBOOptimiser, 1e-6),
    (SineCosineOptimiser, 1e-3),
    (JayaOptimiser, 1e-2),
    (ACOROptimiser, 1e-4),
]
IDS = [c.__name__ for c, _ in CONTINUOUS]


def _run(cls, fn=sphere, bounds=BOUNDS_5D, **kw):
    kw.setdefault("max_evaluations", 6000)
    kw.setdefault("max_iterations", None)
    kw.setdefault("seed", 0)
    return cls(**kw).optimise(fn, bounds)


@pytest.mark.parametrize("cls,tol", CONTINUOUS, ids=IDS)
class TestContinuousAlgorithms:
    def test_sphere_converges(self, cls, tol):
        result = _run(cls)
        assert isinstance(result, OptimisationResult)
        assert result.best_value < tol

    def test_respects_budget_and_bounds(self, cls, tol):
        result = _run(cls, max_evaluations=1000)
        assert result.n_evaluations <= 1000
        assert len(result.best_solution) == 5
        assert all(-5.0 <= v <= 5.0 for v in result.best_solution)
        for sol in result.population:
            assert all(-5.0 <= v <= 5.0 for v in sol)

    def test_reproducible_with_seed(self, cls, tol):
        a = _run(cls, max_evaluations=800)
        b = _run(cls, max_evaluations=800)
        assert a.best_value == b.best_value
        assert a.best_solution == b.best_solution

    def test_history_monotone_and_population_sorted(self, cls, tol):
        result = _run(cls, max_evaluations=800)
        assert all(x >= y for x, y in zip(result.history, result.history[1:]))
        assert result.history[-1] == result.best_value
        vals = result.population_values
        assert vals == sorted(vals)

    def test_maximise(self, cls, tol):
        result = cls(max_evaluations=1500, max_iterations=None, seed=1).optimise(
            lambda x: -sphere(x), BOUNDS_5D, maximise=True
        )
        assert result.best_value <= 0
        assert result.best_value == pytest.approx(-sphere(result.best_solution))
        assert all(x <= y for x, y in zip(result.history, result.history[1:]))

    def test_warm_start_is_used(self, cls, tol):
        opt = cls(max_evaluations=60, max_iterations=None, seed=0)
        result = opt.optimise(sphere, BOUNDS_5D, initial_solution=[0.0] * 5)
        assert result.best_value == 0.0


class TestPopulationOptimiserBase:
    def test_requires_bounds(self):
        with pytest.raises(ValueError):
            GreyWolfOptimiser().optimise(sphere)

    def test_requires_a_budget(self):
        with pytest.raises(ValueError):
            GreyWolfOptimiser(max_iterations=None).optimise(sphere, BOUNDS_5D)

    def test_max_iterations(self):
        result = GreyWolfOptimiser(max_iterations=7, seed=0).optimise(sphere, BOUNDS_5D)
        assert len(result.history) == 8  # initial + one per iteration

    def test_max_no_improve(self):
        result = JayaOptimiser(max_iterations=10_000, max_no_improve=5, seed=0).optimise(
            lambda x: 1.0, BOUNDS_5D
        )
        assert len(result.history) == 6

    def test_callback_can_stop(self):
        calls = []

        def cb(it, best, value):
            calls.append(it)
            return it >= 3

        TLBOOptimiser(seed=0).optimise(sphere, BOUNDS_5D, callback=cb)
        assert calls == [1, 2, 3]

    def test_nan_objective_treated_as_worst(self):
        result = DifferentialEvolutionOptimiser(max_evaluations=500, seed=0).optimise(
            lambda x: float("nan") if x[0] > 0 else sphere(x), BOUNDS_5D
        )
        assert result.best_solution[0] <= 0

    def test_population_too_small(self):
        with pytest.raises(ValueError):
            DifferentialEvolutionOptimiser(population_size=3)

    def test_all_continuous_are_population_optimisers(self):
        for cls, _ in CONTINUOUS:
            assert issubclass(cls, PopulationOptimiser)


def test_de_strategies():
    for strategy in ("rand/1", "best/1", "current-to-best/1", "rand/2"):
        r = _run(DifferentialEvolutionOptimiser, strategy=strategy, F=0.6)
        assert r.best_value < 0.1, strategy
    with pytest.raises(ValueError):
        DifferentialEvolutionOptimiser(strategy="nope")


def test_cmaes_handles_different_scales():
    bounds = [(-1e-3, 1e-3), (-1e3, 1e3)]
    r = CMAESOptimiser(max_evaluations=3000, max_iterations=None, seed=0).optimise(
        lambda x: (x[0] * 1e3) ** 2 + (x[1] / 1e3) ** 2, bounds
    )
    assert r.best_value < 1e-6


def test_gwo_solves_rastrigin():
    r = _run(GreyWolfOptimiser, fn=rastrigin, bounds=[(-5.12, 5.12)] * 5, max_evaluations=10_000)
    assert r.best_value < 1.0


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

def _tsp(n=10, seed=0):
    pts = np.random.default_rng(seed).random((n, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)

    def cost(tour):
        return float(sum(dist[tour[i - 1], tour[i]] for i in range(len(tour))))

    return dist, cost


class TestTabuSearch:
    def test_real(self):
        r = TabuSearchOptimiser(seed=0, step_size=0.05).optimise(sphere, [(-5, 5)] * 3)
        assert r.best_value < 0.05

    def test_binary_maximise(self):
        r = TabuSearchOptimiser(encoding="binary", seed=0).optimise(
            sum, n_genes=15, maximise=True
        )
        assert r.best_solution == [1] * 15
        assert r.best_value == 15

    def test_permutation(self):
        _, cost = _tsp()
        r = TabuSearchOptimiser(encoding="permutation", seed=0, max_iterations=300).optimise(
            cost, n_genes=10
        )
        assert sorted(r.best_solution) == list(range(10))
        rnd = np.mean([cost(list(np.random.default_rng(i).permutation(10))) for i in range(20)])
        assert r.best_value < 0.7 * rnd

    def test_budget(self):
        r = TabuSearchOptimiser(seed=0, max_evaluations=100).optimise(sphere, [(-5, 5)] * 3)
        assert r.n_evaluations <= 100

    def test_invalid_encoding(self):
        with pytest.raises(ValueError):
            TabuSearchOptimiser(encoding="tree")


# ---------------------------------------------------------------------------
# Ant Colony (permutations)
# ---------------------------------------------------------------------------

class TestAntColony:
    def test_tsp_with_heuristic(self):
        dist, cost = _tsp()
        with np.errstate(divide="ignore"):
            eta = np.where(dist > 0, 1.0 / dist, 0.0)
        r = AntColonyOptimiser(seed=0, max_iterations=60).optimise(cost, heuristic=eta)
        assert sorted(r.best_solution) == list(range(10))
        tabu = TabuSearchOptimiser(encoding="permutation", seed=0, max_iterations=500).optimise(
            cost, n_genes=10
        )
        assert r.best_value <= tabu.best_value * 1.05

    def test_ordering_problem_without_heuristic(self):
        # Cost is minimised by the identity permutation.
        target = list(range(8))
        r = AntColonyOptimiser(seed=1, max_iterations=150, max_no_improve=None).optimise(
            lambda p: sum(abs(a - b) for a, b in zip(p, target)), n_genes=8
        )
        assert r.best_value <= 4

    def test_warm_start_and_maximise(self):
        r = AntColonyOptimiser(seed=0, max_iterations=1).optimise(
            lambda p: -p[0], n_genes=5, maximise=True, initial_solution=[0, 1, 2, 3, 4]
        )
        assert r.best_value == 0

    def test_needs_size(self):
        with pytest.raises(ValueError):
            AntColonyOptimiser().optimise(lambda p: 0.0)


# ---------------------------------------------------------------------------
# Registry & benchmarks
# ---------------------------------------------------------------------------

def test_registry_contains_catalogue():
    for key in ("de", "cmaes", "gwo", "woa", "firefly", "cuckoo", "abc", "bat",
                "harmony", "tlbo", "sca", "jaya", "acor", "aco", "tabu"):
        assert key in OPTIMISERS


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmarks_at_optimum(name):
    bench = BENCHMARKS[name]
    optimum_x = {
        "rosenbrock": [1.0] * 4,
        "levy": [1.0] * 4,
        "schwefel": [420.9687] * 4,
        "styblinski_tang": [-2.903534] * 4,
    }.get(name, [0.0] * 4)
    assert bench.fn(optimum_x) == pytest.approx(bench.optimum(4), abs=1e-3)


def test_compare_harness():
    rows = compare(
        {"DE": DifferentialEvolutionOptimiser(max_iterations=None),
         "TLBO": TLBOOptimiser(max_iterations=None)},
        problems=["sphere"], dim=3, n_runs=2, max_evaluations=500,
    )
    assert [r["optimiser"] for r in rows] == ["DE", "TLBO"]
    assert all(r["mean_evaluations"] <= 500 for r in rows)
    assert "sphere" in format_table(rows)
