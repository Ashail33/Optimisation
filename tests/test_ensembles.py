"""Tests for the ensemble strategies in :mod:`optim.ensembles`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from optim import (
    ENSEMBLES,
    AdaptiveEnsembleOptimiser,
    CMAESOptimiser,
    CooperativeCoevolutionOptimiser,
    DifferentialEvolutionOptimiser,
    EnsembleOptimiser,
    EnsembleResult,
    GeneticOptimiser,
    GreyWolfOptimiser,
    IslandModelOptimiser,
    LocalSearchOptimiser,
    MemeticOptimiser,
    PSOOptimiser,
    SimulatedAnnealingOptimiser,
    TabuSearchOptimiser,
    TLBOOptimiser,
)
from optim.benchmarks import rastrigin, sphere
from optim.ensembles._common import SolutionPool, run_optimiser

BOUNDS = [(-5.12, 5.12)] * 6


def _algos():
    return [
        DifferentialEvolutionOptimiser(),
        GreyWolfOptimiser(),
        CMAESOptimiser(),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestCommon:
    def test_pool_keeps_best_unique(self):
        pool = SolutionPool(3)
        pool.add([[1.0], [2.0], [1.0], [0.5], [3.0]], [1, 4, 1, 0.25, 9])
        assert pool.solutions == [[0.5], [1.0], [2.0]]
        assert pool.values == [0.25, 1, 4]

    def test_pool_ignores_inf(self):
        pool = SolutionPool(3)
        pool.add([[1.0], [2.0]], [float("inf"), 2.0])
        assert pool.solutions == [[2.0]]

    def test_run_optimiser_passes_supported_kwargs(self):
        # DE accepts initial_solutions + max_evaluations
        r = run_optimiser(
            DifferentialEvolutionOptimiser(), sphere, BOUNDS,
            population=[[0.0] * 6], max_evaluations=50, seed=1,
        )
        assert r.best_value == 0.0 and r.n_evaluations <= 50
        # SA only accepts initial_solution
        r = run_optimiser(
            SimulatedAnnealingOptimiser(max_epochs=1, initial_temp=1e-9), sphere, BOUNDS,
            population=[[0.0] * 6], seed=1,
        )
        assert r.best_value == 0.0

    def test_run_optimiser_does_not_mutate_seed(self):
        opt = GreyWolfOptimiser(seed=5)
        run_optimiser(opt, sphere, BOUNDS, max_evaluations=40, seed=99)
        assert opt.seed == 5


# ---------------------------------------------------------------------------
# Island model
# ---------------------------------------------------------------------------

class TestIslandModel:
    @pytest.mark.parametrize("topology", ["ring", "fully_connected", "star", "random"])
    def test_topologies(self, topology):
        ens = IslandModelOptimiser(_algos(), topology=topology, n_epochs=5,
                                   max_evaluations=6000, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        assert isinstance(r, EnsembleResult)
        assert r.best_value < 1e-3
        assert r.n_evaluations <= 6000
        assert len(r.info["island_best_values"]) == 3

    def test_migration_shares_best(self):
        ens = IslandModelOptimiser(_algos(), topology="fully_connected", n_epochs=3,
                                   max_evaluations=3000, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        # after migration every island has seen a solution at least as good as
        # the best one present before the final epoch
        assert max(r.info["island_best_values"]) <= r.history[-2]

    def test_works_with_legacy_optimisers(self):
        ens = IslandModelOptimiser(
            [PSOOptimiser(n_particles=10, max_iterations=20),
             GeneticOptimiser(population_size=20, max_generations=20)],
            n_epochs=3, seed=0,
        )
        r = ens.optimise(sphere, BOUNDS)
        assert r.best_value < sphere([5.12] * 6)
        assert len(r.run_results) == 6

    def test_reproducible(self):
        mk = lambda: IslandModelOptimiser(_algos(), n_epochs=3, max_evaluations=2000, seed=3)
        assert mk().optimise(sphere, BOUNDS).best_value == mk().optimise(sphere, BOUNDS).best_value

    def test_validation(self):
        with pytest.raises(ValueError):
            IslandModelOptimiser([GreyWolfOptimiser()])
        with pytest.raises(ValueError):
            IslandModelOptimiser(_algos(), topology="mesh")


# ---------------------------------------------------------------------------
# Adaptive (bandit) ensemble
# ---------------------------------------------------------------------------

class TestAdaptive:
    @pytest.mark.parametrize(
        "selection", ["ucb", "epsilon_greedy", "probability_matching", "round_robin"]
    )
    def test_selection_rules(self, selection):
        ens = AdaptiveEnsembleOptimiser(_algos(), selection=selection, n_rounds=12,
                                        max_evaluations=6000, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        assert r.best_value < 1e-3
        assert r.n_evaluations <= 6000
        assert sum(r.info["selection_counts"].values()) == len(r.info["selections"])

    def test_round_robin_cycles(self):
        ens = AdaptiveEnsembleOptimiser(_algos(), selection="round_robin", n_rounds=6,
                                        max_evaluations=3000, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        assert r.info["selections"][:3] == [
            "DifferentialEvolutionOptimiser", "GreyWolfOptimiser", "CMAESOptimiser"
        ]

    def test_learns_to_prefer_useful_algorithm(self):
        class Useless(GreyWolfOptimiser):
            def optimise(self, objective_fn, bounds=None, **kw):
                kw["max_evaluations"] = 1
                return super().optimise(objective_fn, bounds, **kw)

        ens = AdaptiveEnsembleOptimiser([Useless(), TLBOOptimiser()], n_rounds=30,
                                        round_evaluations=200, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        counts = r.info["selection_counts"]
        assert counts["TLBOOptimiser"] > counts["Useless"]

    def test_maximise(self):
        ens = AdaptiveEnsembleOptimiser(_algos(), n_rounds=6, max_evaluations=2000, seed=0)
        r = ens.optimise(lambda x: -sphere(x), BOUNDS, maximise=True)
        assert r.best_value <= 0
        assert r.best_value == pytest.approx(-sphere(r.best_solution))
        assert all(x <= y for x, y in zip(r.history, r.history[1:]))

    def test_validation(self):
        with pytest.raises(ValueError):
            AdaptiveEnsembleOptimiser([])
        with pytest.raises(ValueError):
            AdaptiveEnsembleOptimiser(_algos(), selection="thompson")


# ---------------------------------------------------------------------------
# Memetic
# ---------------------------------------------------------------------------

class TestMemetic:
    def test_global_plus_tabu(self):
        ens = MemeticOptimiser(DifferentialEvolutionOptimiser(),
                               TabuSearchOptimiser(step_size=0.01),
                               n_generations=5, max_evaluations=8000, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        assert r.best_value < 1e-2
        assert r.n_evaluations <= 8000
        assert len(r.run_results) == 5 * 4  # 1 global + 3 local per generation

    def test_with_legacy_local_search(self):
        ens = MemeticOptimiser(GreyWolfOptimiser(max_iterations=20), LocalSearchOptimiser(),
                               n_generations=2, n_refine=1, seed=0)
        r = ens.optimise(sphere, BOUNDS)
        assert r.best_value < 1e-2

    def test_refinement_improves_on_global(self):
        ens = MemeticOptimiser(DifferentialEvolutionOptimiser(),
                               CMAESOptimiser(sigma0=0.05),
                               n_generations=3, max_evaluations=4000, seed=1)
        r = ens.optimise(sphere, BOUNDS)
        global_best = min(rr.best_value for rr in r.run_results[::4])
        assert r.best_value <= global_best


# ---------------------------------------------------------------------------
# Cooperative co-evolution
# ---------------------------------------------------------------------------

class TestCooperativeCoevolution:
    def test_separable_high_dimensional(self):
        bounds = [(-5.12, 5.12)] * 30
        cc = CooperativeCoevolutionOptimiser(DifferentialEvolutionOptimiser(), n_groups=6,
                                             max_evaluations=20_000, seed=0)
        r = cc.optimise(rastrigin, bounds)
        assert r.n_evaluations <= 20_000
        start = rastrigin([2.5] * 30)
        assert r.best_value < 0.25 * start
        assert len(r.best_solution) == 30

    @pytest.mark.parametrize("grouping", ["random", "sequential"])
    def test_groupings_and_multiple_optimisers(self, grouping):
        cc = CooperativeCoevolutionOptimiser(
            [DifferentialEvolutionOptimiser(), TLBOOptimiser()],
            n_groups=3, grouping=grouping, n_cycles=4, max_evaluations=4000, seed=0,
        )
        r = cc.optimise(sphere, BOUNDS)
        assert r.best_value < 0.1
        assert all(x >= y for x, y in zip(r.history, r.history[1:]))

    def test_warm_start(self):
        cc = CooperativeCoevolutionOptimiser(GreyWolfOptimiser(), n_cycles=1,
                                             max_evaluations=100, seed=0)
        r = cc.optimise(sphere, BOUNDS, initial_solution=[0.0] * 6)
        assert r.best_value == 0.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_ensembles_nest():
    inner = MemeticOptimiser(DifferentialEvolutionOptimiser(), CMAESOptimiser(sigma0=0.05),
                             n_generations=2, max_evaluations=1500)
    outer = IslandModelOptimiser([inner, GreyWolfOptimiser()], n_epochs=2, seed=0)
    r = outer.optimise(sphere, BOUNDS)
    assert r.best_value < 1e-2


def test_portfolio_with_catalogue():
    ens = EnsembleOptimiser([
        DifferentialEvolutionOptimiser(max_evaluations=1000, seed=0),
        GreyWolfOptimiser(max_evaluations=1000, seed=0),
    ], strategy="chain")
    r = ens.optimise(sphere, BOUNDS)
    assert r.best_value <= r.run_results[0].best_value


def test_ensemble_registry():
    assert set(ENSEMBLES) == {"portfolio", "island", "adaptive", "memetic", "cooperative"}
