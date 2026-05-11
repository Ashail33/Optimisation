"""
Tests for the optim generalised optimisation library.

Every optimiser is tested against simple, well-known benchmark functions so
that the expected optima are easy to reason about:

* **Sphere**      f(x) = sum(xi^2),  minimum at x=0, f(0)=0
* **Rosenbrock**  f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1), f=0
* **Binary sum**  f(x) = sum(xi),  maximised at x=[1,...,1]
* **TSP tiny**    3-city fixed-distance problem with known optimal tour
"""

from __future__ import annotations

import math
import sys
import os

# Ensure the repo root is on the path so `import optim` works regardless of
# how the test runner is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from optim import (
    OPTIMISERS,
    DBMOSAOptimiser,
    DifferentialEvolutionOptimiser,
    EnsembleOptimiser,
    EnsembleResult,
    GeneticOptimiser,
    LocalSearchOptimiser,
    OptimisationResult,
    PSOOptimiser,
    RandomSearchOptimiser,
    SimulatedAnnealingOptimiser,
    TabuSearchOptimiser,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SPHERE_BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]
SPHERE_BOUNDS_3D = [(-5.0, 5.0)] * 3


def sphere(x):
    return sum(v ** 2 for v in x)


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def binary_sum(x):
    return sum(x)


# ---------------------------------------------------------------------------
# OptimisationResult
# ---------------------------------------------------------------------------

class TestOptimisationResult:
    def test_fields(self):
        r = OptimisationResult(best_solution=[0.0, 0.0], best_value=0.0)
        assert r.best_solution == [0.0, 0.0]
        assert r.best_value == 0.0
        assert r.history == []
        assert r.n_evaluations == 0

    def test_history_populated(self):
        r = OptimisationResult(
            best_solution=[1.0], best_value=1.0, history=[5.0, 3.0, 1.0], n_evaluations=30
        )
        assert r.history == [5.0, 3.0, 1.0]
        assert r.n_evaluations == 30


# ---------------------------------------------------------------------------
# GeneticOptimiser
# ---------------------------------------------------------------------------

class TestGeneticOptimiser:
    def test_real_encoding_minimises_sphere(self):
        ga = GeneticOptimiser(
            population_size=30, max_no_improve=30, max_generations=300,
            encoding="real", seed=0
        )
        result = ga.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
        assert result.best_value < 5.0  # should have improved significantly
        assert result.n_evaluations > 0
        assert len(result.history) > 0

    def test_real_encoding_maximise(self):
        # Maximise -sphere → best_value should be close to 0 (from above)
        ga = GeneticOptimiser(
            population_size=20, max_no_improve=20, max_generations=200,
            encoding="real", seed=1
        )
        result = ga.optimise(lambda x: -sphere(x), bounds=SPHERE_BOUNDS_2D,
                             maximise=False)
        # Alternatively call with maximise=True on sphere
        result2 = ga.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        # When maximising sphere the best is to push to bounds → large value
        assert result2.best_value > 0

    def test_binary_encoding_maximise_sum(self):
        ga = GeneticOptimiser(
            population_size=20, max_no_improve=20, max_generations=200,
            encoding="binary", seed=2
        )
        result = ga.optimise(binary_sum, bounds=None, n_genes=10, maximise=True)
        assert isinstance(result, OptimisationResult)
        assert result.best_value >= 5  # should find at least half ones

    def test_permutation_encoding_tsp(self):
        # Tiny 3-city distance matrix; any valid permutation should be found
        dist = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

        def tsp_obj(tour):
            total = 0
            for i in range(1, len(tour)):
                total += dist[tour[i]][tour[i - 1]]
            total += dist[tour[0]][tour[-1]]  # return edge to complete the tour
            return total

        ga = GeneticOptimiser(
            population_size=20, max_no_improve=30, max_generations=200,
            encoding="permutation", seed=3
        )
        result = ga.optimise(tsp_obj, n_genes=3)
        assert isinstance(result, OptimisationResult)
        assert len(result.best_solution) == 3
        assert sorted(result.best_solution) == [0, 1, 2]

    def test_requires_bounds_for_real(self):
        ga = GeneticOptimiser(encoding="real")
        with pytest.raises(ValueError):
            ga.optimise(sphere)

    def test_requires_n_genes_for_permutation(self):
        ga = GeneticOptimiser(encoding="permutation")
        with pytest.raises(ValueError):
            ga.optimise(sphere)

    def test_invalid_encoding_raises(self):
        with pytest.raises(ValueError):
            GeneticOptimiser(encoding="unknown")

    def test_invalid_n_parents_raises(self):
        with pytest.raises(ValueError):
            GeneticOptimiser(n_parents=3)  # must be even

    def test_custom_crossover_and_mutation(self):
        """Custom operators are called (smoke test)."""

        def my_crossover(p1, p2):
            mid = len(p1) // 2
            return p1[:mid] + p2[mid:], p2[:mid] + p1[mid:]

        def my_mutation(sol):
            import copy
            s = copy.copy(sol)
            s[0] = 0.0
            return s

        ga = GeneticOptimiser(
            population_size=10, max_no_improve=10, max_generations=50,
            encoding="real", crossover_fn=my_crossover, mutation_fn=my_mutation,
            seed=4
        )
        result = ga.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)


# ---------------------------------------------------------------------------
# PSOOptimiser
# ---------------------------------------------------------------------------

class TestPSOOptimiser:
    def test_minimises_sphere(self):
        pso = PSOOptimiser(n_particles=20, max_no_improve=50, seed=0)
        result = pso.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
        assert result.best_value < 5.0
        assert result.n_evaluations > 0
        assert len(result.best_solution) == 2

    def test_requires_bounds(self):
        pso = PSOOptimiser()
        with pytest.raises(ValueError):
            pso.optimise(sphere)

    def test_maximise(self):
        # When maximising sphere over [-5,5]^2, optimum ≈ 50
        pso = PSOOptimiser(n_particles=15, max_no_improve=30, seed=1)
        result = pso.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        assert result.best_value > 20.0

    def test_warm_start(self):
        pso = PSOOptimiser(n_particles=10, max_no_improve=30, seed=2)
        seed_solutions = [[0.1, 0.1]]
        result = pso.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                              initial_solutions=seed_solutions)
        assert isinstance(result, OptimisationResult)

    def test_history_length(self):
        pso = PSOOptimiser(n_particles=10, max_no_improve=20,
                           max_iterations=50, seed=3)
        result = pso.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert len(result.history) > 0

    def test_w_decay(self):
        pso = PSOOptimiser(n_particles=10, w_decay=0.99, max_no_improve=30, seed=4)
        result = pso.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)


# ---------------------------------------------------------------------------
# LocalSearchOptimiser
# ---------------------------------------------------------------------------

class TestLocalSearchOptimiser:
    def test_real_minimises_sphere(self):
        ls = LocalSearchOptimiser(step_size=0.1, seed=0)
        result = ls.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[3.0, -2.0])
        assert isinstance(result, OptimisationResult)
        assert result.best_value < sphere([3.0, -2.0])

    def test_binary_maximises_sum(self):
        ls = LocalSearchOptimiser(encoding="binary", seed=0)
        result = ls.optimise(binary_sum, bounds=[None] * 5,
                             initial_solution=[0, 0, 0, 0, 0],
                             maximise=True)
        assert result.best_value >= 1  # must have flipped at least one bit

    def test_generates_random_initial_solution(self):
        ls = LocalSearchOptimiser(step_size=0.5, seed=5)
        result = ls.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)

    def test_constraints_respected(self):
        """Feasibility filter: neighbour must have first coord ≥ 0."""
        ls = LocalSearchOptimiser(step_size=0.5, max_no_improve=5, seed=6)

        def feasible(x):
            return x[0] >= 0.0

        result = ls.optimise(sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)],
                             initial_solution=[1.0, 1.0],
                             constraints_fn=feasible)
        assert result.best_solution[0] >= -0.001  # tolerance for float steps

    def test_custom_neighbourhood(self):
        def neighbourhood(sol):
            return [[sol[0] + 0.01, sol[1]], [sol[0], sol[1] + 0.01]]

        ls = LocalSearchOptimiser(neighbourhood_fn=neighbourhood, seed=7)
        result = ls.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[2.0, 2.0])
        assert isinstance(result, OptimisationResult)

    def test_requires_bounds_or_initial(self):
        ls = LocalSearchOptimiser(encoding="real")
        with pytest.raises(ValueError):
            ls.optimise(sphere)

    def test_invalid_encoding(self):
        with pytest.raises(ValueError):
            LocalSearchOptimiser(encoding="permutation")

    def test_max_iterations_respected(self):
        ls = LocalSearchOptimiser(step_size=0.1, max_iterations=5, seed=8)
        result = ls.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[3.0, 3.0])
        assert result.n_evaluations <= 100  # rough upper bound


# ---------------------------------------------------------------------------
# SimulatedAnnealingOptimiser
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingOptimiser:
    def test_minimises_sphere(self):
        sa = SimulatedAnnealingOptimiser(
            initial_temp=1000, cooling_rate=0.99,
            max_accepted=100, max_rejected=50, max_epochs=500,
            termination="epoch", seed=0
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
        assert result.best_value < sphere([5.0, 5.0])
        assert result.n_evaluations > 0

    def test_maximise(self):
        sa = SimulatedAnnealingOptimiser(
            initial_temp=1000, cooling_rate=0.99,
            max_accepted=100, max_rejected=50, max_epochs=200,
            termination="epoch", seed=1
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        assert result.best_value > 0

    def test_temperature_termination(self):
        sa = SimulatedAnnealingOptimiser(
            initial_temp=100, cooling_rate=0.9,
            termination="temperature", min_temp=1.0,
            schedule="Geometric", epoch_type="Static",
            static_epoch_length=20, max_epochs=10000,
            seed=2
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)

    def test_static_epoch(self):
        sa = SimulatedAnnealingOptimiser(
            initial_temp=100, cooling_rate=0.95,
            epoch_type="Static", static_epoch_length=10,
            max_epochs=50, termination="epoch", seed=3
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)

    def test_linear_schedule(self):
        sa = SimulatedAnnealingOptimiser(
            initial_temp=100, cooling_rate=0.5,
            schedule="Linear", max_epochs=200, termination="epoch", seed=4
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)

    def test_custom_neighbour_fn(self):
        def step(sol, bounds):
            import random
            n = sol[:]
            i = random.randrange(len(n))
            n[i] += random.uniform(-0.1, 0.1)
            return n

        sa = SimulatedAnnealingOptimiser(
            initial_temp=50, max_epochs=100, termination="epoch",
            neighbour_fn=step, seed=5
        )
        result = sa.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)

    def test_requires_bounds_without_custom_fn(self):
        sa = SimulatedAnnealingOptimiser()
        with pytest.raises(ValueError):
            sa.optimise(sphere)


# ---------------------------------------------------------------------------
# DBMOSAOptimiser
# ---------------------------------------------------------------------------

class TestDBMOSAOptimiser:
    def test_returns_pareto_archive(self):
        def bi_obj(x):
            """Bi-objective: minimise x^2 and (x-2)^2."""
            return [x[0] ** 2, (x[0] - 2) ** 2]

        dbmosa = DBMOSAOptimiser(
            initial_temp=1e4, cooling_rate=0.9,
            max_accepted=10, max_rejected=10, max_epochs=50,
            termination="epoch", max_archive_size=20, seed=0
        )
        result = dbmosa.optimise(bi_obj, bounds=[(-5.0, 5.0)])
        assert isinstance(result, OptimisationResult)
        # best_solution is the Pareto archive
        assert isinstance(result.best_solution, list)
        assert len(result.best_solution) >= 1
        assert result.n_evaluations > 0

    def test_diversity_histogram(self):
        def bi_obj(x):
            return [x[0] ** 2, (x[0] - 2) ** 2]

        dbmosa = DBMOSAOptimiser(
            initial_temp=1e3, cooling_rate=0.9,
            max_accepted=10, max_rejected=10, max_epochs=30,
            termination="epoch",
            diversity_method="Histogram", diversity_threshold=3,
            min_archive_for_diversity=3, max_archive_size=20, seed=1
        )
        result = dbmosa.optimise(bi_obj, bounds=[(-3.0, 3.0)])
        assert isinstance(result, OptimisationResult)

    def test_diversity_kernel(self):
        def bi_obj(x):
            return [x[0] ** 2, (x[0] - 2) ** 2]

        dbmosa = DBMOSAOptimiser(
            initial_temp=1e3, max_epochs=30, termination="epoch",
            diversity_method="Kernel", min_archive_for_diversity=2,
            max_archive_size=20, seed=2
        )
        result = dbmosa.optimise(bi_obj, bounds=[(-3.0, 3.0)])
        assert isinstance(result, OptimisationResult)

    def test_requires_bounds_without_custom_fn(self):
        dbmosa = DBMOSAOptimiser()
        with pytest.raises(ValueError):
            dbmosa.optimise(lambda x: [x[0] ** 2, (x[0] - 2) ** 2])


# ---------------------------------------------------------------------------
# EnsembleOptimiser
# ---------------------------------------------------------------------------

class TestEnsembleOptimiser:
    def _make_ga(self, seed=0):
        return GeneticOptimiser(
            population_size=15, max_no_improve=20, max_generations=100,
            encoding="real", seed=seed
        )

    def _make_pso(self, seed=0):
        return PSOOptimiser(n_particles=10, max_no_improve=30,
                            max_iterations=100, seed=seed)

    def _make_sa(self, seed=0):
        return SimulatedAnnealingOptimiser(
            initial_temp=100, cooling_rate=0.95,
            max_accepted=50, max_rejected=30, max_epochs=200,
            termination="epoch", seed=seed
        )

    # ----- Best strategy -----

    def test_best_strategy_returns_ensemble_result(self):
        ens = EnsembleOptimiser(
            [self._make_ga(), self._make_pso()], strategy="best"
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, EnsembleResult)
        assert len(result.run_results) == 2
        assert result.best_value <= min(r.best_value for r in result.run_results) + 1e-9

    def test_best_strategy_total_evals(self):
        ens = EnsembleOptimiser(
            [self._make_pso(0), self._make_pso(1)], strategy="best"
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert result.n_evaluations == sum(r.n_evaluations for r in result.run_results)

    def test_best_strategy_maximise(self):
        ens = EnsembleOptimiser(
            [self._make_ga(), self._make_pso()], strategy="best"
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        assert result.best_value > 0

    # ----- Chain strategy -----

    def test_chain_strategy_runs_in_order(self):
        ens = EnsembleOptimiser(
            [self._make_pso(), self._make_sa()], strategy="chain"
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, EnsembleResult)
        assert len(result.run_results) == 2

    def test_chain_strategy_total_evals(self):
        ens = EnsembleOptimiser(
            [self._make_pso(), self._make_sa()], strategy="chain"
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert result.n_evaluations == sum(r.n_evaluations for r in result.run_results)

    def test_chain_ga_then_ls(self):
        """GA → LocalSearch refinement chain."""
        ga = self._make_ga()
        ls = LocalSearchOptimiser(step_size=0.05, max_no_improve=10, seed=0)
        ens = EnsembleOptimiser([ga, ls], strategy="chain")
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, EnsembleResult)

    # ----- Random restart strategy -----

    def test_random_restart_runs_n_times(self):
        ens = EnsembleOptimiser(
            [self._make_sa()], strategy="random_restart", n_restarts=3
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, EnsembleResult)
        assert len(result.run_results) == 3

    def test_random_restart_best_is_min(self):
        ens = EnsembleOptimiser(
            [self._make_pso()], strategy="random_restart", n_restarts=4
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert result.best_value == min(r.best_value for r in result.run_results)

    def test_random_restart_maximise(self):
        ens = EnsembleOptimiser(
            [self._make_ga()], strategy="random_restart", n_restarts=3
        )
        result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        assert result.best_value == max(r.best_value for r in result.run_results)

    # ----- Error handling -----

    def test_empty_optimisers_raises(self):
        with pytest.raises(ValueError):
            EnsembleOptimiser([])

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            EnsembleOptimiser([self._make_ga()], strategy="vote")

    def test_per_optimiser_kwargs(self):
        ga = self._make_ga()
        pso = self._make_pso()
        ens = EnsembleOptimiser([ga, pso], strategy="best")
        # Pass dummy extra kwargs (should be ignored gracefully via **kwargs)
        result = ens.optimise(
            sphere, bounds=SPHERE_BOUNDS_2D,
            optimiser_kwargs=[{}, {}]
        )
        assert isinstance(result, EnsembleResult)

    # ----- Canonical strategy names + aliases -----

    def test_canonical_strategy_names(self):
        ga = self._make_ga()
        pso = self._make_pso()
        for name in ("portfolio", "pipeline", "multi_start"):
            opts = [ga] if name == "multi_start" else [ga, pso]
            ens = EnsembleOptimiser(opts, strategy=name, n_restarts=2)
            result = ens.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
            assert isinstance(result, EnsembleResult)
            assert ens.strategy == name

    def test_aliases_resolve_to_canonical(self):
        for alias, canonical in (
            ("best", "portfolio"),
            ("chain", "pipeline"),
            ("random_restart", "multi_start"),
        ):
            ens = EnsembleOptimiser([self._make_ga()], strategy=alias, n_restarts=2)
            assert ens.strategy == canonical


# ---------------------------------------------------------------------------
# TabuSearchOptimiser
# ---------------------------------------------------------------------------

class TestTabuSearchOptimiser:
    def test_real_minimises_sphere(self):
        ts = TabuSearchOptimiser(
            step_size=0.2, tabu_tenure=8,
            max_iterations=300, max_no_improve=80, seed=0
        )
        result = ts.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[3.0, -2.0])
        assert isinstance(result, OptimisationResult)
        assert result.best_value < sphere([3.0, -2.0])
        assert result.n_evaluations > 0

    def test_permutation_tsp(self):
        dist = [[0, 10, 20, 15], [10, 0, 12, 8], [20, 12, 0, 9], [15, 8, 9, 0]]

        def tsp_obj(tour):
            z = sum(dist[tour[i]][tour[i - 1]] for i in range(1, len(tour)))
            z += dist[tour[0]][tour[-1]]
            return z

        ts = TabuSearchOptimiser(
            encoding="permutation", tabu_tenure=4,
            max_iterations=100, seed=1
        )
        result = ts.optimise(tsp_obj, n_genes=4)
        assert sorted(result.best_solution) == [0, 1, 2, 3]

    def test_maximise(self):
        ts = TabuSearchOptimiser(
            step_size=0.5, max_iterations=100, max_no_improve=30, seed=2
        )
        result = ts.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[0.1, 0.1], maximise=True)
        assert result.best_value > 0.0

    def test_custom_neighbourhood(self):
        def nb(sol):
            return [
                ([sol[0] + 0.1, sol[1]], (sol[0] + 0.1, sol[1])),
                ([sol[0] - 0.1, sol[1]], (sol[0] - 0.1, sol[1])),
            ]

        ts = TabuSearchOptimiser(neighbourhood_fn=nb, max_iterations=50, seed=3)
        result = ts.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solution=[2.0, 0.0])
        assert isinstance(result, OptimisationResult)

    def test_invalid_encoding(self):
        with pytest.raises(ValueError):
            TabuSearchOptimiser(encoding="binary")

    def test_requires_bounds_for_real(self):
        ts = TabuSearchOptimiser()
        with pytest.raises(ValueError):
            ts.optimise(sphere)


# ---------------------------------------------------------------------------
# DifferentialEvolutionOptimiser
# ---------------------------------------------------------------------------

class TestDifferentialEvolutionOptimiser:
    def test_minimises_sphere(self):
        de = DifferentialEvolutionOptimiser(
            population_size=20, F=0.5, CR=0.9,
            max_generations=200, max_no_improve=30, seed=0
        )
        result = de.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
        assert result.best_value < 1.0
        assert result.n_evaluations > 0

    def test_minimises_rosenbrock(self):
        de = DifferentialEvolutionOptimiser(
            population_size=30, F=0.7, CR=0.9,
            max_generations=500, max_no_improve=80, seed=1
        )
        result = de.optimise(rosenbrock, bounds=[(-2.0, 2.0), (-2.0, 2.0)])
        assert result.best_value < 1.0

    def test_maximise(self):
        de = DifferentialEvolutionOptimiser(
            population_size=15, max_generations=100, max_no_improve=30, seed=2
        )
        result = de.optimise(sphere, bounds=SPHERE_BOUNDS_2D, maximise=True)
        assert result.best_value > 20.0

    def test_warm_start(self):
        de = DifferentialEvolutionOptimiser(
            population_size=10, max_generations=50, max_no_improve=20, seed=3
        )
        result = de.optimise(sphere, bounds=SPHERE_BOUNDS_2D,
                             initial_solutions=[[0.5, 0.5], [-0.5, -0.5]])
        assert isinstance(result, OptimisationResult)

    def test_requires_bounds(self):
        de = DifferentialEvolutionOptimiser()
        with pytest.raises(ValueError):
            de.optimise(sphere)

    def test_invalid_population_size(self):
        with pytest.raises(ValueError):
            DifferentialEvolutionOptimiser(population_size=3)

    def test_invalid_cr(self):
        with pytest.raises(ValueError):
            DifferentialEvolutionOptimiser(CR=1.5)


# ---------------------------------------------------------------------------
# RandomSearchOptimiser
# ---------------------------------------------------------------------------

class TestRandomSearchOptimiser:
    def test_real_baseline(self):
        rs = RandomSearchOptimiser(max_evaluations=200, seed=0)
        result = rs.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
        assert result.n_evaluations == 200
        assert result.best_value < sphere([5.0, 5.0])

    def test_binary(self):
        rs = RandomSearchOptimiser(encoding="binary", max_evaluations=100, seed=1)
        result = rs.optimise(binary_sum, n_genes=10, maximise=True)
        assert result.best_value >= 5
        assert all(b in (0, 1) for b in result.best_solution)

    def test_history_monotonic(self):
        rs = RandomSearchOptimiser(max_evaluations=50, seed=2)
        result = rs.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        # The history (best so far) must never increase as more samples come in.
        assert all(result.history[i] <= result.history[i - 1]
                   for i in range(1, len(result.history)))

    def test_custom_sampler(self):
        import random as _random
        sampler = lambda: [_random.uniform(-1, 1), _random.uniform(-1, 1)]
        rs = RandomSearchOptimiser(sample_fn=sampler, max_evaluations=30, seed=3)
        result = rs.optimise(sphere)
        assert isinstance(result, OptimisationResult)

    def test_requires_bounds_for_real(self):
        rs = RandomSearchOptimiser(max_evaluations=10)
        with pytest.raises(ValueError):
            rs.optimise(sphere)

    def test_invalid_encoding(self):
        with pytest.raises(ValueError):
            RandomSearchOptimiser(encoding="permutation")


# ---------------------------------------------------------------------------
# OPTIMISERS registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_all_optimisers(self):
        expected = {
            "genetic", "pso", "de", "local_search", "tabu",
            "sa", "dbmosa", "random_search", "ensemble",
        }
        assert set(OPTIMISERS.keys()) == expected

    def test_registry_instantiation(self):
        cls = OPTIMISERS["pso"]
        opt = cls(n_particles=10, max_no_improve=10, max_iterations=20, seed=0)
        result = opt.optimise(sphere, bounds=SPHERE_BOUNDS_2D)
        assert isinstance(result, OptimisationResult)
