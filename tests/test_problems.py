"""Tests for the problem suite (optim.problems)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from optim.problems import PROBLEMS, TAGS, as_random_key, decode, make_problem
from optim.problems.base import ProblemInstance

FAMILIES = sorted(PROBLEMS)


@pytest.mark.parametrize("family", FAMILIES)
def test_family_builds_with_valid_metadata(family):
    fam = PROBLEMS[family]
    p = fam.make()
    assert isinstance(p, ProblemInstance)
    assert p.family == family and p.category == fam.category
    assert p.encoding == fam.encoding and p.sense == fam.sense
    assert set(p.tags) <= TAGS
    assert len(p.bounds) == p.dim
    rng = np.random.default_rng(0)
    assert np.isfinite(p.objective(p.random_solution(rng)))
    assert np.isfinite(p.random_reference(n=20))


@pytest.mark.parametrize("family", FAMILIES)
def test_instances_are_reproducible_and_distinct(family):
    a, b = make_problem(family, instance=1), make_problem(family, instance=1)
    x = a.random_solution(np.random.default_rng(1))
    assert a.true_objective(x) == b.true_objective(x)
    assert a.name == b.name
    if family not in ("onemax", "trap", "schwefel"):  # structure-free families
        c = make_problem(family, instance=2)
        assert c.name != a.name
        xs = [a.random_solution(np.random.default_rng(s)) for s in range(5)]
        assert any(a.true_objective(v) != c.true_objective(v) for v in xs)


@pytest.mark.parametrize("family", [f for f in FAMILIES if make_problem(f).optimum is not None])
def test_known_optimum_is_attained_and_not_beaten(family):
    p = make_problem(family)
    assert p.optimum_solution is not None
    assert p.true_objective(p.optimum_solution) == pytest.approx(p.optimum, rel=1e-6, abs=1e-6)
    rng = np.random.default_rng(3)
    # random solutions never beat the optimum
    for _ in range(200):
        v = p.true_objective(p.random_solution(rng))
        assert not p.better(v, p.optimum + (1e-6 * abs(p.optimum) if p.maximise else -1e-6 * abs(p.optimum)) )
    # small perturbations of the optimum don't beat it (real encodings)
    if p.encoding == "real":
        lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
        x = np.array(p.optimum_solution)
        for _ in range(200):
            y = np.clip(x + rng.normal(0, 1e-3, len(x)) * (hi - lo), lo, hi)
            v = p.true_objective(y)
            tol = 1e-9 * max(1.0, abs(p.optimum))
            assert not p.better(v, p.optimum + (tol if p.maximise else -tol))


def test_continuous_shift_moves_optimum_off_centre():
    centred = make_problem("rastrigin", dim=6, shift=False)
    shifted = make_problem("rastrigin", dim=6, shift=True)
    assert centred.true_objective([0.0] * 6) == pytest.approx(0.0)
    assert shifted.true_objective([0.0] * 6) > 1.0
    assert shifted.true_objective(shifted.optimum_solution) == pytest.approx(0.0, abs=1e-9)
    assert "shifted" in shifted.tags and "shifted" not in centred.tags


def test_rotation_changes_tags_and_keeps_optimum():
    p = make_problem("ellipsoid", dim=5, rotate=True)
    assert {"rotated", "non_separable"} <= p.tags and "separable" not in p.tags
    assert p.true_objective(p.optimum_solution) == pytest.approx(0.0, abs=1e-9)


def test_noise_affects_objective_not_true_objective():
    p = make_problem("sphere", dim=4, noise=0.5)
    x = [1.0, 1.0, 1.0, 1.0]
    vals = {p.objective(x) for _ in range(5)}
    assert len(vals) > 1
    assert p.true_objective(x) == p.true_objective(x)
    p.reseed_noise(7); a = [p.objective(x) for _ in range(3)]
    p.reseed_noise(7); b = [p.objective(x) for _ in range(3)]
    assert a == b


def test_knapsack_dp_matches_brute_force():
    p = make_problem("knapsack", dim=12, instance=3, correlation="weak")
    best = max(p.objective(list(bits)) for bits in itertools.product((0, 1), repeat=12))
    assert best == p.optimum


def test_knapsack_infeasible_is_worse_than_empty():
    p = make_problem("knapsack", dim=10)
    assert p.objective([1] * 10) < p.objective([0] * 10) == 0


def test_tsp_heuristic_and_tour_length():
    p = make_problem("tsp", dim=6)
    assert p.extras["heuristic"].shape == (6, 6)
    assert p.objective([0, 1, 2, 3, 4, 5]) == pytest.approx(p.objective([1, 2, 3, 4, 5, 0]))


def test_newsvendor_is_noisy_with_analytic_truth():
    p = make_problem("newsvendor", dim=4)
    assert p.noisy and p.optimum is not None
    sims = np.mean([p.objective(p.optimum_solution) for _ in range(400)])
    assert sims == pytest.approx(p.optimum, rel=0.03)


@pytest.mark.parametrize("family", ["production_planning", "marketing_budget", "fixed_charge"])
def test_penalty_and_repair_agree_on_feasible_points(family):
    pen = make_problem(family, constraint_handling="penalty")
    rep = make_problem(family, constraint_handling="repair")
    x = np.zeros(pen.dim)
    assert pen.objective(x) == rep.objective(x)
    big = np.array([b[1] for b in pen.bounds])
    assert pen.objective(big) < rep.objective(big)  # penalty punishes, repair rescales


def test_random_key_decoding():
    assert decode([0.2, 0.7, 0.5], "binary") == [0, 1, 0]
    assert decode([0.9, 0.1, 0.5], "permutation") == [1, 2, 0]
    p = make_problem("onemax", dim=5)
    rk = as_random_key(p)
    assert rk.encoding == "real" and rk.bounds == [(0.0, 1.0)] * 5
    assert rk.objective([0.9] * 5) == 5 and rk.optimum == p.optimum


def test_unknown_family_and_bad_params():
    with pytest.raises(KeyError):
        make_problem("nope")
    with pytest.raises(ValueError):
        make_problem("knapsack", correlation="bogus")
    with pytest.raises(ValueError):
        make_problem("trap", dim=12, k=5)
