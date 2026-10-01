"""Tests for the benchmark harness, taxonomy and trajectory layer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from optim import (
    AdaptiveEnsembleOptimiser,
    CMAESOptimiser,
    CooperativeCoevolutionOptimiser,
    DifferentialEvolutionOptimiser,
    EnsembleOptimiser,
    GeneticOptimiser,
    GreyWolfOptimiser,
    IslandModelOptimiser,
    MemeticOptimiser,
    PSOOptimiser,
)
from optim.bench import BudgetExhausted, Recorder, load_config, run_benchmark, summarise
from optim.bench.analysis import TARGETS, _gap, average_ranks, friedman
from optim.bench.cli import main as cli_main
from optim.bench.config import Task, prepare
from optim.bench.runner import run_task
from optim.bench.site import build_site
from optim.benchmarks import sphere
from optim.problems import make_problem
from optim.taxonomy import CARDS, ENSEMBLE_CARDS, OPTIMISER_CARDS, taxonomy_tree

SMOKE = {
    "name": "t",
    "budget": {"per_dim": 60},
    "repeats": 2,
    "seed": 0,
    "record_trajectory": True,
    "problems": [
        {"family": "sphere", "dim": 3, "params": {"rotate": [False, True]}},
        {"family": "knapsack", "dim": 10},
        {"family": "tsp", "dim": 6},
        {"family": "pricing", "dim": 3},
    ],
    "optimisers": [
        {"name": "DE", "entry": "de"},
        {"name": "GA", "entry": "genetic"},
        {"name": "Tabu", "entry": "tabu"},
        {"name": "ACO", "entry": "aco"},
        {"name": "Island", "entry": "island",
         "params": {"optimisers": [{"entry": "de"}, {"entry": "gwo"}], "n_epochs": 2}},
    ],
}


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_every_optimiser_and_ensemble_has_a_card():
    from optim import ENSEMBLES, OPTIMISERS

    assert set(OPTIMISERS) - {"ensemble"} <= set(OPTIMISER_CARDS)
    assert {c.cls for c in ENSEMBLE_CARDS.values()} >= set(ENSEMBLES.values())
    for c in ENSEMBLE_CARDS.values():
        assert c.talbi_class in {"LRH", "LTH", "HRH", "HTH"}
    tree = taxonomy_tree()
    assert set(tree["problem_categories"]) == {"continuous", "combinatorial", "profit"}
    json.dumps({k: v.describe() for k, v in CARDS.items()})  # serialisable


def test_cards_report_capabilities():
    assert OPTIMISER_CARDS["de"].budget_control and OPTIMISER_CARDS["de"].warm_startable
    assert not OPTIMISER_CARDS["genetic"].budget_control
    assert OPTIMISER_CARDS["tabu"].supports("permutation")
    assert "centre_biased" in OPTIMISER_CARDS["gwo"].biases


# ---------------------------------------------------------------------------
# Trajectory layer & GA fix
# ---------------------------------------------------------------------------

def test_population_optimisers_record_trajectory():
    r = CMAESOptimiser(max_iterations=5, seed=0).optimise(sphere, [(-5, 5)] * 3, record_trajectory=True)
    assert [s.step_idx for s in r.trajectory] == list(range(6))
    assert r.trajectory[0].action == {"type": "initialise"}
    assert "sigma" in r.trajectory[-1].state
    costs = [s.cost for s in r.trajectory]
    assert costs == sorted(costs, reverse=True)
    assert all(s.delta_cost <= 0 for s in r.trajectory[1:])
    assert DifferentialEvolutionOptimiser(max_iterations=3, seed=0).optimise(sphere, [(-5, 5)] * 3).trajectory is None


def test_genetic_handles_negative_and_mixed_sign_objectives():
    for f in (lambda x: 10 - sphere(x), lambda x: -100 - sphere(x)):
        r = GeneticOptimiser(seed=0, max_generations=60).optimise(f, [(-5, 5)] * 3, maximise=True)
        assert r.best_value == pytest.approx(f([0, 0, 0]), abs=2.0)


@pytest.mark.parametrize("ens", [
    lambda: IslandModelOptimiser([DifferentialEvolutionOptimiser(), GreyWolfOptimiser()], seed=0),
    lambda: AdaptiveEnsembleOptimiser([DifferentialEvolutionOptimiser(), CMAESOptimiser()], seed=0),
    lambda: MemeticOptimiser(DifferentialEvolutionOptimiser(), CMAESOptimiser(sigma0=0.05), seed=0),
    lambda: CooperativeCoevolutionOptimiser(DifferentialEvolutionOptimiser(), n_groups=2, seed=0),
    lambda: EnsembleOptimiser([DifferentialEvolutionOptimiser(), CMAESOptimiser()]),
    lambda: EnsembleOptimiser([CMAESOptimiser()], strategy="random_restart", n_restarts=4),
], ids=["island", "adaptive", "memetic", "cooperative", "portfolio", "multistart"])
def test_ensembles_accept_a_per_call_budget(ens):
    r = ens().optimise(sphere, [(-5, 5)] * 4, max_evaluations=1500)
    assert r.n_evaluations <= 1500
    assert len(r.run_results) > 1  # the budget was shared, not eaten by one member


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

def test_recorder_enforces_budget_and_records_curve():
    p = make_problem("pricing", dim=3)  # maximisation
    rec = Recorder(p, budget=5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = p.random_solution(rng)
        assert rec(x) == pytest.approx(-p.objective(x))  # minimising view
    with pytest.raises(BudgetExhausted):
        rec(p.random_solution(rng))
    assert rec.n_evaluations == 5
    values = [v for _, v in rec.curve]
    assert values == sorted(values)  # profit improves monotonically
    assert rec.best_true == values[-1]


def test_recorder_uses_true_value_on_noisy_problems():
    p = make_problem("sphere", dim=3, noise=0.5)
    rec = Recorder(p, budget=50)
    rng = np.random.default_rng(1)
    for _ in range(50):
        rec(p.random_solution(rng))
    assert rec.best_true == pytest.approx(p.true_objective(rec.best_x))


def test_budget_is_exact_even_for_optimisers_without_budget_control():
    cfg = load_config({**SMOKE, "optimisers": [{"name": "PSO", "entry": "pso"}], "problems": [{"family": "sphere", "dim": 3}]})
    out = run_task(cfg.tasks()[0])
    assert out["row"]["n_evaluations"] == 180
    assert out["row"]["status"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_grid_expansion_and_seeds():
    cfg = load_config(SMOKE)
    assert len(cfg.problems) == 5 and len(cfg.optimisers) == 5
    tasks = cfg.tasks()
    assert len(tasks) == 5 * 5 * 2
    assert len({t.seed for t in tasks}) == len(tasks)
    assert load_config(SMOKE).tasks()[7].seed == tasks[7].seed  # deterministic
    assert tasks[0].budget == 180


def test_config_validation():
    with pytest.raises(KeyError):
        load_config({**SMOKE, "problems": [{"family": "nope"}]})
    with pytest.raises(KeyError):
        load_config({**SMOKE, "optimisers": [{"name": "x", "entry": "nope"}]})
    with pytest.raises(ValueError):
        load_config({**SMOKE, "optimisers": [{"name": "MO", "entry": "dbmosa"}]})
    with pytest.raises(ValueError):
        load_config({**SMOKE, "optimisers": [{"name": "A", "entry": "de"}, {"name": "A", "entry": "gwo"}]})


def test_prepare_selects_native_or_random_key_encoding():
    cfg = load_config(SMOKE)
    by = {(t.optimiser.name, t.problem.family): t for t in cfg.tasks()}
    _, solved, opt, how = prepare(by[("GA", "tsp")])
    assert how == "native" and opt.encoding == "permutation"
    _, solved, opt, how = prepare(by[("DE", "knapsack")])
    assert how == "random_key" and solved.encoding == "real"
    with pytest.raises(ValueError):
        prepare(by[("ACO", "sphere")])
    t = by[("DE", "knapsack")]
    t.random_keys = False
    with pytest.raises(ValueError):
        prepare(t)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    run_benchmark(SMOKE, out, progress=False)
    return out


def test_run_directory_layout(run_dir):
    for name in ("manifest.json", "results.csv", "curves.csv", "trajectories.jsonl",
                 "errors.jsonl", "skipped.jsonl", "summary.json", "scored.csv"):
        assert (run_dir / name).exists(), name
    results = pd.read_csv(run_dir / "results.csv")
    assert (results["status"] != "error").all()
    assert (results["n_evaluations"] <= results["budget"]).all()
    skipped = [json.loads(l) for l in (run_dir / "skipped.jsonl").read_text().splitlines()]
    assert {s["optimiser"] for s in skipped} == {"ACO"}  # ACO is permutation-only
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["n_runs"] == len(results)
    assert set(manifest["optimisers"]["Island"]) >= {"talbi_class", "spec"}


def test_summary_metrics(run_dir):
    s = json.loads((run_dir / "summary.json").read_text())
    scored = pd.read_csv(run_dir / "scored.csv")
    assert ((scored["gap"] >= 0) & (scored["score"] <= 1)).all()
    assert scored["auc"].between(0, 1).all()
    # each problem's best-known reference is attained by some run
    for prob, g in scored.groupby("problem"):
        if g["optimum_known"].iloc[0]:
            continue
        assert g["gap"].min() == pytest.approx(0.0, abs=1e-12)
    names = {r["optimiser"] for r in s["leaderboard"]}
    assert names == {"DE", "GA", "Tabu", "ACO", "Island"}
    assert set(s["ecdf"]["by_category"]) == {"all", "continuous", "combinatorial", "profit"}
    assert all(len(v) == len(s["ecdf"]["grid"]) for v in s["ecdf"]["by_category"]["all"].values())
    assert s["friedman"]["n_optimisers"] == 4  # ACO left out: partial coverage


def test_gap_normalisation():
    # minimisation: optimum 0, random median 10
    assert _gap([0, 5, 10, 20], 1.0, 0.0, 10.0).tolist() == [0, 0.5, 1, 2]
    # maximisation: profit optimum 100, random median 40  -> m* = -100, m_rand = -40
    assert _gap([100, 70, 40], -1.0, -100.0, -40.0).tolist() == [0, 0.5, 1]


def test_ranks_and_friedman():
    df = pd.DataFrame({
        "problem": ["a"] * 3 + ["b"] * 3 + ["c"] * 2,
        "budget": 1,
        "optimiser": ["x", "y", "z", "x", "y", "z", "x", "y"],
        "gap": [0.0, 0.5, 1.0, 0.1, 0.2, 0.9, 0.3, 0.1],
    })
    r = average_ranks(df).set_index("optimiser")
    assert r.loc["x", "norm_rank"] == pytest.approx((0 + 0 + 1) / 3)
    assert r.loc["z", "norm_rank"] == pytest.approx(1.0)
    f = friedman(df)  # z misses task c, so it is dropped -> too few optimisers
    assert f["optimisers"] == ["x", "y"] if "optimisers" in f else f["n_optimisers"] == 2
    assert f["p_value"] is None
    full = df[df["problem"] != "c"]
    f = friedman(full)
    assert f["n_tasks"] == 2 and f["n_optimisers"] == 3 and 0 <= f["p_value"] <= 1


def test_site_is_self_contained(run_dir, tmp_path):
    html = build_site(run_dir, tmp_path / "dash").read_text()
    assert "__DATA__" not in html and "__TITLE__" not in html
    assert "<script src=" not in html  # no external dependencies
    payload = html.split('id="bench-data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(payload)
    assert data["n_runs"] > 0 and data["task_rows"]


def test_cli(tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({**SMOKE, "repeats": 1, "record_trajectory": False}))
    out = tmp_path / "run"
    assert cli_main(["run", str(cfg), "--out", str(out), "--site", str(out / "index.html"), "--quiet"]) == 0
    assert (out / "index.html").exists()
    assert cli_main(["list", "problems"]) == 0
    assert "knapsack" in capsys.readouterr().out
    assert cli_main(["analyse", str(out)]) == 0


def test_yaml_configs_in_repo_load():
    from pathlib import Path

    for path in sorted(Path(__file__).parent.parent.joinpath("configs").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.tasks(), path.name
