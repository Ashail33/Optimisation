# Architecture

`optim` is organised like [clustbench](https://github.com/Ashail33/clustbench).
It has a registry of **problems** (clustbench's datasets) and a registry of
**algorithms** described by machine-readable **cards**. A config-driven
**benchmark** writes tidy run directories. An **analysis** layer turns raw
facts into comparable metrics and statistics, and a static **dashboard**
shows them. Every iterative algorithm can also emit a **state-action
trajectory**, which later work can learn from.

```
                ┌───────────────────────── library (numpy only) ──────────────────────────┐
                │                                                                         │
 optim.problems │  PROBLEMS registry ──► ProblemInstance   (sense, encoding, optimum, tags)│
                │   continuous · combinatorial · profit      reproducible from its spec    │
                │                                                                         │
 optim.algorithms, ensembles                                                              │
                │  BaseOptimiser.optimise(f, bounds, maximise, initial_solutions,         │
                │                         max_evaluations, record_trajectory, ...)        │
                │      └─► OptimisationResult(best, history, population, trajectory)      │
                │                                                                         │
 optim.taxonomy │  OPTIMISER_CARDS / ENSEMBLE_CARDS  (family, encodings, mechanisms,      │
                │                                     biases, Talbi class, ...)           │
                └─────────────────────────────────────────────────────────────────────────┘
                                         │
                ┌────────────────── optim.bench (extra: pandas, pyyaml, scipy, psutil) ───┐
 config.py      │  YAML ─► BenchmarkConfig ─► Task grid (instance × optimiser × repeat)   │
                │           prepare(): card picks native encoding, or random keys         │
 recorder.py    │  Recorder: the only view of f — exact budget, incumbent, anytime curve  │
 runner.py      │  run_task() ─► run dir: results.csv · curves.csv · trajectories.jsonl   │
                │                         manifest.json · skipped.jsonl · errors.jsonl    │
 analysis.py    │  derive(): gap, score, solved, evals-to-target, AUC                     │
                │  summarise(): ranks, Friedman, ECDFs, median curves ─► summary.json     │
 site.py        │  build_site(): summary.json embedded in one self-contained HTML page    │
 cli.py         │  optbench run | analyse | site | list                                   │
                └─────────────────────────────────────────────────────────────────────────┘
```

## How clustbench maps onto optim

| clustbench | optim | Notes |
|---|---|---|
| `datasets.py` + `DataSpec` knobs | `optim.problems` + `make_problem(family, dim, instance, **params)` | Each family is a registered factory; its parameters are grid axes |
| `Algorithm` + `@register` | `BaseOptimiser` subclasses + `OPTIMISERS` / `ENSEMBLES` | Unchanged public API |
| `algorithm_cards.py` | `optim.taxonomy` cards | Encodings, mechanisms, biases, Talbi class |
| `Step(state, action, cost, delta_cost)` | `optim.base.Step`, `record_trajectory=True` | Emitted by every `PopulationOptimiser` |
| `benchmark.py` run loop + `measure_resources` | `bench.runner.run_task` + `bench.recorder.Recorder` | Recorder enforces the budget exactly |
| `Record` / `StepRecord` | rows of `results.csv` / `trajectories.jsonl` | One row per run / per step |
| ARI, NMI, silhouette … | normalised gap, score, solved, evals-to-target, AUC | Scale-free, so profits and costs are comparable |
| average ranks + Friedman | `bench.analysis.average_ranks`, `friedman` | Ranks normalised for partial coverage |
| `docs/index.html` dashboard | `optbench site` → one self-contained HTML file | No CDN, works offline and on Pages |
| YAML config grids | `configs/bench.*.yaml` | Same Cartesian-product semantics |

## Design decisions

### 1. The recorder is the single source of truth
Optimisers never see the problem directly. They see a `Recorder`, which:

- **enforces the evaluation budget exactly** for every optimiser, by raising
  `BudgetExhausted` on the first call over budget. Optimisers with no budget
  control (GA, PSO, SA, local search) therefore cannot overspend, and every
  comparison is at an equal budget;
- tracks the incumbent and records the **anytime curve** (evaluations → best
  value) at every improvement;
- on noisy problems, records the **noise-free** value of the incumbent, so
  scores measure real quality rather than lucky noise.

The best value and the curve come from the recorder, not from what the
optimiser reports. That makes very different algorithms, and ensembles that
nest them, measurable in the same way.

### 2. Problems are reproducible from their spec
`make_problem("tsp", dim=20, instance=3, layout="clustered")` builds the same
instance anywhere: in a worker process, on another machine, or in a future
external (non-Python) optimiser. Each family seeds its random structure (city
positions, rotation matrices, knapsack items) from `(family, dim, instance)`.
This is what makes parallel runs and an ask/tell protocol possible.

Problem **instances** are separate from optimiser **repeats**. A task is an
instance at a budget, and repeats are independent runs of a stochastic
optimiser on that same instance.

### 3. Every metric goes through one normalised gap
```
gap = (m(v) − m*) / (m_rand − m*)      m = value in minimisation orientation
```
`m*` is the optimum when it is known: closed-form for most profit problems,
dynamic programming for knapsack, enumeration for small NK and max-cut.
Otherwise `m*` is the **best-known** value, the best any run found on that
instance. `m_rand` is the median value of random solutions.

`gap = 0` is optimal and `gap = 1` is no better than random sampling. Because
of this one definition, costs and profits, a TSP in tour units and a
portfolio in utility units can sit in the same leaderboard. Targets
(`1e-1 … 1e-8`), the ECDF and the anytime AUC are all defined on the gap.

### 4. Shifted optima by default
Continuous problems default to `shift=True`. GWO, WOA and SCA are biased
towards the centre of the box. On centred benchmarks they look unbeatable; on
shifted ones, measured here, GWO goes from 0 to about 9–10 on Rastrigin.
A benchmark, and any router trained on it, should not reward that bias.
Rotation (`rotate=True`) separates rotation-invariant algorithms (CMA-ES)
from coordinate-wise ones (DE, ABC, Harmony Search).

### 5. Encodings are resolved through cards, with random keys as a bridge
`prepare(task)` reads the optimiser card:
- if the optimiser supports the problem's encoding, it is configured
  natively (for example `GeneticOptimiser(encoding="permutation")`);
- otherwise, if it is real-valued and `random_keys: true`, the problem is
  wrapped in a [random-key](../optim/problems/encodings.py) view (argsort for
  permutations, a threshold for bit strings), so all 13 continuous
  metaheuristics and continuous ensembles also run on TSP, knapsack and so on;
- otherwise the pair is **skipped**, recorded in `skipped.jsonl`, never as an
  error.

### 6. Ensembles accept a per-call budget
Every ensemble's `optimise()` takes `max_evaluations` and splits it across its
members, epochs or rounds. Without that, the first member would spend the
whole budget, and an island model would only be its first island. A test
checks that every ensemble type shares its budget.

### 7. Ranks are normalised for partial coverage
Specialists such as ACO (permutations only) run on fewer tasks. Per task,
the rank is rescaled to `[0, 1]` among the optimisers that ran (0 is best),
then averaged. Coverage is reported next to the rank, and full-coverage
optimisers are listed first. The Friedman test uses the optimisers with full
coverage on the tasks they all share.

### 8. Raw facts and derived metrics are kept apart
The runner writes only what happened (`results.csv`, `curves.csv`).
Everything comparative — best-known references, gaps, ranks, statistics —
is computed by `optbench analyse` into `scored.csv` and `summary.json`.
Changing a metric, or adding runs, means re-analysing rather than re-running.

### 9. The core library has no new dependencies
`optim`, `optim.problems` and `optim.taxonomy` need only NumPy (SciPy is
optional, for two LP/QP reference optima). The harness's extras live under
`pip install -e ".[bench]"`.

## Data model

**`results.csv`** has one row per run: `run_id, optimiser, entry, kind,
optimiser_family, talbi_class, problem, problem_family, category, encoding,
solved_as, sense, dim, instance, tags, problem_params, repeat, seed, budget,
optimum, random_reference, best_value, n_evaluations, status, error,
wall_time_s, cpu_time_s, rss_delta_mb, n_steps`.

**`curves.csv`**: `run_id, evaluations, value`, with one row per improvement
of the incumbent.

**`trajectories.jsonl`**: `run_id, step_idx, cost, delta_cost, accepted,
action, state`, with one row per iteration (`record_trajectory: true`).
`cost` is always in minimisation orientation.

**`scored.csv`** is `results.csv` plus `reference_value, gap, score, solved,
evals_to_1e-01 … evals_to_1e-08, auc`.

**`summary.json`**, which feeds the dashboard, holds the leaderboard (overall
and per category), the Friedman test, per-task rows for client-side
filtering, median convergence curves per instance, ECDFs per category,
problem metadata, optimiser cards and the config.

## Extension points

| To add… | Do this |
|---|---|
| A problem family | Write a factory `fn(dim, instance, **params) -> ProblemInstance` and decorate it with `@register_problem(...)`. Give it tags from `TAGS` and an optimum when one is computable. |
| A profit-function type | Add a family in `optim/problems/profit.py` with a `profit_*` tag (add the tag to `TAGS`). |
| A continuous metaheuristic | Subclass `PopulationOptimiser`, implement `_step` (optionally `_state_summary` / `state.action` for trajectories), and add an `OptimiserCard`. |
| A discrete optimiser | Subclass `BaseOptimiser`, accept `n_genes`, `initial_solution(s)` and `max_evaluations`, and add a card with its `encodings` and `encoding_param`. |
| An ensemble type | Subclass `BaseOptimiser`, call members through `ensembles._common.run_optimiser`, accept `max_evaluations`, and add an `EnsembleCard` with its Talbi class. |
| A metric | Compute it in `analysis.derive()` from `results` and the gap curves, then surface it in `summarise()`. |

## Roadmap: building from here

The architecture leaves room for the clustbench-style next steps:

1. **Landscape fingerprints.** Compute exploratory-landscape features
   (ruggedness, fitness–distance correlation, dispersion, a separability
   probe) per instance with a small sampling budget, stored next to `tags`.
   This is the analogue of clustbench's data fingerprint.
2. **Learned router.** Train on `scored.csv` (fingerprint → best optimiser)
   and evaluate on held-out instances: the offline counterpart of
   `AdaptiveEnsembleOptimiser`.
3. **Gap finder and algorithm mutator.** Search fingerprint space for regimes
   where no card does well, and propose new hybrids by recombining cards'
   mechanisms.
4. **Learned hyper-heuristics.** Use `trajectories.jsonl` (state, action,
   delta) to learn which operator to apply next. The RL-environment idea from
   clustbench applies directly.
5. **Ask/tell external protocol.** Since problems are reproducible from their
   spec, a non-Python optimiser can be driven over stdin/stdout, one batch of
   candidate solutions per message.
