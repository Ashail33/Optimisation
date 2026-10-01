# Optimisation

A generalised, extensible metaheuristic optimisation library in Python.

## Table of Contents

- [Included Optimisers](#included-optimisers)
- [Installation](#installation)
- [Structuring Your Problem](#structuring-your-problem)
  - [Step 1 — Write your objective function](#step-1--write-your-objective-function)
  - [Step 2 — Define your decision variables](#step-2--define-your-decision-variables)
  - [Step 3 — Choose an encoding](#step-3--choose-an-encoding)
  - [Step 4 — Choose an optimiser](#step-4--choose-an-optimiser)
- [Quick Start](#quick-start)
- [Optimiser Reference](#optimiser-reference)
  - [GeneticOptimiser](#geneticoptimiser)
  - [PSOOptimiser](#psooptimiser)
  - [LocalSearchOptimiser](#localsearchoptimiser)
  - [SimulatedAnnealingOptimiser](#simulatedannealingoptimiser)
  - [DBMOSAOptimiser](#dbmosaoptimiser)
  - [EnsembleOptimiser](#ensembleoptimiser)
- [Metaheuristic Catalogue](#metaheuristic-catalogue)
- [Ensemble Types](#ensemble-types)
- [Problem Suite](#problem-suite)
- [Taxonomy](#taxonomy)
- [Benchmarking & Dashboard](#benchmarking--dashboard)
- [OptimisationResult](#optimisationresult)
- [Custom Operators](#custom-operators)
- [Parameter Tuning Tips](#parameter-tuning-tips)
- [Integration](#integration)
- [Running Tests](#running-tests)

---

## Included Optimisers

| Class | Algorithm | Search space |
|---|---|---|
| `GeneticOptimiser` | Genetic Algorithm | real / binary / permutation |
| `PSOOptimiser` | Particle Swarm Optimisation | continuous |
| `LocalSearchOptimiser` | Best-improvement Local Search | real / binary |
| `SimulatedAnnealingOptimiser` | Simulated Annealing | continuous |
| `DBMOSAOptimiser` | Dominance-Based Multi-Objective SA | continuous (multi-obj) |
| `EnsembleOptimiser` | Combine optimisers (best / chain / restart) | any |

It also includes a [catalogue of 15 more metaheuristics](#metaheuristic-catalogue),
a [problem suite](#problem-suite) of continuous, combinatorial and
profit-function problems, a machine-readable [taxonomy](#taxonomy), and a
[benchmark harness with a dashboard](#benchmarking--dashboard) — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The catalogue:
Differential Evolution, CMA-ES, Grey Wolf, Whale, Firefly, Cuckoo Search,
Artificial Bee Colony, Bat, ACO_R, Harmony Search, TLBO, Jaya, Sine Cosine,
Ant Colony (permutations) and Tabu Search. There are also
[seven ensemble types](#ensemble-types): portfolio, relay chain, multi-start,
island model, adaptive bandit / hyper-heuristic, memetic, and cooperative
co-evolution.

---

## Installation

```bash
pip install -e ".[dev]"         # editable install with test dependencies
pip install -e ".[dev,bench]"   # plus the benchmark harness (pandas, pyyaml, scipy, psutil)
```

Requires Python ≥ 3.10 and NumPy ≥ 1.26. Tested on Python 3.10 – 3.14. No
other runtime dependencies.

After install, verify the package is importable:

```python
import optim
print(optim.__version__)      # '0.3.0'
print(optim.__all__)          # list of public classes
```

The legacy reference scripts in the repository root (`DBMOSA algorithm.py`,
`Genetic search algorithm.py`, `Local Search function`,
`Particle swarm optimisation algorithm`) are the original un-packaged
implementations kept for historical reference only. **Always use the `optim`
package** — it is the integration target.

---

## Structuring Your Problem

All optimisers in this library share the same four-step workflow.

### Step 1 — Write your objective function

Your objective function takes a single argument — the **solution** (a list of
values) — and returns a **scalar** (or a list of scalars for multi-objective
problems).  It must be self-contained: no side-effects, no shared mutable
state.

```python
# Single-objective: minimise the sphere function
def sphere(x):
    return sum(v**2 for v in x)

# Multi-objective: two competing objectives
def bi_objective(x):
    return [x[0]**2, (x[0] - 2)**2]

# Combinatorial: Travelling Salesman Problem cost
distances = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

def tsp_cost(tour):
    return sum(distances[tour[i]][tour[i-1]] for i in range(len(tour)))
```

To **maximise** instead of minimise, pass `maximise=True` to `optimise()` —
you do **not** need to negate your function.

```python
result = optimiser.optimise(my_fn, bounds=..., maximise=True)
```

### Step 2 — Define your decision variables

**Continuous / real-valued variables** are defined via `bounds`, a list of
`(min, max)` tuples — one per variable.

```python
# Two variables: x ∈ [-5, 5], y ∈ [0, 10]
bounds = [(-5.0, 5.0), (0.0, 10.0)]
```

**Binary variables** are represented as a list of 0s and 1s.  Provide
`n_genes` (the number of bits) or a `bounds` list whose length sets the gene
count.

**Permutation variables** are a list containing each integer from `0` to
`n_genes - 1` exactly once (useful for sequencing / routing problems).
Provide `n_genes` to the `optimise()` call.

### Step 3 — Choose an encoding

| Problem type | Encoding | Optimiser(s) |
|---|---|---|
| Real-valued continuous variables | `'real'` | GA, PSO, LS, SA, Tabu, and every continuous algorithm in the [catalogue](#metaheuristic-catalogue) |
| Bit strings / feature selection | `'binary'` | GA, LS, Tabu |
| Permutations / sequencing (e.g. TSP) | `'permutation'` | GA, Tabu, Ant Colony |
| Multiple competing objectives | multi-objective | DBMOSA |

### Step 4 — Choose an optimiser

| Situation | Recommended optimiser |
|---|---|
| Continuous landscape, fast convergence needed | `PSOOptimiser` |
| Mixed or unknown landscape, need flexibility | `GeneticOptimiser` |
| Good starting point known, need local refinement | `LocalSearchOptimiser` |
| Rugged landscape, risk of local optima | `SimulatedAnnealingOptimiser` |
| Multiple competing objectives | `DBMOSAOptimiser` |
| Unsure which algorithm to use | `EnsembleOptimiser` (strategy `'best'`) |
| Want global search then precise local refinement | `EnsembleOptimiser` (strategy `'chain'`) or `MemeticOptimiser` |
| Strong general-purpose continuous optimiser | `CMAESOptimiser`, `DifferentialEvolutionOptimiser` |
| Don't know which algorithm suits the problem | `AdaptiveEnsembleOptimiser` (learns online) |
| Multimodal landscape, want to keep diversity | `IslandModelOptimiser` |
| Many (hundreds of) variables | `CooperativeCoevolutionOptimiser` |

---

## Quick Start

Every optimiser shares the same call signature:

```python
result = optimiser.optimise(objective_fn, bounds=..., maximise=False)
print(result.best_solution, result.best_value)
```

```python
from optim import PSOOptimiser

pso = PSOOptimiser(n_particles=30, max_no_improve=200)
result = pso.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)
print(result)  # OptimisationResult(best_value=..., n_evaluations=...)
```

---

## Optimiser Reference

### GeneticOptimiser

Genetic Algorithm supporting real-valued, binary, and permutation encodings.
Uses fitness-proportionate (roulette-wheel) parent selection and steady-state
replacement.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `population_size` | `int` | `50` | Number of individuals per generation. |
| `elite_size` | `int` | `10` | Top solutions kept each generation (steady-state replacement). |
| `n_parents` | `int` | `6` | Parents selected per reproduction step; must be a positive even number. |
| `max_no_improve` | `int` | `100` | Stop after this many consecutive generations with no improvement. |
| `max_generations` | `int\|None` | `1000` | Hard upper limit on generations. `None` = no hard limit. |
| `encoding` | `str` | `'real'` | Solution representation: `'real'`, `'binary'`, or `'permutation'`. |
| `crossover_fn` | `callable\|None` | `None` | Custom crossover `(parent1, parent2) -> (child1, child2)`. Uses built-in default when `None`. |
| `mutation_fn` | `callable\|None` | `None` | Custom mutation `(solution) -> mutated`. Uses built-in default when `None`. |
| `seed` | `int\|None` | `None` | Random seed for reproducibility. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Function to optimise; accepts a solution list and returns a scalar. |
| `bounds` | `list of (min, max)` | `None` | Required for `encoding='real'`. Also sets `n_genes` for `'binary'` if not given. |
| `maximise` | `bool` | `False` | Set `True` to maximise instead of minimise. |
| `n_genes` | `int\|None` | `None` | Gene count for `'binary'` or `'permutation'` encodings. |

#### Built-in operators by encoding

| Encoding | Default crossover | Default mutation |
|---|---|---|
| `'real'` | Arithmetic (blend) crossover | Gaussian perturbation clamped to bounds |
| `'binary'` | Single-point crossover | Bit-flip (one random gene) |
| `'permutation'` | Order crossover (OX) | Swap two random positions |

#### Examples

```python
from optim import GeneticOptimiser

# --- Real-valued: minimise sphere function ---
ga = GeneticOptimiser(population_size=50, max_no_improve=100, encoding='real')
result = ga.optimise(lambda x: sum(v**2 for v in x), bounds=[(-5, 5)] * 3)
print(result.best_solution, result.best_value)

# --- Permutation: Travelling Salesman Problem ---
dist = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
def tsp(tour):
    return sum(dist[tour[i]][tour[i-1]] for i in range(1, len(tour)))

ga_tsp = GeneticOptimiser(encoding='permutation', max_no_improve=50)
result = ga_tsp.optimise(tsp, n_genes=3)

# --- Binary: maximise number of 1-bits ---
ga_bin = GeneticOptimiser(encoding='binary', max_no_improve=30)
result = ga_bin.optimise(sum, n_genes=10, maximise=True)
```

---

### PSOOptimiser

Classic *gbest* Particle Swarm Optimisation for continuous decision spaces.
Particles clamp to bounds on collision and reflect their velocity.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_particles` | `int` | `30` | Number of particles in the swarm. |
| `c1` | `float` | `1.5` | Cognitive (personal-best) acceleration coefficient. |
| `c2` | `float` | `1.5` | Social (global-best) acceleration coefficient. |
| `w` | `float` | `0.7` | Inertia weight — balances exploration vs. exploitation. |
| `w_decay` | `float` | `1.0` | Factor multiplied to `w` each iteration. `1.0` = no decay. |
| `max_no_improve` | `int` | `200` | Stop after this many consecutive swarm-level non-improving steps. |
| `max_iterations` | `int\|None` | `5000` | Hard upper limit on iterations. `None` = no hard limit. |
| `precision` | `int` | `4` | Decimal places to round randomly initialised positions to. |
| `seed` | `int\|None` | `None` | Random seed for reproducibility. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Objective function; accepts a list and returns a scalar. |
| `bounds` | `list of (min, max)` | — | Required. One tuple per decision variable. |
| `maximise` | `bool` | `False` | Set `True` to maximise. |
| `initial_solutions` | `list of lists\|None` | `None` | Warm-start seeds for some particles; remaining are random. |

#### Example

```python
from optim import PSOOptimiser

pso = PSOOptimiser(n_particles=30, c1=1.5, c2=1.5, w=0.7, max_no_improve=200)
result = pso.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)
print(result)  # OptimisationResult(best_value=..., n_evaluations=...)
```

---

### LocalSearchOptimiser

Best-improvement local search.  At each step every neighbour of the current
solution is evaluated and the algorithm moves to the best improving one.
Terminates when no improving neighbour exists (strict local optimum) or the
budget is exhausted.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `encoding` | `str` | `'real'` | `'real'` (step neighbourhood) or `'binary'` (bit-flip neighbourhood). |
| `step_size` | `float` | `0.01` | Step size for the real-valued neighbourhood (distance added/subtracted per dimension). |
| `max_no_improve` | `int\|None` | `None` | Stop after this many non-improving moves. `None` = stop only at a strict local optimum. |
| `max_iterations` | `int\|None` | `10000` | Hard upper limit on local search steps. |
| `neighbourhood_fn` | `callable\|None` | `None` | Custom neighbourhood generator `(solution) -> list[solution]`. Overrides `encoding` and `step_size` when set. |
| `seed` | `int\|None` | `None` | Random seed for reproducibility. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Objective function. |
| `bounds` | `list of (min, max)\|None` | `None` | Used to generate random initial solutions and clamp the step neighbourhood. |
| `maximise` | `bool` | `False` | Set `True` to maximise. |
| `initial_solution` | `list\|None` | `None` | Starting solution. Random if `None` (requires `bounds`). |
| `constraints_fn` | `callable\|None` | `None` | `constraints_fn(solution) -> bool`. Infeasible neighbours are skipped. |

#### Example

```python
from optim import LocalSearchOptimiser

ls = LocalSearchOptimiser(step_size=0.05)
result = ls.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5, 5), (-5, 5)],
    initial_solution=[3.0, -2.0],
    constraints_fn=lambda x: x[0] >= 0,   # optional feasibility filter
)
print(result.best_solution, result.best_value)
```

---

### SimulatedAnnealingOptimiser

Single-objective Simulated Annealing for continuous decision spaces.  Uses a
random step move by default (perturb one dimension by a fraction of its
range).  Accepts/rejects candidates probabilistically via the Metropolis
criterion.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_temp` | `float` | `1e6` | Starting temperature. Higher values allow more uphill moves early on. |
| `cooling_rate` | `float` | `0.9999` | Cooling factor α. Interpretation depends on `schedule` (see below). |
| `reheating_rate` | `float` | `0.5` | Reheating factor for Dynamic epoch when `max_rejected` is hit. |
| `max_accepted` | `int` | `200` | Max accepted moves per epoch (Dynamic epoch only). |
| `max_rejected` | `int` | `150` | Max rejected moves per epoch (Dynamic epoch only). Triggers reheating. |
| `static_epoch_length` | `int` | `100` | Moves per epoch when `epoch_type='Static'`. |
| `max_epochs` | `int` | `10000` | Maximum number of epochs. |
| `min_temp` | `float` | `1e-6` | Temperature at which the algorithm stops (temperature termination). |
| `schedule` | `str` | `'Geometric'` | Cooling schedule: `'Linear'`, `'Geometric'`, `'Logarithmic'`, or `'Very slow cooling'`. |
| `epoch_type` | `str` | `'Dynamic'` | Epoch length strategy: `'Dynamic'` (accept/reject driven) or `'Static'` (fixed length). |
| `termination` | `str` | `'epoch'` | Stop on `'epoch'` count or `'temperature'` threshold. |
| `neighbour_fn` | `callable\|None` | `None` | Custom move generator `(solution, bounds) -> new_solution`. |
| `move_scale` | `float` | `0.05` | Scale of default random step (fraction of dimension range). |
| `seed` | `int\|None` | `None` | Random seed for reproducibility. |

#### Cooling schedule formulae

| Schedule | Cool update | Notes |
|---|---|---|
| `'Geometric'` | `T ← T × α` | Most common. `α` near 1 (e.g. 0.9999) gives slow cooling. |
| `'Linear'` | `T ← T − α` | `α` is the fixed decrement. Use small values relative to `initial_temp`. |
| `'Logarithmic'` | `T ← T / ln(step)` | Theoretically convergent; can be slow in practice. |
| `'Very slow cooling'` | `T ← T / (1 + α)` | Slowest schedule; use tiny `α` values. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Scalar objective function. |
| `bounds` | `list of (min, max)` | — | Required unless a custom `neighbour_fn` is provided. |
| `maximise` | `bool` | `False` | Set `True` to maximise. |
| `initial_solution` | `list\|None` | `None` | Starting solution. Randomly generated within `bounds` if `None`. |

#### Example

```python
from optim import SimulatedAnnealingOptimiser

sa = SimulatedAnnealingOptimiser(
    initial_temp=1e6,
    cooling_rate=0.9999,
    schedule='Geometric',          # 'Linear', 'Geometric', 'Logarithmic', 'Very slow cooling'
    epoch_type='Dynamic',          # 'Dynamic' or 'Static'
    termination='epoch',           # 'epoch' or 'temperature'
    max_epochs=5000,
)
result = sa.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
print(result.best_solution, result.best_value)
```

---

### DBMOSAOptimiser

Dominance-Based Multi-Objective Simulated Annealing.  Maintains a Pareto
archive and uses a dominance-count ratio (ΔE) to drive acceptance.  Supports
optional diversity-preservation methods to spread solutions across the Pareto
front.

Your **objective function must return a list** of scalar values (one per
objective).  All objectives are minimised by default; pass `maximise=True` to
maximise all of them.

The returned `result.best_solution` is the **Pareto archive** — a list of
non-dominated solutions.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_temp` | `float` | `1e9` | Starting temperature. |
| `cooling_rate` | `float` | `0.9999` | Cooling factor α. |
| `reheating_rate` | `float` | `0.5` | Reheating factor (Dynamic epoch). |
| `max_accepted` | `int` | `200` | Max accepted moves per epoch (Dynamic). |
| `max_rejected` | `int` | `150` | Max rejected moves per epoch (Dynamic). |
| `static_epoch_length` | `int` | `100` | Moves per epoch (Static). |
| `max_epochs` | `int` | `20000` | Maximum epochs. |
| `min_temp` | `float` | `1e-4` | Temperature threshold for `termination='temperature'`. |
| `schedule` | `str` | `'Geometric'` | Cooling schedule (same options as SA). |
| `epoch_type` | `str` | `'Dynamic'` | `'Dynamic'` or `'Static'`. |
| `termination` | `str` | `'temperature'` | `'epoch'` or `'temperature'`. |
| `diversity_method` | `str\|None` | `None` | Diversity-preservation strategy: `'Kernel'`, `'NN'`, `'Histogram'`, or `None`. |
| `diversity_threshold` | `float` | `5.0` | Density threshold for the `'Histogram'` method. |
| `min_archive_for_diversity` | `int` | `5` | Diversity criterion activates once the archive reaches this size. |
| `max_archive_size` | `int\|None` | `100` | Maximum archive size. Most-crowded solution is pruned when exceeded. `None` = unlimited. |
| `neighbour_fn` | `callable\|None` | `None` | Custom move generator `(solution, bounds) -> new_solution`. |
| `move_scale` | `float` | `0.1` | Scale of default random step. |
| `seed` | `int\|None` | `None` | Random seed. |

#### Diversity methods

| Method | Behaviour |
|---|---|
| `None` | No diversity preservation (archive pruned by crowding distance only). |
| `'Kernel'` | Kernel density estimate; divides ΔE by local density to reward sparse regions. |
| `'Histogram'` | Rejects candidates landing in over-populated histogram cells (threshold set by `diversity_threshold`). |
| `'NN'` | Nearest-neighbour crowding; penalises the most-crowded candidate. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Multi-objective function returning a list of scalars. |
| `bounds` | `list of (min, max)` | — | Required unless a custom `neighbour_fn` is given. |
| `maximise` | `bool` | `False` | Set `True` to maximise all objectives. |
| `initial_solution` | `list\|None` | `None` | Starting solution. Random if `None`. |

#### Example

```python
from optim import DBMOSAOptimiser

def bi_objective(x):
    return [x[0]**2, (x[0] - 2)**2]   # two competing objectives

dbmosa = DBMOSAOptimiser(
    initial_temp=1e7,
    max_epochs=1000,
    termination='epoch',
    diversity_method='Histogram',       # 'Kernel', 'NN', 'Histogram', or None
    max_archive_size=50,
)
result = dbmosa.optimise(bi_objective, bounds=[(-5.0, 5.0)])
pareto_front = result.best_solution    # list of non-dominated solutions
print(f"Pareto front size: {len(pareto_front)}")
```

---

### EnsembleOptimiser

Combines multiple optimisers using one of three strategies.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimisers` | `list[BaseOptimiser]` | — | Constituent optimisers. For `'random_restart'`, provide a list with one entry. |
| `strategy` | `str` | `'best'` | `'best'`, `'chain'`, or `'random_restart'`. |
| `n_restarts` | `int` | `5` | Number of restarts for `strategy='random_restart'`. Ignored for other strategies. |

#### Strategies

| Strategy | Behaviour |
|---|---|
| `'best'` | Run all optimisers independently; return the single best result found. |
| `'chain'` | Run optimisers sequentially, warm-starting each with the previous optimiser's best solution. |
| `'random_restart'` | Re-run the same optimiser `n_restarts` times from random starts; return the best result. |

#### `optimise()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective_fn` | `callable` | — | Objective function passed to every constituent optimiser. |
| `bounds` | `list of (min, max)\|None` | `None` | Passed to each constituent optimiser. |
| `maximise` | `bool` | `False` | Passed to each constituent optimiser. |
| `optimiser_kwargs` | `list[dict]\|None` | `None` | Per-optimiser extra keyword arguments. The i-th dict is passed to the i-th optimiser. |

The return value is an `EnsembleResult` (subclass of `OptimisationResult`)
that also exposes a `run_results` list with the individual result from each
constituent run.

#### Examples

```python
from optim import GeneticOptimiser, PSOOptimiser, SimulatedAnnealingOptimiser, EnsembleOptimiser

# --- Strategy: 'best' — run GA and PSO, take the best ---
ga  = GeneticOptimiser(population_size=30, max_no_improve=50)
pso = PSOOptimiser(n_particles=20, max_no_improve=100)

ens = EnsembleOptimiser([ga, pso], strategy='best')
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
print(result.run_results)  # per-optimiser results

# --- Strategy: 'chain' — PSO for global search, SA for local refinement ---
pso = PSOOptimiser(n_particles=20, max_no_improve=50)
sa  = SimulatedAnnealingOptimiser(initial_temp=100, max_epochs=500)

ens = EnsembleOptimiser([pso, sa], strategy='chain')
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])

# --- Strategy: 'random_restart' — run SA five times, keep the best ---
sa  = SimulatedAnnealingOptimiser(initial_temp=1000, max_epochs=3000)
ens = EnsembleOptimiser([sa], strategy='random_restart', n_restarts=5)
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
```

---

## Metaheuristic Catalogue

`optim.algorithms` holds a catalogue of well-known metaheuristics, all
importable from the top-level `optim` package.

| Class | Algorithm | Family | Search space | Algorithm-specific parameters (defaults) |
|---|---|---|---|---|
| `DifferentialEvolutionOptimiser` | Differential Evolution | evolutionary | continuous | `strategy='rand/1'` (`'best/1'`, `'current-to-best/1'`, `'rand/2'`), `F=(0.5, 1.0)` (tuple = dither), `CR=0.9` |
| `CMAESOptimiser` | CMA-ES | evolution strategy | continuous | `sigma0=0.3` (fraction of each range) |
| `GreyWolfOptimiser` | Grey Wolf Optimizer | swarm | continuous | — |
| `WhaleOptimiser` | Whale Optimization Algorithm | swarm | continuous | `spiral_b=1.0` |
| `FireflyOptimiser` | Firefly Algorithm | swarm | continuous | `beta0=1`, `gamma=1`, `alpha=0.2`, `alpha_decay=0.97` |
| `CuckooSearchOptimiser` | Cuckoo Search (Lévy flights) | swarm | continuous | `pa=0.25`, `step_scale=0.01`, `levy_beta=1.5` |
| `ArtificialBeeColonyOptimiser` | Artificial Bee Colony | swarm | continuous | `limit=None` (→ `pop × dim / 2`) |
| `BatOptimiser` | Bat Algorithm | swarm | continuous | `f_min=0`, `f_max=2`, `loudness=1`, `pulse_rate=0.5`, `alpha=gamma=0.9` |
| `ACOROptimiser` | Ant colony for continuous domains (ACO_R) | swarm | continuous | `n_ants=None`, `q=0.2`, `xi=0.85` |
| `HarmonySearchOptimiser` | Harmony Search | music-inspired | continuous | `hmcr=0.9`, `par=0.3`, `bandwidth=(0.1, 0.001)` |
| `TLBOOptimiser` | Teaching–Learning-Based Optimization | human-inspired | continuous | — (parameter-free) |
| `JayaOptimiser` | Jaya | human-inspired | continuous | — (parameter-free) |
| `SineCosineOptimiser` | Sine Cosine Algorithm | math-inspired | continuous | `a=2.0` |
| `AntColonyOptimiser` | Rank-based Ant System | swarm | permutation | `n_ants=20`, `alpha=1`, `beta=2`, `evaporation=0.1`; `optimise(..., heuristic=eta)` |
| `TabuSearchOptimiser` | Tabu Search | trajectory | real / binary / permutation | `encoding`, `n_neighbours=30`, `tabu_tenure=10`, `step_size=0.1` |

Each module's docstring cites the original paper.

### Shared options for the continuous algorithms

All continuous algorithms subclass
[`PopulationOptimiser`](optim/population.py), so they share these options:

| Constructor parameter | Default | Description |
|---|---|---|
| `population_size` | algorithm-specific | Individuals / particles / nests / food sources / archive size. |
| `max_iterations` | `1000` | Iteration (generation) limit. |
| `max_evaluations` | `None` | Objective-evaluation budget. Set this to compare algorithms fairly. At least one of the two limits is required. |
| `max_no_improve` | `None` | Stop after this many iterations without improvement (by more than `tol`). |
| `seed` | `None` | Seed for the algorithm's private `numpy.random.Generator`. It never touches global random state. |

| `optimise()` parameter | Description |
|---|---|
| `initial_solutions` / `initial_solution` | Warm-start the population. |
| `max_evaluations`, `max_iterations` | Override the budget for one call. |
| `callback(iteration, best_solution, best_value)` | Called every iteration. Return `True` to stop. |

Results include the final `population` (best first). The ensembles use it
to pass a whole population between algorithms.

Algorithms whose coefficients decay over the run (GWO, WOA, SCA, Harmony
Search bandwidth) schedule them on the fraction of the budget used. Set the
budget you actually intend to spend.

```python
from optim import DifferentialEvolutionOptimiser, CMAESOptimiser, GreyWolfOptimiser
from optim.benchmarks import rastrigin

bounds = [(-5.12, 5.12)] * 10
for opt in (DifferentialEvolutionOptimiser(strategy="current-to-best/1"),
            CMAESOptimiser(sigma0=0.3),
            GreyWolfOptimiser(population_size=40)):
    r = opt.optimise(rastrigin, bounds, max_evaluations=20_000, max_iterations=None)
    print(type(opt).__name__, r.best_value)
```

### Discrete examples

```python
import numpy as np
from optim import AntColonyOptimiser, TabuSearchOptimiser

pts = np.random.default_rng(0).random((20, 2))
dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
tour_length = lambda t: sum(dist[t[i - 1], t[i]] for i in range(len(t)))

# ACO with a 1/distance heuristic
with np.errstate(divide="ignore"):
    eta = np.where(dist > 0, 1 / dist, 0)
aco = AntColonyOptimiser(n_ants=30, max_iterations=200, seed=0)
print(aco.optimise(tour_length, n_genes=20, heuristic=eta).best_value)

# Tabu search over swaps
ts = TabuSearchOptimiser(encoding="permutation", tabu_tenure=15, max_iterations=2000)
print(ts.optimise(tour_length, n_genes=20).best_value)

# Tabu search on bit strings (feature selection, knapsack, ...)
ts = TabuSearchOptimiser(encoding="binary")
print(ts.optimise(lambda b: sum(b), n_genes=30, maximise=True).best_value)
```

---

## Ensemble Types

`optim.ensembles` provides one class per standard way of combining
optimisers. Every ensemble is itself a `BaseOptimiser`, so ensembles can be
nested (for example, memetic algorithms on the islands of an island model).

| Ensemble type | Class | How the optimisers interact |
|---|---|---|
| Portfolio (best-of) | `EnsembleOptimiser(strategy='best')` | Independent runs; keep the best. |
| High-level relay hybrid | `EnsembleOptimiser(strategy='chain')` | Run in sequence, handing over the best solution once. |
| Multi-start | `EnsembleOptimiser(strategy='random_restart')` | Restart one algorithm several times. |
| Island model | `IslandModelOptimiser` | Each algorithm evolves its own island. Elites migrate between islands every epoch (`'ring'`, `'fully_connected'`, `'star'`, `'random'`). |
| Adaptive / hyper-heuristic | `AdaptiveEnsembleOptimiser` | A multi-armed bandit (`'ucb'`, `'epsilon_greedy'`, `'probability_matching'`, `'round_robin'`) chooses which algorithm runs each round, warm-started from a shared pool. It rewards algorithms that improve the best value. |
| Memetic (low-level hybrid) | `MemeticOptimiser` | Global search and local refinement of the top `n_refine` elites alternate. Refined solutions go back into the population. |
| Cooperative co-evolution | `CooperativeCoevolutionOptimiser` | Splits the variables into groups and optimises each group with the others held at the best-so-far context vector. Suits high-dimensional, (nearly) separable problems. |

### Budgets and warm starts

Ensembles count every objective call and take a total `max_evaluations`.
That budget is split across epochs, rounds or groups (override with
`epoch_evaluations`, `round_evaluations`, `group_evaluations`,
`global_evaluations` / `local_evaluations`).

For each call, the ensemble passes warm-start solutions and a budget only if
the constituent accepts them (`initial_solutions` → `initial_solution`;
`max_evaluations`). That means **any** optimiser works in any ensemble.
Optimisers without budget control (GA, PSO, LS, SA) run to their own
stopping criteria, so give them short limits when you use them inside
epoch-based ensembles.

When an ensemble has a `seed`, each constituent run gets a different seed
derived from it. Repeated epochs then explore differently, and the whole run
is still reproducible.

### Examples

```python
from optim import (
    AdaptiveEnsembleOptimiser, CMAESOptimiser, CooperativeCoevolutionOptimiser,
    DifferentialEvolutionOptimiser, GreyWolfOptimiser, IslandModelOptimiser,
    MemeticOptimiser, TabuSearchOptimiser,
)
from optim.benchmarks import rastrigin

bounds = [(-5.12, 5.12)] * 20
algos = [DifferentialEvolutionOptimiser(), CMAESOptimiser(), GreyWolfOptimiser()]

# Island model: three heterogeneous islands, best 2 migrate around a ring
island = IslandModelOptimiser(algos, n_epochs=10, topology="ring",
                              migration_size=2, max_evaluations=50_000, seed=0)
r = island.optimise(rastrigin, bounds)
print(r.best_value, r.info["island_best_values"])

# Adaptive selection: UCB bandit learns which algorithm pays off
adaptive = AdaptiveEnsembleOptimiser(algos, n_rounds=25, selection="ucb",
                                     max_evaluations=50_000, seed=0)
r = adaptive.optimise(rastrigin, bounds)
print(r.best_value, r.info["selection_counts"])

# Memetic: DE explores, Tabu Search refines the 3 best each generation
memetic = MemeticOptimiser(DifferentialEvolutionOptimiser(),
                           TabuSearchOptimiser(step_size=0.02),
                           n_generations=10, n_refine=3, max_evaluations=50_000)
print(memetic.optimise(rastrigin, bounds).best_value)

# Cooperative co-evolution: 5 random variable groups, re-shuffled each cycle
coop = CooperativeCoevolutionOptimiser(DifferentialEvolutionOptimiser(), n_groups=5,
                                       grouping="random", max_evaluations=50_000)
print(coop.optimise(rastrigin, bounds).best_value)
```

---

## Problem Suite

`optim.problems` is a registry of benchmark problem *families* — the
optimisation equivalent of clustbench's datasets. Every family builds
reproducible instances from `(family, dim, instance, **params)`. Each
instance knows its `sense` (`'min'` cost or `'max'` profit), its `encoding`,
its optimum (when computable), its `tags`, and the median value of random
solutions.

```python
from optim import CMAESOptimiser
from optim.problems import make_problem, PROBLEMS

p = make_problem("pricing", dim=10, instance=2)        # a profit function
r = CMAESOptimiser(max_evaluations=10_000, max_iterations=None).optimise(
    p.objective, p.bounds, maximise=p.maximise)
print(r.best_value, "vs optimum", p.optimum, sorted(p.tags))
```

| Category | Families | Notes |
|---|---|---|
| **continuous** | `sphere`, `ellipsoid`, `bent_cigar`, `step`, `zakharov`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `levy`, `styblinski_tang`, `schwefel` | Parameters `shift` (default **on**, off-centre optimum), `rotate` (non-separable), `noise` (multiplicative) |
| **combinatorial** | `tsp` (uniform / clustered), `knapsack` (uncorrelated / weak / strong / subset-sum), `onemax`, `trap` (deceptive), `nk` (tunable ruggedness), `maxcut` | Knapsack optimum by DP; NK and max-cut exact for `dim <= 16` |
| **profit** | see below | Constraints handled by `constraint_handling='penalty'` or `'repair'` |

**Profit-function types**, one family each:

| Family | Profit type | Optimum |
|---|---|---|
| `production_planning` | linear margins under shared resource limits | exact (LP) |
| `marketing_budget` | concave, diminishing returns (`a·log(1+s·x) − x`) under a budget | exact (KKT water-filling) |
| `pricing` | quadratic: substitute products with cross-elastic linear demand | exact (linear system) |
| `portfolio` | risk-adjusted mean − λ·variance, optional cardinality limit | exact (QP) without cardinality |
| `newsvendor` | stochastic: simulated sales under uncertain demand (noisy) | exact (critical fractile) |
| `fixed_charge` | discontinuous: set-up costs plus a capacity limit | exact by enumeration (`dim <= 12`) |

Discrete problems can be solved by any real-valued optimiser through
**random keys**: `as_random_key(problem)` decodes `[0, 1]^n` by argsort
(permutations) or by a 0.5 threshold (bits).

`optbench list problems` prints every family with its parameters.

---

## Taxonomy

`optim.taxonomy` gives every optimiser and ensemble a machine-readable
**card**, the analogue of clustbench's `algorithm_cards.py`. The benchmark
uses cards to decide encodings, and the dashboard uses them for grouping.
Later, a router or algorithm mutator can reason over the same features.

- **Optimisers**: `family` (evolutionary, swarm, physics, human, music,
  mathematical, trajectory) × `search` (population, trajectory,
  constructive), plus `encodings`, `mechanisms` and `biases`
  (`rotation_invariant`, `centre_biased`, `separability_exploiting`,
  `parameter_free`, `self_adaptive`, …).
- **Ensembles**: Talbi's hybrid taxonomy, *level* (low = embedded, high =
  self-contained) × *mode* (relay = sequential, teamwork = cooperative):

| Talbi class | Ensembles |
|---|---|
| HTH (high-level teamwork) | `portfolio`, `multistart`, `island` |
| HRH (high-level relay) | `relay`, `adaptive` (bandit hyper-heuristic) |
| LRH (low-level relay) | `memetic` |
| LTH (low-level teamwork) | `cooperative` (co-evolution) |

---

## Benchmarking & Dashboard

`optim.bench` is a benchmark harness modelled on clustbench. It needs the
extra dependencies: `pip install -e ".[bench]"`.

```bash
optbench list optimisers                     # what's registered
optbench run configs/bench.demo.yaml --out runs/demo --site docs/dashboard/index.html
optbench analyse runs/demo                   # recompute metrics without re-running
optbench site runs/demo --out docs/dashboard/index.html
```

A config is a grid, as in clustbench. Every list is an axis of the
Cartesian product, and ensemble members are nested specs:

```yaml
name: my-study
budget: {per_dim: 1000}      # or {evaluations: 20000}
repeats: 5
random_keys: true            # real-valued optimisers may solve discrete problems
problems:
  - {family: rastrigin, dim: [10, 30], params: {rotate: [false, true]}}
  - {family: knapsack, dim: 50, params: {correlation: [uncorrelated, strong]}}
  - {family: portfolio, dim: 20, params: {cardinality: [null, 5]}}
optimisers:
  - {name: DE, entry: de}
  - {name: CMA-ES, entry: cmaes}
  - name: Island
    entry: island
    params: {optimisers: [{entry: de}, {entry: cmaes}], n_epochs: 10}
```

Ready-made configs are in `configs/`: `bench.smoke.yaml`,
`bench.continuous.yaml`, `bench.combinatorial.yaml`, `bench.profit.yaml`,
`bench.ensembles.yaml`, and `bench.demo.yaml` (the run behind the committed
dashboard).

**What you get.** A run directory with `results.csv` (one row per run),
`curves.csv` (anytime curves), `trajectories.jsonl` (state-action steps),
`manifest.json`, `skipped.jsonl` / `errors.jsonl`, plus `scored.csv` and
`summary.json` after analysis.

**Metrics** all derive from one **normalised gap**: 0 = optimum (or
best-known), 1 = no better than random sampling. That puts costs and profits
of any scale on one footing:

- `score = 1 − gap`
- `solved` (gap ≤ 1e-8)
- evaluations to reach gap targets 1e-1 … 1e-8
- an **anytime AUC**
- normalised **average ranks**
- a **Friedman test**
- COCO-style **ECDFs**

**Dashboard.** `docs/dashboard/index.html` is a single self-contained page
with no external dependencies; open it from disk or serve it with GitHub
Pages. It has:

- a filterable leaderboard (by problem category, encoding, dimension, tag,
  and optimiser vs ensemble);
- a problem × optimiser precision heatmap;
- convergence curves per instance;
- anytime ECDFs;
- the taxonomy cards;
- run details.

Every evaluation goes through a recorder that enforces the budget **exactly**
for every optimiser, including those without budget control. Comparisons are
always at equal cost. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
the design and how it maps onto clustbench.

For quick in-memory comparisons without the harness, `optim.benchmarks`
keeps the lightweight `compare()` / `format_table()` helpers and
`python examples/compare_algorithms.py`.

> **Caveat:** GWO, WOA and SCA are biased towards the centre of the search
> box. This is why continuous problems default to `shift=True`: on
> unshifted functions these algorithms look far better than they are.

---

## OptimisationResult

Every `optimise()` call returns an `OptimisationResult` (or `EnsembleResult`
for `EnsembleOptimiser`).

```python
result.best_solution   # the best solution found (list, or Pareto archive for DBMOSA)
result.best_value      # objective value of best_solution (float)
result.history         # list of best values per iteration / epoch
result.n_evaluations   # total number of objective-function calls (int)
result.population      # final population, best first (population-based optimisers; else None)
result.population_values
```

`EnsembleResult` additionally provides:

```python
result.run_results     # list of OptimisationResult — one per constituent optimiser / restart
result.info            # strategy diagnostics, e.g. selection_counts (adaptive), island_best_values (island)
```

---

## Custom Operators

Every optimiser accepts callable overrides for its key internal operators,
letting you tailor the search to your problem without subclassing.

```python
# Custom crossover for GeneticOptimiser
# Signature: (parent1: list, parent2: list) -> (child1, child2)
def my_crossover(parent1, parent2):
    mid = len(parent1) // 2
    return parent1[:mid] + parent2[mid:], parent2[:mid] + parent1[mid:]

ga = GeneticOptimiser(crossover_fn=my_crossover, encoding='real')

# Custom mutation for GeneticOptimiser
# Signature: (solution: list) -> mutated_solution
def my_mutation(solution):
    import random, copy
    s = copy.copy(solution)
    i = random.randrange(len(s))
    s[i] += random.gauss(0, 0.5)
    return s

ga = GeneticOptimiser(mutation_fn=my_mutation, encoding='real')

# Custom move for SimulatedAnnealingOptimiser / DBMOSAOptimiser
# Signature: (solution: list, bounds: list) -> new_solution
def my_move(solution, bounds):
    import random, copy
    s = copy.copy(solution)
    s[0] += random.gauss(0, 0.1)
    return s

sa = SimulatedAnnealingOptimiser(neighbour_fn=my_move)

# Custom neighbourhood for LocalSearchOptimiser
# Signature: (solution: list) -> list[solution]
def my_neighbourhood(solution):
    return [[solution[0] + 0.1], [solution[0] - 0.1]]

ls = LocalSearchOptimiser(neighbourhood_fn=my_neighbourhood)
```

---

## Parameter Tuning Tips

### GeneticOptimiser

- **`population_size`** — larger populations explore more of the space but are
  slower per generation.  Start at 50 and scale up for high-dimensional or
  multi-modal problems.
- **`elite_size`** — keep at roughly 10–20 % of `population_size` to balance
  selection pressure and diversity.
- **`max_no_improve`** — the primary stopping rule.  Increase (e.g. 200–500)
  if the algorithm terminates too early on hard problems.
- **`encoding`** — use `'real'` for continuous problems, `'binary'` for
  combinatorial problems with on/off decisions, and `'permutation'` for
  sequencing / routing problems.

### PSOOptimiser

- **`w`** — values in [0.4, 0.9] work well.  Lower values favour exploitation;
  higher values favour exploration.
- **`c1` and `c2`** — balanced values of 1.5–2.0 are typical.  Increasing
  `c1` makes particles follow their own best; increasing `c2` drives them
  toward the global best.
- **`w_decay`** — set slightly below 1.0 (e.g. 0.999) for linearly decreasing
  inertia, which often improves convergence.
- **`n_particles`** — 20–50 is a good starting range.  Increase for
  high-dimensional or highly multi-modal problems.

### LocalSearchOptimiser

- **`step_size`** — controls neighbourhood granularity.  Large steps escape
  local optima but may overshoot; small steps are precise but slow.  A value
  of 1–5 % of the variable range is typical.
- **`max_no_improve`** — set `None` to run until a strict local optimum is
  found, or a small positive integer to allow a limited plateau.
- Use `LocalSearchOptimiser` as a **final refinement stage** after a global
  search (e.g. via `EnsembleOptimiser` with `strategy='chain'`).

### SimulatedAnnealingOptimiser

- **`initial_temp`** — set high enough that almost all moves are accepted
  initially (acceptance probability ≈ 0.9).  A rough guide:
  `initial_temp ≈ -Δf_avg / ln(0.9)` where Δf_avg is the average uphill move.
- **`cooling_rate`** for `'Geometric'` schedule — values in [0.999, 0.99999]
  work well.  Closer to 1.0 = slower cooling = better quality but more
  evaluations.
- **`epoch_type='Dynamic'`** is generally more efficient than `'Static'`
  because the epoch length adapts to the acceptance rate.
- **`termination='epoch'`** gives predictable runtime; `'temperature'` runs
  until the landscape is frozen.

### DBMOSAOptimiser

- **`max_archive_size`** — limits memory and controls crowding.  Values of
  50–200 work well for 2–3 objective problems.
- **`diversity_method`** — start with `None` to get a quick Pareto front, then
  try `'Histogram'` or `'NN'` if the front is too clustered.
- Use higher `initial_temp` than single-objective SA because the dominance-
  based ΔE is bounded in [−1, 1], so temperatures like `1e7`–`1e9` are
  typical.

### EnsembleOptimiser

- **`strategy='best'`** is the safest choice when you are unsure which
  algorithm will work best — it tries all of them and discards the worst.
- **`strategy='chain'`** is most effective when the first optimiser is a
  fast global explorer (PSO, GA) and the last is a precise local refiner (LS,
  SA with low temperature).
- **`strategy='random_restart'`** is useful when an algorithm is fast but
  sensitive to initialisation (e.g. SA on a rugged landscape).

---

## Integration

Every optimiser in `optim` follows the same contract, which makes them
interchangeable in any downstream pipeline:

```python
class MyOptimiser(BaseOptimiser):
    def optimise(self, objective_fn, bounds=None, *, maximise=False, **kwargs):
        ...
        return OptimisationResult(
            best_solution=...,
            best_value=...,
            history=[...],
            n_evaluations=...,
        )
```

Uniform contract recap:

- Subclass [`BaseOptimiser`][optim.base.BaseOptimiser] and implement
  `optimise(objective_fn, bounds, *, maximise=False, **kwargs)`.
- Accept arbitrary extra `**kwargs` so the optimiser plays nicely with
  `EnsembleOptimiser`, which forwards per-optimiser kwargs.
- Always return an [`OptimisationResult`][optim.base.OptimisationResult] (or a
  subclass thereof).
- Respect `maximise=True` — use `self._wrap_objective(objective_fn, maximise)`
  from the base class to get a function that is always internally minimised.
- Count every objective-function call into `n_evaluations`.
- Optionally accept `initial_solutions` / `initial_solution` and
  `max_evaluations` in `optimise()`. The ensembles detect these and use them
  for warm starts and budget control.

### Adding a new continuous metaheuristic

Subclass [`PopulationOptimiser`](optim/population.py) and implement `_step`
(plus `_initialise` if the algorithm needs extra state). Bounds, budgets,
seeding, warm starts, history, `maximise` and the result object are all
handled for you:

```python
import numpy as np
from optim import PopulationOptimiser

class RandomWalkOptimiser(PopulationOptimiser):
    def __init__(self, step=0.1, population_size=None, **kwargs):
        self.step = step
        super().__init__(population_size=population_size, **kwargs)

    def _step(self, problem, state, rng):
        # state.X: (n, d) positions, state.F: (n,) values; problem.evaluate counts evals
        cand = problem.clip(state.X + rng.normal(0, self.step, state.X.shape) * problem.span)
        self._greedy_replace(state, cand, problem.evaluate(cand))
```

Once a new optimiser follows that contract it can be:

- Called directly with the same signature as every built-in optimiser.
- Dropped into [`EnsembleOptimiser`][optim.ensemble.EnsembleOptimiser] for
  `'best'`, `'chain'`, or `'random_restart'` composition.
- Registered in the `OPTIMISERS` dictionary exported by the package so that
  config-driven code can instantiate it by name:

```python
from optim import OPTIMISERS

cls = OPTIMISERS["pso"]           # PSOOptimiser
opt = cls(n_particles=20, seed=0)
result = opt.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
```

## Running Tests

```bash
python -m pytest tests/ -v
```
