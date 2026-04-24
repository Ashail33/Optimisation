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

---

## Installation

```bash
pip install -e ".[dev]"   # editable install with test dependencies
```

Requires Python ≥ 3.10 and NumPy ≥ 1.26. Tested on Python 3.10 – 3.14. No
other runtime dependencies.

After install, verify the package is importable:

```python
import optim
print(optim.__version__)      # '0.1.0'
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
| Real-valued continuous variables | `'real'` | GA, PSO, LS, SA |
| Bit strings / feature selection | `'binary'` | GA, LS |
| Permutations / sequencing (e.g. TSP) | `'permutation'` | GA |
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
| Want global search then precise local refinement | `EnsembleOptimiser` (strategy `'chain'`) |

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

## OptimisationResult

Every `optimise()` call returns an `OptimisationResult` (or `EnsembleResult`
for `EnsembleOptimiser`).

```python
result.best_solution   # the best solution found (list, or Pareto archive for DBMOSA)
result.best_value      # objective value of best_solution (float)
result.history         # list of best values per iteration / epoch
result.n_evaluations   # total number of objective-function calls (int)
```

`EnsembleResult` additionally provides:

```python
result.run_results     # list of OptimisationResult — one per constituent optimiser / restart
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
