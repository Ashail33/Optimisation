# optim — A Metaheuristic Optimisation Library

A generalised, extensible library of metaheuristic optimisation algorithms
in Python. Every optimiser obeys the same `BaseOptimiser` interface, so
algorithms are interchangeable in any pipeline and can be composed into
ensembles without glue code.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Notation](#notation)
- [The Optimiser Standard](#the-optimiser-standard)
- [Algorithms](#algorithms)
  - [Random Search](#random-search-randomsearchoptimiser)
  - [Local Search](#local-search-localsearchoptimiser)
  - [Tabu Search](#tabu-search-tabusearchoptimiser)
  - [Simulated Annealing](#simulated-annealing-simulatedannealingoptimiser)
  - [DBMOSA (multi-objective SA)](#dbmosa-dbmosaoptimiser)
  - [Genetic Algorithm](#genetic-algorithm-geneticoptimiser)
  - [Differential Evolution](#differential-evolution-differentialevolutionoptimiser)
  - [Particle Swarm Optimisation](#particle-swarm-optimisation-psooptimiser)
- [Ensembles — Three Families](#ensembles--three-families)
  - [Portfolio](#1-portfolio-aliasbest)
  - [Pipeline](#2-pipeline-aliaschain)
  - [Multi-start](#3-multi-start-aliasrandom_restart)
- [OptimisationResult](#optimisationresult)
- [Encodings](#encodings)
- [Custom Operators](#custom-operators)
- [Adding Your Own Optimiser](#adding-your-own-optimiser)
- [Parameter Tuning Tips](#parameter-tuning-tips)
- [Running Tests](#running-tests)
- [References](#references)

---

## Features

| Family | Optimiser | Search space | Multi-objective |
|---|---|---|---|
| Baseline | `RandomSearchOptimiser` | real / binary | — |
| Trajectory | `LocalSearchOptimiser` | real / binary | — |
| Trajectory | `TabuSearchOptimiser` | real / permutation | — |
| Trajectory | `SimulatedAnnealingOptimiser` | real | — |
| Trajectory | `DBMOSAOptimiser` | real | yes (Pareto archive) |
| Population | `GeneticOptimiser` | real / binary / permutation | — |
| Population | `DifferentialEvolutionOptimiser` | real | — |
| Population | `PSOOptimiser` | real | — |
| Ensemble | `EnsembleOptimiser` | any | inherits |

Cross-cutting features:

- **Uniform interface** — every optimiser exposes the same
  `optimise(objective_fn, bounds=..., maximise=False, **kwargs)` call.
- **Minimise or maximise** with a single flag; you never need to negate.
- **Encoding-aware defaults** — real-valued, binary, and permutation
  operators are built in; supply a callable to override any of them.
- **Custom operators** — crossover, mutation, neighbourhood, sampler, and
  move generators are all pluggable.
- **Reproducibility** — every optimiser accepts a `seed`.
- **Stopping rules** — every optimiser exposes both a budget cap
  (iterations/generations/epochs/evaluations) and a stagnation cap
  (`max_no_improve`).
- **Result object** — a single `OptimisationResult` dataclass with
  `best_solution`, `best_value`, `history`, and `n_evaluations`.
- **Ensembles** — three composition families
  (portfolio / pipeline / multi-start) usable on any optimiser mix.
- **Registry** — `optim.OPTIMISERS` maps short names to classes for
  config-driven pipelines.

---

## Installation

```bash
pip install -e ".[dev]"   # editable install with test dependencies
```

Requires Python >= 3.10 and NumPy >= 1.26. Tested on Python 3.10 - 3.14. No
other runtime dependencies.

After install, verify the package is importable:

```python
import optim
print(optim.__version__)      # '0.2.0'
print(sorted(optim.OPTIMISERS))
# ['dbmosa', 'de', 'ensemble', 'genetic', 'local_search',
#  'pso', 'random_search', 'sa', 'tabu']
```

The legacy reference scripts at the repository root (`DBMOSA algorithm.py`,
`Genetic search algorithm.py`, `Local Search function`,
`Particle swarm optimisation algorithm`) are the original un-packaged
implementations kept for historical reference. **Always use the `optim`
package** — it is the integration target.

---

## Quick Start

Every optimiser shares the same call signature:

```python
from optim import PSOOptimiser

pso = PSOOptimiser(n_particles=30, max_no_improve=200, seed=0)
result = pso.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)
print(result.best_solution, result.best_value)
```

Want to compare three algorithms and keep the best? Drop them into a
portfolio ensemble:

```python
from optim import (
    GeneticOptimiser, DifferentialEvolutionOptimiser,
    PSOOptimiser, EnsembleOptimiser,
)

ens = EnsembleOptimiser(
    [GeneticOptimiser(seed=0),
     DifferentialEvolutionOptimiser(seed=0),
     PSOOptimiser(seed=0)],
    strategy="portfolio",
)
result = ens.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)
```

---

## Notation

The same symbols are used in every algorithm description below.

| Symbol | Meaning |
|---|---|
| $x$ | a candidate solution (decision vector) |
| $x_j$ | the $j$-th decision variable of $x$ |
| $x^{(i)}$ | the $i$-th individual in a population |
| $f(x)$ | objective function value at $x$ |
| $D$ | number of decision variables (dimensionality) |
| $N$ | population / swarm size |
| $t$ | iteration / generation / epoch counter |
| $x^\star, f^\star$ | best solution and best value found so far |
| $\mathcal{N}(x)$ | neighbourhood of $x$ |
| $[l_j, u_j]$ | lower and upper bound of variable $j$ |
| $\mathcal{U}(a, b)$ | uniform random number in $[a, b]$ |
| $T$ | temperature (Simulated Annealing) |
| $\alpha$ | cooling / contraction parameter |
| $w, c_1, c_2$ | PSO inertia and acceleration coefficients |
| $p^{(i)}, g$ | PSO personal best of particle $i$, global best |
| $v^{(i)}$ | PSO velocity of particle $i$ |
| $F, CR$ | DE differential weight, crossover probability |

All optimisers internally **minimise**. To maximise, pass `maximise=True` to
`optimise()`; the wrapper negates the objective transparently.

---

## The Optimiser Standard

Every optimiser in `optim` is a concrete subclass of `BaseOptimiser` and
satisfies the following contract:

```python
class BaseOptimiser(ABC):
    @abstractmethod
    def optimise(
        self,
        objective_fn: Callable[[Any], float],
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        **kwargs: Any,
    ) -> OptimisationResult:
        ...
```

Rules of the standard:

1. **Signature** — `optimise()` always accepts `objective_fn` and `bounds`
   positionally, then keyword-only `maximise`, plus `**kwargs` for algorithm-
   specific extras (`n_genes`, `initial_solution`, `initial_solutions`,
   `constraints_fn`, ...).
2. **Direction handling** — every optimiser calls
   `self._wrap_objective(objective_fn, maximise)` so the internal logic is
   always minimisation; the wrapper negates the objective when
   `maximise=True`.
3. **Return type** — always an `OptimisationResult` (or a subclass such as
   `EnsembleResult`).
4. **Stopping rules** — at least one of `max_iterations` /
   `max_generations` / `max_epochs` / `max_evaluations`, plus a stagnation
   cap `max_no_improve` where applicable. Both must be respected.
5. **Evaluation count** — every call to the objective function must be
   counted into `OptimisationResult.n_evaluations`.
6. **History** — `OptimisationResult.history` is the running best objective
   value (one entry per major step). For minimisation it is monotonically
   non-increasing.
7. **Reproducibility** — accept a `seed` in `__init__` and seed all RNGs
   inside `optimise()` if it is set.
8. **Encoding awareness** — when the optimiser supports multiple encodings,
   expose an `encoding` parameter and validate it in `__init__`.
9. **No hidden globals** — operators must be pure functions or methods so
   the optimiser can be safely reused and composed.

If your code follows these rules it is automatically:

- usable interchangeably anywhere `BaseOptimiser` is accepted,
- composable inside any `EnsembleOptimiser` strategy,
- registrable in `optim.OPTIMISERS` for config-driven dispatch.

---

## Algorithms

### Random Search (`RandomSearchOptimiser`)

The canonical baseline. Drawn $N_{\text{eval}}$ samples uniformly from the
search space and keeps the best.

For real-valued problems, sample $x \sim \mathcal{U}(l, u)$. For binary
problems, sample each gene from $\{0, 1\}$. A custom `sample_fn` overrides
both.

```text
for t = 1 ... N_eval:
    x ~ Uniform(l, u)
    if f(x) < f*: x*, f* <- x, f(x)
```

**Key parameters:** `encoding`, `max_evaluations`, `sample_fn`, `seed`.

```python
from optim import RandomSearchOptimiser

rs = RandomSearchOptimiser(max_evaluations=500, seed=0)
result = rs.optimise(lambda x: x[0]**2 + x[1]**2,
                     bounds=[(-5, 5), (-5, 5)])
```

Use it as a sanity check: if a fancier algorithm cannot beat it on your
problem, the algorithm or its tuning is suspect.

---

### Local Search (`LocalSearchOptimiser`)

Best-improvement local search. At every step it evaluates every neighbour
in $\mathcal{N}(x)$ and moves to the best improving one. Terminates at a
strict local optimum or when a budget is exhausted.

Default neighbourhoods:

- **real**: $\mathcal{N}(x) = \{x \pm \delta e_j : j = 1,\dots,D\}$ clamped
  to bounds. $\delta$ is `step_size`.
- **binary**: bit-flip neighbourhood
  $\mathcal{N}(x) = \{x \oplus e_j : j = 1,\dots,D\}$.

```text
repeat:
    N <- neighbourhood(x)               # filtered by constraints_fn
    x' <- argmin_{n in N} f(n)
    if f(x') < f(x): x <- x'
    else: stop (strict local optimum)
```

**Key parameters:** `encoding`, `step_size`, `max_no_improve`,
`max_iterations`, `neighbourhood_fn`, `seed`.

`optimise()` extras: `initial_solution`, `constraints_fn(x) -> bool`.

```python
from optim import LocalSearchOptimiser

ls = LocalSearchOptimiser(step_size=0.05)
result = ls.optimise(lambda x: x[0]**2 + x[1]**2,
                     bounds=[(-5, 5), (-5, 5)],
                     initial_solution=[3.0, -2.0],
                     constraints_fn=lambda x: x[0] >= 0)
```

---

### Tabu Search (`TabuSearchOptimiser`)

Local search with short-term memory (Glover, 1986). Maintains a *tabu list*
of recently visited moves; the best **non-tabu** neighbour is chosen at
each step, even if it worsens $f$. A tabu move is accepted anyway when it
satisfies the **aspiration criterion** (it would improve $f^\star$).

Default neighbourhoods:

- **real**: $\pm\delta$ step in each dimension (move key = rounded
  position).
- **permutation**: all 2-opt swaps (move key = swapped index pair).

```text
repeat:
    N  <- neighbourhood(x)
    x' <- argmin_{n in N \ tabu (or aspirating)} f(n)
    if x' is None: stop
    push move(x -> x') onto tabu list (length = tabu_tenure)
    x <- x'
    if f(x) < f*: x*, f* <- x, f(x)
```

**Key parameters:** `encoding`, `step_size`, `tabu_tenure`,
`max_iterations`, `max_no_improve`, `neighbourhood_fn`, `seed`.

```python
from optim import TabuSearchOptimiser

ts = TabuSearchOptimiser(encoding="permutation", tabu_tenure=8,
                         max_iterations=300, seed=0)
result = ts.optimise(tsp_cost, n_genes=10)
```

---

### Simulated Annealing (`SimulatedAnnealingOptimiser`)

A single-trajectory probabilistic search that accepts uphill moves with
probability $e^{-\Delta f / T}$ (the Metropolis criterion), with $T$
decaying over time.

```text
repeat:
    x' <- neighbour(x)
    df <- f(x') - f(x)
    if df < 0 or U(0,1) < exp(-df / T): x <- x'
    end of epoch: T <- schedule(T)
```

**Cooling schedules** (parameter $\alpha$):

| Schedule | Cool update | Notes |
|---|---|---|
| `Geometric` | $T \leftarrow \alpha T$ | most common; $\alpha$ near 1 = slow |
| `Linear` | $T \leftarrow T - \alpha$ | $\alpha$ is the decrement |
| `Logarithmic` | $T \leftarrow T / \ln(t)$ | theoretically convergent, slow |
| `Very slow cooling` | $T \leftarrow T / (1 + \alpha)$ | tiny $\alpha$ values |

**Epoch types:**

- `Dynamic` — epoch ends after `max_accepted` accepted moves (then cool) or
  `max_rejected` rejected moves (then reheat).
- `Static` — epoch ends after `static_epoch_length` moves (then cool).

**Termination:** `'epoch'` (by epoch count) or `'temperature'` (when
$T \le T_\min$).

**Key parameters:** `initial_temp`, `cooling_rate` ($\alpha$),
`reheating_rate`, `max_accepted`, `max_rejected`, `static_epoch_length`,
`max_epochs`, `min_temp`, `schedule`, `epoch_type`, `termination`,
`neighbour_fn`, `move_scale`, `seed`.

```python
from optim import SimulatedAnnealingOptimiser

sa = SimulatedAnnealingOptimiser(
    initial_temp=1e3, cooling_rate=0.99, schedule='Geometric',
    epoch_type='Dynamic', termination='epoch', max_epochs=5000,
)
result = sa.optimise(lambda x: x[0]**2 + x[1]**2,
                     bounds=[(-5, 5), (-5, 5)])
```

---

### DBMOSA (`DBMOSAOptimiser`)

Dominance-Based Multi-Objective Simulated Annealing
(Bandyopadhyay et al., 2008). The objective returns a vector
$\mathbf{f}(x) = (f_1(x), \dots, f_M(x))$ and the algorithm maintains a
Pareto archive $\mathcal{A}$ of non-dominated solutions.

Acceptance uses a **dominance-count energy** based on how many archive
members dominate the current and candidate solutions:

$$\Delta E = \frac{-\#\text{dom}(x) + \#\text{dom}(x')}{|\mathcal{A}| + 2}$$

A candidate is accepted with probability $\min(1, e^{-\Delta E / T})$.

**Diversity preservation** (optional): `'Kernel'`, `'NN'`, or `'Histogram'`
modify $\Delta E$ to discourage crowded regions of objective space. When
the archive exceeds `max_archive_size`, the most crowded solution
(smallest 1-NN distance in objective space) is pruned.

The returned `best_solution` is the **Pareto archive** — a list of
non-dominated solutions; `best_value` is the mean sum-of-objectives over
the archive (a coarse quality scalar).

```python
from optim import DBMOSAOptimiser

def bi_objective(x):
    return [x[0]**2, (x[0] - 2)**2]

dbmosa = DBMOSAOptimiser(initial_temp=1e7, max_epochs=1000,
                         termination='epoch',
                         diversity_method='Histogram',
                         max_archive_size=50)
result = dbmosa.optimise(bi_objective, bounds=[(-5.0, 5.0)])
print(f"Pareto front size: {len(result.best_solution)}")
```

---

### Genetic Algorithm (`GeneticOptimiser`)

Population-based search with selection, crossover, and mutation. Supports
three encodings out of the box, each with a sensible default operator
pair:

| Encoding | Default crossover | Default mutation |
|---|---|---|
| `real` | arithmetic / blend $c = \lambda p_1 + (1-\lambda) p_2$ | one-coordinate Gaussian perturbation, clamped |
| `binary` | single-point crossover | one-bit flip |
| `permutation` | order crossover (OX) | swap two random positions |

Parent selection is fitness-proportionate (roulette-wheel) with inverse
fitness weights; replacement is steady-state — the best `elite_size` of
$\{\text{population}\}\cup\{\text{children}\}$ survives.

```text
init population X of size N
repeat:
    parents  <- roulette(X, n_parents)
    children <- crossover(parents) + mutation(one child)
    X        <- top elite_size of (X ∪ children)
```

**Key parameters:** `population_size`, `elite_size`, `n_parents`,
`max_no_improve`, `max_generations`, `encoding`, `crossover_fn`,
`mutation_fn`, `seed`.

```python
from optim import GeneticOptimiser

ga = GeneticOptimiser(population_size=50, max_no_improve=100,
                      encoding='real')
result = ga.optimise(lambda x: sum(v**2 for v in x),
                     bounds=[(-5, 5)] * 3)
```

---

### Differential Evolution (`DifferentialEvolutionOptimiser`)

Storn & Price's DE/rand/1/bin (1997) for continuous spaces. For every
parent $x^{(i)}$ we draw three distinct indices $a, b, c$ and build a
**trial vector**:

$$v = x^{(a)} + F \cdot (x^{(b)} - x^{(c)})$$

$$u_j = \begin{cases} v_j & \text{if } \mathcal{U}(0,1) < CR \text{ or } j = j_{\text{rand}} \\ x^{(i)}_j & \text{otherwise} \end{cases}$$

The trial replaces the parent iff $f(u) \le f(x^{(i)})$ (greedy
selection). $j_{\text{rand}}$ guarantees at least one inherited gene.

**Key parameters:** `population_size` ($N \ge 4$), `F` (differential
weight), `CR` (crossover probability), `max_generations`, `max_no_improve`,
`seed`. `optimise()` extras: `initial_solutions`.

```python
from optim import DifferentialEvolutionOptimiser

de = DifferentialEvolutionOptimiser(population_size=30, F=0.5, CR=0.9,
                                    max_generations=500, seed=0)
result = de.optimise(lambda x: x[0]**2 + x[1]**2,
                     bounds=[(-5, 5), (-5, 5)])
```

---

### Particle Swarm Optimisation (`PSOOptimiser`)

Classic *gbest* PSO (Kennedy & Eberhart, 1995). Particles update velocity
and position via:

$$v^{(i)}_{t+1} = w\, v^{(i)}_t + c_1 r_1 (p^{(i)} - x^{(i)}_t) + c_2 r_2 (g - x^{(i)}_t)$$

$$x^{(i)}_{t+1} = x^{(i)}_t + v^{(i)}_{t+1}$$

with $r_1, r_2 \sim \mathcal{U}(0,1)^D$. Positions are clamped to bounds
and velocities are reflected and damped at the walls. The inertia weight
$w$ can be linearly damped each step by `w_decay`.

**Key parameters:** `n_particles` ($N$), `c1`, `c2`, `w`, `w_decay`,
`max_no_improve`, `max_iterations`, `precision`, `seed`.

`optimise()` extras: `initial_solutions` (warm-start seeds).

```python
from optim import PSOOptimiser

pso = PSOOptimiser(n_particles=30, c1=1.5, c2=1.5, w=0.7,
                   max_no_improve=200, seed=0)
result = pso.optimise(lambda x: x[0]**2 + x[1]**2,
                      bounds=[(-5, 5), (-5, 5)])
```

---

## Ensembles — Three Families

`EnsembleOptimiser` provides three composition strategies following the
standard taxonomy of hybrid metaheuristics (Talbi, 2002). Each accepts
either the canonical name or a backward-compatible alias.

```python
from optim import EnsembleOptimiser
EnsembleOptimiser(optimisers, strategy='portfolio'  | 'best',
                              # or 'pipeline'        | 'chain',
                              # or 'multi_start'     | 'random_restart',
                  n_restarts=5)
```

Every individual run is preserved in `result.run_results` (a list of
`OptimisationResult`).

### 1. Portfolio (alias `best`)

Run every constituent **in parallel**, independently, and return the
single best result. A *teamwork-style* ensemble that hedges across
algorithms without coordination.

```text
for opt in optimisers:
    r_opt <- opt.optimise(f, bounds, ...)
return argmin_r r.best_value
```

Use when you do not know which algorithm fits the landscape best.

```python
from optim import (
    GeneticOptimiser, PSOOptimiser,
    DifferentialEvolutionOptimiser, EnsembleOptimiser,
)

ens = EnsembleOptimiser(
    [GeneticOptimiser(seed=0),
     PSOOptimiser(seed=0),
     DifferentialEvolutionOptimiser(seed=0)],
    strategy='portfolio',
)
result = ens.optimise(my_fn, bounds=my_bounds)
```

### 2. Pipeline (alias `chain`)

Run constituents **sequentially**, warm-starting each one with the best
solution of the previous. A *relay-style* ensemble — typically a fast
global search hands off to a precise local refiner.

```text
warm <- None
for opt in optimisers:
    r <- opt.optimise(f, bounds, initial_solution=warm, ...)
    warm <- r.best_solution
return last r
```

Best for "explore then exploit" patterns.

```python
from optim import (
    DifferentialEvolutionOptimiser, LocalSearchOptimiser,
    EnsembleOptimiser,
)

ens = EnsembleOptimiser(
    [DifferentialEvolutionOptimiser(max_generations=200, seed=0),
     LocalSearchOptimiser(step_size=0.01, seed=0)],
    strategy='pipeline',
)
```

### 3. Multi-start (alias `random_restart`)

Run the *same* optimiser `n_restarts` times, each from a fresh random
initialisation, and return the best. Mitigates the sensitivity of
stochastic algorithms to their starting point.

```text
for k in 1..n_restarts:
    r_k <- optimisers[0].optimise(f, bounds, ...)
return argmin_k r_k.best_value
```

Best for rugged landscapes where a single run is variable.

```python
from optim import SimulatedAnnealingOptimiser, EnsembleOptimiser

ens = EnsembleOptimiser(
    [SimulatedAnnealingOptimiser(initial_temp=1e3, max_epochs=2000, seed=0)],
    strategy='multi_start', n_restarts=5,
)
```

---

## OptimisationResult

Every `optimise()` call returns an `OptimisationResult`. `EnsembleOptimiser`
returns an `EnsembleResult` which extends it.

```python
result.best_solution   # best solution found (or Pareto archive for DBMOSA)
result.best_value      # objective value of best_solution (float)
result.history         # running best per iteration / generation / epoch
result.n_evaluations   # total objective-function evaluations
# EnsembleResult only:
result.run_results     # list[OptimisationResult] - one per run
```

For minimisation, `history` is monotonically non-increasing. For
maximisation (via `maximise=True`), the values are reported in the
original direction (non-decreasing).

---

## Encodings

| Encoding | Solution type | Optimisers that support it |
|---|---|---|
| `real` | list of floats in `bounds` | RS, LS, TS, SA, DBMOSA, GA, DE, PSO |
| `binary` | list of 0/1 ints (length `n_genes`) | RS, LS, GA |
| `permutation` | list = permutation of `0..n_genes-1` | TS, GA |

Provide `bounds` for real-valued problems, or `n_genes` for binary /
permutation problems (or both — for binary, `len(bounds)` sets `n_genes`).

---

## Custom Operators

Every optimiser exposes the operators that drive its search behaviour, so
you can adapt the algorithm to a domain-specific encoding without
subclassing.

```python
# Genetic Algorithm: custom crossover and mutation
# crossover_fn(parent1, parent2) -> (child1, child2)
# mutation_fn(solution)          -> mutated_solution
ga = GeneticOptimiser(crossover_fn=my_crossover, mutation_fn=my_mutation)

# Local Search: custom neighbourhood
# neighbourhood_fn(solution) -> list[solution]
ls = LocalSearchOptimiser(neighbourhood_fn=my_nb)

# Tabu Search: custom neighbourhood (returns (neighbour, move_key) pairs)
# neighbourhood_fn(solution) -> list[(solution, move_key)]
ts = TabuSearchOptimiser(neighbourhood_fn=my_nb)

# SA / DBMOSA: custom move generator
# neighbour_fn(solution, bounds) -> new_solution
sa = SimulatedAnnealingOptimiser(neighbour_fn=my_move)

# Random Search: custom sampler
# sample_fn() -> solution
rs = RandomSearchOptimiser(sample_fn=my_sampler)
```

---

## Adding Your Own Optimiser

The standard makes it trivial to add a new optimiser. Subclass
`BaseOptimiser`, implement `optimise`, return an `OptimisationResult`, and
your class is immediately usable everywhere a built-in is — including
inside any ensemble strategy.

```python
from optim import BaseOptimiser, OptimisationResult

class MyOptimiser(BaseOptimiser):
    def __init__(self, max_iterations=100, seed=None):
        self.max_iterations = max_iterations
        self.seed = seed

    def optimise(self, objective_fn, bounds=None, *,
                 maximise=False, initial_solution=None, **kwargs):
        import random
        if self.seed is not None:
            random.seed(self.seed)
        obj = self._wrap_objective(objective_fn, maximise)

        x = initial_solution or [random.uniform(l, u) for l, u in bounds]
        best, best_v = list(x), obj(x)
        history, n_eval = [best_v], 1

        for _ in range(self.max_iterations):
            # ... your search step here ...
            n_eval += 1
            history.append(best_v)

        return OptimisationResult(
            best_solution=best,
            best_value=-best_v if maximise else best_v,
            history=[-v if maximise else v for v in history],
            n_evaluations=n_eval,
        )
```

Register it by short name to make it discoverable to config-driven code:

```python
from optim import OPTIMISERS
OPTIMISERS["mine"] = MyOptimiser

opt = OPTIMISERS["mine"](max_iterations=200, seed=0)
result = opt.optimise(f, bounds=b)
```

Checklist before considering a new optimiser "library-quality":

- [ ] `optimise(objective_fn, bounds=None, *, maximise=False, **kwargs)`
- [ ] Uses `self._wrap_objective(...)` to handle `maximise`
- [ ] Returns an `OptimisationResult`
- [ ] Counts every objective call into `n_evaluations`
- [ ] Records the running best in `history`
- [ ] Supports `seed` for reproducibility
- [ ] At least one budget cap and (where meaningful) a stagnation cap
- [ ] Validates constructor arguments in `__init__`
- [ ] Has tests in `tests/test_optimisers.py` against the sphere / Rosenbrock
      benchmarks

---

## Parameter Tuning Tips

### Genetic Algorithm

- **`population_size`** — larger populations explore more but are slower
  per generation. Start at 50; scale up for high-dimensional or multi-modal
  problems.
- **`elite_size`** — roughly 10-20 % of `population_size` to balance
  selection pressure and diversity.
- **`max_no_improve`** — primary stopping rule. Increase (200-500) on hard
  problems.
- **`encoding`** — `'real'` for continuous, `'binary'` for on/off feature
  selection, `'permutation'` for sequencing / routing.

### Differential Evolution

- **`F`** — usually in `[0.4, 1.0]`. Larger values increase exploration.
- **`CR`** — `0.9` is a robust default; lower `CR` (e.g. `0.1-0.3`) suits
  separable problems.
- **`population_size`** — `5*D` to `10*D` is a common rule of thumb; never
  below 4.

### Particle Swarm Optimisation

- **`w`** — values in `[0.4, 0.9]` work well. Lower favours exploitation;
  higher favours exploration.
- **`c1`, `c2`** — balanced values of `1.5-2.0` are typical.
- **`w_decay`** — slightly below `1.0` (e.g. `0.999`) gives a linearly
  decreasing inertia and usually improves convergence.

### Local Search

- **`step_size`** — 1-5 % of the variable range is typical. Large steps
  escape local optima but may overshoot.
- **`max_no_improve`** — `None` to run until a strict local optimum, or a
  small integer to allow a limited plateau.
- Best used as a **final refinement** inside a pipeline ensemble.

### Tabu Search

- **`tabu_tenure`** — usually `sqrt(N_neighbours)` to `2*sqrt(N_neighbours)`;
  too small and the search cycles, too large and good moves are blocked.
- **`step_size`** for real encoding — same logic as Local Search.

### Simulated Annealing

- **`initial_temp`** — set high enough that almost all moves are accepted
  initially. Rule of thumb:
  $T_0 \approx -\overline{\Delta f} / \ln(0.9)$.
- **`cooling_rate`** for Geometric — values in `[0.999, 0.99999]`. Closer
  to 1 = slower cooling = better quality but more evaluations.
- **`epoch_type='Dynamic'`** is generally more efficient than `'Static'`.

### DBMOSA

- **`max_archive_size`** — 50-200 for 2-3 objective problems.
- **`diversity_method`** — start with `None`; switch to `'Histogram'` or
  `'NN'` if the front clusters.
- Use higher `initial_temp` than single-objective SA: $\Delta E \in [-1, 1]$,
  so `1e7` - `1e9` is typical.

### EnsembleOptimiser

- **`portfolio`** — safest default when the algorithm choice is unclear.
- **`pipeline`** — best when the first optimiser is a global explorer and
  the last is a precise local refiner.
- **`multi_start`** — useful when an algorithm is fast but starting-point-
  sensitive (e.g. SA on a rugged landscape).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The suite includes 70+ tests covering every optimiser against the
sphere / Rosenbrock / binary-sum / TSP benchmarks plus all ensemble
strategies, alias resolution, and the registry.

---

## References

- Bandyopadhyay, S., Saha, S., Maulik, U. & Deb, K. (2008). A Simulated
  Annealing-Based Multiobjective Optimization Algorithm: AMOSA. *IEEE
  Trans. Evolutionary Computation*, 12(3), 269-283.
- Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter
  Optimization. *JMLR*, 13, 281-305.
- Glover, F. (1986). Future paths for integer programming and links to
  artificial intelligence. *Computers & Operations Research*, 13(5),
  533-549.
- Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization.
  *Proceedings of ICNN'95*, Vol. 4, 1942-1948.
- Storn, R. & Price, K. (1997). Differential Evolution — A Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces.
  *Journal of Global Optimization*, 11(4), 341-359.
- Talbi, E.-G. (2002). A Taxonomy of Hybrid Metaheuristics. *Journal of
  Heuristics*, 8(5), 541-564.
