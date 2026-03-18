# Optimisation

A generalised, extensible metaheuristic optimisation library in Python.

## Included Optimisers

| Class | Algorithm | Search space |
|---|---|---|
| `GeneticOptimiser` | Genetic Algorithm | real / binary / permutation |
| `PSOOptimiser` | Particle Swarm Optimisation | continuous |
| `LocalSearchOptimiser` | Best-improvement Local Search | real / binary |
| `SimulatedAnnealingOptimiser` | Simulated Annealing | continuous |
| `DBMOSAOptimiser` | Dominance-Based Multi-Objective SA | continuous (multi-obj) |
| `EnsembleOptimiser` | Ensemble (best / chain / random restart) | any |

## Installation

```bash
pip install -e ".[dev]"   # editable install with test dependencies
```

Requires Python ≥ 3.9, NumPy ≥ 1.21, and pandas ≥ 1.3.

## Quick Start

Every optimiser shares the same interface:

```python
result = optimiser.optimise(objective_fn, bounds=..., maximise=False)
print(result.best_solution, result.best_value)
```

### Particle Swarm Optimisation

```python
from optim import PSOOptimiser

pso = PSOOptimiser(n_particles=30, c1=1.5, c2=1.5, w=0.7, max_no_improve=200)
result = pso.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)
print(result)  # OptimisationResult(best_value=..., n_evaluations=...)
```

### Genetic Algorithm

```python
from optim import GeneticOptimiser

# Real-valued (e.g., minimise sphere function)
ga = GeneticOptimiser(population_size=50, max_no_improve=100, encoding='real')
result = ga.optimise(lambda x: sum(v**2 for v in x), bounds=[(-5, 5)]*3)

# Permutation encoding (e.g., TSP)
dist = [[0,10,20],[10,0,15],[20,15,0]]
def tsp(tour):
    return sum(dist[tour[i]][tour[i-1]] for i in range(1, len(tour)))

ga_tsp = GeneticOptimiser(encoding='permutation', max_no_improve=50)
result = ga_tsp.optimise(tsp, n_genes=3)

# Binary encoding
ga_bin = GeneticOptimiser(encoding='binary', max_no_improve=30)
result = ga_bin.optimise(sum, n_genes=10, maximise=True)
```

### Local Search

```python
from optim import LocalSearchOptimiser

ls = LocalSearchOptimiser(step_size=0.05)
result = ls.optimise(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5, 5), (-5, 5)],
    initial_solution=[3.0, -2.0],
    constraints_fn=lambda x: x[0] >= 0,   # optional feasibility filter
)
```

### Simulated Annealing

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
```

### Multi-Objective SA (DBMOSA)

```python
from optim import DBMOSAOptimiser

def bi_objective(x):
    return [x[0]**2, (x[0] - 2)**2]   # return a list of objective values

dbmosa = DBMOSAOptimiser(
    initial_temp=1e7,
    max_epochs=1000,
    termination='epoch',
    diversity_method='Histogram',       # 'Kernel', 'NN', 'Histogram', or None
    max_archive_size=50,
)
result = dbmosa.optimise(bi_objective, bounds=[(-5.0, 5.0)])
pareto_front = result.best_solution    # list of non-dominated solutions
```

## Ensemble Optimisers

### Strategy: `'best'` — run all, take the best

```python
from optim import GeneticOptimiser, PSOOptimiser, EnsembleOptimiser

ga  = GeneticOptimiser(population_size=30, max_no_improve=50)
pso = PSOOptimiser(n_particles=20, max_no_improve=100)

ens = EnsembleOptimiser([ga, pso], strategy='best')
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
print(result.run_results)  # per-optimiser results available
```

### Strategy: `'chain'` — feed output of one to the next

```python
from optim import PSOOptimiser, SimulatedAnnealingOptimiser, EnsembleOptimiser

pso = PSOOptimiser(n_particles=20, max_no_improve=50)          # global search
sa  = SimulatedAnnealingOptimiser(initial_temp=100, max_epochs=500)  # refine

ens = EnsembleOptimiser([pso, sa], strategy='chain')
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
```

### Strategy: `'random_restart'` — restart the same optimiser N times

```python
from optim import SimulatedAnnealingOptimiser, EnsembleOptimiser

sa  = SimulatedAnnealingOptimiser(initial_temp=1000, max_epochs=3000)
ens = EnsembleOptimiser([sa], strategy='random_restart', n_restarts=5)
result = ens.optimise(lambda x: x[0]**2 + x[1]**2, bounds=[(-5, 5), (-5, 5)])
```

## Custom Operators

Every optimiser accepts callable overrides for its key operators:

```python
# Custom crossover for GeneticOptimiser
def my_crossover(parent1, parent2):
    mid = len(parent1) // 2
    return parent1[:mid] + parent2[mid:], parent2[:mid] + parent1[mid:]

ga = GeneticOptimiser(crossover_fn=my_crossover, encoding='real')

# Custom move for SimulatedAnnealingOptimiser
def my_move(solution, bounds):
    import random, copy
    s = copy.copy(solution)
    s[0] += random.gauss(0, 0.1)
    return s

sa = SimulatedAnnealingOptimiser(neighbour_fn=my_move)

# Custom neighbourhood for LocalSearchOptimiser
def my_neighbourhood(solution):
    return [[solution[0] + 0.1], [solution[0] - 0.1]]

ls = LocalSearchOptimiser(neighbourhood_fn=my_neighbourhood)
```

## OptimisationResult

```python
result.best_solution   # the best solution found
result.best_value      # its objective value
result.history         # list of best values per iteration
result.n_evaluations   # total number of function evaluations
```

## Running Tests

```bash
python -m pytest tests/ -v
```
