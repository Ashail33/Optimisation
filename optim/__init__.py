"""
optim — A generalised optimisation and metaheuristic library.

Available optimisers
--------------------
Single-solution (trajectory) methods
    * :class:`LocalSearchOptimiser` — best-improvement local search
    * :class:`TabuSearchOptimiser` — local search with tabu memory
    * :class:`SimulatedAnnealingOptimiser` — single-objective Simulated Annealing
    * :class:`DBMOSAOptimiser` — Dominance-Based Multi-Objective SA

Population-based methods
    * :class:`GeneticOptimiser` — Genetic Algorithm (real / binary / permutation)
    * :class:`PSOOptimiser` — Particle Swarm Optimisation
    * :class:`DifferentialEvolutionOptimiser` — DE/rand/1/bin

Baseline
    * :class:`RandomSearchOptimiser` — uniform-random sampling

Ensembles (composition of optimisers)
    * :class:`EnsembleOptimiser` — Portfolio / Pipeline / Multi-start

Data containers
---------------
* :class:`OptimisationResult` — result returned by every optimiser
* :class:`EnsembleResult` — extended result including per-run data

Quick start
-----------
>>> from optim import PSOOptimiser
>>> pso = PSOOptimiser(n_particles=20, max_no_improve=100, seed=42)
>>> result = pso.optimise(lambda x: x[0]**2 + x[1]**2,
...                       bounds=[(-5.0, 5.0), (-5.0, 5.0)])
>>> round(result.best_value, 4)  # doctest: +SKIP
0.0
"""

from .base import BaseOptimiser, OptimisationResult
from .de import DifferentialEvolutionOptimiser
from .ensemble import EnsembleOptimiser, EnsembleResult
from .genetic import GeneticOptimiser
from .local_search import LocalSearchOptimiser
from .pso import PSOOptimiser
from .random_search import RandomSearchOptimiser
from .sa import DBMOSAOptimiser, SimulatedAnnealingOptimiser
from .tabu import TabuSearchOptimiser

__version__ = "0.2.0"

# Registry of concrete optimiser classes keyed by short name.  Useful for
# building CLIs / config-driven pipelines that instantiate optimisers
# dynamically, e.g. ``OPTIMISERS["pso"](n_particles=20)``.
OPTIMISERS = {
    "genetic": GeneticOptimiser,
    "pso": PSOOptimiser,
    "de": DifferentialEvolutionOptimiser,
    "local_search": LocalSearchOptimiser,
    "tabu": TabuSearchOptimiser,
    "sa": SimulatedAnnealingOptimiser,
    "dbmosa": DBMOSAOptimiser,
    "random_search": RandomSearchOptimiser,
    "ensemble": EnsembleOptimiser,
}

__all__ = [
    "BaseOptimiser",
    "OptimisationResult",
    "GeneticOptimiser",
    "PSOOptimiser",
    "DifferentialEvolutionOptimiser",
    "LocalSearchOptimiser",
    "TabuSearchOptimiser",
    "SimulatedAnnealingOptimiser",
    "DBMOSAOptimiser",
    "RandomSearchOptimiser",
    "EnsembleOptimiser",
    "EnsembleResult",
    "OPTIMISERS",
    "__version__",
]
