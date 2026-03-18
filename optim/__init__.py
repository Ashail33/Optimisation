"""
optim — A generalised optimisation library.

Available optimisers
--------------------
* :class:`GeneticOptimiser` — Genetic Algorithm (real / binary / permutation)
* :class:`PSOOptimiser` — Particle Swarm Optimisation
* :class:`LocalSearchOptimiser` — Best-improvement Local Search
* :class:`SimulatedAnnealingOptimiser` — Single-objective Simulated Annealing
* :class:`DBMOSAOptimiser` — Dominance-Based Multi-Objective SA
* :class:`EnsembleOptimiser` — Combine optimisers (best / chain / restart)

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
from .ensemble import EnsembleOptimiser, EnsembleResult
from .genetic import GeneticOptimiser
from .local_search import LocalSearchOptimiser
from .pso import PSOOptimiser
from .sa import DBMOSAOptimiser, SimulatedAnnealingOptimiser

__all__ = [
    "BaseOptimiser",
    "OptimisationResult",
    "GeneticOptimiser",
    "PSOOptimiser",
    "LocalSearchOptimiser",
    "SimulatedAnnealingOptimiser",
    "DBMOSAOptimiser",
    "EnsembleOptimiser",
    "EnsembleResult",
]
