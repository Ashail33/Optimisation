"""
optim — A generalised metaheuristic optimisation library.

Optimisers
----------
Core (real / binary / permutation encodings):

* :class:`GeneticOptimiser` — Genetic Algorithm
* :class:`PSOOptimiser` — Particle Swarm Optimisation
* :class:`LocalSearchOptimiser` — Best-improvement Local Search
* :class:`SimulatedAnnealingOptimiser` — Simulated Annealing
* :class:`DBMOSAOptimiser` — Dominance-Based Multi-Objective SA

Metaheuristic catalogue (:mod:`optim.algorithms`):

* Evolutionary — :class:`DifferentialEvolutionOptimiser`, :class:`CMAESOptimiser`
* Swarm — :class:`GreyWolfOptimiser`, :class:`WhaleOptimiser`,
  :class:`FireflyOptimiser`, :class:`CuckooSearchOptimiser`,
  :class:`ArtificialBeeColonyOptimiser`, :class:`BatOptimiser`,
  :class:`ACOROptimiser`, :class:`AntColonyOptimiser` (permutations)
* Other — :class:`HarmonySearchOptimiser`, :class:`TLBOOptimiser`,
  :class:`SineCosineOptimiser`, :class:`JayaOptimiser`,
  :class:`TabuSearchOptimiser`

Ensembles (:mod:`optim.ensembles`):

* :class:`EnsembleOptimiser` — portfolio ``'best'``, relay ``'chain'``,
  ``'random_restart'``
* :class:`IslandModelOptimiser` — islands with migration
* :class:`AdaptiveEnsembleOptimiser` — bandit / hyper-heuristic selection
* :class:`MemeticOptimiser` — global search + local refinement
* :class:`CooperativeCoevolutionOptimiser` — variable decomposition

Problems, taxonomy and benchmarking
-----------------------------------
* :mod:`optim.problems` — continuous, combinatorial and profit-function
  problem families (``make_problem``)
* :mod:`optim.taxonomy` — machine-readable cards for every optimiser and
  ensemble (Talbi classes for ensembles)
* :mod:`optim.bench` — config-driven benchmark harness, metrics and
  dashboard (``optbench`` CLI; needs the ``bench`` extra)

Data containers
---------------
* :class:`OptimisationResult` — result returned by every optimiser
* :class:`Step` — one state-action step of a recorded trajectory
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

from .algorithms import (
    ACOROptimiser,
    AntColonyOptimiser,
    ArtificialBeeColonyOptimiser,
    BatOptimiser,
    CMAESOptimiser,
    CuckooSearchOptimiser,
    DifferentialEvolutionOptimiser,
    FireflyOptimiser,
    GreyWolfOptimiser,
    HarmonySearchOptimiser,
    JayaOptimiser,
    SineCosineOptimiser,
    TabuSearchOptimiser,
    TLBOOptimiser,
    WhaleOptimiser,
)
from .base import BaseOptimiser, OptimisationResult, Step
from .ensemble import EnsembleOptimiser, EnsembleResult
from .ensembles import (
    AdaptiveEnsembleOptimiser,
    CooperativeCoevolutionOptimiser,
    IslandModelOptimiser,
    MemeticOptimiser,
)
from .genetic import GeneticOptimiser
from .local_search import LocalSearchOptimiser
from .population import PopulationOptimiser
from .pso import PSOOptimiser
from .sa import DBMOSAOptimiser, SimulatedAnnealingOptimiser

__version__ = "0.3.0"

# Registry of single-algorithm optimiser classes keyed by short name.  Useful
# for CLIs / config-driven pipelines, e.g. ``OPTIMISERS["pso"](n_particles=20)``.
OPTIMISERS = {
    "genetic": GeneticOptimiser,
    "pso": PSOOptimiser,
    "local_search": LocalSearchOptimiser,
    "sa": SimulatedAnnealingOptimiser,
    "dbmosa": DBMOSAOptimiser,
    "de": DifferentialEvolutionOptimiser,
    "cmaes": CMAESOptimiser,
    "gwo": GreyWolfOptimiser,
    "woa": WhaleOptimiser,
    "firefly": FireflyOptimiser,
    "cuckoo": CuckooSearchOptimiser,
    "abc": ArtificialBeeColonyOptimiser,
    "bat": BatOptimiser,
    "harmony": HarmonySearchOptimiser,
    "tlbo": TLBOOptimiser,
    "sca": SineCosineOptimiser,
    "jaya": JayaOptimiser,
    "acor": ACOROptimiser,
    "aco": AntColonyOptimiser,
    "tabu": TabuSearchOptimiser,
    "ensemble": EnsembleOptimiser,
}

# Registry of ensemble strategies (each wraps other optimisers).
ENSEMBLES = {
    "portfolio": EnsembleOptimiser,
    "island": IslandModelOptimiser,
    "adaptive": AdaptiveEnsembleOptimiser,
    "memetic": MemeticOptimiser,
    "cooperative": CooperativeCoevolutionOptimiser,
}

__all__ = [
    "BaseOptimiser",
    "PopulationOptimiser",
    "OptimisationResult",
    "EnsembleResult",
    "Step",
    # core
    "GeneticOptimiser",
    "PSOOptimiser",
    "LocalSearchOptimiser",
    "SimulatedAnnealingOptimiser",
    "DBMOSAOptimiser",
    # catalogue
    "ACOROptimiser",
    "AntColonyOptimiser",
    "ArtificialBeeColonyOptimiser",
    "BatOptimiser",
    "CMAESOptimiser",
    "CuckooSearchOptimiser",
    "DifferentialEvolutionOptimiser",
    "FireflyOptimiser",
    "GreyWolfOptimiser",
    "HarmonySearchOptimiser",
    "JayaOptimiser",
    "SineCosineOptimiser",
    "TabuSearchOptimiser",
    "TLBOOptimiser",
    "WhaleOptimiser",
    # ensembles
    "EnsembleOptimiser",
    "IslandModelOptimiser",
    "AdaptiveEnsembleOptimiser",
    "MemeticOptimiser",
    "CooperativeCoevolutionOptimiser",
    # registries
    "OPTIMISERS",
    "ENSEMBLES",
    "__version__",
]
