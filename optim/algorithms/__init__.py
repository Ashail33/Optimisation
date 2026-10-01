"""
Catalogue of metaheuristic optimisation algorithms.

Continuous (box-bounded) population-based algorithms — all subclass
:class:`optim.population.PopulationOptimiser` and share the same budget,
seeding and warm-start options:

=====================================  ===============================================
Class                                  Algorithm
=====================================  ===============================================
:class:`DifferentialEvolutionOptimiser` Differential Evolution (Storn & Price, 1997)
:class:`CMAESOptimiser`                 CMA-ES (Hansen, 2016)
:class:`GreyWolfOptimiser`              Grey Wolf Optimizer (Mirjalili et al., 2014)
:class:`WhaleOptimiser`                 Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
:class:`FireflyOptimiser`               Firefly Algorithm (Yang, 2009)
:class:`CuckooSearchOptimiser`          Cuckoo Search (Yang & Deb, 2009)
:class:`ArtificialBeeColonyOptimiser`   Artificial Bee Colony (Karaboga & Basturk, 2007)
:class:`BatOptimiser`                   Bat Algorithm (Yang, 2010)
:class:`HarmonySearchOptimiser`         Harmony Search (Geem et al., 2001)
:class:`TLBOOptimiser`                  Teaching–Learning-Based Optimization (Rao et al., 2011)
:class:`SineCosineOptimiser`            Sine Cosine Algorithm (Mirjalili, 2016)
:class:`JayaOptimiser`                  Jaya (Rao, 2016)
:class:`ACOROptimiser`                  Ant colony for continuous domains (Socha & Dorigo, 2008)
=====================================  ===============================================

Discrete / trajectory algorithms:

* :class:`TabuSearchOptimiser` — Tabu Search (real, binary, permutation)
* :class:`AntColonyOptimiser` — rank-based Ant System for permutations
"""

from .aco_continuous import ACOROptimiser
from .ant_colony import AntColonyOptimiser
from .bat import BatOptimiser
from .bee_colony import ArtificialBeeColonyOptimiser
from .cmaes import CMAESOptimiser
from .cuckoo import CuckooSearchOptimiser
from .differential_evolution import DifferentialEvolutionOptimiser
from .firefly import FireflyOptimiser
from .grey_wolf import GreyWolfOptimiser
from .harmony_search import HarmonySearchOptimiser
from .jaya import JayaOptimiser
from .sine_cosine import SineCosineOptimiser
from .tabu_search import TabuSearchOptimiser
from .tlbo import TLBOOptimiser
from .whale import WhaleOptimiser

__all__ = [
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
]
