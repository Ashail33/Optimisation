"""
Machine-readable taxonomy of every optimiser and ensemble in the library.

The clustbench analogue is ``algorithm_cards.py``: instead of describing
algorithms only in prose, each one gets a *card* — structured metadata that
code can query.  Cards drive:

* the benchmark (which optimisers can run on which encodings, and how to
  configure them for a problem — :func:`configure_for`);
* the dashboard's taxonomy view and leaderboard grouping;
* later, algorithm selection / fingerprint matching (mechanisms and biases
  are the features a router or an algorithm mutator can reason over).

Optimisers are classified by **family** (where the idea comes from) and
**search model** (what is iterated).  Ensembles follow Talbi's hybrid
metaheuristic taxonomy — *level* (low: one algorithm embedded inside
another; high: algorithms kept self-contained) × *mode* (relay: run in
sequence; teamwork: run cooperatively) — refined by how the members
*cooperate*.

References
----------
Talbi, E.-G. (2002). A taxonomy of hybrid metaheuristics. *Journal of
Heuristics*, 8, 541–564.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type

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
from .base import BaseOptimiser
from .ensemble import EnsembleOptimiser
from .ensembles import (
    AdaptiveEnsembleOptimiser,
    CooperativeCoevolutionOptimiser,
    IslandModelOptimiser,
    MemeticOptimiser,
)
from .genetic import GeneticOptimiser
from .local_search import LocalSearchOptimiser
from .pso import PSOOptimiser
from .sa import DBMOSAOptimiser, SimulatedAnnealingOptimiser

#: Where an algorithm's idea comes from.
FAMILIES = {
    "evolutionary": "Variation + selection on a population (GA, DE, ES)",
    "swarm": "Agents following leaders / each other (PSO, GWO, ACO, ...)",
    "physics": "Physical processes (annealing)",
    "human": "Human social behaviour (teaching-learning, Jaya)",
    "music": "Musical improvisation (Harmony Search)",
    "mathematical": "Mathematical functions (Sine Cosine)",
    "trajectory": "Single-solution neighbourhood search (local / tabu search)",
}

#: Inductive biases and properties an optimiser may have.
BIASES = {
    "rotation_invariant": "Behaviour unchanged by rotating the coordinate system",
    "separability_exploiting": "Coordinate-wise moves; strong on separable, weak on rotated problems",
    "centre_biased": "Drifts towards the centre of the box; flattered by centred optima",
    "parameter_free": "No algorithm-specific parameters to tune",
    "self_adaptive": "Adapts its own step sizes / parameters online",
    "elitist": "Never loses the best solution found",
    "schedule_driven": "Parameters follow a schedule over the run budget",
    "uses_problem_heuristic": "Can exploit problem-specific desirability information",
}


@dataclass(frozen=True)
class OptimiserCard:
    """Structured description of one optimiser."""

    key: str
    cls: Type[BaseOptimiser]
    name: str
    family: str
    search: str  # 'population' | 'trajectory' | 'constructive'
    encodings: Tuple[str, ...]
    year: int
    reference: str
    mechanisms: FrozenSet[str] = frozenset()
    biases: FrozenSet[str] = frozenset()
    encoding_param: Optional[str] = None  # constructor kwarg selecting the encoding
    single_objective: bool = True
    defaults: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "optimiser"

    def supports(self, encoding: str) -> bool:
        return encoding in self.encodings

    @property
    def warm_startable(self) -> bool:
        params = inspect.signature(self.cls.optimise).parameters
        return "initial_solutions" in params or "initial_solution" in params

    @property
    def budget_control(self) -> bool:
        return "max_evaluations" in inspect.signature(self.cls.optimise).parameters

    def describe(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cls"] = self.cls.__name__
        d["mechanisms"] = sorted(self.mechanisms)
        d["biases"] = sorted(self.biases)
        d["warm_startable"] = self.warm_startable
        d["budget_control"] = self.budget_control
        d["kind"] = self.kind
        return d


@dataclass(frozen=True)
class EnsembleCard:
    """Structured description of one ensemble strategy (Talbi taxonomy)."""

    key: str
    cls: Type[BaseOptimiser]
    name: str
    category: str
    level: str  # 'high' | 'low'
    mode: str  # 'relay' | 'teamwork'
    cooperation: str
    adaptive: bool
    year: int
    reference: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    members_param: str = "optimisers"

    @property
    def kind(self) -> str:
        return "ensemble"

    @property
    def talbi_class(self) -> str:
        return f"{'L' if self.level == 'low' else 'H'}{'R' if self.mode == 'relay' else 'T'}H"

    def describe(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cls"] = self.cls.__name__
        d["talbi_class"] = self.talbi_class
        d["kind"] = self.kind
        return d


_R = "real"
_B = "binary"
_P = "permutation"

OPTIMISER_CARDS: Dict[str, OptimiserCard] = {c.key: c for c in [
    OptimiserCard("genetic", GeneticOptimiser, "Genetic Algorithm", "evolutionary", "population",
                  (_R, _B, _P), 1975, "Holland (1975)",
                  frozenset({"crossover", "mutation", "roulette_selection"}), frozenset({"elitist"}),
                  encoding_param="encoding"),
    OptimiserCard("de", DifferentialEvolutionOptimiser, "Differential Evolution", "evolutionary",
                  "population", (_R,), 1997, "Storn & Price (1997)",
                  frozenset({"difference_vector_mutation", "binomial_crossover", "greedy_replacement"}),
                  frozenset({"elitist", "separability_exploiting"})),
    OptimiserCard("cmaes", CMAESOptimiser, "CMA-ES", "evolutionary", "population", (_R,), 2001,
                  "Hansen & Ostermeier (2001)",
                  frozenset({"gaussian_sampling", "covariance_adaptation", "step_size_adaptation"}),
                  frozenset({"rotation_invariant", "self_adaptive"})),
    OptimiserCard("pso", PSOOptimiser, "Particle Swarm Optimisation", "swarm", "population", (_R,),
                  1995, "Kennedy & Eberhart (1995)",
                  frozenset({"velocity", "personal_best", "global_best"}), frozenset({"elitist"})),
    OptimiserCard("gwo", GreyWolfOptimiser, "Grey Wolf Optimizer", "swarm", "population", (_R,), 2014,
                  "Mirjalili et al. (2014)", frozenset({"leader_following", "encircling"}),
                  frozenset({"centre_biased", "schedule_driven", "elitist"})),
    OptimiserCard("woa", WhaleOptimiser, "Whale Optimization Algorithm", "swarm", "population", (_R,),
                  2016, "Mirjalili & Lewis (2016)",
                  frozenset({"leader_following", "encircling", "spiral_move"}),
                  frozenset({"centre_biased", "schedule_driven"})),
    OptimiserCard("firefly", FireflyOptimiser, "Firefly Algorithm", "swarm", "population", (_R,), 2009,
                  "Yang (2009)", frozenset({"attraction", "random_walk"}), frozenset()),
    OptimiserCard("cuckoo", CuckooSearchOptimiser, "Cuckoo Search", "swarm", "population", (_R,), 2009,
                  "Yang & Deb (2009)", frozenset({"levy_flight", "abandonment", "greedy_replacement"}),
                  frozenset({"elitist"})),
    OptimiserCard("abc", ArtificialBeeColonyOptimiser, "Artificial Bee Colony", "swarm", "population",
                  (_R,), 2007, "Karaboga & Basturk (2007)",
                  frozenset({"coordinate_perturbation", "fitness_proportional_selection", "scout_restart"}),
                  frozenset({"separability_exploiting", "elitist"})),
    OptimiserCard("bat", BatOptimiser, "Bat Algorithm", "swarm", "population", (_R,), 2010,
                  "Yang (2010)", frozenset({"velocity", "frequency_tuning", "local_random_walk"}),
                  frozenset()),
    OptimiserCard("acor", ACOROptimiser, "Ant Colony for Continuous Domains", "swarm", "population",
                  (_R,), 2008, "Socha & Dorigo (2008)",
                  frozenset({"solution_archive", "gaussian_kernel_sampling"}), frozenset({"elitist"})),
    OptimiserCard("aco", AntColonyOptimiser, "Rank-based Ant System", "swarm", "constructive", (_P,),
                  1999, "Bullnheimer et al. (1999)",
                  frozenset({"pheromone", "solution_construction", "evaporation"}),
                  frozenset({"uses_problem_heuristic", "elitist"})),
    OptimiserCard("harmony", HarmonySearchOptimiser, "Harmony Search", "music", "population", (_R,),
                  2001, "Geem et al. (2001)",
                  frozenset({"memory_consideration", "pitch_adjustment"}),
                  frozenset({"separability_exploiting", "schedule_driven", "elitist"})),
    OptimiserCard("tlbo", TLBOOptimiser, "Teaching-Learning-Based Optimization", "human", "population",
                  (_R,), 2011, "Rao et al. (2011)",
                  frozenset({"mean_attraction", "peer_learning", "greedy_replacement"}),
                  frozenset({"parameter_free", "elitist"})),
    OptimiserCard("jaya", JayaOptimiser, "Jaya", "human", "population", (_R,), 2016, "Rao (2016)",
                  frozenset({"best_attraction", "worst_repulsion", "greedy_replacement"}),
                  frozenset({"parameter_free", "elitist"})),
    OptimiserCard("sca", SineCosineOptimiser, "Sine Cosine Algorithm", "mathematical", "population",
                  (_R,), 2016, "Mirjalili (2016)", frozenset({"oscillation", "leader_following"}),
                  frozenset({"centre_biased", "schedule_driven"})),
    OptimiserCard("sa", SimulatedAnnealingOptimiser, "Simulated Annealing", "physics", "trajectory",
                  (_R,), 1983, "Kirkpatrick et al. (1983)",
                  frozenset({"metropolis_acceptance", "cooling_schedule"}), frozenset({"schedule_driven"})),
    OptimiserCard("local_search", LocalSearchOptimiser, "Best-improvement Local Search", "trajectory",
                  "trajectory", (_R, _B), 1958, "Croes (1958)",
                  frozenset({"neighbourhood_enumeration", "best_improvement"}),
                  frozenset({"elitist", "separability_exploiting"}), encoding_param="encoding"),
    OptimiserCard("tabu", TabuSearchOptimiser, "Tabu Search", "trajectory", "trajectory", (_R, _B, _P),
                  1989, "Glover (1989)",
                  frozenset({"neighbourhood_sampling", "tabu_memory", "aspiration"}), frozenset(),
                  encoding_param="encoding"),
    OptimiserCard("dbmosa", DBMOSAOptimiser, "Dominance-Based Multi-Objective SA", "physics",
                  "trajectory", (_R,), 2008, "Smith et al. (2008)",
                  frozenset({"pareto_archive", "metropolis_acceptance"}), frozenset(),
                  single_objective=False),
]}

ENSEMBLE_CARDS: Dict[str, EnsembleCard] = {c.key: c for c in [
    EnsembleCard("portfolio", EnsembleOptimiser, "Algorithm portfolio (best-of)", "portfolio",
                 "high", "teamwork", "none", False, 2001, "Gomes & Selman (2001)",
                 defaults={"strategy": "best"}),
    EnsembleCard("relay", EnsembleOptimiser, "Sequential relay (chain)", "relay", "high", "relay",
                 "sequential_handover", False, 2002, "Talbi (2002)", defaults={"strategy": "chain"}),
    EnsembleCard("multistart", EnsembleOptimiser, "Multi-start", "multistart", "high", "teamwork",
                 "none", False, 1987, "Boender & Rinnooy Kan (1987)",
                 defaults={"strategy": "random_restart"}),
    EnsembleCard("island", IslandModelOptimiser, "Island model", "island", "high", "teamwork",
                 "migration", False, 1999, "Whitley et al. (1999)"),
    EnsembleCard("adaptive", AdaptiveEnsembleOptimiser, "Bandit selection hyper-heuristic",
                 "selection_hyper_heuristic", "high", "relay", "shared_pool", True, 2013,
                 "Burke et al. (2013)"),
    EnsembleCard("memetic", MemeticOptimiser, "Memetic algorithm", "memetic", "low", "relay",
                 "embedded_refinement", False, 1989, "Moscato (1989)", members_param="global_optimiser"),
    EnsembleCard("cooperative", CooperativeCoevolutionOptimiser, "Cooperative co-evolution",
                 "decomposition", "low", "teamwork", "context_vector", False, 1994,
                 "Potter & De Jong (1994)"),
]}

CARDS: Dict[str, Any] = {**OPTIMISER_CARDS, **ENSEMBLE_CARDS}


def get_card(key: str):
    if key not in CARDS:
        raise KeyError(f"unknown optimiser/ensemble {key!r}; known: {sorted(CARDS)}")
    return CARDS[key]


def configure_for(card: OptimiserCard, params: Dict[str, Any], encoding: str) -> Dict[str, Any]:
    """Return constructor params with the encoding selected for ``encoding``."""
    params = dict(card.defaults, **params)
    if card.encoding_param and encoding != "real":
        params.setdefault(card.encoding_param, encoding)
    return params


def taxonomy_tree() -> Dict[str, Any]:
    """Nested summary used by docs and the dashboard."""
    from .problems import PROBLEMS

    by_family: Dict[str, List[str]] = {}
    for c in OPTIMISER_CARDS.values():
        by_family.setdefault(c.family, []).append(c.key)
    by_talbi: Dict[str, List[str]] = {}
    for c in ENSEMBLE_CARDS.values():
        by_talbi.setdefault(c.talbi_class, []).append(c.key)
    problems: Dict[str, List[str]] = {}
    for p in PROBLEMS.values():
        problems.setdefault(p.category, []).append(p.id)
    return {
        "optimiser_families": by_family,
        "ensemble_talbi_classes": by_talbi,
        "problem_categories": problems,
        "families": FAMILIES,
        "biases": BIASES,
    }
