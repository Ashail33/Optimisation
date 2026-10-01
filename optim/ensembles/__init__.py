"""
Ensemble strategies — ways of combining several optimisers.

=================================================  ==========================================
Class                                              Ensemble type
=================================================  ==========================================
:class:`EnsembleOptimiser` (``'best'``)            Portfolio: independent runs, keep the best
:class:`EnsembleOptimiser` (``'chain'``)           High-level relay hybrid (sequential hand-off)
:class:`EnsembleOptimiser` (``'random_restart'``)  Multi-start of one algorithm
:class:`IslandModelOptimiser`                      Cooperative islands with migration
:class:`AdaptiveEnsembleOptimiser`                 Hyper-heuristic / bandit algorithm selection
:class:`MemeticOptimiser`                          Low-level hybrid: global search + local refinement
:class:`CooperativeCoevolutionOptimiser`           Variable decomposition, sub-problems in turn
=================================================  ==========================================

All of them are themselves :class:`~optim.base.BaseOptimiser` instances and
return an :class:`~optim.ensemble.EnsembleResult`, so ensembles can be
nested (e.g. a memetic algorithm on each island).
"""

from ..ensemble import EnsembleOptimiser, EnsembleResult
from .adaptive import AdaptiveEnsembleOptimiser
from .cooperative import CooperativeCoevolutionOptimiser
from .island import IslandModelOptimiser
from .memetic import MemeticOptimiser

__all__ = [
    "AdaptiveEnsembleOptimiser",
    "CooperativeCoevolutionOptimiser",
    "EnsembleOptimiser",
    "EnsembleResult",
    "IslandModelOptimiser",
    "MemeticOptimiser",
]
