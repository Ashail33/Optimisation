"""
Problem abstraction and registry.

A *problem family* (e.g. ``rastrigin``, ``tsp``, ``pricing``) is a factory
registered with :func:`register_problem`.  Calling it with a size, an
``instance`` number and family-specific parameters builds a concrete,
fully reproducible :class:`ProblemInstance`: the same arguments always give
the same instance, which is what lets a benchmark be re-created in a worker
process or by an external program from its spec alone.

Every instance carries the metadata analysis needs:

* ``sense``  — ``'min'`` (cost) or ``'max'`` (profit);
* ``encoding`` — ``'real'``, ``'binary'`` or ``'permutation'``;
* ``optimum`` — the known optimal value, when there is one (otherwise the
  benchmark uses the best value any optimiser found: the *best-known*);
* ``tags`` — landscape / structure descriptors drawn from :data:`TAGS`
  (modality, separability, conditioning, noise, constraints, profit type);
* ``random_reference`` — the median value of random solutions, used as the
  "zero skill" end of the normalised score.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Vocabulary for problem tags.  Kept small and fixed so tags can be used as
#: features (fingerprints) and as dashboard filters.
TAGS = {
    # modality / landscape
    "unimodal", "multimodal", "deceptive", "plateaus", "rugged",
    # structure
    "separable", "non_separable", "ill_conditioned", "rotated", "shifted",
    # uncertainty and feasibility
    "noisy", "constrained",
    # profit-function types (applied problems)
    "profit_linear", "profit_concave", "profit_quadratic",
    "profit_risk_adjusted", "profit_stochastic", "profit_discontinuous",
    # combinatorial structure
    "routing", "packing", "graph", "pseudo_boolean",
}

ENCODINGS = ("real", "binary", "permutation")


@dataclass
class ProblemInstance:
    """A concrete optimisation problem.

    Attributes
    ----------
    name : str
        Unique, human-readable key, e.g. ``"rastrigin-d10-i1"``.
    family : str
        Registered family id, e.g. ``"rastrigin"``.
    category : str
        ``'continuous'``, ``'combinatorial'`` or ``'profit'``.
    encoding : str
        ``'real'``, ``'binary'`` or ``'permutation'``.
    sense : str
        ``'min'`` or ``'max'``.
    dim : int
        Number of decision variables (problem size).
    objective : callable
        The (possibly noisy) objective the optimiser sees.
    bounds : list of (low, high), optional
        Box bounds (``real``); ``[(0, 1)] * dim`` for discrete encodings so
        bound-driven optimisers can infer the size.
    optimum : float, optional
        Known optimal value of ``true_objective``.
    optimum_solution : list, optional
        A solution attaining ``optimum``, when known.
    tags : frozenset of str
    params : dict
        Family parameters used to build the instance (for the record).
    instance : int
        Instance number (seeds the instance's random structure).
    true_objective : callable, optional
        Noise-free objective, used for scoring.  Defaults to ``objective``.
    extras : dict
        Extra hints for optimisers that can use them, e.g. the ``heuristic``
        desirability matrix for ant colony on routing problems.
    description : str
    """

    name: str
    family: str
    category: str
    encoding: str
    sense: str
    dim: int
    objective: Callable[[Sequence], float]
    bounds: Optional[List[Tuple[float, float]]] = None
    optimum: Optional[float] = None
    optimum_solution: Optional[List[Any]] = None
    tags: FrozenSet[str] = frozenset()
    params: Dict[str, Any] = field(default_factory=dict)
    instance: int = 1
    true_objective: Optional[Callable[[Sequence], float]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    _noise_rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        unknown = set(self.tags) - TAGS
        if unknown:
            raise ValueError(f"unknown tags: {sorted(unknown)}")
        if self.true_objective is None:
            self.true_objective = self.objective
        if self.bounds is None and self.encoding != "real":
            self.bounds = [(0.0, 1.0)] * self.dim

    # ------------------------------------------------------------------
    @property
    def maximise(self) -> bool:
        return self.sense == "max"

    @property
    def noisy(self) -> bool:
        return "noisy" in self.tags

    def __call__(self, x: Sequence) -> float:
        return self.objective(x)

    def reseed_noise(self, seed: int) -> None:
        """Re-seed the noise stream of a noisy problem (one stream per run)."""
        if self._noise_rng is not None:
            self._noise_rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state

    def random_solution(self, rng: np.random.Generator) -> List[Any]:
        if self.encoding == "real":
            lo = np.array([b[0] for b in self.bounds])
            hi = np.array([b[1] for b in self.bounds])
            return (lo + rng.random(self.dim) * (hi - lo)).tolist()
        if self.encoding == "binary":
            return rng.integers(0, 2, self.dim).tolist()
        return rng.permutation(self.dim).tolist()

    def random_reference(self, n: int = 200, seed: int = 12345) -> float:
        """Median true objective of ``n`` uniformly random solutions.

        This is the "no skill" anchor of the normalised score: an optimiser
        that does no better than random sampling scores 0.
        """
        rng = np.random.default_rng(seed)
        values = [self.true_objective(self.random_solution(rng)) for _ in range(n)]
        return float(np.median(values))

    def better(self, a: float, b: float) -> bool:
        """``True`` when value ``a`` is strictly better than ``b``."""
        return a > b if self.maximise else a < b

    def optimise_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments every optimiser may find useful."""
        kw: Dict[str, Any] = {"bounds": self.bounds}
        if self.encoding != "real":
            kw["n_genes"] = self.dim
        kw.update(self.extras)
        return kw

    def describe(self) -> Dict[str, Any]:
        """JSON-friendly metadata (no callables)."""
        return {
            "name": self.name,
            "family": self.family,
            "category": self.category,
            "encoding": self.encoding,
            "sense": self.sense,
            "dim": self.dim,
            "optimum": self.optimum,
            "tags": sorted(self.tags),
            "params": self.params,
            "instance": self.instance,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemFamily:
    """Registry entry describing a problem family."""

    id: str
    factory: Callable[..., ProblemInstance]
    category: str
    encoding: str
    sense: str
    tags: FrozenSet[str]
    description: str
    default_dim: int
    parameters: Dict[str, Any]

    def make(self, dim: Optional[int] = None, instance: int = 1, **params: Any) -> ProblemInstance:
        return self.factory(dim=dim or self.default_dim, instance=instance, **params)

    def describe(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "encoding": self.encoding,
            "sense": self.sense,
            "tags": sorted(self.tags),
            "description": self.description,
            "default_dim": self.default_dim,
            "parameters": {k: repr(v) for k, v in self.parameters.items()},
        }


PROBLEMS: Dict[str, ProblemFamily] = {}


def register_problem(
    id: str,
    *,
    category: str,
    encoding: str,
    sense: str,
    tags: Iterable[str] = (),
    default_dim: int = 10,
):
    """Decorator registering a factory ``fn(dim, instance, **params)``.

    The family's parameters (and their defaults) are read from the
    factory's signature; the first line of its docstring is the
    description.
    """

    def deco(fn: Callable[..., ProblemInstance]) -> Callable[..., ProblemInstance]:
        sig = inspect.signature(fn)
        parameters = {
            k: p.default for k, p in sig.parameters.items()
            if k not in ("dim", "instance")
        }
        unknown = set(tags) - TAGS
        if unknown:
            raise ValueError(f"unknown tags for {id}: {sorted(unknown)}")
        PROBLEMS[id] = ProblemFamily(
            id=id,
            factory=fn,
            category=category,
            encoding=encoding,
            sense=sense,
            tags=frozenset(tags),
            description=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
            default_dim=default_dim,
            parameters=parameters,
        )
        return fn

    return deco


def make_problem(family: str, dim: Optional[int] = None, instance: int = 1, **params: Any) -> ProblemInstance:
    """Build a problem instance from the registry.

    >>> p = make_problem("rastrigin", dim=5, instance=2, rotate=True)
    >>> p.sense, p.encoding, p.optimum
    ('min', 'real', 0.0)
    """
    if family not in PROBLEMS:
        raise KeyError(f"unknown problem family {family!r}; known: {sorted(PROBLEMS)}")
    return PROBLEMS[family].make(dim=dim, instance=instance, **params)


def instance_rng(family: str, dim: int, instance: int, salt: int = 0) -> np.random.Generator:
    """Deterministic generator for an instance's random structure."""
    key = sum(ord(c) * (i + 1) for i, c in enumerate(family))
    return np.random.default_rng([key, dim, instance, salt])


def instance_name(family: str, dim: int, instance: int, **params: Any) -> str:
    parts = [family, f"d{dim}"]
    for k in sorted(params):
        v = params[k]
        if isinstance(v, bool):
            if v:
                parts.append(k)
        elif v is not None:
            parts.append(f"{k}{v}")
    parts.append(f"i{instance}")
    return "-".join(str(p) for p in parts)


def penalised(
    profit: Callable[[np.ndarray], float],
    violation: Callable[[np.ndarray], float],
    penalty: float,
    sense: str,
) -> Callable[[np.ndarray], float]:
    """Static-penalty constraint handling: worsen the objective by
    ``penalty * violation`` (subtract for profits, add for costs)."""
    sign = -1.0 if sense == "max" else 1.0

    def f(x):
        x = np.asarray(x, dtype=float)
        v = violation(x)
        return float(profit(x) + sign * penalty * v) if v > 0 else float(profit(x))

    return f


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)
