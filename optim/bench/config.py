"""
Benchmark configuration: a YAML/dict spec expanded into a task grid.

Example (YAML)::

    name: smoke
    budget: {per_dim: 500}         # or {evaluations: 5000}
    repeats: 3                     # independent runs per (optimiser, instance)
    seed: 0
    random_keys: true              # let real-valued optimisers solve discrete problems
    problems:
      - family: rastrigin
        dim: [5, 10]
        instances: 2               # instance ids 1..2  (or a list [1, 5])
        params: {rotate: [false, true]}
      - family: knapsack
        dim: 30
        params: {correlation: [uncorrelated, strong]}
    optimisers:
      - {name: DE, entry: de, params: {strategy: rand/1}}
      - name: Island(DE+GWO)
        entry: island
        params:
          optimisers: [{entry: de}, {entry: gwo}]
          n_epochs: 5

Every list value in a problem's ``dim`` / ``params`` is a grid axis (the
Cartesian product is taken, as in clustbench).  Optimiser ``params`` may
contain nested ``{entry: ..., params: ...}`` specs — that is how ensemble
members are declared.
"""

from __future__ import annotations

import itertools
import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..problems import PROBLEMS, ProblemInstance, as_random_key, make_problem
from ..taxonomy import CARDS, EnsembleCard, OptimiserCard, configure_for, get_card


@dataclass(frozen=True)
class ProblemSpec:
    """A concrete problem instance, rebuildable anywhere from this spec."""

    family: str
    dim: int
    instance: int
    params: Tuple[Tuple[str, Any], ...] = ()

    def build(self) -> ProblemInstance:
        return make_problem(self.family, dim=self.dim, instance=self.instance, **dict(self.params))

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "dim": self.dim, "instance": self.instance,
                "params": dict(self.params)}


@dataclass
class OptimiserSpec:
    """An optimiser or ensemble declared in the config."""

    name: str
    entry: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def card(self):
        return get_card(self.entry)

    def encodings(self) -> Tuple[str, ...]:
        """Encodings this spec can solve natively (ensembles: all members)."""
        return _encodings(self.entry, self.params)

    def build(self, encoding: str):
        """Instantiate for problems of ``encoding``."""
        return _build(self.entry, self.params, encoding)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "entry": self.entry, "params": self.params}


@dataclass
class Task:
    problem: ProblemSpec
    optimiser: OptimiserSpec
    repeat: int
    budget: int
    seed: int
    random_keys: bool = True
    record_trajectory: bool = False


@dataclass
class BenchmarkConfig:
    name: str
    problems: List[ProblemSpec]
    optimisers: List[OptimiserSpec]
    budget: Dict[str, int]
    repeats: int = 3
    seed: int = 0
    random_keys: bool = True
    record_trajectory: bool = False
    n_jobs: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)

    def budget_for(self, problem: ProblemSpec) -> int:
        if "evaluations" in self.budget:
            return int(self.budget["evaluations"])
        return int(self.budget["per_dim"]) * problem.dim

    def tasks(self) -> List[Task]:
        out = []
        for p, o, r in itertools.product(self.problems, self.optimisers, range(self.repeats)):
            out.append(Task(
                problem=p, optimiser=o, repeat=r, budget=self.budget_for(p),
                seed=derive_seed(self.seed, o.name, p, r),
                random_keys=self.random_keys, record_trajectory=self.record_trajectory,
            ))
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(source: Union[str, Path, Dict[str, Any]]) -> BenchmarkConfig:
    """Load a benchmark config from a YAML/JSON file path or a dict."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text()
        if str(source).endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("YAML configs need PyYAML: pip install 'optim[bench]'") from exc
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
    else:
        raw = dict(source)

    problems: List[ProblemSpec] = []
    for entry in raw.get("problems", []):
        problems.extend(_expand_problem(entry))
    optimisers = [
        OptimiserSpec(name=o.get("name", o["entry"]), entry=o["entry"], params=o.get("params", {}) or {})
        for o in raw.get("optimisers", [])
    ]
    if not problems or not optimisers:
        raise ValueError("config needs at least one problem and one optimiser")
    names = [o.name for o in optimisers]
    if len(set(names)) != len(names):
        raise ValueError("optimiser names must be unique")
    for o in optimisers:
        _validate(o.entry, o.params)

    budget = raw.get("budget", {"per_dim": 1000})
    if isinstance(budget, int):
        budget = {"evaluations": budget}
    if not ({"evaluations", "per_dim"} & set(budget)):
        raise ValueError("budget must give 'evaluations' or 'per_dim'")

    return BenchmarkConfig(
        name=raw.get("name", "benchmark"),
        problems=problems,
        optimisers=optimisers,
        budget=budget,
        repeats=int(raw.get("repeats", 3)),
        seed=int(raw.get("seed", 0)),
        random_keys=bool(raw.get("random_keys", True)),
        record_trajectory=bool(raw.get("record_trajectory", False)),
        n_jobs=int(raw.get("n_jobs", 1)),
        raw=raw,
    )


def _as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _expand_problem(entry: Dict[str, Any]) -> List[ProblemSpec]:
    family = entry["family"]
    if family not in PROBLEMS:
        raise KeyError(f"unknown problem family {family!r}; known: {sorted(PROBLEMS)}")
    dims = _as_list(entry.get("dim", PROBLEMS[family].default_dim))
    inst = entry.get("instances", 1)
    instances = list(range(1, inst + 1)) if isinstance(inst, int) else list(inst)
    params = entry.get("params", {}) or {}
    keys = sorted(params)
    grids = [_as_list(params[k]) for k in keys]
    out = []
    for d, i, values in itertools.product(dims, instances, itertools.product(*grids)):
        out.append(ProblemSpec(family, int(d), int(i), tuple(zip(keys, values))))
    return out


def derive_seed(base: int, name: str, problem: ProblemSpec, repeat: int) -> int:
    """Order-independent run seed (stable across processes and machines)."""
    key = json.dumps([base, name, problem.to_dict(), repeat], sort_keys=True, default=str)
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Building optimisers (recursively, for ensembles)
# ---------------------------------------------------------------------------

def _is_spec(v: Any) -> bool:
    return isinstance(v, dict) and "entry" in v


def _members(params: Dict[str, Any]):
    for v in params.values():
        if _is_spec(v):
            yield v
        elif isinstance(v, list):
            yield from (x for x in v if _is_spec(x))


def _validate(entry: str, params: Dict[str, Any]) -> None:
    card = get_card(entry)
    if isinstance(card, OptimiserCard) and not card.single_objective:
        raise ValueError(f"{entry!r} is multi-objective and cannot be benchmarked on scalar problems")
    for m in _members(params):
        _validate(m["entry"], m.get("params", {}) or {})


def _encodings(entry: str, params: Dict[str, Any]) -> Tuple[str, ...]:
    card = get_card(entry)
    if isinstance(card, OptimiserCard):
        return card.encodings
    members = list(_members(params))
    if not members:
        return ("real",)
    common = set(_encodings(members[0]["entry"], members[0].get("params", {}) or {}))
    for m in members[1:]:
        common &= set(_encodings(m["entry"], m.get("params", {}) or {}))
    # Cooperative co-evolution decomposes box-bounded vectors only.
    if entry == "cooperative":
        common &= {"real"}
    return tuple(sorted(common))


def _build_value(v: Any, encoding: str) -> Any:
    if _is_spec(v):
        return _build(v["entry"], v.get("params", {}) or {}, encoding)
    if isinstance(v, list):
        return [_build_value(x, encoding) for x in v]
    return v


def _build(entry: str, params: Dict[str, Any], encoding: str):
    card = get_card(entry)
    built = {k: _build_value(v, encoding) for k, v in params.items()}
    if isinstance(card, OptimiserCard):
        built = configure_for(card, built, encoding)
    else:
        built = dict(card.defaults, **built)
    return card.cls(**built)


def prepare(task: Task) -> Tuple[ProblemInstance, ProblemInstance, Any, str]:
    """Build ``(native_problem, solved_problem, optimiser, solved_as)``.

    When the optimiser cannot handle the problem's encoding natively and
    ``random_keys`` is on, the problem is wrapped as a random-key real
    problem.  Raises ``ValueError`` when the pair is incompatible.
    """
    native = task.problem.build()
    encodings = task.optimiser.encodings()
    if native.encoding in encodings:
        return native, native, task.optimiser.build(native.encoding), "native"
    if task.random_keys and "real" in encodings and native.encoding != "real":
        return native, as_random_key(native), task.optimiser.build("real"), "random_key"
    raise ValueError(
        f"{task.optimiser.name} cannot solve {native.encoding} problems"
        + ("" if task.random_keys else " (random_keys is off)")
    )
