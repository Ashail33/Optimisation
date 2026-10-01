"""
Random-key encodings: let real-valued optimisers solve discrete problems.

A random-key vector ``x in [0, 1]^n`` is decoded to

* a **bit string** by thresholding at 0.5, or
* a **permutation** by ``argsort(x)`` (the rank order of the keys).

:func:`as_random_key` wraps a binary or permutation :class:`ProblemInstance`
as an equivalent real-valued one, so all continuous metaheuristics (and
continuous ensembles) can be benchmarked on combinatorial problems.

Reference: Bean, J. C. (1994). Genetics and random keys for sequencing and
optimization. *ORSA Journal on Computing*, 6(2), 154–160.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Sequence

import numpy as np

from .base import ProblemInstance


def decode(keys: Sequence[float], encoding: str) -> List[Any]:
    """Decode a random-key vector into ``encoding`` (binary/permutation)."""
    keys = np.asarray(keys, dtype=float)
    if encoding == "binary":
        return (keys > 0.5).astype(int).tolist()
    if encoding == "permutation":
        return np.argsort(keys, kind="stable").tolist()
    if encoding == "real":
        return keys.tolist()
    raise ValueError(f"unknown encoding {encoding!r}")


def as_random_key(problem: ProblemInstance) -> ProblemInstance:
    """Return a real-valued view of a discrete problem over ``[0, 1]^dim``."""
    if problem.encoding == "real":
        return problem
    enc = problem.encoding
    objective, true_objective = problem.objective, problem.true_objective
    return replace(
        problem,
        name=problem.name + "-rk",
        encoding="real",
        objective=lambda x: objective(decode(x, enc)),
        true_objective=lambda x: true_objective(decode(x, enc)),
        bounds=[(0.0, 1.0)] * problem.dim,
        optimum_solution=None,
        extras={},
        params={**problem.params, "decoded_from": enc},
    )
