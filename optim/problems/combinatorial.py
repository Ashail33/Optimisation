"""
Combinatorial benchmark problems (binary and permutation encodings).

=============  ===========  =====  ===========================================
Family         Encoding     Sense  Optimum
=============  ===========  =====  ===========================================
``tsp``        permutation  min    best-known (from the benchmark)
``knapsack``   binary       max    exact (dynamic programming)
``onemax``     binary       max    ``dim``
``trap``       binary       max    ``dim`` (deceptive)
``nk``         binary       max    exact by enumeration when ``dim <= 16``
``maxcut``     binary       max    exact by enumeration when ``dim <= 16``
=============  ===========  =====  ===========================================
"""

from __future__ import annotations

import numpy as np

from .base import ProblemInstance, instance_name, instance_rng, register_problem

_ENUMERATION_LIMIT = 16


def _all_bitstrings(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


# ---------------------------------------------------------------------------
# Travelling salesman
# ---------------------------------------------------------------------------

@register_problem("tsp", category="combinatorial", encoding="permutation", sense="min",
                  tags={"routing", "multimodal", "non_separable"}, default_dim=20)
def tsp(dim: int, instance: int = 1, layout: str = "uniform") -> ProblemInstance:
    """Euclidean travelling salesman tour (closed) over random cities."""
    if layout not in ("uniform", "clustered"):
        raise ValueError("layout must be 'uniform' or 'clustered'")
    rng = instance_rng("tsp", dim, instance)
    if layout == "uniform":
        pts = rng.random((dim, 2))
    else:
        centres = rng.random((max(2, dim // 8), 2))
        pts = centres[rng.integers(len(centres), size=dim)] + rng.normal(0, 0.05, (dim, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)

    def length(tour) -> float:
        t = np.asarray(tour, dtype=int)
        return float(dist[t, np.roll(t, -1)].sum())

    with np.errstate(divide="ignore"):
        eta = np.where(dist > 0, 1.0 / dist, 0.0)
    return ProblemInstance(
        name=instance_name("tsp", dim, instance, layout=layout),
        family="tsp", category="combinatorial", encoding="permutation", sense="min",
        dim=dim, objective=length,
        tags=frozenset({"routing", "multimodal", "non_separable"}),
        params={"layout": layout}, instance=instance,
        extras={"heuristic": eta},
        description=f"{dim}-city {layout} Euclidean TSP",
    )


# ---------------------------------------------------------------------------
# 0/1 knapsack
# ---------------------------------------------------------------------------

def _knapsack_dp(weights: np.ndarray, profits: np.ndarray, capacity: int):
    n = len(weights)
    best = np.zeros(capacity + 1)
    keep = np.zeros((n, capacity + 1), dtype=bool)
    for i in range(n):
        w, p = int(weights[i]), profits[i]
        if w > capacity:
            continue
        cand = best[: capacity + 1 - w] + p
        better = cand > best[w:]
        keep[i, w:] = better
        best[w:] = np.where(better, cand, best[w:])
    x = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            x[i] = 1
            c -= int(weights[i])
    return float(best[capacity]), x.tolist()


@register_problem("knapsack", category="combinatorial", encoding="binary", sense="max",
                  tags={"packing", "constrained", "profit_linear"}, default_dim=50)
def knapsack(dim: int, instance: int = 1, correlation: str = "uncorrelated",
             capacity_ratio: float = 0.5) -> ProblemInstance:
    """0/1 knapsack: maximise packed profit within a weight capacity."""
    kinds = ("uncorrelated", "weak", "strong", "subset_sum")
    if correlation not in kinds:
        raise ValueError(f"correlation must be one of {kinds}")
    rng = instance_rng("knapsack", dim, instance)
    w = rng.integers(1, 101, dim)
    if correlation == "uncorrelated":
        p = rng.integers(1, 101, dim)
    elif correlation == "weak":
        p = np.clip(w + rng.integers(-10, 11, dim), 1, None)
    elif correlation == "strong":
        p = w + 10
    else:
        p = w.copy()
    p = p.astype(float)
    cap = int(capacity_ratio * w.sum())
    opt, opt_x = _knapsack_dp(w, p, cap)

    def value(x) -> float:
        x = np.asarray(x, dtype=float) > 0.5
        weight = w[x].sum()
        # Infeasible packings score the (negative) overweight, so they are
        # always worse than any feasible packing.
        return float(p[x].sum()) if weight <= cap else float(cap - weight)

    return ProblemInstance(
        name=instance_name("knapsack", dim, instance, corr=correlation, cap=capacity_ratio),
        family="knapsack", category="combinatorial", encoding="binary", sense="max",
        dim=dim, objective=value, optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"packing", "constrained", "profit_linear"}),
        params={"correlation": correlation, "capacity_ratio": capacity_ratio},
        instance=instance,
        description=f"{dim}-item {correlation} knapsack",
    )


# ---------------------------------------------------------------------------
# Pseudo-Boolean functions
# ---------------------------------------------------------------------------

@register_problem("onemax", category="combinatorial", encoding="binary", sense="max",
                  tags={"pseudo_boolean", "unimodal", "separable"}, default_dim=50)
def onemax(dim: int, instance: int = 1) -> ProblemInstance:
    """OneMax: number of ones in the bit string (unimodal sanity check)."""
    return ProblemInstance(
        name=instance_name("onemax", dim, instance), family="onemax",
        category="combinatorial", encoding="binary", sense="max", dim=dim,
        objective=lambda x: float(np.sum(np.asarray(x) > 0.5)),
        optimum=float(dim), optimum_solution=[1] * dim,
        tags=frozenset({"pseudo_boolean", "unimodal", "separable"}), instance=instance,
        description=f"OneMax on {dim} bits",
    )


@register_problem("trap", category="combinatorial", encoding="binary", sense="max",
                  tags={"pseudo_boolean", "deceptive", "multimodal"}, default_dim=50)
def trap(dim: int, instance: int = 1, k: int = 5) -> ProblemInstance:
    """Concatenated deceptive trap functions of order ``k``."""
    if dim % k:
        raise ValueError("dim must be a multiple of k")

    def value(x) -> float:
        u = (np.asarray(x) > 0.5).reshape(-1, k).sum(axis=1)
        return float(np.where(u == k, k, k - 1 - u).sum())

    return ProblemInstance(
        name=instance_name("trap", dim, instance, k=k), family="trap",
        category="combinatorial", encoding="binary", sense="max", dim=dim,
        objective=value, optimum=float(dim), optimum_solution=[1] * dim,
        tags=frozenset({"pseudo_boolean", "deceptive", "multimodal"}),
        params={"k": k}, instance=instance,
        description=f"{dim // k} concatenated order-{k} traps",
    )


@register_problem("nk", category="combinatorial", encoding="binary", sense="max",
                  tags={"pseudo_boolean", "rugged", "multimodal", "non_separable"}, default_dim=16)
def nk(dim: int, instance: int = 1, k: int = 2) -> ProblemInstance:
    """NK landscape: tunable ruggedness via ``k`` epistatic interactions."""
    if not 0 <= k < dim:
        raise ValueError("k must be in [0, dim)")
    rng = instance_rng("nk", dim, instance, salt=k)
    links = np.array([np.concatenate([[i], rng.choice(np.delete(np.arange(dim), i), k, replace=False)])
                      for i in range(dim)])
    tables = rng.random((dim, 2 ** (k + 1)))
    powers = 2 ** np.arange(k + 1)

    def batch(X: np.ndarray) -> np.ndarray:
        idx = (X[:, links] * powers).sum(axis=2)  # (m, dim)
        return tables[np.arange(dim), idx].mean(axis=1)

    opt = opt_x = None
    if dim <= _ENUMERATION_LIMIT:
        allx = _all_bitstrings(dim)
        vals = batch(allx)
        opt, opt_x = float(vals.max()), allx[int(vals.argmax())].tolist()

    return ProblemInstance(
        name=instance_name("nk", dim, instance, k=k), family="nk",
        category="combinatorial", encoding="binary", sense="max", dim=dim,
        objective=lambda x: float(batch((np.asarray(x)[None, :] > 0.5).astype(int))[0]),
        optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"pseudo_boolean", "rugged", "multimodal", "non_separable"}),
        params={"k": k}, instance=instance,
        description=f"NK landscape N={dim}, K={k}",
    )


@register_problem("maxcut", category="combinatorial", encoding="binary", sense="max",
                  tags={"graph", "multimodal", "non_separable"}, default_dim=16)
def maxcut(dim: int, instance: int = 1, density: float = 0.5) -> ProblemInstance:
    """Max-cut on a random unweighted graph."""
    rng = instance_rng("maxcut", dim, instance)
    upper = np.triu(rng.random((dim, dim)) < density, 1)
    A = (upper | upper.T).astype(float)

    def batch(X: np.ndarray) -> np.ndarray:
        s = 2.0 * X - 1.0
        return 0.25 * (A.sum() - np.einsum("mi,ij,mj->m", s, A, s))

    opt = opt_x = None
    if dim <= _ENUMERATION_LIMIT:
        allx = _all_bitstrings(dim)
        vals = batch(allx)
        opt, opt_x = float(vals.max()), allx[int(vals.argmax())].tolist()

    return ProblemInstance(
        name=instance_name("maxcut", dim, instance, density=density), family="maxcut",
        category="combinatorial", encoding="binary", sense="max", dim=dim,
        objective=lambda x: float(batch((np.asarray(x)[None, :] > 0.5).astype(float))[0]),
        optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"graph", "multimodal", "non_separable"}),
        params={"density": density}, instance=instance,
        description=f"Max-cut on G({dim}, {density})",
    )
