"""
Applied *profit-function* problems — one family per profit-function type.

=======================  ========================  =================================
Family                   Profit-function type      Optimum
=======================  ========================  =================================
``production_planning``  linear (+ resources)      exact (LP, needs SciPy)
``marketing_budget``     concave, diminishing      exact (KKT water-filling)
                         returns
``pricing``              quadratic (linear demand  exact (linear system)
                         with cross-elasticities)
``portfolio``            risk-adjusted (mean −     exact without cardinality
                         λ·variance)               (convex QP, needs SciPy)
``newsvendor``           stochastic (noisy         exact (critical fractile)
                         simulated profit)
``fixed_charge``         discontinuous (set-up     exact by enumeration for
                         costs)                    ``dim <= 12``
=======================  ========================  =================================

All are *maximised* real-valued problems.  Resource constraints are handled
by ``constraint_handling='penalty'`` (static penalty, the default) or
``'repair'`` (scale the plan back onto the feasible region), so the same
problem can test both styles.
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Optional

import numpy as np

from .base import ProblemInstance, instance_name, instance_rng, penalised, register_problem

_HANDLING = ("penalty", "repair")


def _check_handling(h: str) -> None:
    if h not in _HANDLING:
        raise ValueError(f"constraint_handling must be one of {_HANDLING}")


def _packing_repair(A: np.ndarray, b: np.ndarray):
    """Scale ``x`` down so that ``A x <= b`` (``A >= 0``)."""

    def repair(x: np.ndarray) -> np.ndarray:
        use = A @ x
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(use > b, b / use, 1.0)
        return x * float(ratio.min())

    return repair


def _packing_violation(A: np.ndarray, b: np.ndarray):
    return lambda x: float(np.sum(np.maximum(0.0, A @ x - b) / b))


def _constrained(profit, A, b, handling, penalty):
    if handling == "repair":
        repair = _packing_repair(A, b)
        return lambda x: float(profit(repair(np.asarray(x, dtype=float))))
    return penalised(profit, _packing_violation(A, b), penalty, "max")


# ---------------------------------------------------------------------------
# Linear profit: production planning
# ---------------------------------------------------------------------------

@register_problem("production_planning", category="profit", encoding="real", sense="max",
                  tags={"profit_linear", "constrained"}, default_dim=10)
def production_planning(dim: int, instance: int = 1, n_resources: int = 3,
                        constraint_handling: str = "penalty") -> ProblemInstance:
    """Linear profit: choose production quantities under shared resource limits."""
    _check_handling(constraint_handling)
    rng = instance_rng("production_planning", dim, instance, salt=n_resources)
    margin = rng.uniform(5, 50, dim)
    upper = rng.uniform(50, 150, dim)
    A = rng.uniform(0.5, 5.0, (n_resources, dim))
    b = 0.35 * (A @ upper)
    profit = lambda x: float(margin @ x)
    penalty = 10.0 * float(margin @ upper)

    opt = opt_x = None
    try:
        from scipy.optimize import linprog

        res = linprog(-margin, A_ub=A, b_ub=b, bounds=list(zip(np.zeros(dim), upper)), method="highs")
        if res.success:
            opt, opt_x = float(-res.fun), res.x.tolist()
    except ImportError:  # pragma: no cover - SciPy is optional
        pass

    return ProblemInstance(
        name=instance_name("production_planning", dim, instance, r=n_resources, ch=constraint_handling),
        family="production_planning", category="profit", encoding="real", sense="max", dim=dim,
        objective=_constrained(profit, A, b, constraint_handling, penalty),
        bounds=list(zip([0.0] * dim, upper.tolist())),
        optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"profit_linear", "constrained"}),
        params={"n_resources": n_resources, "constraint_handling": constraint_handling},
        instance=instance,
        description=f"{dim} products, {n_resources} shared resources, linear margins",
    )


# ---------------------------------------------------------------------------
# Concave profit: marketing budget allocation (diminishing returns)
# ---------------------------------------------------------------------------

@register_problem("marketing_budget", category="profit", encoding="real", sense="max",
                  tags={"profit_concave", "constrained"}, default_dim=10)
def marketing_budget(dim: int, instance: int = 1, budget_ratio: float = 0.5,
                     constraint_handling: str = "repair") -> ProblemInstance:
    """Concave profit: split a budget across channels with diminishing returns."""
    _check_handling(constraint_handling)
    rng = instance_rng("marketing_budget", dim, instance)
    a = rng.uniform(20, 100, dim)       # saturation revenue scale
    s = rng.uniform(0.02, 0.2, dim)     # response steepness
    unconstrained = np.maximum(0.0, a - 1.0 / s)
    B = budget_ratio * unconstrained.sum()

    def profit(x):
        return float(np.sum(a * np.log1p(s * x)) - np.sum(x))

    # KKT water-filling: a s / (1 + s x) = 1 + lam
    def alloc(lam):
        return np.maximum(0.0, a / (1.0 + lam) - 1.0 / s)

    lo, hi = 0.0, float(np.max(a * s))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if alloc(mid).sum() > B else (lo, mid)
    x_opt = alloc(hi)

    A, b = np.ones((1, dim)), np.array([B])
    return ProblemInstance(
        name=instance_name("marketing_budget", dim, instance, br=budget_ratio, ch=constraint_handling),
        family="marketing_budget", category="profit", encoding="real", sense="max", dim=dim,
        objective=_constrained(profit, A, b, constraint_handling, penalty=float(a.sum())),
        bounds=[(0.0, float(B))] * dim,
        optimum=profit(x_opt), optimum_solution=x_opt.tolist(),
        tags=frozenset({"profit_concave", "constrained"}),
        params={"budget_ratio": budget_ratio, "constraint_handling": constraint_handling},
        instance=instance,
        description=f"{dim}-channel budget allocation, log response",
    )


# ---------------------------------------------------------------------------
# Quadratic profit: multi-product pricing with cross-elasticities
# ---------------------------------------------------------------------------

@register_problem("pricing", category="profit", encoding="real", sense="max",
                  tags={"profit_quadratic", "non_separable"}, default_dim=10)
def pricing(dim: int, instance: int = 1, cross_elasticity: float = 0.5) -> ProblemInstance:
    """Quadratic profit: set prices for substitute products under linear demand."""
    rng = instance_rng("pricing", dim, instance)
    cost = rng.uniform(5, 20, dim)
    own = rng.uniform(1.0, 3.0, dim)
    cross = rng.uniform(0, 1, (dim, dim)) * own[:, None] * cross_elasticity / dim
    np.fill_diagonal(cross, 0.0)
    Bm = np.diag(own) - cross              # d = a - B p ; substitutes raise demand
    target = cost + rng.uniform(2, 10, dim)  # optimal prices
    a = (Bm + Bm.T) @ target - Bm.T @ cost

    def profit(p):
        p = np.asarray(p, dtype=float)
        return float((p - cost) @ (a - Bm @ p))

    return ProblemInstance(
        name=instance_name("pricing", dim, instance, x=cross_elasticity),
        family="pricing", category="profit", encoding="real", sense="max", dim=dim,
        objective=profit,
        bounds=list(zip(cost.tolist(), (cost + 3 * (target - cost)).tolist())),
        optimum=profit(target), optimum_solution=target.tolist(),
        tags=frozenset({"profit_quadratic", "non_separable"}),
        params={"cross_elasticity": cross_elasticity}, instance=instance,
        description=f"{dim} substitute products, linear demand",
    )


# ---------------------------------------------------------------------------
# Risk-adjusted profit: mean-variance portfolio
# ---------------------------------------------------------------------------

@register_problem("portfolio", category="profit", encoding="real", sense="max",
                  tags={"profit_risk_adjusted", "constrained", "non_separable"}, default_dim=10)
def portfolio(dim: int, instance: int = 1, risk_aversion: float = 3.0,
              cardinality: Optional[int] = None) -> ProblemInstance:
    """Risk-adjusted profit: mean-variance utility of a long-only portfolio."""
    rng = instance_rng("portfolio", dim, instance, salt=cardinality or 0)
    n_factors = max(1, dim // 4)
    loadings = rng.normal(0, 0.15, (dim, n_factors))
    cov = loadings @ loadings.T + np.diag(rng.uniform(0.01, 0.05, dim))
    mu = 0.02 + loadings @ rng.uniform(0.0, 0.3, n_factors) + rng.normal(0, 0.01, dim)

    def weights(x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        if cardinality is not None and cardinality < dim:
            x = np.where(x >= np.sort(x)[-cardinality], x, 0.0)
        total = x.sum()
        return x / total if total > 0 else np.full(dim, 1.0 / dim)

    def utility(x):
        w = weights(x)
        return float(mu @ w - risk_aversion * w @ cov @ w)

    opt = opt_x = None
    if cardinality is None:
        try:
            from scipy.optimize import minimize

            res = minimize(
                lambda w: -(mu @ w - risk_aversion * w @ cov @ w),
                np.full(dim, 1.0 / dim),
                jac=lambda w: -(mu - 2 * risk_aversion * cov @ w),
                bounds=[(0, 1)] * dim,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(dim)}],
                method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
            )
            if res.success:
                opt_x = np.maximum(res.x, 0) / np.maximum(res.x, 0).sum()
                opt = float(mu @ opt_x - risk_aversion * opt_x @ cov @ opt_x)
                opt_x = opt_x.tolist()
        except ImportError:  # pragma: no cover
            pass

    return ProblemInstance(
        name=instance_name("portfolio", dim, instance, ra=risk_aversion, K=cardinality),
        family="portfolio", category="profit", encoding="real", sense="max", dim=dim,
        objective=utility, bounds=[(0.0, 1.0)] * dim,
        optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"profit_risk_adjusted", "constrained", "non_separable"}),
        params={"risk_aversion": risk_aversion, "cardinality": cardinality}, instance=instance,
        description=f"{dim}-asset mean-variance portfolio"
                    + (f", at most {cardinality} holdings" if cardinality else ""),
    )


# ---------------------------------------------------------------------------
# Stochastic profit: multi-product newsvendor
# ---------------------------------------------------------------------------

@register_problem("newsvendor", category="profit", encoding="real", sense="max",
                  tags={"profit_stochastic", "noisy", "separable"}, default_dim=10)
def newsvendor(dim: int, instance: int = 1, scenarios: int = 10) -> ProblemInstance:
    """Stochastic profit: order quantities under uncertain demand (simulated)."""
    rng = instance_rng("newsvendor", dim, instance)
    mean = rng.uniform(50, 200, dim)
    sd = mean * rng.uniform(0.1, 0.4, dim)
    cost = rng.uniform(2, 10, dim)
    price = cost * rng.uniform(1.5, 3.0, dim)
    salvage = cost * rng.uniform(0.0, 0.5, dim)
    N = NormalDist()

    def expected(q):
        q = np.asarray(q, dtype=float)
        z = (q - mean) / sd
        Phi = np.array([N.cdf(v) for v in z])
        phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        leftover = sd * (z * Phi + phi)
        return float(np.sum((price - cost) * q - (price - salvage) * leftover))

    noise_rng = np.random.default_rng([instance, 104729])

    def simulated(q):
        q = np.asarray(q, dtype=float)
        D = np.maximum(0.0, noise_rng.normal(mean, sd, (scenarios, dim)))
        sold = np.minimum(q, D)
        return float(np.mean(np.sum(price * sold + salvage * (q - sold) - cost * q, axis=1)))

    ratio = (price - cost) / (price - salvage)
    q_opt = mean + sd * np.array([N.inv_cdf(r) for r in ratio])
    return ProblemInstance(
        name=instance_name("newsvendor", dim, instance, sc=scenarios),
        family="newsvendor", category="profit", encoding="real", sense="max", dim=dim,
        objective=simulated, true_objective=expected,
        bounds=[(0.0, float(m + 4 * s)) for m, s in zip(mean, sd)],
        optimum=expected(q_opt), optimum_solution=q_opt.tolist(),
        tags=frozenset({"profit_stochastic", "noisy", "separable"}),
        params={"scenarios": scenarios}, instance=instance,
        description=f"{dim}-product newsvendor, {scenarios} demand scenarios per evaluation",
        _noise_rng=noise_rng,
    )


# ---------------------------------------------------------------------------
# Discontinuous profit: fixed-charge production
# ---------------------------------------------------------------------------

@register_problem("fixed_charge", category="profit", encoding="real", sense="max",
                  tags={"profit_discontinuous", "constrained", "multimodal", "plateaus"},
                  default_dim=10)
def fixed_charge(dim: int, instance: int = 1, constraint_handling: str = "penalty") -> ProblemInstance:
    """Discontinuous profit: per-product set-up costs plus a shared capacity."""
    _check_handling(constraint_handling)
    rng = instance_rng("fixed_charge", dim, instance)
    margin = rng.uniform(5, 30, dim)
    usage = rng.uniform(1, 5, dim)
    upper = rng.uniform(20, 100, dim)
    setup = margin * upper * rng.uniform(0.1, 0.6, dim)
    cap = 0.4 * float(usage @ upper)
    threshold = 1e-3 * upper

    def profit(x):
        x = np.asarray(x, dtype=float)
        made = x > threshold
        return float(margin @ (x * made) - setup @ made)

    opt = opt_x = None
    if dim <= 12:
        best, best_x = 0.0, np.zeros(dim)
        order = np.argsort(-margin / usage)
        for mask in itertools.product((0, 1), repeat=dim):
            on = np.array(mask, dtype=bool)
            x = np.zeros(dim)
            left = cap
            for i in order:  # fractional knapsack over opened products
                if on[i]:
                    x[i] = min(upper[i], left / usage[i])
                    left -= x[i] * usage[i]
            v = profit(x)
            if v > best:
                best, best_x = v, x
        opt, opt_x = best, best_x.tolist()

    A, b = usage[None, :], np.array([cap])
    return ProblemInstance(
        name=instance_name("fixed_charge", dim, instance, ch=constraint_handling),
        family="fixed_charge", category="profit", encoding="real", sense="max", dim=dim,
        objective=_constrained(profit, A, b, constraint_handling, penalty=10 * float(margin @ upper)),
        bounds=list(zip([0.0] * dim, upper.tolist())),
        optimum=opt, optimum_solution=opt_x,
        tags=frozenset({"profit_discontinuous", "constrained", "multimodal", "plateaus"}),
        params={"constraint_handling": constraint_handling}, instance=instance,
        description=f"{dim} products with set-up costs and shared capacity",
    )
