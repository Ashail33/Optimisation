"""
Continuous benchmark functions with instance transforms.

Each function is evaluated on ``z = R (x - o) + x0`` where ``x0`` is the
base function's optimiser, ``o`` a random *shift* (so the optimum is not at
the centre of the box) and ``R`` a random *rotation* (which makes separable
functions non-separable).  Both are drawn from the instance number, so
instance ``i`` is identical everywhere it is created.

``shift`` defaults to ``True``: several swarm algorithms (GWO, WOA, SCA)
are biased towards the centre of the search box and look unbeatable on
unshifted benchmarks — exactly the kind of overfitting a benchmark should
not reward.

Optional multiplicative Gaussian ``noise`` gives a noisy objective
``f(x) * (1 + noise * N(0, 1))``; scoring always uses the noise-free value.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .. import benchmarks as bf
from .base import ProblemInstance, instance_name, instance_rng, register_problem


def _ellipsoid(z: np.ndarray) -> float:
    d = len(z)
    w = 10.0 ** (6.0 * np.arange(d) / max(1, d - 1))
    return float(np.sum(w * z ** 2))


def _bent_cigar(z: np.ndarray) -> float:
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def _step(z: np.ndarray) -> float:
    return float(np.sum(np.floor(z + 0.5) ** 2))


# id: (fn, half-width of box, x0 (scalar), f* per dim, tags, transformable)
_FUNCTIONS = {
    "sphere": (bf.sphere, 5.0, 0.0, 0.0, {"unimodal", "separable"}, True),
    "ellipsoid": (_ellipsoid, 5.0, 0.0, 0.0, {"unimodal", "separable", "ill_conditioned"}, True),
    "bent_cigar": (_bent_cigar, 5.0, 0.0, 0.0, {"unimodal", "separable", "ill_conditioned"}, True),
    "step": (_step, 5.0, 0.0, 0.0, {"unimodal", "separable", "plateaus"}, True),
    "zakharov": (bf.zakharov, 5.0, 0.0, 0.0, {"unimodal", "non_separable"}, True),
    "rosenbrock": (bf.rosenbrock, 5.0, 1.0, 0.0, {"non_separable", "ill_conditioned"}, True),
    "rastrigin": (bf.rastrigin, 5.12, 0.0, 0.0, {"multimodal", "separable"}, True),
    "ackley": (bf.ackley, 32.768, 0.0, 0.0, {"multimodal", "non_separable"}, True),
    "griewank": (bf.griewank, 600.0, 0.0, 0.0, {"multimodal", "non_separable"}, True),
    "levy": (bf.levy, 10.0, 1.0, 0.0, {"multimodal", "non_separable"}, True),
    "styblinski_tang": (bf.styblinski_tang, 5.0, -2.9035340286202334, -39.16616570377141, {"multimodal", "separable"}, True),
    # Schwefel's optimum sits near the box edge and better values exist
    # outside the box, so it is used untransformed.
    "schwefel": (bf.schwefel, 500.0, 420.9687, 0.0, {"multimodal", "separable", "deceptive"}, False),
}


def _make_factory(fid: str):
    fn, half, x0, fstar, tags, transformable = _FUNCTIONS[fid]

    def factory(
        dim: int,
        instance: int = 1,
        shift: bool = True,
        rotate: bool = False,
        noise: float = 0.0,
    ) -> ProblemInstance:
        if dim < 2:
            raise ValueError("dim must be at least 2")
        if not transformable and (shift or rotate):
            shift = rotate = False
        rng = instance_rng(fid, dim, instance)
        o = rng.uniform(-0.8 * half, 0.8 * half, dim) if shift else np.full(dim, x0)
        if rotate:
            q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
            R = q * np.sign(np.diag(r))
        else:
            R = None

        def true_f(x) -> float:
            x = np.asarray(x, dtype=float)
            z = x - o
            if R is not None:
                z = R @ z
            return fn(z + x0)

        noise_rng = np.random.default_rng([instance, 7919]) if noise > 0 else None
        if noise > 0:
            def objective(x) -> float:
                return true_f(x) * (1.0 + noise * noise_rng.standard_normal())
        else:
            objective = true_f

        all_tags = set(tags)
        if shift:
            all_tags.add("shifted")
        if rotate:
            all_tags |= {"rotated", "non_separable"}
            all_tags.discard("separable")
        if noise > 0:
            all_tags.add("noisy")

        params = dict(shift=shift, rotate=rotate, noise=noise)
        return ProblemInstance(
            name=instance_name(fid, dim, instance, **{k: v for k, v in params.items() if v}),
            family=fid,
            category="continuous",
            encoding="real",
            sense="min",
            dim=dim,
            objective=objective,
            true_objective=true_f,
            bounds=[(-half, half)] * dim,
            optimum=fstar * dim,
            optimum_solution=o.tolist(),
            tags=frozenset(all_tags),
            params=params,
            instance=instance,
            description=f"{fid} function" + (" (shifted)" if shift else "") + (" (rotated)" if rotate else ""),
            _noise_rng=noise_rng,
        )

    factory.__doc__ = f"{fid.replace('_', ' ').title()} function ({', '.join(sorted(tags))})."
    return factory


for _fid, (_fn, _half, _x0, _fs, _tags, _tr) in _FUNCTIONS.items():
    register_problem(_fid, category="continuous", encoding="real", sense="min",
                     tags=_tags, default_dim=10)(_make_factory(_fid))
