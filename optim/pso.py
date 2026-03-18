"""
Particle Swarm Optimisation (PSO) for continuous decision spaces.

The classic *gbest* PSO is implemented with the standard velocity update:

    v(t+1) = w * v(t)
             + c1 * r1 * (p_best - x(t))
             + c2 * r2 * (g_best - x(t))

Particles that leave the search space have their position clamped to the
nearest bound and their velocity damped (reflection bounce).

References
----------
Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization.
*Proceedings of ICNN'95*, Vol. 4, pp. 1942–1948.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .base import BaseOptimiser, OptimisationResult


class PSOOptimiser(BaseOptimiser):
    """Particle Swarm Optimisation for continuous decision spaces.

    Parameters
    ----------
    n_particles : int
        Number of particles in the swarm.  Default: 30.
    c1 : float
        Cognitive (personal-best) acceleration coefficient.  Default: 1.5.
    c2 : float
        Social (global-best) acceleration coefficient.  Default: 1.5.
    w : float
        Inertia weight.  Default: 0.7.
    w_decay : float
        Factor by which *w* is multiplied each iteration (linear damping).
        Use ``1.0`` for no decay.  Default: 1.0.
    max_no_improve : int
        Stop after this many consecutive swarm-level non-improving steps.
        Default: 200.
    max_iterations : int or None
        Hard upper limit on the number of iterations.  ``None`` means no
        hard limit (use ``max_no_improve`` only).  Default: 5000.
    precision : int
        Number of decimal places to round initial particle positions to.
        Default: 4.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import PSOOptimiser
    >>> pso = PSOOptimiser(n_particles=20, max_no_improve=100)
    >>> result = pso.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                       bounds=[(-5, 5), (-5, 5)])
    >>> result.best_value  # close to 0
    """

    def __init__(
        self,
        n_particles: int = 30,
        c1: float = 1.5,
        c2: float = 1.5,
        w: float = 0.7,
        w_decay: float = 1.0,
        max_no_improve: int = 200,
        max_iterations: Optional[int] = 5000,
        precision: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_decay = w_decay
        self.max_no_improve = max_no_improve
        self.max_iterations = max_iterations
        self.precision = precision
        self.seed = seed

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run PSO.

        Parameters
        ----------
        objective_fn : callable
            Function to minimise (or maximise if *maximise=True*).
        bounds : list of (min, max) tuples
            Bounds for each decision variable.  Required.
        maximise : bool
            If ``True``, maximise the objective.  Default: ``False``.
        initial_solutions : list of lists, optional
            Seed the swarm with these solutions (warm start).  Any remaining
            particles are initialised randomly.

        Returns
        -------
        OptimisationResult
        """
        if bounds is None:
            raise ValueError("bounds must be provided for PSOOptimiser")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)
        n_dims = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # ----- Initialise positions -----
        X = self._init_positions(lo, hi, initial_solutions)

        # Personal bests
        P = deepcopy(X)
        fitness = np.array([obj(list(x)) for x in X])
        n_eval = self.n_particles

        g_best_idx = int(np.argmin(fitness))
        g_best = X[g_best_idx].copy()
        g_best_val = float(fitness[g_best_idx])

        p_fitness = fitness.copy()

        # ----- Velocities -----
        v_max = hi - lo
        V = np.array(
            [
                np.random.uniform(-v_max, v_max)
                for _ in range(self.n_particles)
            ]
        )

        history = [g_best_val]
        no_improve = 0
        iteration = 0
        w = float(self.w)

        while no_improve < self.max_no_improve:
            if self.max_iterations is not None and iteration >= self.max_iterations:
                break

            for i in range(self.n_particles):
                r1 = np.random.random(n_dims)
                r2 = np.random.random(n_dims)

                # Standard gbest velocity update
                V[i] = (
                    w * V[i]
                    + self.c1 * r1 * (P[i] - X[i])
                    + self.c2 * r2 * (g_best - X[i])
                )

                X[i] = X[i] + V[i]

                # Clamp to bounds and reflect velocity at walls
                over_hi = X[i] > hi
                over_lo = X[i] < lo
                X[i] = np.clip(X[i], lo, hi)
                V[i][over_hi | over_lo] *= -0.5

                f = obj(list(X[i]))
                n_eval += 1

                # Update personal best
                if f < p_fitness[i]:
                    P[i] = X[i].copy()
                    p_fitness[i] = f

                # Update global best
                if f < g_best_val:
                    g_best = X[i].copy()
                    g_best_val = f

            # Check for improvement at the end of the full sweep
            if g_best_val < history[-1]:
                no_improve = 0
            else:
                no_improve += 1

            history.append(g_best_val)
            w *= self.w_decay
            iteration += 1

        reported_best = -g_best_val if maximise else g_best_val
        reported_history = [-v if maximise else v for v in history]

        return OptimisationResult(
            best_solution=list(g_best),
            best_value=reported_best,
            history=reported_history,
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_positions(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        seeds: Optional[List[List[float]]],
    ) -> np.ndarray:
        X = np.zeros((self.n_particles, len(lo)))
        start = 0
        if seeds:
            for k, s in enumerate(seeds[: self.n_particles]):
                X[k] = np.clip(s, lo, hi)
            start = min(len(seeds), self.n_particles)
        for k in range(start, self.n_particles):
            X[k] = lo + np.random.random(len(lo)) * (hi - lo)
            X[k] = np.round(X[k], self.precision)
        return X
