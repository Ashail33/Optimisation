"""
Ant Colony Optimisation (ACO) for permutation problems.

Each ant builds a permutation one element at a time, choosing the next
element ``j`` after ``i`` with probability proportional to
``tau[i, j]**alpha * eta[i, j]**beta`` where ``tau`` is the learned
pheromone and ``eta`` an optional problem-specific heuristic (e.g.
``1 / distance`` for the TSP).  Pheromone evaporates every iteration and is
reinforced on the edges of the best ants (rank-based Ant System) and of the
best-so-far solution.  Pheromone is bounded below so no edge becomes
impossible (as in MAX–MIN Ant System).  A virtual start node with its own
pheromone row lets the colony learn which element should come first, so
the algorithm also suits open sequencing problems, not just closed tours.

Deposits are rank-based rather than proportional to ``1 / cost``, so the
objective may be negative or maximised.

References
----------
Dorigo, M., Maniezzo, V. & Colorni, A. (1996). Ant system: optimization by a
colony of cooperating agents. *IEEE Trans. SMC — Part B*, 26(1), 29–41.

Bullnheimer, B., Hartl, R. F. & Strauss, C. (1999). A new rank based version
of the Ant System. *Central European Journal of Operations Research*, 7,
25–38.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult


class AntColonyOptimiser(BaseOptimiser):
    """Rank-based Ant System for permutation (sequencing / routing) problems.

    Parameters
    ----------
    n_ants : int
        Ants (solutions built) per iteration.  Default 20.
    alpha : float
        Pheromone influence.  Default 1.0.
    beta : float
        Heuristic influence (ignored without a ``heuristic``).  Default 2.0.
    evaporation : float
        Fraction of pheromone that evaporates each iteration.  Default 0.1.
    n_elite : int, optional
        Number of ranked ants that deposit pheromone.  Default
        ``max(1, n_ants // 4)``.
    min_pheromone : float
        Lower bound on pheromone values.  Default 0.01.
    max_iterations : int or None
        Default 200.
    max_no_improve : int or None
        Default 50.
    max_evaluations : int or None
        Hard evaluation budget.
    seed : int, optional
        Seed for the private random generator.
    """

    def __init__(
        self,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.1,
        n_elite: Optional[int] = None,
        min_pheromone: float = 0.01,
        max_iterations: Optional[int] = 200,
        max_no_improve: Optional[int] = 50,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 < evaporation <= 1.0:
            raise ValueError("evaporation must be in (0, 1]")
        if max_iterations is None and max_evaluations is None and max_no_improve is None:
            raise ValueError("set at least one stopping criterion")
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.n_elite = n_elite
        self.min_pheromone = min_pheromone
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.max_evaluations = max_evaluations
        self.seed = seed

    # ------------------------------------------------------------------
    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        n_genes: Optional[int] = None,
        heuristic: Optional[Sequence[Sequence[float]]] = None,
        initial_solution: Optional[Sequence[int]] = None,
        initial_solutions: Optional[Sequence[Sequence[int]]] = None,
        max_evaluations: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run the colony.

        Parameters
        ----------
        objective_fn : callable
            Cost of a permutation of ``range(n_genes)`` (a list of ints).
        bounds : ignored
            Accepted for interface compatibility; its length is used as
            ``n_genes`` if nothing else gives the problem size.
        maximise : bool
            Maximise instead of minimise.
        n_genes : int, optional
            Permutation length.
        heuristic : (n, n) array-like, optional
            Desirability of placing ``j`` straight after ``i`` (e.g.
            ``1 / distance``).  Must be non-negative.
        initial_solution, initial_solutions : optional
            Known good permutations; they seed the pheromone trails.
        max_evaluations : int, optional
            Override the evaluation budget for this call.
        """
        rng = np.random.default_rng(self.seed)
        sign = -1.0 if maximise else 1.0
        budget = self.max_evaluations if max_evaluations is None else max_evaluations

        seeds = [list(s) for s in (initial_solutions or [])]
        if initial_solution is not None:
            seeds.insert(0, list(initial_solution))

        n = n_genes
        if n is None and heuristic is not None:
            n = len(heuristic)
        if n is None and seeds:
            n = len(seeds[0])
        if n is None and bounds is not None:
            n = len(bounds)
        if n is None or n < 2:
            raise ValueError("n_genes (>= 2) must be provided")

        # Row ``n`` of tau / eta is the virtual start node.
        eta_b = np.ones((n + 1, n))
        if heuristic is not None:
            eta = np.asarray(heuristic, dtype=float)
            if eta.shape != (n, n) or np.any(eta < 0):
                raise ValueError("heuristic must be a non-negative (n, n) matrix")
            eta_b[:n] = eta ** self.beta
        tau = np.ones((n + 1, n))
        n_elite = self.n_elite or max(1, self.n_ants // 4)

        n_eval = 0
        best, best_f = None, math.inf

        def evaluate(tour) -> float:
            nonlocal n_eval
            n_eval += 1
            f = sign * float(objective_fn(tour.tolist()))
            return math.inf if math.isnan(f) else f

        def exhausted() -> bool:
            return budget is not None and n_eval >= budget

        for s in seeds:
            if exhausted():
                break
            tour = np.asarray(s, dtype=int)
            f = evaluate(tour)
            self._deposit(tau, tour, 1.0)
            if f < best_f:
                best, best_f = tour, f

        history = [best_f] if best is not None else []
        iteration = no_improve = 0
        while not exhausted():
            if self.max_iterations is not None and iteration >= self.max_iterations:
                break
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

            weights = tau ** self.alpha * eta_b
            tours, costs = [], []
            for _ in range(self.n_ants):
                if exhausted():
                    break
                tour = self._construct(weights, n, rng)
                tours.append(tour)
                costs.append(evaluate(tour))
            if not tours:
                break

            previous = best_f
            order = np.argsort(costs, kind="stable")
            if costs[order[0]] < best_f:
                best, best_f = tours[order[0]], costs[order[0]]

            tau *= 1.0 - self.evaporation
            for rank, k in enumerate(order[:n_elite]):
                self._deposit(tau, tours[k], (n_elite - rank) / n_elite)
            self._deposit(tau, best, 1.0)
            np.maximum(tau, self.min_pheromone, out=tau)

            iteration += 1
            no_improve = 0 if best_f < previous else no_improve + 1
            history.append(best_f)

        return OptimisationResult(
            best_solution=[int(v) for v in best],
            best_value=sign * best_f,
            history=[sign * h for h in history],
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _construct(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        tour = np.empty(n, dtype=int)
        unvisited = np.ones(n, dtype=bool)
        current = n  # virtual start node
        for k in range(n):
            w = weights[current] * unvisited
            total = w.sum()
            if total > 0:
                current = rng.choice(n, p=w / total)
            else:
                current = rng.choice(np.flatnonzero(unvisited))
            tour[k] = current
            unvisited[current] = False
        return tour

    @staticmethod
    def _deposit(tau: np.ndarray, tour: np.ndarray, amount: float) -> None:
        tau[len(tau) - 1, tour[0]] += amount
        tau[tour[:-1], tour[1:]] += amount
