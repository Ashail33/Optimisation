"""
Adaptive (hyper-heuristic) ensemble.

A selection hyper-heuristic treats the algorithms as the arms of a
multi-armed bandit.  The run is split into rounds; each round one algorithm
is chosen, warm-started from a shared pool of the best solutions found so
far, and rewarded by how much it improved the best value.  Algorithms that
pay off are chosen more often, so the ensemble learns online which
algorithm suits the problem — and can switch as the search moves from
exploration to exploitation.

Selection rules
---------------
``'ucb'``                   UCB1: mean reward + ``exploration`` × confidence bonus.
``'epsilon_greedy'``        best mean reward, random arm with prob. ``exploration``.
``'probability_matching'``  pick in proportion to recency-weighted rewards,
                            each arm keeping at least ``exploration / k``.
``'round_robin'``           cycle through algorithms (non-adaptive baseline).

References
----------
Burke, E. K. et al. (2013). Hyper-heuristics: a survey of the state of the
art. *Journal of the Operational Research Society*, 64, 1695–1724.

Fialho, Á. et al. (2010). Analyzing bandit-based adaptive operator selection
mechanisms. *Annals of Mathematics and Artificial Intelligence*, 60, 25–64.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..base import BaseOptimiser, OptimisationResult
from ..ensemble import EnsembleResult
from ._common import CountingObjective, SolutionPool, derive_seed, run_optimiser, split_budget

_RULES = ("ucb", "epsilon_greedy", "probability_matching", "round_robin")
_DEFAULT_EXPLORATION = {
    "ucb": 0.5,
    "epsilon_greedy": 0.2,
    "probability_matching": 0.2,
    "round_robin": 0.0,
}


class AdaptiveEnsembleOptimiser(BaseOptimiser):
    """Bandit-based adaptive selection between optimisers.

    Parameters
    ----------
    optimisers : list of BaseOptimiser
        Candidate algorithms (the bandit's arms).
    n_rounds : int
        Number of selection rounds.  Default 20.
    round_evaluations : int, optional
        Evaluation budget per round (for optimisers that accept
        ``max_evaluations``).  Default: ``max_evaluations / n_rounds``.
    selection : {'ucb', 'epsilon_greedy', 'probability_matching', 'round_robin'}
        Default ``'ucb'``.
    exploration : float, optional
        UCB constant, epsilon, or probability-matching floor.
    adaptation_rate : float
        Recency weight for probability matching.  Default 0.3.
    pool_size : int
        Size of the shared warm-start pool.  Default 30.
    max_evaluations : int, optional
        Total evaluation budget.
    seed : int, optional
        Seeds the ensemble; each run gets a derived seed.
    """

    def __init__(
        self,
        optimisers: Sequence[BaseOptimiser],
        n_rounds: int = 20,
        round_evaluations: Optional[int] = None,
        selection: str = "ucb",
        exploration: Optional[float] = None,
        adaptation_rate: float = 0.3,
        pool_size: int = 30,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not optimisers:
            raise ValueError("optimisers must be a non-empty list")
        if selection not in _RULES:
            raise ValueError(f"selection must be one of {_RULES}")
        self.optimisers = list(optimisers)
        self.n_rounds = n_rounds
        self.round_evaluations = round_evaluations
        self.selection = selection
        self.exploration = (
            _DEFAULT_EXPLORATION[selection] if exploration is None else exploration
        )
        self.adaptation_rate = adaptation_rate
        self.pool_size = pool_size
        self.max_evaluations = max_evaluations
        self.seed = seed

    def optimise(
        self,
        objective_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        *,
        maximise: bool = False,
        initial_solutions: Optional[Sequence[Any]] = None,
        optimiser_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Run the adaptive ensemble."""
        rng = np.random.default_rng(self.seed)
        objective = CountingObjective(objective_fn, maximise)
        pool = SolutionPool(self.pool_size)
        k = len(self.optimisers)
        counts = np.zeros(k, dtype=int)
        reward_sum = np.zeros(k)
        quality = np.full(k, 1.0)  # optimistic start for probability matching
        budget = self.round_evaluations or split_budget(self.max_evaluations, self.n_rounds)

        run_results: List[OptimisationResult] = []
        selections: List[int] = []
        rewards: List[float] = []
        history: List[float] = []

        for round_ in range(self.n_rounds):
            if self.max_evaluations is not None and objective.n_evaluations >= self.max_evaluations:
                break
            arm = self._select(round_, counts, reward_sum, quality, rng)
            before = pool.best_value if len(pool) else math.inf
            seeds = pool.solutions or (list(initial_solutions) if initial_solutions else None)
            cap = budget
            if self.max_evaluations is not None:
                remaining = self.max_evaluations - objective.n_evaluations
                cap = remaining if cap is None else min(cap, remaining)

            result = run_optimiser(
                self.optimisers[arm],
                objective,
                bounds,
                population=seeds,
                max_evaluations=cap,
                seed=derive_seed(rng),
                extra_kwargs=optimiser_kwargs[arm] if optimiser_kwargs else None,
            )
            pool.add_result(result)
            reward = self._reward(before, pool.best_value)

            counts[arm] += 1
            reward_sum[arm] += reward
            quality[arm] += self.adaptation_rate * (reward - quality[arm])
            selections.append(arm)
            rewards.append(reward)
            run_results.append(result)
            history.append(pool.best_value)

        sign = objective.sign
        names = [type(o).__name__ for o in self.optimisers]
        return EnsembleResult(
            best_solution=pool.best_solution,
            best_value=sign * pool.best_value,
            history=[sign * h for h in history],
            n_evaluations=objective.n_evaluations,
            population=pool.solutions,
            population_values=[sign * v for v in pool.values],
            run_results=run_results,
            info={
                "selections": [names[a] for a in selections],
                "selection_counts": {names[i]: int(counts[i]) for i in range(k)},
                "mean_rewards": {
                    names[i]: float(reward_sum[i] / counts[i]) if counts[i] else 0.0
                    for i in range(k)
                },
                "rewards": rewards,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _reward(before: float, after: float) -> float:
        """Relative improvement of the best value, clipped to ``[0, 1]``."""
        if math.isinf(before):
            return 1.0
        gain = before - after
        if gain <= 0:
            return 0.0
        return min(1.0, gain / (abs(before) + 1e-12))

    def _select(self, round_, counts, reward_sum, quality, rng) -> int:
        k = len(counts)
        if self.selection == "round_robin":
            return round_ % k
        if self.selection == "probability_matching":
            p_min = self.exploration / k
            p = p_min + (1 - k * p_min) * quality / quality.sum() if quality.sum() > 0 else None
            return int(rng.choice(k, p=p))
        untried = np.flatnonzero(counts == 0)
        if len(untried):
            return int(untried[0])
        means = reward_sum / counts
        if self.selection == "epsilon_greedy":
            if rng.random() < self.exploration:
                return int(rng.integers(k))
            return int(rng.choice(np.flatnonzero(means == means.max())))
        bonus = self.exploration * np.sqrt(2 * math.log(counts.sum()) / counts)
        return int(np.argmax(means + bonus))
