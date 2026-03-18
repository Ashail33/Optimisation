"""
Simulated Annealing (SA) optimisers.

Two classes are provided:

``SimulatedAnnealingOptimiser``
    Single-objective SA for continuous decision spaces.  A random step in one
    decision-variable dimension is used as the default move.  A custom move
    generator can be supplied via ``neighbour_fn``.

``DBMOSAOptimiser``
    Dominance-Based Multi-Objective SA (DBMOSA).  Accepts a multi-objective
    function returning a sequence of objective values and maintains a Pareto
    archive.  Optionally applies a diversity-preservation criterion.

Both optimisers support the same cooling / reheating schedules:
    'Linear', 'Geometric', 'Logarithmic', 'Very slow cooling'.

References
----------
Bandyopadhyay, S. et al. (2008). A Simulated Annealing-Based
Multiobjective Optimization Algorithm: AMOSA.  *IEEE Trans. Evolutionary
Computation*, 12(3), 269–283.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .base import BaseOptimiser, OptimisationResult


# ---------------------------------------------------------------------------
# Shared temperature-schedule helper
# ---------------------------------------------------------------------------

def _apply_schedule(
    T: float,
    change_factor: float,
    direction: str,
    schedule: str,
    step: int,
) -> float:
    """Apply one step of the selected temperature schedule.

    Parameters
    ----------
    T : float
        Current temperature.
    change_factor : float
        Alpha (cooling) or Beta (reheating) parameter.
    direction : {'Cool', 'Heat'}
        Which direction to adjust the temperature.
    schedule : {'Linear', 'Geometric', 'Logarithmic', 'Very slow cooling'}
        Type of schedule.
    step : int
        Current iteration number (used for Logarithmic schedule).
    """
    if schedule == "Linear":
        if direction == "Cool":
            return max(1e-10, T - change_factor)
        return T + change_factor

    if schedule == "Geometric":
        return T * change_factor

    if schedule == "Logarithmic":
        log_step = math.log(max(step, 2))
        if direction == "Cool":
            return T / log_step
        return T * log_step

    if schedule == "Very slow cooling":
        if direction == "Cool":
            return T / (1 + change_factor)
        return T * (1 + change_factor)

    raise ValueError(f"Unknown schedule: {schedule!r}")


# ---------------------------------------------------------------------------
# Default continuous-space move generator
# ---------------------------------------------------------------------------

def _random_step_move(
    solution: List[float],
    bounds: List[Tuple[float, float]],
    scale: float = 0.05,
) -> List[float]:
    """Perturb one randomly chosen dimension by a fraction of its range."""
    neighbour = solution[:]
    i = random.randrange(len(solution))
    lo, hi = bounds[i]
    step = scale * (hi - lo)
    neighbour[i] = min(hi, max(lo, neighbour[i] + random.uniform(-step, step)))
    return neighbour


# ===========================================================================
# Single-objective Simulated Annealing
# ===========================================================================

class SimulatedAnnealingOptimiser(BaseOptimiser):
    """Single-objective Simulated Annealing for continuous decision spaces.

    Parameters
    ----------
    initial_temp : float
        Starting temperature.  Default: ``1e6``.
    cooling_rate : float
        Cooling factor (alpha for Geometric, decrement for Linear, etc.).
        Default: ``0.9999``.
    reheating_rate : float
        Reheating factor used with ``epoch_type='Dynamic'`` when max rejects
        is hit.  Default: ``0.5``.
    max_accepted : int
        Maximum number of accepted moves per epoch (Dynamic epoch).
        Default: 200.
    max_rejected : int
        Maximum number of rejected moves per epoch (Dynamic epoch).
        Default: 150.
    static_epoch_length : int
        Number of moves per epoch when ``epoch_type='Static'``.
        Default: 100.
    max_epochs : int
        Maximum number of epochs.  Default: 10000.
    min_temp : float
        Minimum temperature; algorithm stops when T drops below this value.
        Default: ``1e-6``.
    schedule : {'Linear', 'Geometric', 'Logarithmic', 'Very slow cooling'}
        Cooling / reheating schedule.  Default: ``'Geometric'``.
    epoch_type : {'Dynamic', 'Static'}
        How epoch length is determined.  Default: ``'Dynamic'``.
    termination : {'epoch', 'temperature'}
        Whether to terminate on epoch count or temperature.
        Default: ``'epoch'``.
    neighbour_fn : callable, optional
        Custom move generator ``(solution, bounds) -> new_solution``.
    move_scale : float
        Scale of the default random step move (fraction of dimension range).
        Default: ``0.05``.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import SimulatedAnnealingOptimiser
    >>> sa = SimulatedAnnealingOptimiser(initial_temp=1000, max_epochs=5000)
    >>> result = sa.optimise(lambda x: x[0]**2 + x[1]**2,
    ...                      bounds=[(-5, 5), (-5, 5)])
    >>> result.best_value  # close to 0
    """

    def __init__(
        self,
        initial_temp: float = 1e6,
        cooling_rate: float = 0.9999,
        reheating_rate: float = 0.5,
        max_accepted: int = 200,
        max_rejected: int = 150,
        static_epoch_length: int = 100,
        max_epochs: int = 10_000,
        min_temp: float = 1e-6,
        schedule: str = "Geometric",
        epoch_type: str = "Dynamic",
        termination: str = "epoch",
        neighbour_fn: Optional[Callable] = None,
        move_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reheating_rate = reheating_rate
        self.max_accepted = max_accepted
        self.max_rejected = max_rejected
        self.static_epoch_length = static_epoch_length
        self.max_epochs = max_epochs
        self.min_temp = min_temp
        self.schedule = schedule
        self.epoch_type = epoch_type
        self.termination = termination
        self.neighbour_fn = neighbour_fn
        self.move_scale = move_scale
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
        initial_solution: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run Simulated Annealing.

        Parameters
        ----------
        objective_fn : callable
            Scalar objective function.
        bounds : list of (min, max) tuples
            Required for the default move generator.
        maximise : bool
            If ``True``, maximise the objective.  Default: ``False``.
        initial_solution : list, optional
            Starting solution.  Generated randomly within *bounds* if omitted.

        Returns
        -------
        OptimisationResult
        """
        if bounds is None and self.neighbour_fn is None:
            raise ValueError(
                "Either bounds or neighbour_fn must be provided for "
                "SimulatedAnnealingOptimiser"
            )

        if self.seed is not None:
            random.seed(self.seed)

        obj = self._wrap_objective(objective_fn, maximise)

        # Initial solution
        x = self._resolve_initial(initial_solution, bounds)
        z = obj(x)
        n_eval = 1

        best_solution = deepcopy(x)
        best_value = z
        history = [best_value]

        T = float(self.initial_temp)
        epoch = 0
        epoch_move_count = 0
        accepted = 0
        rejected = 0

        while True:
            # Termination checks
            if self.termination == "epoch" and epoch >= self.max_epochs:
                break
            if self.termination == "temperature" and T <= self.min_temp:
                break
            if epoch >= self.max_epochs:
                break

            # Generate and evaluate neighbour
            if self.neighbour_fn is not None:
                x_new = self.neighbour_fn(x, bounds)
            else:
                x_new = _random_step_move(x, bounds, self.move_scale)  # type: ignore[arg-type]

            z_new = obj(x_new)
            n_eval += 1

            delta = z_new - z
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-300)):
                # Accept move
                x = x_new
                z = z_new
                accepted += 1
                if z < best_value:
                    best_value = z
                    best_solution = deepcopy(x)
            else:
                rejected += 1

            epoch_move_count += 1
            history.append(best_value)

            # Epoch management and temperature update
            if self.epoch_type == "Dynamic":
                if accepted >= self.max_accepted:
                    T = _apply_schedule(T, self.cooling_rate, "Cool", self.schedule, epoch)
                    epoch += 1
                    accepted = 0
                    rejected = 0
                elif rejected >= self.max_rejected:
                    T = _apply_schedule(T, self.reheating_rate, "Heat", self.schedule, epoch)
                    epoch += 1
                    accepted = 0
                    rejected = 0
            else:  # Static
                if epoch_move_count >= self.static_epoch_length:
                    T = _apply_schedule(T, self.cooling_rate, "Cool", self.schedule, epoch)
                    epoch += 1
                    epoch_move_count = 0

        reported_best = -best_value if maximise else best_value
        reported_history = [-v if maximise else v for v in history]

        return OptimisationResult(
            best_solution=best_solution,
            best_value=reported_best,
            history=reported_history,
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_initial(
        self,
        initial_solution: Optional[List[float]],
        bounds: Optional[List[Tuple[float, float]]],
    ) -> List[float]:
        if initial_solution is not None:
            return list(initial_solution)
        if bounds is None:
            raise ValueError("bounds must be provided when initial_solution is None")
        return [random.uniform(lo, hi) for lo, hi in bounds]


# ===========================================================================
# Multi-objective DBMOSA
# ===========================================================================

class DBMOSAOptimiser(BaseOptimiser):
    """Dominance-Based Multi-Objective Simulated Annealing (DBMOSA).

    The algorithm maintains a Pareto archive and uses a dominance-count ratio
    (ΔE) to decide whether to accept candidate solutions.  Optionally a
    diversity-preservation criterion (Kernel, Nearest-Neighbour, or Histogram)
    modifies ΔE to discourage crowded regions.

    Parameters
    ----------
    initial_temp : float
        Starting temperature.  Default: ``1e9``.
    cooling_rate : float
        Cooling factor.  Default: ``0.9999``.
    reheating_rate : float
        Reheating factor used in Dynamic epoch when max-rejected is hit.
        Default: ``0.5``.
    max_accepted : int
        Max accepted per epoch (Dynamic epoch).  Default: 200.
    max_rejected : int
        Max rejected per epoch (Dynamic epoch).  Default: 150.
    static_epoch_length : int
        Moves per epoch (Static epoch).  Default: 100.
    max_epochs : int
        Maximum number of epochs.  Default: 20000.
    min_temp : float
        Temperature at which the algorithm stops (temperature termination).
        Default: ``1e-4``.
    schedule : str
        Temperature schedule.  Default: ``'Geometric'``.
    epoch_type : {'Dynamic', 'Static'}
        Default: ``'Dynamic'``.
    termination : {'epoch', 'temperature'}
        Default: ``'temperature'``.
    diversity_method : {'Kernel', 'NN', 'Histogram', None}
        Diversity-preservation strategy.  ``None`` disables it.
        Default: ``None``.
    diversity_threshold : float
        Density threshold for the Histogram method.  Default: ``5``.
    min_archive_for_diversity : int
        Diversity criterion only kicks in once the archive contains at least
        this many solutions.  Default: ``5``.
    max_archive_size : int or None
        Maximum number of solutions kept in the Pareto archive.  When the
        archive exceeds this limit the most-crowded solution (smallest
        nearest-neighbour distance in objective space) is pruned.  ``None``
        means no limit.  Default: ``100``.
    neighbour_fn : callable, optional
        Custom move generator ``(solution, bounds) -> new_solution``.
    move_scale : float
        Scale of the default random step move.  Default: ``0.1``.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from optim import DBMOSAOptimiser
    >>> def bi_obj(x):
    ...     return [x**2, (x - 2)**2]
    >>> dbmosa = DBMOSAOptimiser(max_epochs=5000)
    >>> result = dbmosa.optimise(bi_obj, bounds=[(-5.0, 5.0)])
    >>> len(result.best_solution)  # Pareto front — list of solutions
    """

    def __init__(
        self,
        initial_temp: float = 1e9,
        cooling_rate: float = 0.9999,
        reheating_rate: float = 0.5,
        max_accepted: int = 200,
        max_rejected: int = 150,
        static_epoch_length: int = 100,
        max_epochs: int = 20_000,
        min_temp: float = 1e-4,
        schedule: str = "Geometric",
        epoch_type: str = "Dynamic",
        termination: str = "temperature",
        diversity_method: Optional[str] = None,
        diversity_threshold: float = 5.0,
        min_archive_for_diversity: int = 5,
        max_archive_size: Optional[int] = 100,
        neighbour_fn: Optional[Callable] = None,
        move_scale: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reheating_rate = reheating_rate
        self.max_accepted = max_accepted
        self.max_rejected = max_rejected
        self.static_epoch_length = static_epoch_length
        self.max_epochs = max_epochs
        self.min_temp = min_temp
        self.schedule = schedule
        self.epoch_type = epoch_type
        self.termination = termination
        self.diversity_method = diversity_method
        self.diversity_threshold = diversity_threshold
        self.min_archive_for_diversity = min_archive_for_diversity
        self.max_archive_size = max_archive_size
        self.neighbour_fn = neighbour_fn
        self.move_scale = move_scale
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
        initial_solution: Optional[Any] = None,
        **kwargs: Any,
    ) -> OptimisationResult:
        """Run DBMOSA.

        Parameters
        ----------
        objective_fn : callable
            Multi-objective function returning a list / tuple of objective
            values.  All objectives are minimised (or all maximised when
            *maximise=True*).
        bounds : list of (min, max) tuples
            Required for the default move generator.
        maximise : bool
            If ``True``, maximise all objectives.  Default: ``False``.
        initial_solution : list, optional
            Starting solution.  Randomly generated within *bounds* if omitted.

        Returns
        -------
        OptimisationResult
            ``best_solution`` is the Pareto archive (list of solutions).
            ``best_value`` is the hypervolume indicator of the archive
            (computed as the sum of all objective values — lower is better
            for minimisation).
        """
        if bounds is None and self.neighbour_fn is None:
            raise ValueError(
                "Either bounds or neighbour_fn must be provided for "
                "DBMOSAOptimiser"
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Wrap for maximisation: negate all objectives
        if maximise:
            raw_obj = objective_fn

            def obj_fn(sol: Any) -> Sequence[float]:
                vals = raw_obj(sol)
                return [-v for v in vals]
        else:
            obj_fn = objective_fn  # type: ignore[assignment]

        x = self._resolve_initial(initial_solution, bounds)
        archive: List[Any] = [deepcopy(x)]
        n_eval = 1

        T = float(self.initial_temp)
        epoch = 0
        accepted = 0
        rejected = 0
        epoch_move_count = 0
        total_moves = 0
        # Hard limit: avoid infinite loops when epoch counter stalls
        max_total_moves = self.max_epochs * max(
            self.max_accepted + self.max_rejected,
            self.static_epoch_length,
            1,
        ) * 2

        archive_history: List[int] = [1]

        while True:
            if self.termination == "epoch" and epoch >= self.max_epochs:
                break
            if self.termination == "temperature" and T <= self.min_temp:
                break
            if epoch >= self.max_epochs:
                break
            if total_moves >= max_total_moves:
                break

            # Generate neighbour
            if self.neighbour_fn is not None:
                x_new = self.neighbour_fn(x, bounds)
            else:
                x_new = _random_step_move(x, bounds, self.move_scale)  # type: ignore[arg-type]

            n_eval += 1

            # Dominance-based delta_E
            delta_e, a_tilda, n_dominating_new, x_dominates = (
                self._delta_e(archive, x_new, x, obj_fn)
            )

            # Optional diversity adjustment
            if (
                self.diversity_method is not None
                and len(archive) >= self.min_archive_for_diversity
            ):
                delta_e = self._apply_diversity(
                    delta_e, x_new, archive, obj_fn, T
                )

            # Acceptance criterion
            accept_prob = min(1.0, math.exp(-delta_e / max(T, 1e-300)))
            if random.random() < accept_prob:
                x = x_new
                accepted += 1
                # Archive update — add if not dominated
                if n_dominating_new == 0:
                    dominated = [
                        a_tilda[i]
                        for i, dom in enumerate(x_dominates)
                        if dom
                    ]
                    archive = [s for s in archive if s not in dominated]
                    archive.append(deepcopy(x))
                    # Prune archive if it exceeds max_archive_size
                    if (
                        self.max_archive_size is not None
                        and len(archive) > self.max_archive_size
                    ):
                        archive = self._prune_archive(archive, obj_fn)
            else:
                rejected += 1

            epoch_move_count += 1
            total_moves += 1
            archive_history.append(len(archive))

            # Epoch and temperature management
            if self.epoch_type == "Dynamic":
                if accepted >= self.max_accepted:
                    T = _apply_schedule(T, self.cooling_rate, "Cool", self.schedule, epoch)
                    epoch += 1
                    accepted = 0
                    rejected = 0
                elif rejected >= self.max_rejected:
                    T = _apply_schedule(T, self.reheating_rate, "Heat", self.schedule, epoch)
                    epoch += 1
                    accepted = 0
                    rejected = 0
            else:
                if epoch_move_count >= self.static_epoch_length:
                    T = _apply_schedule(T, self.cooling_rate, "Cool", self.schedule, epoch)
                    epoch += 1
                    epoch_move_count = 0

        # Summarise Pareto front
        best_solution = [deepcopy(s) for s in archive]
        best_value = sum(
            sum(obj_fn(s)) for s in archive
        ) / max(len(archive), 1)

        return OptimisationResult(
            best_solution=best_solution,
            best_value=best_value,
            history=[float(h) for h in archive_history],
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_initial(
        self,
        initial_solution: Optional[Any],
        bounds: Optional[List[Tuple[float, float]]],
    ) -> Any:
        if initial_solution is not None:
            return deepcopy(initial_solution)
        if bounds is None:
            raise ValueError("bounds must be provided when initial_solution is None")
        return [random.uniform(lo, hi) for lo, hi in bounds]

    @staticmethod
    def _dominates(z_a: Sequence[float], z_b: Sequence[float]) -> bool:
        """Return True if solution with objectives *z_a* dominates *z_b*."""
        return all(a <= b for a, b in zip(z_a, z_b)) and any(
            a < b for a, b in zip(z_a, z_b)
        )

    def _delta_e(
        self,
        archive: List[Any],
        x_new: Any,
        x: Any,
        obj_fn: Callable,
    ) -> Tuple[float, List[Any], int, List[bool]]:
        """Compute the dominance-based energy change ΔE."""
        a_tilda = archive + [x_new, x]
        z_x = obj_fn(x)
        z_new = obj_fn(x_new)

        n_dom_x = sum(
            1 for s in a_tilda if self._dominates(obj_fn(s), z_x)
        )
        n_dom_new = sum(
            1 for s in a_tilda if self._dominates(obj_fn(s), z_new)
        )
        x_dominates = [self._dominates(z_new, obj_fn(s)) for s in a_tilda]

        delta_e = (-n_dom_x + n_dom_new) / len(a_tilda)
        return delta_e, a_tilda, n_dom_new, x_dominates

    def _apply_diversity(
        self,
        delta_e: float,
        x_new: Any,
        archive: List[Any],
        obj_fn: Callable,
        T: float,
    ) -> float:
        """Modify ΔE using the selected diversity method."""
        method = self.diversity_method
        if method == "Kernel":
            sigma = 0.001
            z_new = obj_fn(x_new)
            dist_sum = sum(
                max(0.0, 1 - abs(z_new[0] - obj_fn(s)[0]) / sigma)
                for s in archive
                if abs(z_new[0] - obj_fn(s)[0]) < sigma
            )
            if dist_sum > 0:
                delta_e /= dist_sum

        elif method == "Histogram":
            z_new = obj_fn(x_new)
            archive_z = [obj_fn(s) for s in archive]
            # Count solutions in the same histogram cell (resolution 0.1)
            cell = tuple(round(v, 1) for v in z_new)
            count = sum(
                1 for z in archive_z if tuple(round(v, 1) for v in z) == cell
            )
            if count > self.diversity_threshold:
                delta_e = 1e9 * T

        elif method == "NN":
            # Nearest-neighbour crowding: penalise if already worst-ranked
            archive_z = [obj_fn(s) for s in archive]
            z_new = obj_fn(x_new)
            all_z = archive_z + [z_new]
            n = len(all_z)
            if n >= 3:
                # Simple 1-NN distance for each point
                distances = []
                for zi in all_z:
                    dists = [
                        sum((a - b) ** 2 for a, b in zip(zi, zj)) ** 0.5
                        for zj in all_z
                        if zj is not zi
                    ]
                    distances.append(min(dists))
                # If x_new has the smallest nearest-neighbour distance →
                # most crowded → penalise
                if distances[-1] == min(distances):
                    delta_e = 1e9 * T

        return delta_e

    def _prune_archive(
        self, archive: List[Any], obj_fn: Callable
    ) -> List[Any]:
        """Remove the most-crowded solution from the archive.

        The solution with the smallest sum of distances to its two nearest
        neighbours in objective space is removed.
        """
        if len(archive) <= 1:
            return archive
        zs = [obj_fn(s) for s in archive]
        n = len(zs)
        # Compute minimum 1-NN distance for each solution
        min_dists = []
        for i, zi in enumerate(zs):
            dists = [
                sum((a - b) ** 2 for a, b in zip(zi, zj)) ** 0.5
                for j, zj in enumerate(zs)
                if j != i
            ]
            min_dists.append(min(dists))
        # Remove the most crowded (smallest distance)
        remove_idx = min_dists.index(min(min_dists))
        return [s for k, s in enumerate(archive) if k != remove_idx]
