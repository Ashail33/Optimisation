"""Benchmark every continuous metaheuristic and ensemble type on a few
standard functions.

    python examples/compare_algorithms.py [--dim 10] [--runs 3] [--evals 20000]
"""

import argparse

from optim import (
    ACOROptimiser,
    AdaptiveEnsembleOptimiser,
    ArtificialBeeColonyOptimiser,
    BatOptimiser,
    CMAESOptimiser,
    CooperativeCoevolutionOptimiser,
    CuckooSearchOptimiser,
    DifferentialEvolutionOptimiser,
    FireflyOptimiser,
    GreyWolfOptimiser,
    HarmonySearchOptimiser,
    IslandModelOptimiser,
    JayaOptimiser,
    MemeticOptimiser,
    SineCosineOptimiser,
    TLBOOptimiser,
    WhaleOptimiser,
)
from optim.benchmarks import compare, format_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--evals", type=int, default=20_000)
    parser.add_argument("--problems", nargs="*", default=["sphere", "rosenbrock", "rastrigin", "ackley"])
    args = parser.parse_args()

    # Budgets come from compare(max_evaluations=...), so no iteration cap.
    single = {
        "DE": DifferentialEvolutionOptimiser(max_iterations=None),
        "CMA-ES": CMAESOptimiser(max_iterations=None),
        "GWO": GreyWolfOptimiser(max_iterations=None),
        "WOA": WhaleOptimiser(max_iterations=None),
        "Firefly": FireflyOptimiser(max_iterations=None),
        "Cuckoo": CuckooSearchOptimiser(max_iterations=None),
        "ABC": ArtificialBeeColonyOptimiser(max_iterations=None),
        "Bat": BatOptimiser(max_iterations=None),
        "Harmony": HarmonySearchOptimiser(max_iterations=None),
        "TLBO": TLBOOptimiser(max_iterations=None),
        "SCA": SineCosineOptimiser(max_iterations=None),
        "Jaya": JayaOptimiser(max_iterations=None),
        "ACOR": ACOROptimiser(max_iterations=None),
    }

    def trio():
        return [DifferentialEvolutionOptimiser(), CMAESOptimiser(), GreyWolfOptimiser()]

    # Ensembles take their total budget in the constructor.
    ensembles = {
        "Island(DE,CMA,GWO)": IslandModelOptimiser(trio(), max_evaluations=args.evals),
        "Adaptive-UCB(DE,CMA,GWO)": AdaptiveEnsembleOptimiser(trio(), max_evaluations=args.evals),
        "Memetic(DE+CMA)": MemeticOptimiser(
            DifferentialEvolutionOptimiser(), CMAESOptimiser(sigma0=0.05),
            max_evaluations=args.evals,
        ),
        "CoopCoev(DE)": CooperativeCoevolutionOptimiser(
            DifferentialEvolutionOptimiser(), max_evaluations=args.evals,
        ),
    }

    rows = compare(
        {**single, **ensembles},
        problems=args.problems,
        dim=args.dim,
        n_runs=args.runs,
        max_evaluations=args.evals,
    )
    print(format_table(rows))


if __name__ == "__main__":
    main()
