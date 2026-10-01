"""
Derived metrics and statistics for a benchmark run.

Raw facts (best value, evaluations, curves) are written by the runner; this
module turns them into scale-free, comparable metrics.  Everything is
expressed through the **normalised gap**

.. math::

    \\text{gap}(v) = \\frac{m(v) - m^*}{m_{\\text{rand}} - m^*}

where ``m`` is the value in minimisation orientation, ``m*`` the optimum
(or best-known value when the optimum is unknown: the best any optimiser
found on that instance) and ``m_rand`` the median value of random
solutions.  ``gap = 0`` is optimal, ``gap = 1`` is no better than random
sampling.  This makes profits and costs, and problems of any scale,
comparable.

Per-run metrics
---------------
``gap``, ``score = 1 - min(gap, 1)``, ``solved`` (``gap <= 1e-8``),
``evals_to_<target>`` for gap targets ``1e-1 … 1e-8``, and ``auc`` — the
area under the anytime target-hitting curve over log-spaced budgets
(COCO-style anytime performance, in ``[0, 1]``).

Aggregates
----------
Average ranks (lower gap is better) per task — a task is a problem instance
× budget, averaged over repeats — and the Friedman test across optimisers;
ECDFs of (run, target) pairs reached versus evaluations / dimension;
median convergence curves per instance.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

TARGETS = (1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
SOLVED_GAP = 1e-8
_CURVE_POINTS = 30
_ECDF_POINTS = 40


def _target_col(t: float) -> str:
    return f"evals_to_{t:.0e}"


def load_run(run_dir: Union[str, Path]):
    import pandas as pd

    run_dir = Path(run_dir)
    results = pd.read_csv(run_dir / "results.csv")
    curves_path = run_dir / "curves.csv"
    curves = pd.read_csv(curves_path) if curves_path.stat().st_size > 1 else pd.DataFrame(
        columns=["run_id", "evaluations", "value"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return results, curves, manifest


def _gap(values, sign, m_star, m_rand):
    m = sign * np.asarray(values, dtype=float)
    denom = m_rand - m_star
    if not np.isfinite(denom) or denom <= 0:
        denom = max(abs(m_star), 1.0)
    return np.maximum(0.0, (m - m_star) / denom)


def derive(results, curves):
    """Add reference values and per-run derived metrics to ``results``.

    Returns ``(results, gap_curves)`` where ``gap_curves`` maps run_id to
    ``(evaluations, gap)`` arrays.
    """
    df = results.copy()
    df["sign"] = np.where(df["sense"] == "max", -1.0, 1.0)
    df["m_best"] = df["sign"] * df["best_value"]
    best_found = df.groupby("problem")["m_best"].transform("min")
    m_opt = df["sign"] * df["optimum"]
    df["optimum_known"] = df["optimum"].notna()
    # When the optimum is known it is the reference; otherwise the best found.
    m_star = np.where(df["optimum_known"], np.minimum(m_opt, best_found), best_found)
    df["reference_value"] = df["sign"] * m_star
    df["m_star"] = m_star
    df["m_rand"] = df["sign"] * df["random_reference"]
    df["gap"] = [
        float(_gap([v], s, ms, mr)[0]) if np.isfinite(v) else math.inf
        for v, s, ms, mr in zip(df["best_value"].fillna(np.nan), df["sign"], df["m_star"], df["m_rand"])
    ]
    df["score"] = 1.0 - np.minimum(df["gap"], 1.0)
    df["solved"] = df["gap"] <= SOLVED_GAP

    grouped = {rid: g for rid, g in curves.groupby("run_id")} if len(curves) else {}
    gap_curves: Dict[str, Any] = {}
    e2t = {t: [] for t in TARGETS}
    aucs = []
    for row in df.itertuples():
        g = grouped.get(row.run_id)
        if g is None or not len(g):
            ev, gp = np.array([row.budget]), np.array([np.inf])
        else:
            ev = g["evaluations"].to_numpy(dtype=float)
            gp = _gap(g["value"].to_numpy(), row.sign, row.m_star, row.m_rand)
            gp = np.minimum.accumulate(gp)  # incumbent gap is monotone
        gap_curves[row.run_id] = (ev, gp)
        for t in TARGETS:
            hit = np.flatnonzero(gp <= t)
            e2t[t].append(float(ev[hit[0]]) if len(hit) else np.nan)
        checkpoints = np.geomspace(1, row.budget, 25)
        idx = np.searchsorted(ev, checkpoints, side="right") - 1
        best_at = np.where(idx >= 0, gp[np.maximum(idx, 0)], np.inf)
        frac = np.mean([[b <= t for t in TARGETS] for b in best_at])
        aucs.append(float(frac))
    for t in TARGETS:
        df[_target_col(t)] = e2t[t]
    df["auc"] = aucs
    return df, gap_curves


def average_ranks(df, metric: str = "gap", by: Sequence[str] = ("problem", "budget")):
    """Average rank of each optimiser across tasks (lower metric is better).

    Returns ``rank`` (mean raw rank) and ``norm_rank`` — the per-task rank
    rescaled to ``[0, 1]`` (0 = best, 1 = worst of the optimisers that ran
    on that task), which stays comparable when optimisers cover different
    tasks (e.g. permutation-only specialists).
    """
    per_task = df.groupby(list(by) + ["optimiser"], as_index=False)[metric].mean()
    grp = per_task.groupby(list(by))[metric]
    per_task["rank"] = grp.rank(method="average")
    n = grp.transform("count")
    per_task["norm_rank"] = np.where(n > 1, (per_task["rank"] - 1) / (n - 1).clip(lower=1), 0.5)
    return (
        per_task.groupby("optimiser", as_index=False)[["rank", "norm_rank"]].mean()
        .sort_values("norm_rank").reset_index(drop=True)
    )


def friedman(df, metric: str = "gap", by: Sequence[str] = ("problem", "budget")) -> Dict[str, Any]:
    """Friedman test across tasks.

    Uses the optimisers that ran on the most tasks (specialists that only
    solve some encodings are left out) and the tasks all of them ran on.
    """
    table = df.groupby(list(by) + ["optimiser"])[metric].mean().unstack("optimiser")
    coverage = table.notna().sum()
    table = table.loc[:, coverage == coverage.max()].dropna()
    k, n = table.shape[1], table.shape[0]
    if k < 3 or n < 2:
        return {"statistic": None, "p_value": None, "n_tasks": int(n), "n_optimisers": int(k)}
    ranks = table.rank(axis=1, method="average").to_numpy()
    rbar = ranks.mean(axis=0)
    stat = 12 * n / (k * (k + 1)) * float(np.sum((rbar - (k + 1) / 2) ** 2))
    try:
        from scipy.stats import chi2

        p = float(chi2.sf(stat, k - 1))
    except ImportError:  # pragma: no cover
        p = None
    return {"statistic": stat, "p_value": p, "n_tasks": int(n), "n_optimisers": int(k),
            "optimisers": list(table.columns)}


def _ecdf(sub, gap_curves, grid):
    """Fraction of (run, target) pairs reached by evaluations/dim in ``grid``."""
    if not len(sub):
        return [0.0] * len(grid)
    hits = np.zeros(len(grid))
    total = 0
    for row in sub.itertuples():
        ev, gp = gap_curves[row.run_id]
        scaled = ev / max(1, row.dim)
        for t in TARGETS:
            total += 1
            reached = np.flatnonzero(gp <= t)
            if len(reached):
                hits += grid >= scaled[reached[0]]
    return (hits / total).round(4).tolist()


def summarise(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Compute derived metrics + statistics; write ``scored.csv`` and ``summary.json``."""
    run_dir = Path(run_dir)
    results, curves, manifest = load_run(run_dir)
    ok = results[results["status"] != "error"].copy()
    if not len(ok):
        summary = {"name": manifest.get("name"), "n_runs": 0}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    df, gap_curves = derive(ok, curves)
    df.to_csv(run_dir / "scored.csv", index=False)

    optimisers = sorted(df["optimiser"].unique())
    categories = sorted(df["category"].unique())

    def board(sub):
        if not len(sub):
            return []
        ranks = average_ranks(sub)
        agg = sub.groupby("optimiser").agg(
            mean_gap=("gap", "mean"), median_gap=("gap", "median"), mean_score=("score", "mean"),
            mean_auc=("auc", "mean"), solved_rate=("solved", "mean"), runs=("run_id", "count"),
            mean_wall_s=("wall_time_s", "mean"), tasks=("problem", "nunique"),
        ).reset_index()
        merged = ranks.merge(agg, on="optimiser")
        merged["coverage"] = merged["tasks"] / merged["tasks"].max()
        merged = merged.sort_values(["coverage", "norm_rank"], ascending=[False, True])
        return json.loads(merged.to_json(orient="records"))

    # Per-task rows (problem x budget x optimiser, averaged over repeats) for
    # client-side filtering in the dashboard.
    task_rows = df.groupby(
        ["problem", "problem_family", "category", "encoding", "sense", "dim", "budget", "tags", "optimiser"],
        as_index=False,
    ).agg(gap=("gap", "mean"), score=("score", "mean"), auc=("auc", "mean"),
          solved=("solved", "mean"), wall=("wall_time_s", "mean"), runs=("run_id", "count"))

    # Median convergence curves per problem instance (gap vs fraction of budget).
    fractions = np.geomspace(1e-3, 1.0, _CURVE_POINTS)
    curves_out: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for (problem, opt), g in df.groupby(["problem", "optimiser"]):
        mat = []
        for row in g.itertuples():
            ev, gp = gap_curves[row.run_id]
            cps = fractions * row.budget
            idx = np.searchsorted(ev, cps, side="right") - 1
            mat.append(np.where(idx >= 0, gp[np.maximum(idx, 0)], np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN
            med = np.nanmedian(np.array(mat, dtype=float), axis=0) if mat else []
        curves_out.setdefault(problem, {})[opt] = [
            None if not np.isfinite(v) else float(v) for v in med
        ]

    max_scaled = float((df["budget"] / df["dim"]).max())
    grid = np.geomspace(1.0, max_scaled, _ECDF_POINTS)
    ecdf = {"grid": grid.round(3).tolist(), "by_category": {}}
    for cat in ["all"] + categories:
        sub = df if cat == "all" else df[df["category"] == cat]
        ecdf["by_category"][cat] = {o: _ecdf(sub[sub["optimiser"] == o], gap_curves, grid) for o in optimisers}

    summary = {
        "name": manifest.get("name"),
        "created": manifest.get("created"),
        "elapsed_s": manifest.get("elapsed_s"),
        "environment": manifest.get("environment"),
        "n_runs": int(len(df)),
        "n_errors": int(manifest.get("n_errors", 0)),
        "n_skipped": int(manifest.get("n_skipped", 0)),
        "targets": list(TARGETS),
        "optimisers": optimisers,
        "categories": categories,
        "leaderboard": board(df),
        "leaderboard_by_category": {c: board(df[df["category"] == c]) for c in categories},
        "friedman": friedman(df),
        "task_rows": json.loads(task_rows.to_json(orient="records")),
        "curves": {"fractions": fractions.round(5).tolist(), "by_problem": curves_out},
        "ecdf": ecdf,
        "problems": manifest.get("problems", {}),
        "optimiser_cards": manifest.get("optimisers", {}),
        "taxonomy": manifest.get("taxonomy", {}),
        "config": manifest.get("config", {}),
    }
    summary = _sanitise(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_default, allow_nan=False))
    return summary


def _sanitise(o):
    """Replace non-finite floats with ``None`` so the JSON is browser-safe."""
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)
