"""
Run a benchmark: execute every task and persist the run directory.

Run-directory layout (mirrors clustbench)::

    <out>/
      manifest.json        config, environment, problem catalogue, cards
      results.csv          one row per run (raw facts: best value, evals, resources)
      results.parquet      same, when pyarrow is installed
      curves.csv           anytime curves: run_id, evaluations, value
      trajectories.jsonl   per-iteration state-action steps (record_trajectory: true)
      errors.jsonl         runs that raised an exception
      skipped.jsonl        optimiser/problem pairs with incompatible encodings
      summary.json         derived metrics, ranks, Friedman test, ECDFs, curves
                           (written by :func:`optim.bench.analysis.summarise`)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..ensembles._common import run_optimiser
from ..taxonomy import OptimiserCard, taxonomy_tree
from .config import BenchmarkConfig, Task, load_config, prepare
from .recorder import BudgetExhausted, Recorder

_RANDOM_REFERENCE_CACHE: Dict[Any, float] = {}


def _resources():
    try:
        import psutil

        proc = psutil.Process()
        t = proc.cpu_times()
        return t.user + t.system, proc.memory_info().rss
    except Exception:  # pragma: no cover - psutil optional
        return time.process_time(), 0


def run_task(task: Task) -> Dict[str, Any]:
    """Execute one task; never raises (errors are reported in the result)."""
    base = {
        "optimiser": task.optimiser.name,
        "entry": task.optimiser.entry,
        "problem_family": task.problem.family,
        "dim": task.problem.dim,
        "instance": task.problem.instance,
        "repeat": task.repeat,
        "seed": task.seed,
        "budget": task.budget,
    }
    try:
        native, solved, optimiser, solved_as = prepare(task)
    except ValueError as exc:  # incompatible optimiser/encoding pair
        return {"skipped": {**base, "reason": str(exc)[:500]}}
    except Exception as exc:
        return {"error": {**base, "error_type": type(exc).__name__, "error_message": str(exc)[:500]}}

    key = (task.problem,)
    if key not in _RANDOM_REFERENCE_CACHE:
        _RANDOM_REFERENCE_CACHE[key] = native.random_reference()
    random_ref = _RANDOM_REFERENCE_CACHE[key]

    card = task.optimiser.card
    solved.reseed_noise(task.seed)
    recorder = Recorder(solved, task.budget)
    kwargs = solved.optimise_kwargs()
    bounds = kwargs.pop("bounds")
    if task.record_trajectory:
        kwargs["record_trajectory"] = True

    status, error, result = "completed", None, None
    cpu0, rss0 = _resources()
    t0 = time.perf_counter()
    try:
        result = run_optimiser(optimiser, recorder, bounds, max_evaluations=task.budget,
                               seed=task.seed, extra_kwargs=kwargs)
    except BudgetExhausted:
        status = "budget_exhausted"
    except Exception as exc:  # one optimiser failing must not kill the run
        status, error = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
    wall = time.perf_counter() - t0
    cpu1, rss1 = _resources()

    run_id = uuid.uuid4().hex[:16]
    trajectory = getattr(result, "trajectory", None) if result is not None else None
    row = {
        "run_id": run_id,
        **base,
        "kind": card.kind,
        "optimiser_family": getattr(card, "family", None) or getattr(card, "category", None),
        "talbi_class": getattr(card, "talbi_class", None),
        "problem": native.name,
        "category": native.category,
        "encoding": native.encoding,
        "solved_as": solved_as,
        "sense": native.sense,
        "tags": ",".join(sorted(native.tags)),
        "problem_params": json.dumps(native.params, sort_keys=True, default=str),
        "optimum": native.optimum,
        "random_reference": random_ref,
        "best_value": recorder.best_true,
        "n_evaluations": recorder.n_evaluations,
        "status": status if recorder.best_true is not None else "error",
        "error": error,
        "wall_time_s": wall,
        "cpu_time_s": cpu1 - cpu0,
        "rss_delta_mb": (rss1 - rss0) / 2 ** 20,
        "n_steps": len(trajectory) if trajectory else None,
    }
    out: Dict[str, Any] = {"row": row, "curve": recorder.curve}
    if trajectory:
        out["trajectory"] = [asdict(s) for s in trajectory]
    return out


def _environment() -> Dict[str, Any]:
    import optim

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "optim": optim.__version__,
        "cpu_count": os.cpu_count(),
    }


def run_benchmark(
    config: Union[str, Path, Dict[str, Any], BenchmarkConfig],
    out_dir: Union[str, Path],
    *,
    n_jobs: Optional[int] = None,
    progress: bool = True,
    summarise: bool = True,
) -> Path:
    """Run every task of ``config`` and write the run directory.

    Returns the output directory path.
    """
    cfg = config if isinstance(config, BenchmarkConfig) else load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = cfg.tasks()
    jobs = n_jobs or cfg.n_jobs

    started = time.perf_counter()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outcomes = []
            for i, o in enumerate(pool.map(run_task, tasks, chunksize=4)):
                outcomes.append(o)
                if progress and (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  {i + 1}/{len(tasks)} runs", flush=True)
    else:
        outcomes = []
        for i, t in enumerate(tasks):
            outcomes.append(run_task(t))
            if progress and (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"  {i + 1}/{len(tasks)} runs", flush=True)

    rows, curves, trajectories, errors, skipped = [], [], [], [], []
    for o in outcomes:
        if "skipped" in o:
            skipped.append(o["skipped"])
            continue
        if "error" in o:
            errors.append(o["error"])
            continue
        rows.append(o["row"])
        rid = o["row"]["run_id"]
        curves.extend({"run_id": rid, "evaluations": e, "value": v} for e, v in o["curve"])
        for s in o.get("trajectory", []):
            trajectories.append({"run_id": rid, **s})
        if o["row"]["status"] == "error":
            errors.append({k: o["row"][k] for k in ("run_id", "optimiser", "problem", "error")})

    _write_table(out / "results", rows)
    _write_table(out / "curves", curves, parquet=False)
    _write_jsonl(out / "trajectories.jsonl", trajectories)
    _write_jsonl(out / "errors.jsonl", errors)
    _write_jsonl(out / "skipped.jsonl", skipped)

    catalogue = {}
    for p in cfg.problems:
        inst = p.build()
        catalogue[inst.name] = inst.describe()
    cards = {o.name: {**o.card.describe(), "spec": o.to_dict()} for o in cfg.optimisers}
    manifest = {
        "name": cfg.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "n_tasks": len(tasks),
        "n_runs": len(rows),
        "n_errors": len(errors),
        "n_skipped": len(skipped),
        "config": cfg.raw,
        "environment": _environment(),
        "problems": catalogue,
        "optimisers": cards,
        "taxonomy": taxonomy_tree(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if progress:
        print(f"{len(rows)} runs, {len(skipped)} incompatible pairs skipped, {len(errors)} errors "
              f"in {manifest['elapsed_s']}s -> {out}")

    if summarise:
        from .analysis import summarise as _summarise

        _summarise(out)
    return out


def _write_table(stem: Path, rows: List[Dict[str, Any]], parquet: bool = True) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    if parquet and len(df):
        try:
            df.to_parquet(stem.with_suffix(".parquet"), index=False)
        except Exception:
            pass


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
