"""
``optbench`` command-line interface.

    optbench run configs/bench.smoke.yaml --out runs/smoke [--jobs 4]
    optbench analyse runs/smoke                # recompute summary.json
    optbench site runs/smoke --out docs/dashboard/index.html
    optbench list problems|optimisers|ensembles|tags
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _list(what: str) -> None:
    from ..problems import PROBLEMS, TAGS
    from ..taxonomy import ENSEMBLE_CARDS, OPTIMISER_CARDS

    if what == "problems":
        for p in PROBLEMS.values():
            params = ", ".join(f"{k}={v!r}" for k, v in p.parameters.items())
            print(f"{p.id:20s} {p.category:13s} {p.encoding:11s} {p.sense}  {p.description}")
            if params:
                print(f"{'':20s} params: {params}")
    elif what == "optimisers":
        for c in OPTIMISER_CARDS.values():
            print(f"{c.key:13s} {c.family:13s} {c.search:12s} {'/'.join(c.encodings):27s} {c.name}")
    elif what == "ensembles":
        for c in ENSEMBLE_CARDS.values():
            print(f"{c.key:12s} {c.talbi_class}  {c.category:26s} {c.cooperation:20s} {c.name}")
    elif what == "tags":
        print("\n".join(sorted(TAGS)))
    else:
        raise SystemExit(f"unknown list target {what!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="optbench", description="Benchmark optimisers and ensembles.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark config")
    r.add_argument("config")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=None, help="worker processes (default: config n_jobs)")
    r.add_argument("--site", default=None, help="also write the dashboard to this path")
    r.add_argument("--quiet", action="store_true")

    a = sub.add_parser("analyse", help="recompute summary.json for a run")
    a.add_argument("run")

    s = sub.add_parser("site", help="build the dashboard HTML for a run")
    s.add_argument("run")
    s.add_argument("--out", required=True)

    l = sub.add_parser("list", help="list registered problems / optimisers / ensembles / tags")
    l.add_argument("what", choices=["problems", "optimisers", "ensembles", "tags"])

    args = parser.parse_args(argv)
    if args.cmd == "run":
        from .runner import run_benchmark
        from .site import build_site

        out = run_benchmark(args.config, args.out, n_jobs=args.jobs, progress=not args.quiet)
        if args.site:
            print("dashboard:", build_site(out, args.site))
    elif args.cmd == "analyse":
        from .analysis import summarise

        s = summarise(args.run)
        print(json.dumps(s.get("leaderboard", [])[:10], indent=1))
    elif args.cmd == "site":
        from .site import build_site

        print(build_site(args.run, args.out, refresh=True))
    else:
        _list(args.what)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
