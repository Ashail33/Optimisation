"""
Build the static dashboard for a benchmark run.

The dashboard is a single self-contained HTML file: the run's
``summary.json`` is embedded in the page, so it opens straight from disk,
from GitHub Pages, or as a shared artifact, with no server.  Only Chart.js is
loaded from a CDN.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Union

from .analysis import summarise


def build_site(run_dir: Union[str, Path], out: Union[str, Path], *, refresh: bool = False) -> Path:
    """Write the dashboard for ``run_dir`` to ``out`` (an ``.html`` path or a
    directory, which receives ``index.html``).  Returns the HTML path."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if refresh or not summary_path.exists():
        summarise(run_dir)
    summary = json.loads(summary_path.read_text())

    template = resources.files("optim.bench").joinpath("templates/dashboard.html").read_text()
    payload = json.dumps(summary, separators=(",", ":")).replace("</", "<\\/")
    title = f"Benchmark {summary.get('name') or ''}".strip()
    html = template.replace("__TITLE__", title).replace("__DATA__", payload)

    out = Path(out)
    if out.suffix != ".html":
        out.mkdir(parents=True, exist_ok=True)
        out = out / "index.html"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
