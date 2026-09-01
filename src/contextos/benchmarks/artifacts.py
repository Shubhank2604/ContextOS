"""Content-addressed immutable benchmark artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

from contextos.benchmarks.models import BenchmarkRun, ContextOSBenchDataset


def load_dataset(path: Path) -> ContextOSBenchDataset:
    """Load and validate a versioned ContextOS-Bench dataset."""
    return ContextOSBenchDataset.model_validate_json(path.read_text(encoding="utf-8"))


def write_run_artifact(run: BenchmarkRun, output_directory: Path) -> Path:
    """Write one immutable run, refusing conflicting reuse of its content ID."""
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"contextos-bench-{run.run_id}.json"
    content = run.model_dump_json(indent=2)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8")
        if json.loads(existing) != json.loads(content):
            raise ValueError(f"benchmark artifact ID collision at {path}") from None
    return path
