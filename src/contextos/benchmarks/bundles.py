"""Immutable, multi-file benchmark artifact bundles."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from contextos import __version__

REQUIRED_BUNDLE_FILES = frozenset(
    {
        "config.json",
        "environment.json",
        "cases.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "metrics.csv",
        "report.md",
    }
)
CORE_DEPENDENCIES = ("numpy", "pydantic", "tiktoken", "typer")


class BenchmarkEnvironment(BaseModel):
    """Execution provenance required for every meaningful benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recorded_at_utc: datetime
    python_version: str
    contextos_version: str
    git_sha: str
    operating_system: str
    dependency_versions: dict[str, str]
    embedding_provider: str
    embedding_model: str
    llm_provider: str | None = None
    llm_model: str | None = None


class LoadedBenchmarkBundle(BaseModel):
    """Validated contents loaded from the canonical seven-file representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: dict[str, Any]
    environment: BenchmarkEnvironment
    cases: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    metrics: dict[str, Any]
    metric_rows: list[dict[str, str]]
    report: str


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unavailable"


def capture_environment(
    *,
    recorded_at_utc: datetime,
    embedding_provider: str,
    embedding_model: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    dependency_names: Sequence[str] = (),
) -> BenchmarkEnvironment:
    """Capture stable runtime identity without importing optional dependencies."""
    names = sorted(set((*CORE_DEPENDENCIES, *dependency_names)))
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return BenchmarkEnvironment(
        recorded_at_utc=recorded_at_utc.astimezone(UTC),
        python_version=sys.version.split()[0],
        contextos_version=__version__,
        git_sha=_git_sha(),
        operating_system=platform.platform(),
        dependency_versions=versions,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _json_document(value: Any) -> str:
    return json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n"


def _json_lines(values: Sequence[Any]) -> str:
    return "".join(
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )


def _csv_document(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (BaseModel, dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("benchmark bundle labels must contain letters or digits")
    return slug


def _bundle_name(recorded_at_utc: datetime, strategy: str, profile: str) -> str:
    if recorded_at_utc.utcoffset() is None:
        raise ValueError("benchmark bundle timestamp must be timezone-aware")
    timestamp = recorded_at_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{_slug(strategy)}-{_slug(profile)}"


def _verify_existing(path: Path, expected: Mapping[str, str]) -> None:
    if not path.is_dir():
        raise ValueError(f"benchmark bundle path is not a directory: {path}")
    actual_names = {entry.name for entry in path.iterdir()}
    if actual_names != REQUIRED_BUNDLE_FILES:
        raise ValueError(f"benchmark bundle is incomplete or modified: {path}")
    for name, content in expected.items():
        if (path / name).read_text(encoding="utf-8") != content:
            raise ValueError(f"benchmark bundle collision at {path / name}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"benchmark bundle file must contain a JSON object: {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"benchmark JSONL line must be an object: {path}:{line_number}")
        values.append(value)
    if not values:
        raise ValueError(f"benchmark JSONL file contains no records: {path}")
    return values


def load_benchmark_bundle(path: Path) -> LoadedBenchmarkBundle:
    """Validate the exact file contract and deserialize every bundle representation."""
    if not path.is_dir() or {entry.name for entry in path.iterdir()} != REQUIRED_BUNDLE_FILES:
        raise ValueError(f"benchmark bundle is incomplete or modified: {path}")
    report = (path / "report.md").read_text(encoding="utf-8")
    if not report.strip():
        raise ValueError("benchmark report must not be empty")
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as source:
        metric_rows = list(csv.DictReader(source))
    return LoadedBenchmarkBundle(
        config=_load_json_object(path / "config.json"),
        environment=BenchmarkEnvironment.model_validate(
            _load_json_object(path / "environment.json")
        ),
        cases=_load_jsonl_objects(path / "cases.jsonl"),
        predictions=_load_jsonl_objects(path / "predictions.jsonl"),
        metrics=_load_json_object(path / "metrics.json"),
        metric_rows=metric_rows,
        report=report,
    )


def write_benchmark_bundle(
    *,
    output_directory: Path,
    recorded_at_utc: datetime,
    strategy: str,
    profile: str,
    config: Mapping[str, Any] | BaseModel,
    environment: BenchmarkEnvironment,
    cases: Sequence[Any],
    predictions: Sequence[Any],
    metrics: Mapping[str, Any] | BaseModel,
    metric_rows: Sequence[Mapping[str, Any]],
    report: str,
) -> Path:
    """Write exactly seven files and refuse mutation of an existing bundle."""
    if not cases:
        raise ValueError("benchmark bundle requires at least one case")
    if not predictions:
        raise ValueError("benchmark bundle requires at least one prediction")
    if not metric_rows:
        raise ValueError("benchmark bundle requires at least one metric row")
    if not report.strip():
        raise ValueError("benchmark bundle report must not be empty")
    if environment.recorded_at_utc != recorded_at_utc.astimezone(UTC):
        raise ValueError("environment timestamp must match the benchmark run timestamp")
    normalized_report = report.rstrip() + "\n"
    contents = {
        "config.json": _json_document(config),
        "environment.json": _json_document(environment),
        "cases.jsonl": _json_lines(cases),
        "predictions.jsonl": _json_lines(predictions),
        "metrics.json": _json_document(metrics),
        "metrics.csv": _csv_document(metric_rows),
        "report.md": normalized_report,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / _bundle_name(recorded_at_utc, strategy, profile)
    try:
        path.mkdir()
    except FileExistsError:
        _verify_existing(path, contents)
        return path
    for name, content in contents.items():
        with (path / name).open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    return path
