"""Command-line entry point for ContextOS."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from contextos import __version__
from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    SlidingWindowBaseline,
)
from contextos.benchmarking import run_deduplication_benchmark, run_quick_baseline_benchmark
from contextos.config import OptimizationPolicy
from contextos.errors import ContextOSError
from contextos.models import ContextItem
from contextos.tokenization import TiktokenTokenizer

app = typer.Typer(
    name="contextos",
    help="Construct and inspect LLM context under explicit token budgets.",
    no_args_is_help=True,
)
benchmark_app = typer.Typer(
    help="Run reproducible ContextOS benchmark profiles.",
    invoke_without_command=True,
)
app.add_typer(benchmark_app, name="benchmark")


class BaselineName(StrEnum):
    """Baseline strategies currently available through the CLI."""

    FULL = "full"
    LAST_N = "last-n"
    SLIDING_WINDOW = "sliding-window"


@app.callback()
def main() -> None:
    """Run ContextOS commands."""


@app.command()
def version() -> None:
    """Print the installed ContextOS version."""
    typer.echo(__version__)


def _load_items(input_path: Path) -> list[ContextItem]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("input JSON must be a list or an object containing an 'items' list")
    return [ContextItem.model_validate(value) for value in raw_items]


@app.command()
def optimize(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    budget: Annotated[int, typer.Option("--budget", min=1)],
    strategy: Annotated[BaselineName, typer.Option("--strategy")] = BaselineName.FULL,
    reserve_output_tokens: Annotated[int, typer.Option("--reserve-output-tokens", min=0)] = 0,
    task: Annotated[str, typer.Option("--task")] = "",
    window_seconds: Annotated[int, typer.Option("--window-seconds", min=1)] = 3600,
    trace_json: Annotated[Path | None, typer.Option("--trace-json")] = None,
) -> None:
    """Construct context with a deterministic baseline strategy."""
    try:
        items = _load_items(input_path)
        policy = OptimizationPolicy(
            max_input_tokens=budget,
            reserve_output_tokens=reserve_output_tokens,
        )
        tokenizer = TiktokenTokenizer()
        baseline: BaselineStrategy
        if strategy is BaselineName.FULL:
            baseline = FullContextBaseline()
        elif strategy is BaselineName.LAST_N:
            baseline = LastNTokensBaseline()
        else:
            baseline = SlidingWindowBaseline(window_seconds=window_seconds)
        result = baseline.optimize(
            task=task,
            items=items,
            policy=policy,
            tokenizer=tokenizer,
        )
        if trace_json is not None:
            trace_json.parent.mkdir(parents=True, exist_ok=True)
            trace_json.write_text(result.trace.model_dump_json(indent=2), encoding="utf-8")
    except (ContextOSError, OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        typer.echo(f"Optimization failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Strategy: {result.trace.strategy}")
    typer.echo(f"Input tokens: {result.original_token_count}")
    typer.echo(
        f"Final tokens: {result.final_token_count}/{result.budget_allocation.effective_budget}"
    )
    typer.echo(
        f"Selected items: {', '.join(item.id for item in result.selected_items) or '(none)'}"
    )


@benchmark_app.callback()
def benchmark(
    ctx: typer.Context,
    profile: Annotated[str, typer.Option("--profile")] = "quick",
) -> None:
    """Run a benchmark profile when no benchmark subcommand is supplied."""
    if ctx.invoked_subcommand is not None:
        return
    if profile != "quick":
        typer.echo("Benchmark failed: Milestone 1 supports only the 'quick' profile", err=True)
        raise typer.Exit(code=2)
    report = run_quick_baseline_benchmark(TiktokenTokenizer())
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@benchmark_app.command("dedup")
def benchmark_deduplication(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = Path("benchmarks/datasets/deduplication_cases.json"),
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)] = 0.92,
) -> None:
    """Measure deduplication precision, recall, F1, and false positives."""
    try:
        metrics = run_deduplication_benchmark(input_path, threshold=threshold)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        typer.echo(f"Deduplication benchmark failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
