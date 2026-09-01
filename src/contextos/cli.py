"""Command-line entry point for ContextOS."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from contextos import ContextOptimizer, __version__
from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    SlidingWindowBaseline,
)
from contextos.benchmarking import run_deduplication_benchmark, run_quick_benchmark
from contextos.benchmarks.artifacts import load_dataset, write_run_artifact
from contextos.benchmarks.models import BenchmarkRun
from contextos.benchmarks.positional import (
    load_positional_dataset,
    positional_summary,
    run_positional_benchmark,
    write_positional_run_artifact,
)
from contextos.benchmarks.positional_models import PositionalDataset
from contextos.benchmarks.runner import run_contextos_bench
from contextos.config import OptimizationPolicy
from contextos.errors import ContextOSError
from contextos.models import ContextEdge, ContextItem
from contextos.providers import DeterministicRetrievalProvider, LLMProvider, OpenAIProvider
from contextos.store import SQLiteContextStore
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
store_app = typer.Typer(help="Inspect durable ContextOS stores.")
app.add_typer(store_app, name="store")


class BaselineName(StrEnum):
    """Baseline strategies currently available through the CLI."""

    CONTEXTOS = "contextos"
    FULL = "full"
    LAST_N = "last-n"
    SLIDING_WINDOW = "sliding-window"


class PositionalProviderName(StrEnum):
    """Explicit providers available for the controlled positional experiment."""

    DETERMINISTIC = "deterministic"
    OPENAI = "openai"


@app.callback()
def main() -> None:
    """Run ContextOS commands."""


@app.command()
def version() -> None:
    """Print the installed ContextOS version."""
    typer.echo(__version__)


def _load_input(input_path: Path) -> tuple[list[ContextItem], list[ContextEdge]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("input JSON must be a list or an object containing an 'items' list")
    raw_edges = payload.get("edges", []) if isinstance(payload, dict) else []
    if not isinstance(raw_edges, list):
        raise ValueError("input JSON 'edges' must be a list")
    return (
        [ContextItem.model_validate(value) for value in raw_items],
        [ContextEdge.model_validate(value) for value in raw_edges],
    )


@app.command()
def optimize(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    budget: Annotated[int, typer.Option("--budget", min=1)],
    strategy: Annotated[BaselineName, typer.Option("--strategy")] = BaselineName.CONTEXTOS,
    reserve_output_tokens: Annotated[int, typer.Option("--reserve-output-tokens", min=0)] = 0,
    task: Annotated[str, typer.Option("--task")] = "",
    window_seconds: Annotated[int, typer.Option("--window-seconds", min=1)] = 3600,
    trace_json: Annotated[Path | None, typer.Option("--trace-json")] = None,
) -> None:
    """Construct context with ContextOS or a deterministic baseline."""
    try:
        items, edges = _load_input(input_path)
        policy = OptimizationPolicy(
            max_input_tokens=budget,
            reserve_output_tokens=reserve_output_tokens,
        )
        tokenizer = TiktokenTokenizer()
        if strategy is BaselineName.CONTEXTOS:
            result = ContextOptimizer(tokenizer=tokenizer, edges=edges).optimize(
                task, items, policy
            )
        else:
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
        typer.echo(
            "Benchmark failed: the callback supports only 'quick'; use 'benchmark run' "
            "for ContextOS-Bench",
            err=True,
        )
        raise typer.Exit(code=2)
    report = run_quick_benchmark(TiktokenTokenizer())
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


@benchmark_app.command("run")
def benchmark_run_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = Path("benchmarks/datasets/contextos_bench.json"),
    output_directory: Annotated[Path, typer.Option("--output-directory")] = Path(
        "benchmarks/results"
    ),
    case_limit: Annotated[int | None, typer.Option("--case-limit", min=1)] = None,
) -> None:
    """Run ContextOS-Bench and write an immutable raw result artifact."""
    try:
        dataset = load_dataset(input_path)
        run = run_contextos_bench(
            dataset,
            tokenizer=TiktokenTokenizer(),
            case_limit=case_limit,
        )
        artifact_path = write_run_artifact(run, output_directory)
    except (ContextOSError, OSError, ValueError, ValidationError) as exc:
        typer.echo(f"ContextOS-Bench failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "run_id": run.run_id,
                "artifact": str(artifact_path),
                "case_count": run.metadata["case_count"],
                "aggregates": [aggregate.model_dump(mode="json") for aggregate in run.aggregates],
            },
            indent=2,
            sort_keys=True,
        )
    )


@benchmark_app.command("positional")
def benchmark_positional_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = Path("benchmarks/datasets/positional_retrieval.json"),
    output_directory: Annotated[Path, typer.Option("--output-directory")] = Path(
        "benchmarks/results"
    ),
    profile: Annotated[str, typer.Option("--profile")] = "quick",
    provider_name: Annotated[
        PositionalProviderName,
        typer.Option("--provider"),
    ] = PositionalProviderName.DETERMINISTIC,
    model: Annotated[str | None, typer.Option("--model")] = None,
    max_context_tokens: Annotated[
        int,
        typer.Option("--max-context-tokens", min=288),
    ] = 32_800,
) -> None:
    """Run the controlled positional-retrieval experiment."""
    try:
        if profile not in {"quick", "full"}:
            raise ValueError("positional profile must be 'quick' or 'full'")
        dataset = load_positional_dataset(input_path)
        if profile == "quick":
            shortest = min(case.target_context_tokens for case in dataset.cases)
            dataset = PositionalDataset(
                name=dataset.name,
                generator_version=dataset.generator_version,
                generation_seed=dataset.generation_seed,
                cases=[
                    case
                    for case in dataset.cases
                    if case.target_context_tokens == shortest and case.repetition == 0
                ],
            )
        provider: LLMProvider
        if provider_name is PositionalProviderName.OPENAI:
            if model is None or not model.strip():
                raise ValueError("--model is required for an OpenAI positional run")
            provider = OpenAIProvider(model=model, temperature=0.0)
            provider_model = model
        else:
            if model is not None:
                raise ValueError("--model is valid only with --provider openai")
            provider = DeterministicRetrievalProvider()
            provider_model = provider.model
        run = run_positional_benchmark(
            dataset,
            provider=provider,
            provider_name=provider_name.value,
            provider_model=provider_model,
            tokenizer=TiktokenTokenizer(),
            profile=profile,
            max_context_tokens=max_context_tokens,
        )
        artifact = write_positional_run_artifact(run, output_directory)
    except (ContextOSError, OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Positional benchmark failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    summary = positional_summary(run)
    summary["artifact"] = str(artifact)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def inspect(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Inspect input composition without running optimization."""
    try:
        items, edges = _load_input(input_path)
        tokenizer = TiktokenTokenizer()
        by_type: dict[str, int] = {}
        for item in items:
            by_type[item.type.value] = by_type.get(item.type.value, 0) + 1
        report = {
            "item_count": len(items),
            "edge_count": len(edges),
            "mandatory_count": sum(item.mandatory for item in items),
            "token_count": sum(tokenizer.count_tokens(item.content) for item in items),
            "items_by_type": dict(sorted(by_type.items())),
        }
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        typer.echo(f"Inspection failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@benchmark_app.command("compare")
def benchmark_compare(
    left: Annotated[Path, typer.Option("--left", exists=True, dir_okay=False)],
    right: Annotated[Path, typer.Option("--right", exists=True, dir_okay=False)],
) -> None:
    """Compare aggregate metrics from two runs of the same dataset."""
    try:
        left_run = BenchmarkRun.model_validate_json(left.read_text(encoding="utf-8"))
        right_run = BenchmarkRun.model_validate_json(right.read_text(encoding="utf-8"))
        if left_run.dataset_sha256 != right_run.dataset_sha256:
            raise ValueError("benchmark runs use different datasets")
        left_aggregates = {aggregate.strategy: aggregate for aggregate in left_run.aggregates}
        right_aggregates = {aggregate.strategy: aggregate for aggregate in right_run.aggregates}
        shared_strategies = sorted(set(left_aggregates) & set(right_aggregates))
        comparison = {
            strategy: {
                "task_score_delta": right_aggregates[strategy].mean_task_specific_score
                - left_aggregates[strategy].mean_task_specific_score,
                "cir_delta": right_aggregates[strategy].mean_critical_information_recall
                - left_aggregates[strategy].mean_critical_information_recall,
                "input_token_delta": right_aggregates[strategy].mean_input_tokens
                - left_aggregates[strategy].mean_input_tokens,
                "p95_latency_ms_delta": right_aggregates[strategy].p95_optimizer_latency_ms
                - left_aggregates[strategy].p95_optimizer_latency_ms,
            }
            for strategy in shared_strategies
        }
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        typer.echo(f"Benchmark comparison failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(comparison, indent=2, sort_keys=True))


@store_app.command("stats")
def store_stats(
    database: Annotated[Path, typer.Option("--database", exists=True, dir_okay=False)],
) -> None:
    """Print deterministic item, edge, type, and tier counts for a SQLite store."""
    try:
        with SQLiteContextStore(database) as store:
            items = store.list_items()
            edges = store.load_dependencies()
        report = {
            "item_count": len(items),
            "edge_count": len(edges),
            "items_by_type": {
                context_type: sum(item.type.value == context_type for item in items)
                for context_type in sorted({item.type.value for item in items})
            },
            "items_by_tier": {
                tier: sum(item.lifecycle_tier.value == tier for item in items)
                for tier in sorted({item.lifecycle_tier.value for item in items})
            },
        }
    except (ContextOSError, OSError, ValueError) as exc:
        typer.echo(f"Store inspection failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
