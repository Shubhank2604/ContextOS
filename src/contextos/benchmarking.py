"""Small deterministic benchmark harness used by Milestone 1 and CI."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    NaiveExtractiveBaseline,
    RelevanceOnlyBaseline,
    SlidingWindowBaseline,
)
from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.config import OptimizationPolicy
from contextos.dedup.metrics import (
    DeduplicationCase,
    DeduplicationMetrics,
    deduplication_predictions,
    evaluate_deduplication_cases,
)
from contextos.embeddings import DeterministicEmbeddingProvider
from contextos.errors import ContextBudgetOverflow
from contextos.models import ContextItem, ContextType
from contextos.optimizer import ContextOptimizer
from contextos.tokenization import Tokenizer


def _fixture_items() -> list[ContextItem]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        ContextItem(
            id="old-decision",
            content="Use stateless authentication.",
            type=ContextType.DECISION,
            created_at=start,
            updated_at=start,
            importance=0.9,
        ),
        ContextItem(
            id="recent-noise",
            content="The formatting check passed.",
            type=ContextType.TOOL_OUTPUT,
            created_at=start + timedelta(hours=2),
            updated_at=start + timedelta(hours=2),
            importance=0.2,
        ),
        ContextItem(
            id="current-task",
            content="Fix the authentication timeout.",
            type=ContextType.USER_MESSAGE,
            created_at=start + timedelta(hours=3),
            updated_at=start + timedelta(hours=3),
            importance=1.0,
        ),
    ]


def run_quick_baseline_benchmark(tokenizer: Tokenizer) -> dict[str, Any]:
    """Run the same deterministic fixture through every Milestone 1 baseline."""
    items = _fixture_items()
    total_tokens = sum(tokenizer.count_tokens(item.content) for item in items)
    policy = OptimizationPolicy(max_input_tokens=max(total_tokens - 1, 1), reserve_output_tokens=0)
    strategies: list[BaselineStrategy] = [
        FullContextBaseline(),
        LastNTokensBaseline(),
        SlidingWindowBaseline(window_seconds=2 * 60 * 60),
        RelevanceOnlyBaseline(),
        NaiveExtractiveBaseline(),
    ]
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        try:
            result = strategy.optimize(
                task="Fix the authentication timeout.",
                items=items,
                policy=policy,
                tokenizer=tokenizer,
            )
            results.append(
                {
                    "strategy": strategy.name,
                    "status": "ok",
                    "original_tokens": result.original_token_count,
                    "final_tokens": result.final_token_count,
                    "selected_item_ids": [item.id for item in result.selected_items],
                    "compressed_count": result.trace.compressed_count,
                    "warnings": result.trace.warnings,
                }
            )
        except ContextBudgetOverflow as exc:
            results.append(
                {
                    "strategy": strategy.name,
                    "status": "overflow",
                    "required_tokens": exc.required_tokens,
                    "effective_budget": exc.effective_budget,
                }
            )
    return {"profile": "quick", "case_count": 1, "results": results}


def run_quick_benchmark(tokenizer: Tokenizer) -> dict[str, Any]:
    """Run one fixture through baseline and ContextOS strategies consistently."""
    report = run_quick_baseline_benchmark(tokenizer)
    items = _fixture_items()
    total_tokens = sum(tokenizer.count_tokens(item.content) for item in items)
    policy = OptimizationPolicy(
        max_input_tokens=max(total_tokens - 1, 1),
        reserve_output_tokens=0,
        minimum_compressed_tokens=1,
    )
    result = ContextOptimizer(tokenizer=tokenizer).optimize(
        "Fix the authentication timeout.", items, policy
    )
    report["results"].append(
        {
            "strategy": result.trace.strategy,
            "status": "ok",
            "original_tokens": result.original_token_count,
            "final_tokens": result.final_token_count,
            "selected_item_ids": [item.id for item in result.selected_items],
            "compressed_count": result.trace.compressed_count,
            "warnings": result.trace.warnings,
        }
    )
    return report


def run_deduplication_benchmark(
    fixture_path: Path,
    *,
    threshold: float = 0.92,
) -> DeduplicationMetrics:
    """Evaluate Phase 3A deduplication against a labeled JSON fixture."""
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[DeduplicationCase]).validate_python(payload)
    return evaluate_deduplication_cases(
        cases,
        provider=DeterministicEmbeddingProvider(),
        threshold=threshold,
    )


def write_deduplication_benchmark_bundle(
    fixture_path: Path,
    output_directory: Path,
    *,
    threshold: float = 0.92,
) -> tuple[Path, DeduplicationMetrics]:
    """Run the labeled fixture and persist raw decisions plus derived metrics."""
    raw = fixture_path.read_text(encoding="utf-8")
    cases = TypeAdapter(list[DeduplicationCase]).validate_json(raw)
    provider = DeterministicEmbeddingProvider()
    predictions = deduplication_predictions(cases, provider=provider, threshold=threshold)
    metrics = evaluate_deduplication_cases(cases, provider=provider, threshold=threshold)
    recorded_at = datetime.now(UTC)
    path = write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=recorded_at,
        strategy="exact-semantic-deduplication",
        profile="fixture",
        config={
            "schema_version": "1.0",
            "benchmark": "deduplication-quality",
            "profile": "fixture",
            "fixture_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "semantic_threshold": threshold,
        },
        environment=capture_environment(
            recorded_at_utc=recorded_at,
            embedding_provider="deterministic",
            embedding_model=provider.model_name,
        ),
        cases=cases,
        predictions=predictions,
        metrics=metrics,
        metric_rows=[metrics.model_dump(mode="json")],
        report=(
            "# Deduplication Quality Report\n\n"
            f"- Cases: {metrics.case_count}\n"
            f"- Precision: {metrics.precision:.4f}\n"
            f"- Recall: {metrics.recall:.4f}\n"
            f"- F1: {metrics.f1:.4f}\n\n"
            "Raw `cases.jsonl` and `predictions.jsonl` are authoritative; "
            "this report is derived."
        ),
    )
    return path, metrics
