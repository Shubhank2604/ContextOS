"""Controlled positional-retrieval dataset generation and execution."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev, pvariance
from time import perf_counter

from contextos import __version__
from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.benchmarks.metrics import percentile
from contextos.benchmarks.positional_models import (
    EvidencePosition,
    PositionAccuracy,
    PositionalCase,
    PositionalDataset,
    PositionalPerformance,
    PositionalPrediction,
    PositionalRobustness,
    PositionalRun,
    PositionalStrategy,
)
from contextos.layout import PositionAwareLayout, RelevanceDescendingLayout
from contextos.models import ContextItem, ContextType
from contextos.providers import LLMProvider
from contextos.scoring import ScoreBreakdown
from contextos.tokenization import Tokenizer

GENERATOR_VERSION = "1.0.0"
GENERATION_SEED = 2409
REQUIRED_CONTEXT_LENGTHS = (4_096, 8_192, 16_384, 32_768)
REQUIRED_POSITIONS = tuple(EvidencePosition)
DEFAULT_STRATEGIES = tuple(PositionalStrategy)

_INSTRUCTION = (
    "Use the records below to answer the query. Return only the exact value, with no "
    "explanation.\nCONTEXT_START\n"
)
_SUFFIX = "CONTEXT_END\nQUERY_KEY: {key}\nANSWER:"
_RECORD_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def build_positional_dataset(
    *,
    context_lengths: Sequence[int] = REQUIRED_CONTEXT_LENGTHS,
    repetitions: int = 1,
    generation_seed: int = GENERATION_SEED,
) -> PositionalDataset:
    """Build the deterministic context-length by evidence-position parameter grid."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not context_lengths or any(length < 256 for length in context_lengths):
        raise ValueError("context lengths must be non-empty and at least 256 tokens")
    if len(context_lengths) != len(set(context_lengths)):
        raise ValueError("context lengths must be unique")
    cases: list[PositionalCase] = []
    for length in context_lengths:
        for position in REQUIRED_POSITIONS:
            for repetition in range(repetitions):
                case_seed = (
                    generation_seed + length * 101 + int(position.fraction * 100) + repetition
                )
                length_label = f"{length // 1024:02d}k" if length % 1_024 == 0 else f"{length:05d}t"
                position_label = position.value.replace("_percent", "pct")
                case_id = f"pos-{length_label}-{position_label}-r{repetition:02d}"
                identifier = (
                    hashlib.sha256(f"{case_seed}|{case_id}".encode()).hexdigest()[:12].upper()
                )
                cases.append(
                    PositionalCase(
                        id=case_id,
                        target_context_tokens=length,
                        evidence_position=position,
                        repetition=repetition,
                        seed=case_seed,
                        target_key=f"TARGET_{identifier}",
                        expected_value=f"ANSWER_{identifier}",
                    )
                )
    return PositionalDataset(
        generator_version=GENERATOR_VERSION,
        generation_seed=generation_seed,
        cases=cases,
    )


def write_positional_dataset(dataset: PositionalDataset, path: Path) -> None:
    """Write a compact, versioned positional parameter grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def load_positional_dataset(path: Path) -> PositionalDataset:
    """Load and validate a positional parameter grid."""
    return PositionalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _distractor(case: PositionalCase, index: int) -> str:
    digest = hashlib.sha256(f"{case.seed}|{index}".encode()).hexdigest().upper()
    return f"RECORD DISTRACTOR_{index:06d}_{digest[:8]} => VALUE_{digest[8:24]}"


def _evidence(case: PositionalCase) -> str:
    return f"RECORD {case.target_key} => {case.expected_value}"


def _ordered_records(
    case: PositionalCase,
    distractor_count: int,
    strategy: PositionalStrategy,
) -> list[str]:
    distractors = [_distractor(case, index) for index in range(distractor_count)]
    insertion_index = round(case.evidence_position.fraction * distractor_count)
    records = [
        *distractors[:insertion_index],
        _evidence(case),
        *distractors[insertion_index:],
    ]
    if strategy is PositionalStrategy.ORIGINAL_FULL:
        return records
    items = [
        ContextItem(
            id=f"{case.id}-record-{index:06d}",
            content=record,
            type=ContextType.RETRIEVED_DOCUMENT,
            created_at=_RECORD_TIME,
            updated_at=_RECORD_TIME,
            importance=1.0 if record == _evidence(case) else 0.0,
        )
        for index, record in enumerate(records)
    ]
    scores = {
        item.id: ScoreBreakdown(
            relevance=1.0 if item.content == _evidence(case) else 0.0,
            importance=item.importance,
            recency=0.0,
            novelty=0.0,
            dependency=0.0,
            type_priority=0.0,
            composite_utility=1.0 if item.content == _evidence(case) else 0.0,
        )
        for item in items
    }
    original_positions = {item.id: index for index, item in enumerate(items)}
    layout = (
        RelevanceDescendingLayout()
        if strategy is PositionalStrategy.RELEVANCE_DESCENDING
        else PositionAwareLayout()
    )
    return [
        item.content
        for item in layout.arrange(
            items,
            scores=scores,
            original_positions=original_positions,
        )
    ]


def _prompt(case: PositionalCase, records: Sequence[str]) -> str:
    return _INSTRUCTION + "\n".join(records) + "\n" + _SUFFIX.format(key=case.target_key)


def _largest_fitting_distractor_count(
    case: PositionalCase,
    tokenizer: Tokenizer,
) -> int:
    """Find the largest whole-record prompt that does not exceed the target bucket."""

    def token_count(count: int) -> int:
        records = _ordered_records(case, count, PositionalStrategy.ORIGINAL_FULL)
        return tokenizer.count_tokens(_prompt(case, records))

    if token_count(1) > case.target_context_tokens:
        raise ValueError(f"target context length is too small for case {case.id}")
    low = 1
    high = max(2, case.target_context_tokens // 4)
    while token_count(high) <= case.target_context_tokens:
        low = high
        high *= 2
    best = low
    while low <= high:
        middle = (low + high) // 2
        if token_count(middle) <= case.target_context_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def construct_positional_prompt(
    case: PositionalCase,
    *,
    strategy: PositionalStrategy,
    tokenizer: Tokenizer,
    distractor_count: int | None = None,
) -> tuple[str, int, float]:
    """Construct one prompt and return its distractor count and evidence fraction."""
    count = distractor_count or _largest_fitting_distractor_count(case, tokenizer)
    if count <= 0:
        raise ValueError("distractor count must be positive")
    records = _ordered_records(case, count, strategy)
    prompt = _prompt(case, records)
    evidence = _evidence(case)
    evidence_offset = prompt.index(evidence)
    total_tokens = tokenizer.count_tokens(prompt)
    prefix_tokens = tokenizer.count_tokens(prompt[:evidence_offset])
    fraction = 0.0 if total_tokens == 0 else prefix_tokens / total_tokens
    return prompt, count, fraction


def normalize_exact_match(value: str) -> str:
    """Normalize only surrounding whitespace and case for deterministic exact match."""
    return value.strip().casefold()


def _prediction(
    case: PositionalCase,
    strategy: PositionalStrategy,
    *,
    provider: LLMProvider,
    tokenizer: Tokenizer,
    distractor_count: int,
) -> PositionalPrediction:
    prompt, _, evidence_fraction = construct_positional_prompt(
        case,
        strategy=strategy,
        tokenizer=tokenizer,
        distractor_count=distractor_count,
    )
    estimated_tokens = tokenizer.count_tokens(prompt)
    started = perf_counter()
    response = provider.complete(prompt, max_output_tokens=32)
    latency_ms = (perf_counter() - started) * 1_000
    normalized = normalize_exact_match(response.text)
    return PositionalPrediction(
        case_id=case.id,
        strategy=strategy,
        target_context_tokens=case.target_context_tokens,
        evidence_position=case.evidence_position,
        actual_evidence_fraction=evidence_fraction,
        distractor_count=distractor_count,
        expected_value=case.expected_value,
        raw_prediction=response.text,
        normalized_prediction=normalized,
        exact_match=normalized == normalize_exact_match(case.expected_value),
        estimated_input_tokens=estimated_tokens,
        provider_input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cached_tokens,
        model_ttft_ms=response.ttft_ms,
        model_total_latency_ms=latency_ms,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def aggregate_positional_predictions(
    predictions: Sequence[PositionalPrediction],
) -> tuple[list[PositionAccuracy], list[PositionalRobustness]]:
    """Compute accuracy cells and positional dispersion from raw predictions."""
    cells: list[PositionAccuracy] = []
    cell_keys = sorted(
        {
            (prediction.strategy, prediction.target_context_tokens, prediction.evidence_position)
            for prediction in predictions
        },
        key=lambda value: (value[0].value, value[1], value[2].fraction),
    )
    for strategy, target_tokens, position in cell_keys:
        values = [
            prediction
            for prediction in predictions
            if prediction.strategy is strategy
            and prediction.target_context_tokens == target_tokens
            and prediction.evidence_position is position
        ]
        correct = sum(prediction.exact_match for prediction in values)
        cells.append(
            PositionAccuracy(
                strategy=strategy,
                target_context_tokens=target_tokens,
                evidence_position=position,
                case_count=len(values),
                correct_count=correct,
                accuracy=correct / len(values),
                mean_input_tokens=mean(
                    prediction.provider_input_tokens or prediction.estimated_input_tokens
                    for prediction in values
                ),
                mean_model_latency_ms=mean(
                    prediction.model_total_latency_ms for prediction in values
                ),
            )
        )
    robustness: list[PositionalRobustness] = []
    for strategy, target_tokens in sorted(
        {(cell.strategy, cell.target_context_tokens) for cell in cells},
        key=lambda value: (value[0].value, value[1]),
    ):
        matching = [
            cell
            for cell in cells
            if cell.strategy is strategy and cell.target_context_tokens == target_tokens
        ]
        by_position = {cell.evidence_position: cell.accuracy for cell in matching}
        accuracies = list(by_position.values())
        robustness.append(
            PositionalRobustness(
                strategy=strategy,
                target_context_tokens=target_tokens,
                accuracy_by_position=by_position,
                mean_accuracy=mean(accuracies),
                max_min_positional_gap=max(accuracies) - min(accuracies),
                positional_variance=pvariance(accuracies),
                positional_std_dev=pstdev(accuracies),
            )
        )
    return cells, robustness


def aggregate_positional_performance(
    predictions: Sequence[PositionalPrediction],
) -> list[PositionalPerformance]:
    """Aggregate provider telemetry independently for each layout strategy."""
    performance: list[PositionalPerformance] = []
    for strategy in sorted(
        {prediction.strategy for prediction in predictions}, key=lambda x: x.value
    ):
        values = [prediction for prediction in predictions if prediction.strategy is strategy]
        latencies = [prediction.model_total_latency_ms for prediction in values]
        ttfts = [
            prediction.model_ttft_ms
            for prediction in values
            if prediction.model_ttft_ms is not None
        ]
        output_tokens = [
            prediction.output_tokens
            for prediction in values
            if prediction.output_tokens is not None
        ]
        cached_tokens = [
            prediction.cached_tokens
            for prediction in values
            if prediction.cached_tokens is not None
        ]
        performance.append(
            PositionalPerformance(
                strategy=strategy,
                prediction_count=len(values),
                mean_estimated_input_tokens=mean(
                    prediction.provider_input_tokens
                    if prediction.provider_input_tokens is not None
                    else prediction.estimated_input_tokens
                    for prediction in values
                ),
                total_model_latency_ms=sum(latencies),
                p50_model_latency_ms=percentile(latencies, 0.5),
                p95_model_latency_ms=percentile(latencies, 0.95),
                mean_model_ttft_ms=mean(ttfts) if ttfts else None,
                total_output_tokens=sum(output_tokens) if output_tokens else None,
                total_cached_tokens=sum(cached_tokens) if cached_tokens else None,
            )
        )
    return performance


def run_positional_benchmark(
    dataset: PositionalDataset,
    *,
    provider: LLMProvider,
    provider_name: str,
    provider_model: str,
    tokenizer: Tokenizer,
    profile: str,
    max_context_tokens: int,
    strategies: Sequence[PositionalStrategy] = DEFAULT_STRATEGIES,
) -> PositionalRun:
    """Execute supported cases with identical provider settings for every layout."""
    if max_context_tokens <= 0:
        raise ValueError("max context tokens must be positive")
    if not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("positional strategies must be non-empty and unique")
    requested_lengths = sorted({case.target_context_tokens for case in dataset.cases})
    executed_lengths = [length for length in requested_lengths if length + 32 <= max_context_tokens]
    skipped_lengths = sorted(set(requested_lengths) - set(executed_lengths))
    selected_cases = [
        case for case in dataset.cases if case.target_context_tokens in executed_lengths
    ]
    if not selected_cases:
        raise ValueError("provider context limit excludes every requested context length")
    predictions: list[PositionalPrediction] = []
    for case in selected_cases:
        distractor_count = _largest_fitting_distractor_count(case, tokenizer)
        predictions.extend(
            _prediction(
                case,
                strategy,
                provider=provider,
                tokenizer=tokenizer,
                distractor_count=distractor_count,
            )
            for strategy in strategies
        )
    cells, robustness = aggregate_positional_predictions(predictions)
    performance = aggregate_positional_performance(predictions)
    dataset_sha = hashlib.sha256(dataset.model_dump_json().encode("utf-8")).hexdigest()
    recorded_at = datetime.now(UTC)
    identity = "|".join(
        [
            dataset_sha,
            recorded_at.isoformat(),
            provider_name,
            provider_model,
            *(prediction.model_dump_json() for prediction in predictions),
        ]
    )
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return PositionalRun(
        run_id=run_id,
        recorded_at_utc=recorded_at,
        dataset_sha256=dataset_sha,
        package_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        provider=provider_name,
        model=provider_model,
        profile=profile,
        max_context_tokens=max_context_tokens,
        requested_context_lengths=requested_lengths,
        executed_context_lengths=executed_lengths,
        skipped_context_lengths=skipped_lengths,
        strategies=list(strategies),
        predictions=predictions,
        accuracy_cells=cells,
        robustness=robustness,
        performance=performance,
        metadata={
            "controlled_reproduction": True,
            "paper_doi": "10.1162/tacl_a_00638",
            "case_count": len(selected_cases),
            "prediction_count": len(predictions),
            "temperature": 0,
        },
    )


def write_positional_run_artifact(
    run: PositionalRun,
    output_directory: Path,
    *,
    dataset: PositionalDataset,
) -> Path:
    """Persist a complete immutable positional benchmark bundle."""
    case_ids = {prediction.case_id for prediction in run.predictions}
    cases = [case for case in dataset.cases if case.id in case_ids]
    if len(cases) != run.metadata["case_count"]:
        raise ValueError("positional run cases do not match the supplied dataset")
    environment = capture_environment(
        recorded_at_utc=run.recorded_at_utc,
        embedding_provider="not-applicable",
        embedding_model="not-applicable",
        llm_provider=run.provider,
        llm_model=run.model,
        dependency_names=("openai",) if run.provider == "openai" else (),
    )
    config = {
        "schema_version": "1.0",
        "benchmark": "positional-retrieval",
        "run_id": run.run_id,
        "dataset_sha256": run.dataset_sha256,
        "profile": run.profile,
        "provider": run.provider,
        "model": run.model,
        "max_context_tokens": run.max_context_tokens,
        "requested_context_lengths": run.requested_context_lengths,
        "executed_context_lengths": run.executed_context_lengths,
        "skipped_context_lengths": run.skipped_context_lengths,
        "strategies": [strategy.value for strategy in run.strategies],
        "metadata": run.metadata,
        "statistics": {
            "confidence_interval": "not reported",
            "reason": "one observation per context-length, position, and strategy cell",
            "repeat_policy": "repeat only when model nondeterminism could change the conclusion",
        },
        "performance_measurement": {
            "provider_wall_clock": "time.perf_counter",
            "provider_usage": "recorded when exposed by provider",
        },
    }
    metrics = {
        "schema_version": "1.0",
        "run_id": run.run_id,
        "accuracy_cells": [value.model_dump(mode="json") for value in run.accuracy_cells],
        "robustness": [value.model_dump(mode="json") for value in run.robustness],
        "performance": [value.model_dump(mode="json") for value in run.performance],
    }
    report = [
        "# Positional Retrieval Report",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Provider/model: `{run.provider}/{run.model}`",
        f"- Profile: `{run.profile}`",
        f"- Predictions: {len(run.predictions)}",
        "- Confidence intervals: not reported (one observation per experimental cell)",
        "",
        "| Strategy | Target tokens | Accuracy | Max-min gap |",
        "|---|---:|---:|---:|",
    ]
    report.extend(
        "| "
        f"{value.strategy.value} | {value.target_context_tokens} | "
        f"{value.mean_accuracy:.4f} | {value.max_min_positional_gap:.4f} |"
        for value in run.robustness
    )
    report.extend(
        [
            "",
            "## Provider performance",
            "",
            "| Strategy | Calls | Mean input tokens | Total latency ms | p50 ms | p95 ms | "
            "Output tokens | Cached tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    report.extend(
        "| "
        f"{value.strategy.value} | {value.prediction_count} | "
        f"{value.mean_estimated_input_tokens:.2f} | {value.total_model_latency_ms:.3f} | "
        f"{value.p50_model_latency_ms:.3f} | {value.p95_model_latency_ms:.3f} | "
        f"{value.total_output_tokens if value.total_output_tokens is not None else 'n/a'} | "
        f"{value.total_cached_tokens if value.total_cached_tokens is not None else 'n/a'} |"
        for value in run.performance
    )
    report.extend(
        [
            "",
            "Raw `cases.jsonl` and `predictions.jsonl` are authoritative; this report is derived.",
        ]
    )
    return write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=run.recorded_at_utc,
        strategy="layout-comparison",
        profile=run.profile,
        config=config,
        environment=environment,
        cases=cases,
        predictions=run.predictions,
        metrics=metrics,
        metric_rows=[
            *(
                {"record_type": "robustness", **value.model_dump(mode="json")}
                for value in run.robustness
            ),
            *(
                {"record_type": "performance", **value.model_dump(mode="json")}
                for value in run.performance
            ),
        ],
        report="\n".join(report),
    )


def positional_summary(run: PositionalRun) -> dict[str, object]:
    """Return a concise JSON-safe summary while raw predictions remain authoritative."""
    return {
        "run_id": run.run_id,
        "provider": run.provider,
        "model": run.model,
        "profile": run.profile,
        "executed_context_lengths": run.executed_context_lengths,
        "skipped_context_lengths": run.skipped_context_lengths,
        "prediction_count": len(run.predictions),
        "robustness": [metric.model_dump(mode="json") for metric in run.robustness],
    }


def main() -> None:
    """Regenerate the canonical controlled positional parameter grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/datasets/positional_retrieval.json"),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    arguments = parser.parse_args()
    write_positional_dataset(
        build_positional_dataset(repetitions=arguments.repetitions),
        arguments.output,
    )


if __name__ == "__main__":
    main()
