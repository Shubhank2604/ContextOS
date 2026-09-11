"""Lazy external-data adapter and deterministic LongBench subset metrics."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, cast

from pydantic import ValidationError

from contextos.benchmarks.bundles import capture_environment, write_benchmark_bundle
from contextos.benchmarks.longbench_models import (
    LongBenchCase,
    LongBenchCaseScore,
    LongBenchDatasetAggregate,
    LongBenchMetric,
    LongBenchPairedComparison,
    LongBenchPrediction,
    LongBenchProfile,
    LongBenchScoreReport,
    LongBenchSubsetConfig,
    LongBenchTaskConfig,
    PreparedLongBenchSubset,
)
from contextos.benchmarks.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MIN_BOOTSTRAP_SAMPLE_SIZE,
    bootstrap_mean_ci,
)
from contextos.embeddings import DeterministicEmbeddingProvider


class LongBenchSource(Protocol):
    """Injectable source contract that keeps unit tests offline."""

    def load_task(
        self,
        repository: str,
        dataset: str,
        *,
        split: str,
        revision: str,
    ) -> Iterable[Mapping[str, Any]]:
        """Return raw upstream rows for one configured task."""
        ...


class HuggingFaceLongBenchSource:
    """Load upstream rows only after an explicit preparation command."""

    def load_task(
        self,
        repository: str,
        dataset: str,
        *,
        split: str,
        revision: str,
    ) -> Iterable[Mapping[str, Any]]:
        """Lazy-import datasets and load the pinned task configuration."""
        try:
            datasets = importlib.import_module("datasets")
        except ImportError as exc:
            raise RuntimeError(
                "LongBench preparation requires the optional 'benchmark' dependencies"
            ) from exc
        loaded: Any = datasets.load_dataset(
            repository,
            dataset,
            split=split,
            revision=revision,
            trust_remote_code=True,
        )
        return cast(Iterable[Mapping[str, Any]], loaded)


def load_longbench_config(path: Path) -> LongBenchSubsetConfig:
    """Load and validate the pinned external subset configuration."""
    return LongBenchSubsetConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _case_from_row(
    row: Mapping[str, Any],
    task: LongBenchTaskConfig,
) -> LongBenchCase:
    try:
        raw_source_id = row["_id"]
        raw_answers = row["answers"]
        if (
            raw_source_id is None
            or not isinstance(raw_answers, Sequence)
            or isinstance(raw_answers, str)
        ):
            raise ValueError("_id must be present and answers must be a sequence")
        source_id = str(raw_source_id)
        answers = [str(answer) for answer in raw_answers]
        raw_classes = row.get("all_classes")
        all_classes = None if raw_classes is None else [str(value) for value in raw_classes]
        return LongBenchCase(
            dataset=task.dataset,
            source_id=source_id,
            input=str(row["input"]),
            context=str(row["context"]),
            answers=answers,
            source_length=int(row["length"]),
            language=str(row["language"]),
            all_classes=all_classes,
            metric=task.metric,
            prompt_template=task.prompt_template,
            max_output_tokens=task.max_output_tokens,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid LongBench row for {task.dataset}: {exc}") from exc


def _sample_cases(
    cases: Sequence[LongBenchCase],
    *,
    limit: int | None,
    seed: int,
) -> list[LongBenchCase]:
    ranked = sorted(
        cases,
        key=lambda case: (
            hashlib.sha256(f"{seed}|{case.dataset}|{case.source_id}".encode()).hexdigest(),
            case.source_id,
        ),
    )
    if limit is None:
        return sorted(cases, key=lambda case: case.source_id)
    if len(ranked) < limit:
        raise ValueError(f"requested {limit} cases but upstream task contains {len(ranked)}")
    return ranked[:limit]


def prepare_longbench_subset(
    config: LongBenchSubsetConfig,
    *,
    profile: LongBenchProfile,
    source: LongBenchSource,
) -> PreparedLongBenchSubset:
    """Load, validate, and deterministically select the configured external cases."""
    selected: list[LongBenchCase] = []
    for task in config.tasks:
        rows = source.load_task(
            config.source_repository,
            task.dataset,
            split=config.source_split,
            revision=config.source_revision,
        )
        cases = [_case_from_row(row, task) for row in rows]
        identities = [case.source_id for case in cases]
        if len(identities) != len(set(identities)):
            raise ValueError(f"upstream task {task.dataset} contains duplicate source IDs")
        limit = {
            LongBenchProfile.QUICK: task.quick_samples,
            LongBenchProfile.STANDARD: task.standard_samples,
            LongBenchProfile.FULL: None,
        }[profile]
        selected.extend(_sample_cases(cases, limit=limit, seed=config.sampling_seed))
    return PreparedLongBenchSubset(
        profile=profile,
        source_repository=config.source_repository,
        source_revision=config.source_revision,
        source_split=config.source_split,
        sampling_seed=config.sampling_seed,
        cases=selected,
    )


def write_prepared_subset(subset: PreparedLongBenchSubset, path: Path) -> Path:
    """Write prepared external data without silently replacing different content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = subset.model_dump_json(indent=2)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(content):
            raise ValueError(
                f"prepared LongBench output already exists with different content: {path}"
            ) from None
    return path


def load_prepared_subset(path: Path) -> PreparedLongBenchSubset:
    """Load validated locally prepared external cases."""
    return PreparedLongBenchSubset.model_validate_json(path.read_text(encoding="utf-8"))


def load_longbench_predictions(path: Path) -> list[LongBenchPrediction]:
    """Load newline-delimited predictions with line-specific validation errors."""
    predictions: list[LongBenchPrediction] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            predictions.append(LongBenchPrediction.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"invalid prediction at line {line_number}: {exc}") from exc
    if not predictions:
        raise ValueError("prediction file contains no predictions")
    return predictions


def render_longbench_prompt(case: LongBenchCase, *, context: str | None = None) -> str:
    """Render the pinned official task prompt with original or optimized context."""
    return case.prompt_template.format(
        context=case.context if context is None else context,
        input=case.input,
    )


def normalize_qa_answer(value: str) -> str:
    """Apply the official English LongBench QA normalization."""
    lowered = value.lower()
    unpunctuated = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", unpunctuated)
    return " ".join(without_articles.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Return normalized token F1 compatible with the official evaluator."""
    prediction_tokens = normalize_qa_answer(prediction).split()
    truth_tokens = normalize_qa_answer(ground_truth).split()
    if not prediction_tokens or not truth_tokens:
        return float(prediction_tokens == truth_tokens)
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """Apply LongBench's paragraph-number precision metric."""
    match = re.search(r"Paragraph (\d+)", ground_truth)
    if match is None:
        raise ValueError(f"retrieval ground truth lacks a paragraph ID: {ground_truth!r}")
    expected = match.group(1)
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return float(sum(number == expected for number in numbers) / len(numbers))


def code_similarity_score(prediction: str, ground_truth: str) -> float:
    """Score the first uncommented code line using deterministic edit similarity."""
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return round(SequenceMatcher(None, candidate, ground_truth).ratio() * 100) / 100


def score_longbench_case(case: LongBenchCase, prediction: str) -> float:
    """Score against every accepted answer and retain the maximum official-style score."""
    scorer = {
        LongBenchMetric.QA_F1: qa_f1_score,
        LongBenchMetric.RETRIEVAL: retrieval_score,
        LongBenchMetric.CODE_SIMILARITY: code_similarity_score,
    }[case.metric]
    return max(scorer(prediction, answer) for answer in case.answers)


def score_longbench_predictions(
    subset: PreparedLongBenchSubset,
    predictions: Sequence[LongBenchPrediction],
) -> LongBenchScoreReport:
    """Join by preserved IDs, reject incomplete comparisons, and aggregate per dataset."""
    by_identity: dict[tuple[str, str, str], LongBenchPrediction] = {}
    for prediction in predictions:
        identity = (prediction.dataset, prediction.source_id, prediction.strategy)
        if identity in by_identity:
            raise ValueError(
                f"duplicate prediction identity: {identity[0]}/{identity[1]}/{identity[2]}"
            )
        by_identity[identity] = prediction
    strategies = sorted({prediction.strategy for prediction in predictions})
    expected = {
        (case.dataset, case.source_id, strategy) for case in subset.cases for strategy in strategies
    }
    supplied = set(by_identity)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        raise ValueError(
            f"prediction identity mismatch: missing={len(missing)}, unknown={len(unknown)}"
        )
    providers = {prediction.provider for prediction in predictions}
    models = {prediction.model for prediction in predictions}
    if len(providers) != 1 or len(models) != 1:
        raise ValueError("all LongBench predictions must use one provider and model configuration")
    raw_case_scores = [
        LongBenchCaseScore(
            dataset=case.dataset,
            source_id=case.source_id,
            strategy=strategy,
            status=by_identity[(case.dataset, case.source_id, strategy)].status,
            metric=case.metric,
            score=(
                score_longbench_case(
                    case,
                    by_identity[(case.dataset, case.source_id, strategy)].prediction,
                )
                if by_identity[(case.dataset, case.source_id, strategy)].status == "ok"
                else None
            ),
            prediction=by_identity[(case.dataset, case.source_id, strategy)].prediction,
            answers=case.answers,
        )
        for case in subset.cases
        for strategy in strategies
    ]
    full_scores = {
        (score.dataset, score.source_id): score.score
        for score in raw_case_scores
        if score.strategy == "full_context" and score.status == "ok"
    }
    case_scores = [
        score.model_copy(
            update={
                "quality_retention": (
                    score.score / full_score
                    if score.score is not None
                    and (full_score := full_scores.get((score.dataset, score.source_id)))
                    is not None
                    and full_score > 0
                    else None
                )
            }
        )
        for score in raw_case_scores
    ]
    prepared_sha = hashlib.sha256(subset.model_dump_json().encode()).hexdigest()
    aggregates: list[LongBenchDatasetAggregate] = []
    for dataset, strategy in sorted({(score.dataset, score.strategy) for score in case_scores}):
        aggregate_scores = [
            score
            for score in case_scores
            if score.dataset == dataset and score.strategy == strategy
        ]
        successful_scores = [
            score for score in aggregate_scores if score.status == "ok" and score.score is not None
        ]
        quality_retentions = [
            score.quality_retention
            for score in successful_scores
            if score.quality_retention is not None
        ]
        score_values = [score.score for score in successful_scores if score.score is not None]
        seed_material = f"{prepared_sha}|{dataset}|{strategy}|{aggregate_scores[0].metric}".encode()
        aggregate_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        aggregates.append(
            LongBenchDatasetAggregate(
                dataset=dataset,
                strategy=strategy,
                metric=aggregate_scores[0].metric,
                case_count=len(aggregate_scores),
                successful_case_count=len(successful_scores),
                mean_score=(mean(score_values) if score_values else None),
                mean_quality_retention=(mean(quality_retentions) if quality_retentions else None),
                score_ci95=(
                    bootstrap_mean_ci(score_values, seed=aggregate_seed)
                    if len(score_values) >= MIN_BOOTSTRAP_SAMPLE_SIZE
                    else None
                ),
                quality_retention_ci95=(
                    bootstrap_mean_ci(quality_retentions, seed=aggregate_seed + 1)
                    if len(quality_retentions) >= MIN_BOOTSTRAP_SAMPLE_SIZE
                    else None
                ),
            )
        )
    paired_comparisons: list[LongBenchPairedComparison] = []
    datasets = sorted({score.dataset for score in case_scores})
    for dataset in datasets:
        dataset_scores = [score for score in case_scores if score.dataset == dataset]
        available_strategies = {score.strategy for score in dataset_scores}
        comparison_pairs = [
            ("full_context", candidate)
            for candidate in sorted(available_strategies - {"full_context"})
        ]
        comparison_pairs.extend(
            (baseline, "contextos")
            for baseline in sorted(available_strategies - {"full_context", "contextos"})
        )
        for reference_strategy, candidate in comparison_pairs:
            reference: dict[str, float] = {
                score.source_id: score.score
                for score in dataset_scores
                if score.strategy == reference_strategy
                and score.status == "ok"
                and score.score is not None
            }
            candidate_scores: dict[str, float] = {
                score.source_id: score.score
                for score in dataset_scores
                if score.strategy == candidate and score.status == "ok" and score.score is not None
            }
            shared_ids = sorted(set(reference) & set(candidate_scores))
            deltas = [
                candidate_scores[source_id] - reference[source_id] for source_id in shared_ids
            ]
            if not deltas:
                continue
            metric = dataset_scores[0].metric
            seed_material = (
                f"{prepared_sha}|{dataset}|{reference_strategy}|{candidate}|{metric}".encode()
            )
            comparison_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
            paired_comparisons.append(
                LongBenchPairedComparison(
                    dataset=dataset,
                    reference_strategy=reference_strategy,
                    candidate_strategy=candidate,
                    metric=metric,
                    case_count=len(deltas),
                    mean_score_delta=mean(deltas) if deltas else 0.0,
                    score_delta_ci95=(
                        bootstrap_mean_ci(deltas, seed=comparison_seed)
                        if len(deltas) >= MIN_BOOTSTRAP_SAMPLE_SIZE
                        else None
                    ),
                )
            )
    return LongBenchScoreReport(
        prepared_sha256=prepared_sha,
        prediction_count=len(predictions),
        provider=next(iter(providers)),
        model=next(iter(models)),
        case_scores=case_scores,
        dataset_aggregates=aggregates,
        paired_comparisons=paired_comparisons,
    )


def write_longbench_bundle(
    subset: PreparedLongBenchSubset,
    predictions: Sequence[LongBenchPrediction],
    report: LongBenchScoreReport,
    output_directory: Path,
    *,
    execution_config: Mapping[str, Any],
    recorded_at_utc: datetime | None = None,
) -> Path:
    """Write cases, provider outputs, scores, and provenance as one immutable bundle."""
    if len(predictions) != report.prediction_count:
        raise ValueError("LongBench report count does not match predictions")
    recorded_at = recorded_at_utc or datetime.now(UTC)
    embedding = DeterministicEmbeddingProvider()
    environment = capture_environment(
        recorded_at_utc=recorded_at,
        embedding_provider="deterministic",
        embedding_model=embedding.model_name,
        llm_provider=report.provider,
        llm_model=report.model,
        dependency_names=("datasets", "openai"),
    )
    strategies = sorted({prediction.strategy for prediction in predictions})
    config = {
        "schema_version": "1.0",
        "benchmark": "longbench-subset",
        "prepared_sha256": report.prepared_sha256,
        "profile": subset.profile.value,
        "source_repository": subset.source_repository,
        "source_revision": subset.source_revision,
        "source_split": subset.source_split,
        "sampling_seed": subset.sampling_seed,
        "provider": report.provider,
        "model": report.model,
        "strategies": strategies,
        "execution": dict(execution_config),
        "statistics": {
            "method": "percentile bootstrap; paired for strategy deltas",
            "confidence_level": 0.95,
            "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "minimum_sample_size": MIN_BOOTSTRAP_SAMPLE_SIZE,
            "repeat_policy": "repeat only when model nondeterminism could change the conclusion",
        },
    }
    metrics = {
        "schema_version": report.schema_version,
        "prepared_sha256": report.prepared_sha256,
        "prediction_count": report.prediction_count,
        "case_scores": [score.model_dump(mode="json") for score in report.case_scores],
        "dataset_aggregates": [
            aggregate.model_dump(mode="json") for aggregate in report.dataset_aggregates
        ],
        "paired_comparisons": [
            comparison.model_dump(mode="json") for comparison in report.paired_comparisons
        ],
    }
    report_lines = [
        "# LongBench Subset Report",
        "",
        f"- Provider/model: `{report.provider}/{report.model}`",
        f"- Profile: `{subset.profile.value}`",
        f"- Cases: {len(subset.cases)}",
        f"- Predictions: {len(predictions)}",
        "",
        "| Dataset | Strategy | Metric | Successful | Mean score | Quality retention |",
        "|---|---|---|---:|---:|---:|",
    ]
    report_lines.extend(
        "| "
        f"{value.dataset} | {value.strategy} | {value.metric.value} | "
        f"{value.successful_case_count}/{value.case_count} | "
        f"{value.mean_score if value.mean_score is not None else 'n/a'} | "
        f"{value.mean_quality_retention if value.mean_quality_retention is not None else 'n/a'} |"
        for value in report.dataset_aggregates
    )
    if report.paired_comparisons:
        report_lines.extend(
            [
                "",
                "## Paired score deltas",
                "",
                "Candidate minus reference; confidence intervals require 20 paired cases.",
                "",
                "| Dataset | Reference | Candidate | Cases | Mean delta | 95% CI |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        report_lines.extend(
            "| "
            f"{value.dataset} | {value.reference_strategy} | {value.candidate_strategy} | "
            f"{value.case_count} | "
            f"{value.mean_score_delta:.4f} | "
            + (
                f"[{value.score_delta_ci95.low:.4f}, {value.score_delta_ci95.high:.4f}] |"
                if value.score_delta_ci95 is not None
                else "not reported |"
            )
            for value in report.paired_comparisons
        )
    report_lines.extend(
        [
            "",
            "Raw `cases.jsonl` and `predictions.jsonl` are authoritative; this report is derived.",
        ]
    )
    return write_benchmark_bundle(
        output_directory=output_directory,
        recorded_at_utc=recorded_at,
        strategy="comparison",
        profile=subset.profile.value,
        config=config,
        environment=environment,
        cases=subset.cases,
        predictions=predictions,
        metrics=metrics,
        metric_rows=[
            *(
                {"record_type": "aggregate", **value.model_dump(mode="json")}
                for value in report.dataset_aggregates
            ),
            *(
                {"record_type": "paired_comparison", **value.model_dump(mode="json")}
                for value in report.paired_comparisons
            ),
        ],
        report="\n".join(report_lines),
    )
