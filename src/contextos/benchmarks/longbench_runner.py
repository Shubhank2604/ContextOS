"""Same-provider six-strategy execution for prepared LongBench subsets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

from contextos.baselines import (
    BaselineStrategy,
    FullContextBaseline,
    LastNTokensBaseline,
    NaiveExtractiveBaseline,
    RelevanceOnlyBaseline,
    SlidingWindowBaseline,
)
from contextos.benchmarks.longbench import render_longbench_prompt
from contextos.benchmarks.longbench_models import (
    LongBenchCase,
    LongBenchPrediction,
    PreparedLongBenchSubset,
)
from contextos.config import OptimizationPolicy
from contextos.errors import ContextOSError
from contextos.models import ContextItem, ContextType
from contextos.optimizer import ContextOptimizer
from contextos.providers import LLMProvider
from contextos.tokenization import Tokenizer
from contextos.trace import OptimizedContext

_START = datetime(2025, 1, 1, tzinfo=UTC)


class _ContextOSStrategy:
    name = "contextos"

    def optimize(
        self,
        *,
        task: str,
        items: Sequence[ContextItem],
        policy: OptimizationPolicy,
        tokenizer: Tokenizer,
    ) -> OptimizedContext:
        return ContextOptimizer(tokenizer=tokenizer).optimize(task, items, policy)


def default_longbench_strategies() -> list[BaselineStrategy]:
    """Return every required comparison strategy in stable report order."""
    return [
        FullContextBaseline(),
        LastNTokensBaseline(),
        SlidingWindowBaseline(window_seconds=7 * 86_400),
        RelevanceOnlyBaseline(),
        NaiveExtractiveBaseline(),
        _ContextOSStrategy(),
    ]


def _largest_prefix_end(
    text: str,
    start: int,
    *,
    max_tokens: int,
    tokenizer: Tokenizer,
) -> int:
    low = start + 1
    high = len(text)
    best = start
    while low <= high:
        middle = (low + high) // 2
        if tokenizer.count_tokens(text[start:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError("chunk token limit cannot represent one source character")
    return best


def chunk_longbench_context(
    case: LongBenchCase,
    *,
    tokenizer: Tokenizer,
    max_chunk_tokens: int,
) -> list[ContextItem]:
    """Split context into exact contiguous chunks while preserving source text and order."""
    if max_chunk_tokens <= 0:
        raise ValueError("max chunk tokens must be positive")
    context_type = (
        ContextType.CODE if case.dataset == "repobench-p" else ContextType.RETRIEVED_DOCUMENT
    )
    chunks: list[ContextItem] = []
    start = 0
    while start < len(case.context):
        end = _largest_prefix_end(
            case.context,
            start,
            max_tokens=max_chunk_tokens,
            tokenizer=tokenizer,
        )
        content = case.context[start:end]
        timestamp = _START + timedelta(days=len(chunks))
        chunks.append(
            ContextItem(
                id=f"{case.dataset}-{case.source_id}-chunk-{len(chunks):05d}",
                content=content,
                type=context_type,
                source=f"longbench:{case.dataset}:{case.source_id}",
                created_at=timestamp,
                updated_at=timestamp,
                metadata={
                    "dataset": case.dataset,
                    "source_id": case.source_id,
                    "character_start": start,
                    "character_end": end,
                },
            )
        )
        start = end
    if "".join(chunk.content for chunk in chunks) != case.context:
        raise AssertionError("LongBench chunking failed to preserve source context")
    return chunks


def _policy(budget: int) -> OptimizationPolicy:
    return OptimizationPolicy(
        max_input_tokens=budget,
        minimum_compressed_tokens=max(1, min(64, budget)),
    )


def _failed_prediction(
    case: LongBenchCase,
    *,
    strategy: str,
    status: str,
    warning: str,
    provider_name: str,
    provider_model: str,
    original_tokens: int,
    optimizer_latency_ms: float | None = None,
) -> LongBenchPrediction:
    return LongBenchPrediction(
        dataset=case.dataset,
        source_id=case.source_id,
        strategy=strategy,
        status=status,
        prediction="",
        provider=provider_name,
        model=provider_model,
        original_context_tokens=original_tokens,
        optimizer_latency_ms=optimizer_latency_ms,
        warnings=[warning],
    )


def _run_strategy(
    case: LongBenchCase,
    items: Sequence[ContextItem],
    strategy: BaselineStrategy,
    *,
    provider: LLMProvider,
    provider_name: str,
    provider_model: str,
    tokenizer: Tokenizer,
    context_budget_tokens: int,
    max_context_tokens: int,
) -> LongBenchPrediction:
    original_tokens = tokenizer.count_tokens(case.context)
    max_prompt_tokens = max_context_tokens - case.max_output_tokens
    full_prompt = render_longbench_prompt(case)
    if strategy.name == "full_context" and tokenizer.count_tokens(full_prompt) > max_prompt_tokens:
        return _failed_prediction(
            case,
            strategy=strategy.name,
            status="context_overflow",
            warning="full context exceeds the declared provider context limit",
            provider_name=provider_name,
            provider_model=provider_model,
            original_tokens=original_tokens,
        )
    empty_prompt_tokens = tokenizer.count_tokens(render_longbench_prompt(case, context=""))
    available_context_tokens = max_prompt_tokens - empty_prompt_tokens
    if available_context_tokens <= 0:
        return _failed_prediction(
            case,
            strategy=strategy.name,
            status="context_overflow",
            warning="task prompt and output bound exhaust the provider context limit",
            provider_name=provider_name,
            provider_model=provider_model,
            original_tokens=original_tokens,
        )
    if strategy.name == "full_context":
        strategy_budget = max(sum(tokenizer.count_tokens(item.content) for item in items), 1)
    else:
        strategy_budget = min(context_budget_tokens, available_context_tokens)
    optimize_started = perf_counter()
    try:
        result = strategy.optimize(
            task=case.input,
            items=items,
            policy=_policy(strategy_budget),
            tokenizer=tokenizer,
        )
    except (ContextOSError, ValueError) as exc:
        return _failed_prediction(
            case,
            strategy=strategy.name,
            status="optimization_error",
            warning=str(exc),
            provider_name=provider_name,
            provider_model=provider_model,
            original_tokens=original_tokens,
            optimizer_latency_ms=(perf_counter() - optimize_started) * 1_000,
        )
    optimizer_latency_ms = (perf_counter() - optimize_started) * 1_000
    selected_context = "".join(item.content for item in result.selected_items)
    input_context_tokens = tokenizer.count_tokens(selected_context)
    prompt = render_longbench_prompt(case, context=selected_context)
    prompt_tokens = tokenizer.count_tokens(prompt)
    if prompt_tokens > max_prompt_tokens:
        return _failed_prediction(
            case,
            strategy=strategy.name,
            status="context_overflow",
            warning="constructed prompt exceeds the declared provider context limit",
            provider_name=provider_name,
            provider_model=provider_model,
            original_tokens=original_tokens,
            optimizer_latency_ms=optimizer_latency_ms,
        )
    provider_started = perf_counter()
    try:
        response = provider.complete(prompt, max_output_tokens=case.max_output_tokens)
    except ContextOSError as exc:
        return _failed_prediction(
            case,
            strategy=strategy.name,
            status="provider_error",
            warning=str(exc),
            provider_name=provider_name,
            provider_model=provider_model,
            original_tokens=original_tokens,
            optimizer_latency_ms=optimizer_latency_ms,
        )
    provider_latency_ms = (perf_counter() - provider_started) * 1_000
    reduction = 0.0 if original_tokens == 0 else 1 - input_context_tokens / original_tokens
    return LongBenchPrediction(
        dataset=case.dataset,
        source_id=case.source_id,
        strategy=strategy.name,
        status="ok",
        prediction=response.text,
        provider=provider_name,
        model=provider_model,
        original_context_tokens=original_tokens,
        input_context_tokens=input_context_tokens,
        prompt_input_tokens=prompt_tokens,
        provider_input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cached_tokens,
        context_reduction=max(0.0, min(reduction, 1.0)),
        optimizer_latency_ms=optimizer_latency_ms,
        provider_latency_ms=provider_latency_ms,
        selected_item_ids=[item.id for item in result.selected_items],
        warnings=result.trace.warnings,
    )


def run_longbench_comparison(
    subset: PreparedLongBenchSubset,
    *,
    provider: LLMProvider,
    provider_name: str,
    provider_model: str,
    tokenizer: Tokenizer,
    context_budget_tokens: int,
    max_context_tokens: int,
    max_chunk_tokens: int = 256,
    strategies: Sequence[BaselineStrategy] | None = None,
) -> list[LongBenchPrediction]:
    """Run every selected case and strategy through one fixed provider/model configuration."""
    if context_budget_tokens <= 0 or max_context_tokens <= 0 or max_chunk_tokens <= 0:
        raise ValueError("LongBench token limits must be positive")
    selected_strategies = list(strategies or default_longbench_strategies())
    strategy_names = [strategy.name for strategy in selected_strategies]
    if not selected_strategies or len(strategy_names) != len(set(strategy_names)):
        raise ValueError("LongBench strategies must be non-empty and unique")
    predictions: list[LongBenchPrediction] = []
    for case in subset.cases:
        items = chunk_longbench_context(
            case,
            tokenizer=tokenizer,
            max_chunk_tokens=max_chunk_tokens,
        )
        predictions.extend(
            _run_strategy(
                case,
                items,
                strategy,
                provider=provider,
                provider_name=provider_name,
                provider_model=provider_model,
                tokenizer=tokenizer,
                context_budget_tokens=context_budget_tokens,
                max_context_tokens=max_context_tokens,
            )
            for strategy in selected_strategies
        )
    return predictions


def write_longbench_predictions(
    predictions: Sequence[LongBenchPrediction],
    path: Path,
) -> Path:
    """Write complete raw strategy outputs as immutable newline-delimited JSON."""
    if not predictions:
        raise ValueError("cannot write an empty LongBench prediction set")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(prediction.model_dump_json() for prediction in predictions) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"LongBench prediction output already exists: {path}") from None
    return path
