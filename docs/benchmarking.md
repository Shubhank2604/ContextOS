# Benchmarking Methodology

ContextOS is evaluated against deterministic baselines using project-owned required-fact cases. Phase 4B adds controlled positional retrieval, and Phase 4C adds a configured LongBench subset for external validation.

Raw per-case results and environment metadata—not generated plots—will be the source of truth.

## Deduplication fixture

Phase 3A includes a small labeled development fixture for duplicate precision, recall, F1, and false-positive counts. Run it with:

```bash
contextos benchmark dedup --input benchmarks/datasets/deduplication_cases.json
```

The default semantic threshold is `0.92`. This curated fixture is a regression and configuration aid, not evidence of general semantic-deduplication quality. Final claims require the immutable v0.4.0 benchmark artifacts.

## Phase 4A project-owned benchmark

The canonical ContextOS-Bench dataset contains 50 templated base cases across coding, research, and support/operations agents. Templates are deterministic and versioned; the schema distinguishes base cases from any future generated variants and requires reproducibility metadata for generated data.

Task-specific score is exact annotated-fact recall over the constructed input context. Critical Information Recall is stricter: a required item counts only when all exact facts attached to it remain represented. This evaluator measures context construction, not final language-model answer quality, and reports it as such.

Full Context is executed with enough budget to provide the quality reference. Budget-constrained strategies use the same configured case policy. Quality retention is calculated only when the Full Context task score is non-zero; a zero reference is reported as unavailable rather than divided implicitly.

Every run retains raw per-case results. Aggregates are derived from successful cases, keep unlike metrics separate, report p50/p95 optimizer latency, and use a seeded percentile bootstrap for 95% confidence intervals when at least 20 cases are available.

## Phase 4B controlled positional experiment

This is a **controlled reproduction inspired by the paper**, not a reproduction of every experiment in *Lost in the Middle*. It tests exact key-value retrieval while varying target input length (4K, 8K, 16K, and 32K tokens), original evidence position (beginning, 25%, middle, 75%, and end), and layout (original/full, relevance descending, and ContextOS position aware).

All layouts receive exactly the same records, query, provider/model, decoding bound, and evaluator. The original layout preserves the requested evidence position. The other layouts use the production relevance-descending and position-aware layout implementations. Exact match normalizes only surrounding whitespace and character case.

Reports retain accuracy by position and context length, the max-minus-min positional accuracy gap, population variance and standard deviation, estimated and provider-reported input tokens, raw predictions, prompt hashes, cached/output token counts when exposed, and total provider latency. Context buckets beyond the user-declared model limit are skipped and recorded rather than silently truncated.

The deterministic provider is an offline oracle used only to validate dataset, layout, evaluator, aggregation, and artifact plumbing. It is position invariant by design and cannot establish the positional degradation phenomenon. Only a recorded real-provider run may support an empirical claim.

## Phase 4C LongBench subset

The configured subset covers HotpotQA, 2WikiMQA, PassageRetrieval-en, and RepoBench-P from the pinned `zai-org/LongBench` source revision. It represents multi-document QA, synthetic retrieval, and code completion while keeping provider cost tractable.

Sampling ranks preserved `(dataset, _id)` identities with a versioned SHA-256 seed, so quick and standard selections do not depend on upstream row order. Standard contains exactly 100 examples. Full retains every row from the configured tasks. The preparation artifact records repository, revision, split, profile, seed, original IDs, answers, and source length.

The evaluator follows the official LongBench task mapping: normalized English token F1 for the two QA datasets, paragraph-number precision for retrieval, and first uncommented-line similarity for RepoBench-P. Each prediction is scored against all accepted answers and keeps the maximum, matching the official evaluation convention. No LLM judge is used.

Prediction scoring requires complete identity coverage and one provider/model configuration. Phase 4D adds the complete strategy comparison runner; Phase 4C deliberately does not fabricate model predictions when the external dependency or credentials are unavailable.

## Phase 4D baseline comparison

All applicable project-owned and LongBench comparisons now use six strategies: Full Context, Last-N, Sliding Window, Relevance Only, Naive Extractive, and ContextOS. Relevance Only is deliberately restricted to embedding relevance over whole items. Naive Extractive is deliberately restricted to sentence relevance and source-order restoration. They do not borrow ContextOS's typed retention, deduplication, multi-factor scoring, dependency propagation, allocation, or position-aware policy.

Within one LongBench run, all strategies share the exact prepared case IDs, pinned prompt templates, provider, model, temperature-zero configuration, output bounds, context limit, and evaluator. Constrained strategies share one declared context budget. Full Context is called only when the unmodified prompt fits the declared provider limit; otherwise its raw status is `context_overflow` and its task score and quality reference remain unavailable.

Raw predictions include strategy, status, original and constructed context tokens, local and provider prompt tokens, output/cached tokens when exposed, optimizer and provider latency, context reduction, selected chunk IDs, warnings, and model identity. Scoring requires the complete case-by-strategy matrix, never substitutes a missing prediction, and calculates quality retention only against a non-zero successful Full Context score for the same case and metric.

The current offline ContextOS-Bench comparison is diagnostic, not a finalized empirical claim. Newly added simple baselines may outperform the current integrated configuration; those results must remain visible and motivate the Phase 4E ablation study rather than being filtered from reports.
