# Benchmarking Methodology

ContextOS is evaluated against deterministic baselines using project-owned required-fact cases. Phase 4B adds controlled positional retrieval; a selected LongBench subset follows in the next v0.4 phase.

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
