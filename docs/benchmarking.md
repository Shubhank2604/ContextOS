# Benchmarking Methodology

ContextOS is evaluated against deterministic baselines using project-owned required-fact cases. Controlled positional retrieval and a selected LongBench subset follow in later v0.4 phases.

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
