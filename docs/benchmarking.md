# Benchmarking Methodology

ContextOS will be evaluated against deterministic baselines using project-owned required-fact cases, a controlled positional retrieval benchmark, and a selected LongBench subset. Benchmark implementation begins after the repository foundation is accepted.

Raw per-case results and environment metadata—not generated plots—will be the source of truth.

## Deduplication fixture

Phase 3A includes a small labeled development fixture for duplicate precision, recall, F1, and false-positive counts. Run it with:

```bash
contextos benchmark dedup --input benchmarks/datasets/deduplication_cases.json
```

The default semantic threshold is `0.92`. This curated fixture is a regression and configuration aid, not evidence of general semantic-deduplication quality. Final claims require the immutable v0.4.0 benchmark artifacts.
