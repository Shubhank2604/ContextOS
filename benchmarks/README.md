# ContextOS Benchmarks

The deterministic smoke profile runs with:

```bash
contextos benchmark --profile quick
```

It executes one fixed mixed-context case through Full Context, Last-N, Sliding Window, and the integrated ContextOS optimizer. The report contains deterministic token and selection outcomes; timing measurements remain in optimization traces and are intentionally excluded from deterministic equality checks.

`datasets/contextos_bench_seed.json` uses the versioned ContextOS-Bench schema and covers initial deduplication, allocation, and regression families. `datasets/deduplication_cases.json` supplies the larger labeled deduplication measurement fixture. External-model datasets, immutable result artifacts, and statistical comparisons are added in v0.4. No benchmark result is hand-authored.
