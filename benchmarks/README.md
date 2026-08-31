# ContextOS Benchmarks

The Milestone 1 deterministic smoke profile runs with:

```bash
contextos benchmark --profile quick
```

It executes one fixed mixed-context case through Full Context, Last-N, and Sliding Window. The report contains only deterministic token and selection outcomes; timing measurements remain in optimization traces and are intentionally excluded from deterministic equality checks.

Larger datasets, metrics, providers, immutable result directories, and comparison reports are added in later milestones. No benchmark result is hand-authored.
