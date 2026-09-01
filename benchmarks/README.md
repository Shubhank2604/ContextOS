# ContextOS Benchmarks

The deterministic smoke profile runs with:

```bash
contextos benchmark --profile quick
```

It executes one fixed mixed-context case through Full Context, Last-N, Sliding Window, and the integrated ContextOS optimizer. The report contains deterministic token and selection outcomes; timing measurements remain in optimization traces and are intentionally excluded from deterministic equality checks.

## ContextOS-Bench

`datasets/contextos_bench.json` is generated deterministically from versioned templates and contains exactly 50 base cases:

- 18 coding-agent cases;
- 16 research-agent cases;
- 16 support/operations cases.

The cases are labeled as templated base data—not hand-authored or generated variants. Every case includes annotated required facts plus exact values, dates or identifiers, old critical evidence, recent irrelevant evidence, duplicate paraphrases, changed-number and negation traps, supersession, contradictions, and one-/two-hop dependencies.

Regenerate and verify the canonical dataset with:

```bash
python -m contextos.benchmarks.dataset \
  --output benchmarks/datasets/contextos_bench.json
pytest tests/unit/test_benchmark_schema.py
```

Run the benchmark with:

```bash
contextos benchmark run \
  --input benchmarks/datasets/contextos_bench.json \
  --output-directory benchmarks/results
```

The shared runner evaluates Full Context, Last-N, Sliding Window, and ContextOS. Full Context receives enough budget to act as the quality reference; other strategies use the case's configured optimization budget. Each raw result contains task-specific required-fact score, quality retention, Critical Information Recall, input tokens, context reduction, compression ratio, optimizer/embedding/compression timing, selected IDs, and decision reasons.

Aggregate reports retain p50/p95 optimizer latency and deterministic bootstrap 95% confidence intervals when at least 20 successful cases are available. Generated artifacts are content-addressed and ignored by default until a later validation phase explicitly approves an immutable result for version control. No benchmark result is hand-authored.

`datasets/deduplication_cases.json` remains the focused deduplication regression fixture.
