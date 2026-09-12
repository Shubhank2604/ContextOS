# Resume evidence ledger

This ledger separates implemented capabilities from claims that have reproducible, reviewable measurements. Benchmark bundles under `benchmarks/results/` are local and ignored by Git; therefore the metric claims below are **not yet safe to publish as resume numbers** until the corresponding artifacts are intentionally retained and independently reviewed.

## Claim: Context reduction

Claim: ContextOS reduces average LLM input context while retaining task quality.
Metric: 54.3% mean reduction (69.84 vs 153.04 tokens); task score 0.68 and critical-information recall 0.52.
Benchmark: ContextOS-Bench deterministic full profile, 50 cases.
Result artifact: `benchmarks/results/20260912T053613.813462Z-comparison-full/` (`metrics.json`).
Configuration: deterministic provider; six strategies; 1,000-resample paired bootstrap intervals.
Sample size: 50 paired cases per strategy.
Limitations: Synthetic templated cases; ContextOS trails full, relevance-only, and naive extractive task/CIR scores on this run; no external model quality measurement.
Safe to use on resume: no.

## Claim: Component ablation

Claim: Dependency-aware scoring is measurable on the controlled benchmark.
Metric: Removing dependency scoring lowers task score by 0.08 (0.68 to 0.60); p95 latency changes by -0.69 ms.
Benchmark: ContextOS-Bench ablation, 50 cases.
Result artifact: `benchmarks/results/20260912T055044.020663Z-ablation-full/` (`metrics.json`).
Configuration: one component disabled at a time; deterministic provider; seeded bootstrap intervals.
Sample size: 50 cases for each of six variants.
Limitations: Effect is fixture-specific and CIR was unchanged; it does not establish generalization.
Safe to use on resume: no.

## Claim: Optimizer overhead

Claim: The deterministic optimizer has low local runtime overhead.
Metric: p50 2.03 ms; p95 2.88 ms across 50 cases.
Benchmark: Same ContextOS-Bench full profile.
Result artifact: `benchmarks/results/20260912T053613.813462Z-comparison-full/` (`metrics.json`).
Configuration: native wall-clock probe; process-lifetime peak RSS reported separately (not strategy-attributable).
Sample size: 50 optimizer invocations.
Limitations: Local deterministic execution; excludes network/model latency and is not a production capacity result.
Safe to use on resume: no.

## Claim: Positional robustness

Claim: The position-aware layout preserves accuracy across tested evidence positions.
Metric: 1.0 accuracy and 0.0 max-min positional gap in the quick deterministic fixture.
Benchmark: Controlled positional retrieval quick profile.
Result artifact: `benchmarks/results/20260911T145313.548996Z-layout-comparison-quick/` (`metrics.json`).
Configuration: deterministic provider; 4K bucket; five evidence positions; three layouts.
Sample size: 15 position/layout cells (one case per cell).
Limitations: Deterministic callback is position-invariant and cannot reproduce model positional degradation; this is plumbing validation, not a Lost-in-the-Middle result.
Safe to use on resume: no.
