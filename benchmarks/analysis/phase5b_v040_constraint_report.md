# Phase 5B v0.4 Constraint Baseline

## Scope and provenance

This report records the mandatory run of the unmodified v0.4 optimization strategies against ContextOS-Bench Constraints. The frozen optimizer baseline is Git SHA `4fdd88391300c56ad17af5458897ccdd08d6f7bf`; Phase 5B only adds the separate benchmark track and is recorded at Git SHA `07f343ccc135b22df82fefc01082e429744409d9`.

The deterministic full-profile run contains 90 cases, ten in each of nine categories. It used generator version `1.0.0`, dataset SHA-256 `7adc50c2c732ee1bb71e9e214fa3981c152a0692d407ec721f88d00e6b759305`, and no LLM judge. The authoritative local artifact is `benchmarks/results/20260913T005340.434288Z-constraint-comparison-full`; the tracked machine-readable summary is `phase5b_v040_constraint_baseline.json`.

## Aggregate results

| Strategy | Task score | CIR | Reduction | Violation rate | Dependency closure | State consistency | Contradiction leakage | Exact values | Identifiers | Citations | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContextOS v0.4 | 1.0000 | 1.0000 | 0.0175 | 0.5556 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.677 |
| Full Context | 1.0000 | 1.0000 | 0.0000 | 0.5556 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.065 |
| Last-N | 0.9111 | 0.9111 | 0.0327 | 0.6444 | 0.8667 | 0.3750 | 1.0000 | 0.7333 | 1.0000 | 1.0000 | 0.683 |
| Naive Extractive | 1.0000 | 1.0000 | 0.0175 | 0.5556 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.161 |
| Relevance Only | 1.0000 | 1.0000 | 0.0175 | 0.5556 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.470 |
| Sliding Window | 0.1481 | 0.0926 | 0.8523 | 1.0000 | 0.0000 | 0.6250 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.761 |

All 90 executions succeeded for every strategy. Latency is local Windows wall-clock telemetry from one deterministic run, not a cross-machine performance claim.

## ContextOS category findings

| Category | Cases | Violations | Finding |
|---|---:|---:|---|
| Dependency | 10 | 10 | Required closure is preserved, but an explicitly stale authentication record remains visible. |
| Supersession | 10 | 10 | Current and superseded values are retained together. |
| Contradiction | 10 | 10 | Canonical and conflicting noncanonical values are retained together. |
| Exact numeric | 10 | 0 | Every annotated amount, rate, and date is preserved exactly. |
| Identifier | 10 | 0 | Every annotated user, instruction, and function identifier is preserved. |
| Negation/policy | 10 | 10 | The negative policy is preserved, but its conflicting unsafe instruction is also retained. |
| Tool state | 10 | 10 | Current and obsolete tool states remain simultaneously visible. |
| Citation/evidence | 10 | 0 | Required evidence and source identifiers are preserved. |
| Multi-hop | 10 | 0 | The complete three-hop required-item chain is preserved. |

The dependency-category violations must not be misreported as dependency-closure failures: closure is 100%. They arise because those cases also annotate an obsolete authentication item, and v0.4 does not remove it. Likewise, zero contradiction leakage for Sliding Window is not evidence of superior conflict resolution; that strategy usually drops both sides of the conflict and fails all 90 contracts.

## Interpretation

The main result is a metric blind spot. ContextOS v0.4 obtains perfect task score and Critical Information Recall while violating 50 of 90 preservation contracts. Task score and CIR reward retaining required facts, but they do not penalize simultaneously retaining stale, superseded, or contradictory facts. Full Context has the same failure rate for the same reason: more context is not necessarily a legally consistent context.

The benchmark therefore exposes a meaningful gap that ordinary fact recall does not measure. v0.4 is strong at literal preservation and relation closure on this fixture, but it lacks a mechanism that converts supersession, contradiction, and staleness annotations into an enforceable model-visible state. Relevance Only and Naive Extractive matching ContextOS confirms that topical selection alone cannot resolve this gap.

## Limitations

- The cases are deterministic templates, not a natural-distribution estimate of production failure frequency.
- ContextOS reduces tokens by only 1.75% on this track. The benchmark is primarily a correctness stress test, not evidence of compression efficiency.
- Most cases fit near the configured budget, so the run strongly tests co-retention legality but only lightly tests tradeoffs under severe budget pressure.
- Exact substring scoring tests representational preservation, not whether a downstream model follows the preserved constraint.
- Several categories deliberately combine more than one annotation family. The per-metric values, not the category label alone, identify the actual violated property.
- ContextOS, Full Context, Relevance Only, and Naive Extractive produce identical aggregate correctness values here; the benchmark establishes a shared gap but does not yet demonstrate a ContextOS advantage.

## Stop/go decision

**Go to Phase 5C.** The 55.6% v0.4 violation rate is far from near-perfect, and the disagreement between perfect task/CIR scores and poor legal-state consistency directly justifies explicit preservation contracts. Phase 5C should define those contracts and their failure semantics before the optimizer is changed.

This decision does not claim that the planned intervention will improve LongBench, existing ContextOS-Bench quality, or production workloads. Those claims require later controlled comparisons against this frozen baseline and regression checks against v0.4.
