# v0.4 Context Failure Taxonomy

## Scope and provenance

This analysis freezes ContextOS v0.4.0 as the research baseline at Git SHA `4fdd88391300c56ad17af5458897ccdd08d6f7bf`. It analyzes the immutable local artifact `benchmarks/results/20260912T053613.813462Z-comparison-full` without rerunning or changing the optimizer.

A loss is any case where the v0.4 ContextOS task score is lower than Full Context. Relevance Only and Naive Extractive are also recorded whenever they score higher than ContextOS. All 50 cases were inspected: 18 have no ContextOS task-score loss and 32 are represented in `v040_failure_taxonomy.jsonl`. No loss was excluded.

## Summary

| Failure class | Primary count | Contributing count |
|---|---:|---:|
| `dependency_not_preserved` | 32 | 32 |
| `critical_item_under_ranked` | 0 | 32 |
| `stale_state_retained` | 0 | 32 |
| `contradictory_state_retained` | 0 | 16 |
| All other declared classes | 0 | 0 |
| `unknown` | 0 | 0 |

| Family | Cases | ContextOS losses | Repeated missing role | Mean ContextOS score | Mean ContextOS CIR |
|---|---:|---:|---|---:|---:|
| Coding agent | 18 | 0 | — | 1.00 | 1.00 |
| Research agent | 16 | 16 | `old-source` | 0.50 | 0.00 |
| Support/operations | 16 | 16 | `commitment` | 0.50 | 0.50 |

Full Context scores 1.0 on all 32 losses. Relevance Only and Naive Extractive both score 1.0 on all 16 research losses and 0.75 on all 16 support losses, compared with ContextOS at 0.5.

## Root cause

The dominant failure is relational rather than ordinary topical irrelevance. In every loss, the missing critical item is reachable from selected context through a directed relation, but v0.4 uses relations as soft score propagation rather than as a legal-state constraint. The allocator then rejects the critical item with `insufficient_compression_budget` while admitting lower-value stale or conflicting state.

For research cases, the task requires a synthesis record and the synthesis is `DERIVED_FROM` the authoritative primary source. ContextOS omits that source while retaining a contradiction and a superseded summary. The source contains the exact DOI, publication date, sample size, and effect required by the answer key.

For support cases, the selected task requires a history record, which in turn `REQUIRES` the original customer commitment. ContextOS retains the history but drops its required commitment while selecting the obsolete policy and recent delivery noise. The commitment contains the exact date and credit amount required by the answer key.

The terminal allocation reason is therefore a symptom: budget pressure exposes the absence of hard dependency closure. Merely increasing relevance weights could change these fixtures, but it would not guarantee a legal context state under a different budget or ordering.

## Representative cases

### `research-001`

- Missing item: `research-001-old-source`.
- Selected problematic state: `research-001-contradiction` and `research-001-superseded-summary`.
- ContextOS: task score 0.5, CIR 0.0.
- Full Context, Relevance Only, and Naive Extractive: task score 1.0.
- Primary classification: `dependency_not_preserved`.

### `support-001`

- Missing item: `support-001-commitment`.
- Selected problematic state: `support-001-old-policy` and `support-001-recent-noise`.
- ContextOS: task score 0.5, CIR 0.5.
- Full Context: 1.0; Relevance Only and Naive Extractive: 0.75.
- Primary classification: `dependency_not_preserved`.

## Subsystem attribution

The failures cross three v0.4 boundaries:

1. The dependency graph propagates utility but does not construct or enforce directed requirement closure.
2. The allocator treats related critical items as independently removable candidates.
3. Compression planning can exhaust its reservation and reject a required representation without a preservation validation or explicit failure state.

Deduplication, position-aware layout, and the evaluator are not supported as primary causes by these records. The required items are present before optimization, absent from final selection, and accompanied by explicit allocator rejection reasons.

## Stop/go decision

**Go.** All 32 observed v0.4 losses are plausibly addressable by the Phase 5 hypothesis: explicit preservation contracts, directed relational closure, validated transformations, and explicit overflow/failure semantics. Phase 5B remains justified.

This conclusion is intentionally narrow. The fixture is templated, producing two highly repeated failure shapes. It establishes a reproducible preservation gap but does not show that hard constraints will improve ordinary QA or every workload. Phase 5B must introduce diverse constraint-sensitive cases and must first run the unmodified v0.4 baseline against them.
