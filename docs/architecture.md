# Architecture

ContextOS sits between application state and an LLM provider. Its staged optimizer will validate and tokenize candidate context, enforce mandatory retention, remove safe redundancy, score optional items, allocate budget, compress where allowed, arrange the final layout, and emit a complete decision trace.

Semantic deduplication has an explicit default-on policy switch so ablation runs can bypass it without abusing similarity thresholds. Zero-weight recency and dependency configurations return neutral component scores and renormalize the remaining composite signals. These controls exist to isolate measured component effects; normal presets retain the full pipeline.

Detailed component boundaries will be documented as each sequential milestone is implemented and verified.

## Baseline boundary

All strategies accept an `OptimizationPolicy`, tokenize isolated copies of input items, and return an `OptimizedContext` with budget accounting and an `OptimizationTrace`:

- Full Context retains input order and raises `ContextBudgetOverflow` instead of truncating.
- Last-N ranks by `updated_at`, then `created_at`, then item ID, and retains the newest contiguous suffix that fits while preserving original order in the output.
- Sliding Window uses the newest `updated_at` as its deterministic reference time, removes items outside the configured window, and applies Last-N whole-item selection if the window exceeds budget.
- Relevance Only ranks whole items solely by task-to-item embedding relevance and greedily admits those that fit. It has no type, recency, dependency, novelty, deduplication, or compression logic.
- Naive Extractive ranks source sentences solely by task relevance, greedily admits sentences under the common budget, and restores source-item and sentence order. It does not use ContextOS composite scoring or allocation.

Mandatory retention is intentionally absent from these naive comparison strategies and is identified by a trace warning. Hard retention belongs to the integrated ContextOS optimizer pipeline.

## Semantic selection boundary

Embedding providers expose one batch-oriented interface and remain independent from the optimizer. Tests and deterministic local measurements use a stable feature-hash provider; the optional sentence-transformer implementation loads its configured model lazily. Embeddings are cached by the SHA-256 hash of Unicode-normalized content.

Exact deduplication runs before embeddings and may remove only optional items. Mandatory duplicates always survive. Semantic deduplication is scoped to matching context types and uses a configurable cosine threshold, defaulting to `0.92`. Similarity alone is insufficient: differing numbers, dates, identifiers, URLs, paths, code, or negation force both items to survive.

## Scoring and dependency boundary

Every scoring component is independently normalized to `[0,1]`. Composite utility uses policy weights normalized to sum to one. Recency uses exponential half-life decay from an explicit reference time or, for deterministic standalone scoring, the newest item timestamp.

Application-provided importance always wins. If it is omitted, the context model applies these deterministic fallbacks; they are configuration defaults, not empirical claims:

| Context type | Importance fallback | Type priority |
|---|---:|---:|
| System instruction | 1.00 | 1.00 |
| Tool definition | 0.70 | 0.50 |
| User message | 0.70 | 0.70 |
| Assistant message | 0.40 | 0.50 |
| Tool output | 0.40 | 0.60 |
| Retrieved document | 0.50 | 0.65 |
| Memory | 0.50 | 0.60 |
| Decision | 0.80 | 0.85 |
| Error | 0.80 | 0.80 |
| Plan | 0.60 | 0.55 |
| Code | 0.70 | 0.75 |
| Task state | 0.90 | 0.95 |

Dependency edges are application supplied and retained with their relation type. Score propagation treats an edge as a traversable connection in both directions, multiplies weights along a path, takes the strongest reachable neighbor score, stops at the configured depth (default `2`), and tracks visited IDs to terminate cycles. `SUPERSEDES` and `CONTRADICTS` remain explicit graph evidence; scoring does not silently delete either endpoint.

## Allocation boundary

Mandatory tokens are reserved before optional allocation. Static policy errors, mandatory overflow, and context-dependent class-minimum infeasibility are separate typed failures. Class minima are soft floors applied only to optional types present after deduplication; whole-item granularity can leave a floor unmet, which is recorded rather than silently treated as a policy error.

The allocator performs a stable per-type floor pass followed by a global value-density pass. Class maxima apply to raw selections and planned compressed representations. Items that fail raw selection are ranked separately for compression, and the resulting `AllocationPlan` partitions every optional candidate into exactly one outcome: direct selection, a reserved `CompressionRequest`, or rejection with a reason.

The allocator never invokes a compressor. It also never evicts a raw selection to make room for a compressed candidate. This deterministic greedy behavior is intentionally not claimed to be globally optimal. The plan retains the full ranked compression-candidate order so the compression stage can reuse returned reservations without recomputing or reordering allocator decisions.

## Compression and layout boundary

Compression operates on one item at a time and returns source provenance, strategy, original and compressed token counts, and an explicit failure reason. Extractive compression retains exact source sentences in source order. Tool-output compression retains exact critical/task/boundary lines. Optional LLM summarization is disabled by default and cannot process protected content. A failed or under-target attempt releases its unused reservation for later ranked candidates.

Layout is independent from selection. Original-order and relevance-descending layouts serve as controls. Position-aware layout places mandatory system information first, high-utility evidence early, and recent task state or user messages near the end. Layout never modifies item content, so internal code order is preserved.

## Persistence and integrated runtime boundary

The in-memory and SQLite stores share item, time/type/tier query, edge, tier-update, and explicit-delete operations. SQLite records its schema version and migrates version-zero stores to the current schema. Lifecycle transitions are deterministic, accept explicit application overrides, and never automatically delete archived records.

`ContextOptimizer.optimize(task, items, policy)` owns the complete budgeted pipeline. It validates and tokenizes isolated copies, reserves mandatory content, deduplicates, scores, allocates, compresses, lays out, validates invariants, emits a complete trace, and optionally persists items and edges. Provider fallback is visible in trace warnings.
