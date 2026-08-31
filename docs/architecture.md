# Architecture

ContextOS sits between application state and an LLM provider. Its staged optimizer will validate and tokenize candidate context, enforce mandatory retention, remove safe redundancy, score optional items, allocate budget, compress where allowed, arrange the final layout, and emit a complete decision trace.

Detailed component boundaries will be documented as each sequential milestone is implemented and verified.

## Baseline boundary

All strategies accept an `OptimizationPolicy`, tokenize isolated copies of input items, and return an `OptimizedContext` with budget accounting and an `OptimizationTrace`. The current baselines operate on whole context items:

- Full Context retains input order and raises `ContextBudgetOverflow` instead of truncating.
- Last-N ranks by `updated_at`, then `created_at`, then item ID, and retains the newest contiguous suffix that fits while preserving original order in the output.
- Sliding Window uses the newest `updated_at` as its deterministic reference time, removes items outside the configured window, and applies Last-N whole-item selection if the window exceeds budget.

Mandatory retention is intentionally absent from these naive comparison strategies and is identified by a trace warning. Hard retention belongs to the ContextOS optimizer pipeline introduced in later milestones.

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
