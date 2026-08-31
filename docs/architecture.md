# Architecture

ContextOS sits between application state and an LLM provider. Its staged optimizer will validate and tokenize candidate context, enforce mandatory retention, remove safe redundancy, score optional items, allocate budget, compress where allowed, arrange the final layout, and emit a complete decision trace.

Detailed component boundaries will be documented as each sequential milestone is implemented and verified.

## Baseline boundary

All strategies accept an `OptimizationPolicy`, tokenize isolated copies of input items, and return an `OptimizedContext` with budget accounting and an `OptimizationTrace`. The current baselines operate on whole context items:

- Full Context retains input order and raises `ContextBudgetOverflow` instead of truncating.
- Last-N ranks by `updated_at`, then `created_at`, then item ID, and retains the newest contiguous suffix that fits while preserving original order in the output.
- Sliding Window uses the newest `updated_at` as its deterministic reference time, removes items outside the configured window, and applies Last-N whole-item selection if the window exceeds budget.

Mandatory retention is intentionally absent from these naive comparison strategies and is identified by a trace warning. Hard retention belongs to the ContextOS optimizer pipeline introduced in later milestones.
