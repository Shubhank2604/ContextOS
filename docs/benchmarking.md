# Benchmarking Methodology

ContextOS is evaluated against deterministic baselines using project-owned required-fact cases. Phase 4B adds controlled positional retrieval, and Phase 4C adds a configured LongBench subset for external validation.

Raw per-case results and environment metadata—not generated plots—will be the source of truth.

## Deduplication fixture

Phase 3A includes a small labeled development fixture for duplicate precision, recall, F1, and false-positive counts. Run it with:

```bash
contextos benchmark dedup --input benchmarks/datasets/deduplication_cases.json
```

The default semantic threshold is `0.92`. This curated fixture is a regression and configuration aid, not evidence of general semantic-deduplication quality. Final claims require the immutable v0.4.0 benchmark artifacts.

## Phase 4A project-owned benchmark

The canonical ContextOS-Bench dataset contains 50 templated base cases across coding, research, and support/operations agents. Templates are deterministic and versioned; the schema distinguishes base cases from any future generated variants and requires reproducibility metadata for generated data.

Task-specific score is exact annotated-fact recall over the constructed input context. Critical Information Recall is stricter: a required item counts only when all exact facts attached to it remain represented. This evaluator measures context construction, not final language-model answer quality, and reports it as such.

Full Context is executed with enough budget to provide the quality reference. Budget-constrained strategies use the same configured case policy. Quality retention is calculated only when the Full Context task score is non-zero; a zero reference is reported as unavailable rather than divided implicitly.

Every run retains raw per-case results. Aggregates are derived from successful cases, keep unlike metrics separate, report p50/p95 optimizer latency, and use a seeded percentile bootstrap for 95% confidence intervals when at least 20 cases are available.

## Phase 4G statistical protocol

ContextOS-Bench reports 1,000-resample, seeded percentile-bootstrap 95% confidence intervals for aggregate task score and Critical Information Recall when at least 20 successful cases are available. Strategy deltas are calculated within each shared case before bootstrapping, so pairing is preserved. Reports include every constrained strategy minus Full Context and, separately, ContextOS minus each simple baseline. Input-token deltas remain signed: a negative value means the candidate used fewer tokens.

LongBench uses the same 20-case threshold and paired bootstrap independently within each dataset and official task metric. It reports score and quality-retention intervals per dataset/strategy, every candidate minus Full Context, and ContextOS minus each simple baseline. It never combines QA F1, retrieval precision, and code similarity into one unexplained score. Failed cases and Full Context overflows remain in raw predictions but do not enter a successful-case interval; the paired sample count is always reported.

The positional experiment currently has one observation per context-length, evidence-position, and strategy cell, while the labeled deduplication development fixture has 10 cases. Neither supports a reported confidence interval under this protocol, so their artifacts state `not reported` and give the reason. No interval is fabricated from token-level or position-level pseudo-replication.

Real-model runners use temperature `0` where the provider supports it. Every immutable bundle records UTC date/time, provider, model, decoding configuration when executed by ContextOS, and the environment needed to identify the run. Imported LongBench prediction files record provider/model but cannot prove an unrecorded decoding configuration; comparative claims from imported data therefore require matching external provenance. Results from different provider, model, prompt, evaluator, case set, or decoding configurations must be treated as separate experiments and must never be attributed to ContextOS.

Runs are not repeated mechanically. A critical comparison is repeated with the identical configuration only when residual model nondeterminism is large enough to change the qualitative conclusion; every repetition remains a separate immutable bundle. Deterministic evaluators and seeded bootstrap calculations are reproducible from the retained per-case records.

## Phase 4B controlled positional experiment

This is a **controlled reproduction inspired by the paper**, not a reproduction of every experiment in *Lost in the Middle*. It tests exact key-value retrieval while varying target input length (4K, 8K, 16K, and 32K tokens), original evidence position (beginning, 25%, middle, 75%, and end), and layout (original/full, relevance descending, and ContextOS position aware).

All layouts receive exactly the same records, query, provider/model, decoding bound, and evaluator. The original layout preserves the requested evidence position. The other layouts use the production relevance-descending and position-aware layout implementations. Exact match normalizes only surrounding whitespace and character case.

Reports retain accuracy by position and context length, the max-minus-min positional accuracy gap, population variance and standard deviation, estimated and provider-reported input tokens, raw predictions, prompt hashes, cached/output token counts when exposed, and total provider latency. Context buckets beyond the user-declared model limit are skipped and recorded rather than silently truncated.

The deterministic provider is an offline oracle used only to validate dataset, layout, evaluator, aggregation, and artifact plumbing. It is position invariant by design and cannot establish the positional degradation phenomenon. Only a recorded real-provider run may support an empirical claim.

## Phase 4C LongBench subset

The configured subset covers HotpotQA, 2WikiMQA, PassageRetrieval-en, and RepoBench-P from the pinned `zai-org/LongBench` source revision. It represents multi-document QA, synthetic retrieval, and code completion while keeping provider cost tractable.

Sampling ranks preserved `(dataset, _id)` identities with a versioned SHA-256 seed, so quick and standard selections do not depend on upstream row order. Standard contains exactly 100 examples. Full retains every row from the configured tasks. The preparation artifact records repository, revision, split, profile, seed, original IDs, answers, and source length.

The evaluator follows the official LongBench task mapping: normalized English token F1 for the two QA datasets, paragraph-number precision for retrieval, and first uncommented-line similarity for RepoBench-P. Each prediction is scored against all accepted answers and keeps the maximum, matching the official evaluation convention. No LLM judge is used.

Prediction scoring requires complete identity coverage and one provider/model configuration. Phase 4D adds the complete strategy comparison runner; Phase 4C deliberately does not fabricate model predictions when the external dependency or credentials are unavailable.

## Phase 4D baseline comparison

All applicable project-owned and LongBench comparisons now use six strategies: Full Context, Last-N, Sliding Window, Relevance Only, Naive Extractive, and ContextOS. Relevance Only is deliberately restricted to embedding relevance over whole items. Naive Extractive is deliberately restricted to sentence relevance and source-order restoration. They do not borrow ContextOS's typed retention, deduplication, multi-factor scoring, dependency propagation, allocation, or position-aware policy.

Within one LongBench run, all strategies share the exact prepared case IDs, pinned prompt templates, provider, model, temperature-zero configuration, output bounds, context limit, and evaluator. Constrained strategies share one declared context budget. Full Context is called only when the unmodified prompt fits the declared provider limit; otherwise its raw status is `context_overflow` and its task score and quality reference remain unavailable.

Raw predictions include strategy, status, original and constructed context tokens, local and provider prompt tokens, output/cached tokens when exposed, optimizer and provider latency, context reduction, selected chunk IDs, warnings, and model identity. Scoring requires the complete case-by-strategy matrix, never substitutes a missing prediction, and calculates quality retention only against a non-zero successful Full Context score for the same case and metric.

The current offline ContextOS-Bench comparison is diagnostic, not a finalized empirical claim. Newly added simple baselines may outperform the current integrated configuration; those results must remain visible and motivate the Phase 4E ablation study rather than being filtered from reports.

## Phase 4E ablation study

`contextos benchmark ablation` executes full ContextOS plus five single-component variants: without semantic deduplication, recency, dependency scoring, compression, or position-aware layout. Every variant receives the same 50 cases and per-case budget. The artifact records the exact policy override for each strategy and retains raw task score, Critical Information Recall, input tokens, optimizer latency, aggregate bootstrap intervals, and deltas against full ContextOS.

The current deterministic run is diagnostic. Full ContextOS scores `0.68` on annotated-fact recall and `0.52` CIR at a mean `69.84` input tokens. Removing dependency scoring lowers task score to `0.60` while CIR remains `0.52` and mean input falls to `68.56` tokens. The other removals are neutral on task score, CIR, and input tokens in this dataset. Single-run millisecond latency differences are retained raw but are too small and noisy to support a component-value claim.

These neutral results define benchmark sensitivity rather than proving that the components have no value. The presence-based evaluator cannot measure layout effects, and these cases do not materially exercise compression, semantic duplicate removal, or recency-sensitive selection. Accordingly, no component is removed on this evidence alone. Position-aware layout must be judged by the positional provider experiment; the other components require targeted or external cases before a simplification decision.

## Phase 4F immutable artifacts

Every meaningful ContextOS-Bench, ablation, deduplication, positional, and LongBench run writes one timestamped strategy/profile directory containing exactly:

```text
config.json
environment.json
cases.jsonl
predictions.jsonl
metrics.json
metrics.csv
report.md
```

`environment.json` records the Python and ContextOS versions, Git SHA, operating system, relevant installed dependency versions, embedding provider/model, and LLM provider/model when applicable. `config.json` records track-specific dataset identity, strategies, budgets, thresholds, source revisions, and decoding settings. Existing bundle paths are accepted only when all seven files match byte-for-byte; missing, added, or changed files are rejected as mutation or collision.

Raw cases, predictions, and JSON metrics are the source of truth. CSV and Markdown are deterministic derived views for analysis and review. `contextos benchmark --profile quick` is intentionally excluded because it is an ephemeral CI smoke test rather than a meaningful research run. Generated bundles remain ignored until an explicit evidence review approves them for version control.
