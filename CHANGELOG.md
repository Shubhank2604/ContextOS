# Changelog

All notable changes to ContextOS are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Phase 4A ContextOS-Bench schema and deterministic generator with 50 templated base cases across coding, research, and support/operations scenarios.
- Exact required-fact annotations covering dates, identifiers, changed-number and negation traps, supersession, contradictions, recent noise, and one-/two-hop dependencies.
- Shared Full Context, Last-N, Sliding Window, and ContextOS benchmark runner with raw task, CIR, token, compression, and latency measurements.
- Deterministic bootstrap confidence intervals and content-addressed local benchmark artifacts.
- `contextos benchmark run` and aggregate-aware `contextos benchmark compare` commands.

## [0.3.0] - 2026-08-31

### Added

- Configurable semantic deduplication threshold with a conservative default of `0.92`.
- Deterministic and optional sentence-transformer embedding providers with normalized-content caching.
- Mandatory-aware exact and semantic deduplication with numeric, date, identifier, path, URL, code, and negation safety guards.
- Provider-driven relevance and novelty scoring.
- Labeled deduplication fixture metrics and `contextos benchmark dedup`.
- Complete serializable optimization-policy fields, normalized weights, and quality/balanced/economy presets.
- Deterministic importance, exponential-recency, type-priority, dependency-propagation, and composite scoring.
- Typed dependency edges with bounded cycle-safe traversal and explicit unknown-reference failures.
- Contextual mandatory/minimum budget validation with distinct typed failures.
- Deterministic class-floor, class-maximum, value-density, and compression-reservation allocation plans.
- Explicit allocator/compressor handoff models with complete candidate partition and budget invariants.
- Safe extractive, tool-output, no-op, and optional LLM-summary compression contracts.
- Deterministic returned-reservation reuse after failed or under-target compression.
- Original-order, relevance-descending, and position-aware layout strategies.
- Versioned SQLite persistence, dependency edges, migrations, and lifecycle transitions.
- The authoritative integrated `ContextOptimizer` pipeline with complete per-item traces.
- Public optimizer CLI, input inspection, store statistics, and benchmark comparison commands.
- ContextOS-Bench schemas, initial deduplication/allocation/regression fixtures, and a quick runner that invokes both baseline and ContextOS strategies.

## [0.2.0] - 2026-08-30

### Added

- Token-budget `OptimizationPolicy` foundation.
- Full Context, Last-N, and Sliding Window deterministic baselines.
- Per-item and whole-run optimization traces.
- Baseline CLI and deterministic quick benchmark smoke profile.

## [0.1.0] - 2026-08-30

### Added

- Python package configuration and development tooling.
- Validated typed context model.
- Token-counting abstraction and tiktoken implementation.
- In-memory context store.
- CLI skeleton, documentation skeleton, tests, and CI.

### Fixed

- Bound NumPy below 2.5 so Python 3.11-targeted mypy checks do not parse Python 3.12-only NumPy stubs.
- Updated GitHub's checkout and Python setup actions to their Node.js 24 releases.
