# Changelog

All notable changes to ContextOS are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Configurable semantic deduplication threshold with a conservative default of `0.92`.
- Deterministic and optional sentence-transformer embedding providers with normalized-content caching.
- Mandatory-aware exact and semantic deduplication with numeric, date, identifier, path, URL, code, and negation safety guards.
- Provider-driven relevance and novelty scoring.
- Labeled deduplication fixture metrics and `contextos benchmark dedup`.

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
