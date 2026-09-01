# Known Limitations

- The positional benchmark's deterministic provider is an oracle for offline plumbing validation and cannot measure real long-context degradation. No positional-robustness claim is valid until a real-provider artifact is reviewed.
- Local positional token targets use the configured tiktoken encoding; provider-reported input tokens are retained when available and may differ slightly from local estimates.
- LongBench records are external data with upstream task-specific provenance. They are downloaded only on explicit request, kept out of Git by default, and require the optional `benchmark` dependencies. This local environment has not downloaded them, so Phase 4C currently validates the pinned adapter, profiles, schemas, and evaluator against offline official-shaped fixtures rather than reporting model scores.
- The Phase 4D LongBench comparison runner is implemented and offline-tested, but no real-provider six-strategy result exists locally because external records and credentials were not supplied. Full Context can legitimately be unavailable for examples exceeding a declared model limit, in which case quality retention is not computed.
- The deterministic local embedding provider is intended for offline operation and testing, not as a claim of semantic quality. Optional sentence-transformer embeddings require the `semantic` extra and may download model weights.
- LLM summarization is lossy, disabled by default, and requires an explicitly injected provider. Protected system, tool-definition, code, mandatory, non-compressible, identifier-bearing, and secret-like content is rejected.
- LLMLingua is not a core dependency or v0.3 adapter because a reliable integration would add model/download coupling to the offline runtime; it can be evaluated as an isolated optional adapter in v0.4.
- The v0.3 allocator is deterministic and auditable but is not claimed to be globally optimal.
- SQLite is the only durable v0.3 backend; Redis and vector databases are intentionally outside scope.
- ContextOS-Bench v0.3 provides schemas and labeled seed fixtures. Reproducible external-model measurements and statistical comparisons are the v0.4 milestone.
- ContextOS does not verify model answers or provide prompt-injection security.
