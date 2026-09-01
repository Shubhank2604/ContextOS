# ContextOS

ContextOS is a model-agnostic Python runtime for constructing an LLM's next input context under a fixed token budget. Version 0.3.0 delivers the integrated runtime: safe deduplication, multi-factor scoring, dependency propagation, deterministic allocation, guarded compression, position-aware layout, complete traces, and SQLite persistence. Benchmark claims will be added only when reproducible result artifacts exist.

The v0.4 development benchmark suite now includes ContextOS-Bench and a controlled positional-retrieval experiment inspired by *Lost in the Middle*. Offline runs validate deterministic construction, layout, scoring, and artifact plumbing; empirical model-behavior claims require an explicitly requested real-provider run.

The external-validation adapter configures a reproducible LongBench subset spanning multi-document QA, synthetic passage retrieval, and repository-level code completion. External examples are downloaded only by an explicit command and are not committed to this repository.

The detailed build specification and live project status are maintained locally during development. Checked-in version changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Development setup

ContextOS requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy
pytest
contextos --help
```

## Python SDK

```python
from contextos import ContextItem, ContextOptimizer, ContextType, OptimizationPolicy

optimizer = ContextOptimizer()
result = optimizer.optimize(
    "Fix authentication timeout",
    items,
    OptimizationPolicy.balanced(max_input_tokens=8_000, reserve_output_tokens=1_000),
)
```

`ContextOptimizer.optimize(task, items, policy)` is the sole authority for the final input budget. Inputs are copied; caller-owned items are not mutated.

## CLI

```bash
contextos optimize \
  --input examples/data/coding_context.json \
  --task "Fix authentication timeout" \
  --budget 100 \
  --strategy contextos \
  --trace-json out/trace.json

contextos benchmark --profile quick
contextos benchmark run \
  --input benchmarks/datasets/contextos_bench.json \
  --output-directory benchmarks/results
contextos inspect --input examples/data/coding_context.json
contextos store stats --database out/contextos.sqlite
```

Full Context, Last-N, and Sliding Window remain available with `--strategy` for controlled comparisons. The naive baselines deliberately do not enforce typed mandatory retention; that limitation is emitted in their traces.

ContextOS-Bench contains 50 deterministic templated base cases across coding, research, and support/operations agents. Benchmark runs retain raw per-case task score, critical-information recall, token reduction, compression ratio, decision reasons, and optimizer timings. Result artifacts are generated locally and are not treated as measured project claims until reviewed and committed through the later v0.4 validation phases.

## License

MIT
