# ContextOS

ContextOS is a model-agnostic Python runtime for constructing an LLM's next input context under a fixed token budget. Version 0.2.0 adds deterministic Full Context, Last-N, and Sliding Window baselines with complete decision traces. Benchmark claims will be added only when reproducible result artifacts exist.

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

## Baseline CLI

```bash
contextos optimize \
  --input examples/data/coding_context.json \
  --task "Fix authentication timeout" \
  --budget 100 \
  --strategy last-n \
  --trace-json out/trace.json

contextos benchmark --profile quick
```

The Last-N baseline retains the newest contiguous suffix of whole items that fits. Sliding Window first removes items older than its configured time window, then applies the same whole-item budget behavior. These naive baselines deliberately do not enforce typed mandatory retention; that limitation is emitted in their traces and allows later ContextOS policies to be compared against untyped recency baselines.

## License

MIT
