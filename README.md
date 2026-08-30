# ContextOS

ContextOS is a model-agnostic Python runtime for constructing an LLM's next input context under a fixed token budget. The project is currently at the repository-foundation milestone; benchmark claims will be added only when reproducible result artifacts exist.

The authoritative requirements are in [`MASTER_BUILD_SPEC.md`](MASTER_BUILD_SPEC.md), and current implementation progress is recorded in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

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

## Current scope

Version 0.1.0 establishes validated context models, configurable token counting, an in-memory store, CLI scaffolding, tests, and CI. Later milestones add optimization behavior and benchmarks in the order specified by the master build specification.

## License

MIT
