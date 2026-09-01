# ContextOS Benchmarks

The deterministic smoke profile runs with:

```bash
contextos benchmark --profile quick
```

It executes one fixed mixed-context case through Full Context, Last-N, Sliding Window, and the integrated ContextOS optimizer. The report contains deterministic token and selection outcomes; timing measurements remain in optimization traces and are intentionally excluded from deterministic equality checks.

## ContextOS-Bench

`datasets/contextos_bench.json` is generated deterministically from versioned templates and contains exactly 50 base cases:

- 18 coding-agent cases;
- 16 research-agent cases;
- 16 support/operations cases.

The cases are labeled as templated base data—not hand-authored or generated variants. Every case includes annotated required facts plus exact values, dates or identifiers, old critical evidence, recent irrelevant evidence, duplicate paraphrases, changed-number and negation traps, supersession, contradictions, and one-/two-hop dependencies.

Regenerate and verify the canonical dataset with:

```bash
python -m contextos.benchmarks.dataset \
  --output benchmarks/datasets/contextos_bench.json
pytest tests/unit/test_benchmark_schema.py
```

Run the benchmark with:

```bash
contextos benchmark run \
  --input benchmarks/datasets/contextos_bench.json \
  --output-directory benchmarks/results
```

The shared runner evaluates Full Context, Last-N, Sliding Window, and ContextOS. Full Context receives enough budget to act as the quality reference; other strategies use the case's configured optimization budget. Each raw result contains task-specific required-fact score, quality retention, Critical Information Recall, input tokens, context reduction, compression ratio, optimizer/embedding/compression timing, selected IDs, and decision reasons.

Aggregate reports retain p50/p95 optimizer latency and deterministic bootstrap 95% confidence intervals when at least 20 successful cases are available. Generated artifacts are content-addressed and ignored by default until a later validation phase explicitly approves an immutable result for version control. No benchmark result is hand-authored.

`datasets/deduplication_cases.json` remains the focused deduplication regression fixture.

## Controlled positional retrieval

`datasets/positional_retrieval.json` is a compact deterministic parameter grid covering 4K, 8K, 16K, and 32K token targets and evidence at the beginning, 25%, middle, 75%, and end. It compares identical records under original/full, relevance-descending, and the production ContextOS position-aware layout.

Run the offline plumbing check without credentials:

```bash
contextos benchmark positional --profile quick --provider deterministic --output-directory out/positional-quick
```

Run the complete grid against an explicitly selected OpenAI model only when credentials and the stated provider context limit are available:

```bash
contextos benchmark positional --profile full --provider openai --model MODEL_ID --max-context-tokens MODEL_CONTEXT_LIMIT --output-directory out/positional-real
```

The real-provider command requires `OPENAI_API_KEY` and the `openai` optional dependency. Unsupported context lengths are recorded as skipped. Artifacts retain raw predictions, exact-match results, token counts, prompt hashes, model identity, total provider latency, and positional aggregates. Offline deterministic accuracy is not evidence about long-context model behavior.

Regenerate the parameter grid with:

```bash
python -m contextos.benchmarks.positional --output benchmarks/datasets/positional_retrieval.json
```

## LongBench subset

`config/longbench_subset.json` pins the external source revision and configures four representative English tasks:

- `hotpotqa` and `2wikimqa`: multi-document QA, scored with normalized token F1;
- `passage_retrieval_en`: synthetic retrieval, scored with the official paragraph-number metric;
- `repobench-p`: code completion, scored with the official first-code-line similarity method.

The [official LongBench repository](https://github.com/THUDM/LongBench) is MIT licensed and documents the provenance of constituent tasks. External records remain subject to their upstream acknowledgements and terms; this project does not redistribute them.

Profiles are deterministic and preserve LongBench `_id` values:

- `quick`: 2 examples per task, 8 total, for development only;
- `standard`: 25 examples per task, 100 total;
- `full`: every example exposed by the four pinned task configurations.

Install the optional dependencies and explicitly prepare external data:

```bash
python -m pip install -e ".[benchmark]"
contextos benchmark longbench prepare \
  --config benchmarks/config/longbench_subset.json \
  --profile standard \
  --output out/longbench/prepared-standard.json
```

No download occurs on import, during normal tests, or in standard CI. Prepared datasets remain under ignored local output directories and must not be committed without a separate license/size review.

Predictions are newline-delimited JSON objects keyed by the preserved identity:

```json
{"dataset":"hotpotqa","source_id":"UPSTREAM_ID","prediction":"...","provider":"PROVIDER","model":"MODEL_ID"}
```

Score one complete prediction file with:

```bash
contextos benchmark longbench score \
  --prepared out/longbench/prepared-standard.json \
  --predictions out/longbench/predictions.jsonl \
  --output out/longbench/scores.json
```

Scoring rejects duplicate, missing, or unknown source IDs and mixed provider/model configurations. Aggregates remain separate by dataset and metric; unlike metrics are never collapsed into an unexplained overall average.
