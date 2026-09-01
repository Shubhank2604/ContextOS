# Research Foundation

> **Treating long context as structured runtime state rather than an undifferentiated token sequence can preserve downstream task quality at lower token budgets by jointly applying hard retention, task relevance, redundancy removal, dependency awareness, safe compression, and position-aware layout.**

ContextOS will test this hypothesis using the controlled benchmark tracks defined in the master build specification. No empirical claim is made before corresponding result artifacts exist.

## Positional sensitivity

The Phase 4B experiment is a controlled reproduction inspired by *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024, DOI `10.1162/tacl_a_00638`). It reproduces only a practical key-value retrieval setup that varies context length and evidence position; it does not claim to reproduce the paper's full methodology.

Offline deterministic runs demonstrate reproducibility of prompt construction, layout transformations, exact-match scoring, and raw artifacts. They are not model-behavior results. Any statement about middle-position degradation or mitigation must cite a real-provider artifact produced with one fixed provider, model, and decoding configuration across compared layouts.

## External validation

[LongBench](https://aclanthology.org/2024.acl-long.172/) supplies the external task distribution used in Phase 4C. ContextOS configures a reproducible subset rather than claiming coverage of the entire benchmark. The adapter preserves upstream IDs, uses deterministic profiles, and applies task-appropriate automated metrics without an LLM judge. Model-quality claims remain blocked until complete same-model prediction artifacts are produced and reviewed.
