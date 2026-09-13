# ContextOS Phase 5 Research Implementation Plan

## Purpose

Phase 5 is the research evolution of ContextOS after the completion of the v0.4.0 engineering baseline.

ContextOS v0.4.0 should be treated as a frozen reference implementation and benchmark baseline. Phase 5 must not rewrite history, overwrite v0.4 benchmark artifacts, weaken baselines, or fabricate improvements.

The central research goal is to evolve ContextOS from:

> heuristic context ranking under a token budget

into:

> constraint-aware construction of heterogeneous LLM runtime context under a token budget.

The working research hypothesis is:

> Explicit preservation contracts, relational consistency, and validated context transformations can reduce critical-information and state-consistency failures versus relevance-only and heuristic context selection while retaining substantial context reduction.

This hypothesis must be tested, not assumed.

---

# 1. Current Baseline

ContextOS v0.4.0 already provides:

- typed context items;
- exact and semantic deduplication;
- relevance, novelty, importance, recency, type-priority, and dependency-aware scoring;
- dependency propagation;
- token-budget validation and allocation;
- safe extractive and tool-output compression;
- optional LLM-summary compression;
- position-aware layout;
- optimization traces;
- SQLite persistence;
- six-strategy benchmarking;
- ContextOS-Bench;
- LongBench subset evaluation;
- positional experiments;
- ablations;
- immutable benchmark bundles;
- bootstrap confidence intervals;
- paired comparisons;
- performance telemetry;
- real-model benchmark support.

Measured v0.4.0 results currently include approximately:

- 54.3% average context reduction on the deterministic 50-case ContextOS-Bench;
- 0.68 task score;
- 0.52 Critical Information Recall;
- ~2.9 ms p95 optimizer latency;
- dependency scoring ablation: 0.68 full ContextOS vs 0.60 without dependency scoring;
- relevance-only and naive extractive outperforming ContextOS on some quality metrics;
- mixed LongBench results.

These results are the research starting point.

They must remain preserved as the Phase 5 baseline.

---

# 2. Phase 5 Research Objective

Phase 5 should investigate whether ContextOS can improve the reliability-efficiency frontier by moving beyond purely soft ranking.

The key conceptual shift is:

## v0.4 formulation

```text
candidate context
    ↓
score items
    ↓
rank / allocate under token budget
    ↓
compress
    ↓
layout
    ↓
final context
```

## Phase 5 formulation

```text
heterogeneous runtime state
        ↓
identify preservation requirements
        ↓
resolve hard relations
        ↓
determine legal context states
        ↓
score utility + omission risk
        ↓
allocate under token budget
        ↓
perform type-aware transformations
        ↓
validate preservation contracts
        ↓
fallback if validation fails
        ↓
layout
        ↓
optimized context
+ auditable decision trace
```

---

# 3. Research Targets

The following are research targets only.

They are not acceptance guarantees and must not be used publicly unless measured.

Approximate Phase 5 targets:

| Metric | v0.4 Baseline | Phase 5 Research Target |
|---|---:|---:|
| Context reduction | 54.3% | 40–50% |
| Critical Information Recall | 52% | 85–90% |
| Full-context task-quality retention | 68% deterministic | 88–93% |
| Constraint/state failure reduction vs relevance-only | Not measured | 60–80% lower |
| Local optimizer p95 | ~2.9 ms | Preferably <10 ms |

Phase 5 may intentionally sacrifice some compression in exchange for much stronger retention and consistency.

The project must not tune specifically to force these values.

---

# 4. Branching and Baseline Preservation

Before Phase 5 begins:

1. Confirm v0.4.0 is tagged and reproducible.
2. Record the exact baseline Git SHA.
3. Create a research branch, for example:

```text
research/phase5-constraint-context
```

4. Define a constant or benchmark metadata field:

```text
PHASE5_BASELINE_SHA
```

5. Every Phase 5 benchmark artifact should record:

```text
baseline_v040_sha
current_research_sha
```

6. Existing v0.4 benchmark artifacts must never be modified.

Baseline validation before any Phase 5 work:

```bash
pytest
ruff check .
ruff format --check .
mypy
```

All must pass.

---

# 5. Phase 5A — v0.4 Failure Taxonomy

## Goal

Before introducing new architecture, determine why v0.4 loses to stronger baselines.

Analyze every ContextOS-Bench case where:

```text
ContextOS fails AND Relevance Only succeeds
ContextOS fails AND Naive Extractive succeeds
ContextOS fails AND Full Context succeeds
```

Do not change the optimizer during this phase.

## Required output

Create:

```text
benchmarks/analysis/v040_failure_taxonomy.jsonl
benchmarks/analysis/v040_failure_report.md
```

Each failed case should include at least:

```json
{
  "case_id": "...",
  "contextos_score": 0.0,
  "reference_strategy": "relevance_only",
  "reference_score": 1.0,
  "missing_required_items": [],
  "selected_items": [],
  "failure_class": "...",
  "root_cause": "...",
  "notes": "..."
}
```

## Initial failure taxonomy

Use explicit classes:

```text
critical_item_under_ranked
class_budget_starvation
dependency_not_preserved
stale_state_retained
contradictory_state_retained
compression_information_loss
incorrect_deduplication
recency_failure
position_failure
type_priority_failure
evaluator_limitation
unknown
```

Additional categories may be added if real failures require them.

## Required report

The Markdown report should summarize:

- total cases analyzed;
- failure counts per category;
- representative cases;
- which subsystem caused the failure;
- whether the failure could plausibly be addressed by hard constraints;
- whether the current Phase 5 hypothesis remains supported.

Example:

```text
Failure type                    Count
-------------------------------------
critical_item_under_ranked        11
dependency_not_preserved           5
class_budget_starvation            4
compression_information_loss       2
unknown                            1
```

## Acceptance gate

Every v0.4 loss must be:

- assigned a failure class; or
- explicitly marked `unknown`.

No failed case should be silently excluded.

## Stop/go rule

If the majority of important failures are unrelated to preservation, dependencies, stale state, contradiction, or critical information loss, stop and reconsider the remaining Phase 5 roadmap before implementing it.

## Suggested commit

```text
research: add v0.4 context failure taxonomy
```

---

# 6. Phase 5B — Constraint-Sensitive Benchmark

## Goal

Create a benchmark that explicitly exercises failure modes not sufficiently covered by the existing 50-case fixture.

Do not replace the current ContextOS-Bench.

Add a separate track.

Working name:

```text
ContextOS-Bench Constraints
```

The name may change later.

## Initial benchmark size

Target approximately:

```text
80–120 deterministic cases
```

The initial benchmark must support deterministic scoring without requiring an LLM judge.

## Required categories

### 6.1 Dependency cases

Example:

```text
A: API requires authentication token X.
B: Endpoint /payments uses API A.
Question: call /payments.

B REQUIRES A
```

Success requires both relevant pieces to remain representable.

### 6.2 Supersession cases

Example:

```text
Turn 3:
deployment_region = us-east-1

Turn 27:
deployment_region = us-west-2
```

Relation:

```text
turn_27 SUPERSEDES turn_3
```

Expected state:

```text
deployment_region = us-west-2
```

### 6.3 Contradiction cases

Example:

```text
Document A:
maximum refund = $500

Document B:
maximum refund = $750
```

Metadata must indicate authority, timestamp, version, or canonical status.

### 6.4 Exact numeric preservation

Examples:

```text
$1,947.36
4.75%
2026-09-12
```

### 6.5 Identifier preservation

Examples:

```text
usr_72B91
payment_instruction_v2
validateAccountRisk()
```

### 6.6 Negation and policy preservation

Example:

```text
Do NOT execute transfer without approval.
```

A compressed form that changes the polarity must fail.

### 6.7 Tool-state cases

Example:

```json
{
  "reservation_status": "cancelled",
  "reservation_id": "R9284",
  "timestamp": "..."
}
```

Current state must override obsolete tool output.

### 6.8 Citation/evidence cases

The optimizer must preserve:

- evidence span;
- source identifier;
- citation relationship;
- provenance where required.

### 6.9 Multi-hop relation cases

Example:

```text
C REQUIRES B
B REQUIRES A
```

Selecting C must preserve the dependency closure.

## Required case annotations

Every benchmark case should include machine-readable ground truth such as:

```json
{
  "critical_item_ids": [],
  "relations": [],
  "required_exact_values": [],
  "required_identifiers": [],
  "required_citations": [],
  "forbidden_item_combinations": [],
  "expected_current_state": {}
}
```

## New metrics

Add at least:

```text
Critical Information Recall
Constraint Violation Rate
Dependency Closure Rate
State Consistency Rate
Contradiction Leakage Rate
Exact Value Preservation Rate
Identifier Preservation Rate
Citation Preservation Rate
```

Do not remove:

```text
task score
input tokens
context reduction
optimizer latency
```

## Baseline run

Run unmodified v0.4 against the new benchmark.

This is mandatory.

The benchmark should expose meaningful failures.

If v0.4 scores nearly perfectly, the benchmark is likely not testing a useful research gap.

## Suggested commit

```text
research: add constraint-sensitive context benchmark
```

---

# 7. Phase 5C — Preservation Contracts

## Goal

Replace the coarse boolean-only preservation model with explicit preservation semantics.

Do not remove legacy fields yet.

Current concepts such as:

```text
mandatory
compressible
evictable
```

should remain backwards-compatible during Phase 5.

## Proposed module

```text
src/contextos/contracts/
```

## Proposed model

Conceptually:

```python
class PreservationContract(BaseModel):
    retention: RetentionPolicy
    preserve_numbers: bool = False
    preserve_dates: bool = False
    preserve_identifiers: bool = False
    preserve_citations: bool = False
    preserve_negation: bool = False
    preserve_structure: bool = False
    required_keys: tuple[str, ...] = ()
```

Possible retention enum:

```python
class RetentionPolicy(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    REQUIRED_IF_REFERENCED = "required_if_referenced"
```

Extend `ContextItem` with:

```python
contract: PreservationContract | None = None
```

## Initial supported contracts

Only implement contracts that can be validated deterministically.

Start with:

```text
retention
numbers
dates
identifiers
citations
negation
structured keys
```

Do not add vague options such as:

```text
preserve_meaning=True
```

unless there is a defensible validator.

## Required tests

Test at least:

```text
required item cannot be silently evicted
required values survive transformations
required identifiers survive transformations
required structured keys survive transformations
invalid contract configuration is rejected
legacy ContextItem behavior remains valid
serialization round-trip remains stable
```

## Suggested commit

```text
feat: add context preservation contracts
```

---

# 8. Phase 5D — Hard Relational Semantics

## Goal

Separate soft dependency scoring from hard relation enforcement.

The current v0.4 dependency graph should remain intact for baseline behavior.

Do not mutate existing soft score propagation into incompatible hard semantics.

Add a separate constraint layer.

## Proposed abstraction

For example:

```text
ContextConstraintGraph
```

or:

```text
ConstraintResolver
```

## Relation semantics

### 8.1 REQUIRES

If:

```text
B REQUIRES A
```

then:

```text
B selected => A represented
```

A may be:

- retained verbatim; or
- represented by a validated transformed form.

Otherwise B must not be selected.

### 8.2 SUPERSEDES

If:

```text
B SUPERSEDES A
```

and B is current and valid:

```text
A should not enter the final model-visible context
```

unless an explicit historical-context policy allows it.

### 8.3 CONTRADICTS

Contradictions must not be silently resolved.

Use deterministic metadata when available:

```text
timestamp
authority
version
canonical
source priority
```

If deterministic resolution is impossible:

```text
emit unresolved_conflict
```

Policy may then choose:

- retain both;
- fail optimization;
- require caller resolution.

Do not silently pick one.

### 8.4 DERIVED_FROM

Preserve derivation provenance.

If B is a compressed or derived representation of A, traces must record the relation.

## Directionality

Hard relations must preserve direction.

Example:

```text
A REQUIRES B
```

is not equivalent to:

```text
B REQUIRES A
```

Do not reuse an undirected propagation graph for hard-constraint resolution.

## Required algorithms

Implement:

```text
dependency closure
supersession filtering
conflict detection
cycle detection
unsatisfiable-constraint reporting
```

## Cycles

If:

```text
A REQUIRES B
B REQUIRES A
```

treat the strongly connected requirement group as an atomic requirement unit.

Do not recurse infinitely.

## Required errors / result states

Add explicit states such as:

```text
ConstraintUnsatisfiable
RequiredContextOverflow
UnresolvedConflict
```

If required context exceeds the token budget, do not silently violate the constraint.

## Required tests

Test at least:

```text
single requires
multi-hop requires
requirement cycles
supersession
contradiction detection
unknown references
dependency closure over budget
validated compressed dependency satisfying closure
deterministic relation resolution
```

## Suggested commit

```text
feat: enforce relational context constraints
```

---

# 9. Phase 5E — Type-Aware Transformations

## Goal

Stop treating all compressible text as equivalent.

Different context types should use different transformation policies.

Extend the existing compression executor rather than replacing the runtime.

## Transformation families

### 9.1 Conversation / memory

Allowed:

```text
extractive reduction
optional abstractive summary
```

Must validate configured contracts.

### 9.2 Tool output

Prefer deterministic schema pruning.

Example input:

```json
{
  "status": "cancelled",
  "id": "A728",
  "debug_trace": "...large...",
  "request_metadata": {...}
}
```

Possible reduced form:

```json
{
  "status": "cancelled",
  "id": "A728"
}
```

Only if required fields remain.

### 9.3 Code

Do not default to generic prose summarization.

Initial safe transformations may include:

```text
remove comments
collapse irrelevant bodies
retain signatures
retain imports
retain referenced symbols
```

Avoid building a large AST platform unless experiments justify it.

### 9.4 System instructions

Default to:

```text
lossless / no compression
```

unless a specifically validated transformation exists.

### 9.5 Retrieved evidence

Prefer extractive compression.

Preserve:

```text
source ID
citation relation
required evidence spans
```

### 9.6 Task state / structured state

Prefer deterministic normalization or pruning.

Avoid lossy natural-language summarization where exact state is required.

---

# 10. Phase 5F — Contract Validators

## Goal

Every lossy transformation should be validated against the item's preservation contract.

## Proposed module

```text
src/contextos/validation/
```

## Initial validators

Create reusable validators such as:

```text
NumericPreservationValidator
DatePreservationValidator
IdentifierPreservationValidator
NegationPreservationValidator
CitationPreservationValidator
StructuredFieldValidator
DependencyReferenceValidator
```

## Validation result

Conceptually:

```python
ValidationResult(
    passed=True,
    violations=[],
)
```

or:

```python
ValidationResult(
    passed=False,
    violations=[
        "missing_identifier: usr_72B91"
    ],
)
```

## Transformation pipeline

```text
candidate item
    ↓
preferred transformation
    ↓
contract validation
   / \
PASS FAIL
 |     |
use   fallback
```

Fallback sequence may be:

```text
aggressive transform
        ↓ fail
safe extractive transform
        ↓ fail
original representation
```

If the original representation is required but cannot fit:

```text
optimization must fail explicitly
```

Do not silently drop the required item.

## Trace requirements

Record:

```text
transformation attempted
validator names
validator outcomes
failure reason
fallback path
final representation type
```

## Suggested commit

```text
feat: add contract-validated context transformations
```

---

# 11. Phase 5G — Omission Risk

## Goal

Extend soft optimization from utility-only ranking to utility plus omission/transformation risk.

Hard constraints must remain outside the score.

A hard requirement must never become removable because of a low numeric score.

## Initial formulation

Conceptually:

```text
selection_value =
    expected_utility
    - lambda * omission_risk
    - gamma * transformation_risk
```

Do not over-engineer the formula initially.

## Deterministic risk features

Possible omission-risk signals:

```text
required preservation contract
dependency fan-out
context type
current-state marker
contradiction involvement
exact-value requirement
tool state
decision state
error state
citation dependency
```

Possible transformation-risk signals:

```text
lossy transformation
high exact-value density
identifier density
negation present
structured schema
tool state
source code
```

## Important rule

Do not create dozens of new tunable weights.

Every parameter must have:

- a clear rationale;
- a documented default;
- an ablation plan.

## Required comparisons

Evaluate:

```text
v0.4 heuristic ContextOS
constraint-only ContextOS
constraint + validators
constraint + validators + risk
full Phase 5
```

## Suggested commit

```text
feat: add omission-risk-aware allocation
```

---

# 12. Phase 5H — Constraint-Aware Tracing

## Goal

Turn the optimization trace into a first-class research and debugging artifact.

Current traces already contain ranking and selection information.

Extend them with constraint reasoning.

## Required per-item fields

Record where applicable:

```text
preservation contracts
hard constraints triggered
required_by
dependency closure
superseded items
contradiction/conflict status
omission risk
transformation risk
transformation attempted
validators executed
validator results
fallback used
final representation
final position
```

## Example trace entry

```json
{
  "item_id": "tool_state_27",
  "decision": "retained",
  "decision_reason": "required_dependency",
  "required_by": ["decision_31"],
  "contracts": ["preserve_identifiers"],
  "transformation": "tool_schema_prune",
  "validation": {
    "identifier_preservation": "pass"
  }
}
```

## Counterfactual trace fields

Where computationally cheap, add:

```text
would_have_been_removed_without_constraints
would_have_been_compressed_without_contract
```

These are useful for both research and developer-facing explanations.

## Suggested commit

```text
feat: add constraint-aware optimization traces
```

---

# 13. Phase 5I — Controlled Ablation Matrix

## Goal

Determine which Phase 5 components actually contribute.

Do not compare only:

```text
old ContextOS vs new ContextOS
```

## Required variants

At minimum:

```text
v0.4 heuristic ContextOS
+ hard relations only
+ preservation contracts
+ validated transformations
+ omission risk
full Phase 5
```

Also retain existing baselines:

```text
Full Context
Last-N
Sliding Window
Relevance Only
Naive Extractive
```

External compression baselines may be added only when integration is technically fair and reproducible.

## Fair-comparison requirements

Every compared strategy must receive:

```text
same case
same token budget
same model
same decoding settings
same evaluator
same provider where applicable
```

## Primary metrics

Report:

```text
task score
Critical Information Recall
context reduction
Constraint Violation Rate
Dependency Closure Rate
State Consistency Rate
Contradiction Leakage Rate
Exact Value Preservation Rate
Identifier Preservation Rate
Citation Preservation Rate
p50 optimizer latency
p95 optimizer latency
```

## Budget sweep

Do not evaluate only one arbitrary context budget.

Evaluate a frontier such as:

```text
25%
35%
50%
65%
80%
100%
```

of original context length.

The research question is whether Phase 5 improves the reliability-efficiency Pareto frontier.

## Statistical reporting

Reuse the existing Phase 4 statistical system:

```text
per-case retained measurements
paired deltas
seeded bootstrap intervals
minimum sample-size rules
immutable bundles
```

Do not invent a separate statistics layer.

## Suggested commit

```text
research: add phase5 ablation and budget sweep
```

---

# 14. Phase 5J — External Model Validation

## Goal

Validate whether deterministic improvements translate to real model behavior.

Do this only after deterministic constraint behavior is stable.

## Benchmark selection

Keep LongBench for external long-context evaluation.

However, do not pretend LongBench measures all Phase 5 failure classes.

Add a model-backed version of the constraint benchmark where useful.

## Model setup

Prefer at least:

```text
one strong model
one smaller / cheaper model
```

if budget permits.

For critical comparisons:

```text
same provider
same model
same prompt
same temperature / decoding
same case set
same budget
```

## Required measurements

Capture:

```text
task quality
input tokens
output tokens
cached tokens
TTFT
provider latency
optimizer latency
embedding latency
compression / transformation latency
estimated cost where valid
```

Do not pool unrelated dataset metrics into one unexplained aggregate.

## Repetition policy

Use repeated runs only when provider nondeterminism can materially change the conclusion.

Preserve each repeated run as a separate immutable artifact.

## Suggested commit

```text
research: add phase5 model-backed evaluation
```

---

# 15. Phase 5K — Paper Readiness Gate

## Goal

Stop feature development and determine whether the research hypothesis is supported.

Do not automatically write a paper simply because Phase 5 is implemented.

A strong paper case would require evidence of:

```text
substantial context reduction
+
high critical-information retention
+
materially fewer context/constraint failures
+
reasonable optimizer overhead
+
consistent effects across more than one workload
```

Approximate interesting outcome territory may look like:

```text
40–50% context reduction
85–90% Critical Information Recall
88–93% Full Context quality retention
60–80% fewer structural/context failures
<10 ms p95 local optimizer latency
```

Again:

These are not guaranteed targets or publication claims.

If Phase 5 produces little reliability improvement for substantial added complexity, the research direction should be reconsidered rather than forcing a paper.

---

# 16. Artifact Requirements

Every meaningful Phase 5 benchmark must continue using the existing immutable seven-file bundle:

```text
config.json
environment.json
cases.jsonl
predictions.jsonl
metrics.json
metrics.csv
report.md
```

Add Phase 5 metadata rather than replacing the artifact format.

Record at least:

```text
contract schema version
constraint resolver version
validator versions
risk policy
baseline v0.4 SHA
current Phase 5 SHA
provider/model where applicable
tokenizer
embedding model
decoding configuration
benchmark version
```

---

# 17. Compatibility Requirements

Phase 5 should preserve backwards compatibility where practical.

Specifically:

- old `ContextItem` payloads without preservation contracts must continue to load;
- old benchmark fixtures must still execute;
- v0.4 baseline strategies must remain available;
- current CLI functionality must not silently change semantics;
- serialization schema changes must be versioned;
- new errors must be explicit;
- old benchmark artifacts must remain readable.

If backwards compatibility must be broken, document it explicitly and add migration tests.

---

# 18. Testing Requirements

Every Phase 5 sub-phase must include focused tests before moving to the next phase.

Minimum quality gates after every meaningful commit:

```bash
pytest
ruff check .
ruff format --check .
mypy
```

Do not lower coverage thresholds simply to get Phase 5 merged.

New logic should include:

- happy path;
- edge cases;
- invalid configurations;
- deterministic ordering;
- cycle behavior;
- overflow behavior;
- serialization;
- trace correctness;
- failure/fallback paths.

---

# 19. What the AI Coding Agent Must NOT Do

The AI agent must obey all of the following:

```text
DO NOT fabricate benchmark improvements.

DO NOT alter existing v0.4 benchmark artifacts.

DO NOT weaken or delete strong baselines.

DO NOT optimize directly against held-out test cases repeatedly.

DO NOT claim SOTA.

DO NOT introduce an LLM judge where deterministic scoring is possible.

DO NOT convert every relation into a hard relation automatically.

DO NOT silently violate a preservation contract because of token pressure.

DO NOT silently resolve contradictions without evidence.

DO NOT silently drop required context.

DO NOT mix old and new benchmark versions without explicit metadata.

DO NOT combine unlike metrics into one headline score.

DO NOT update README performance claims unless immutable artifacts support them.

DO NOT perform large unrelated refactors during a Phase 5 sub-phase.

DO NOT make one giant Phase 5 commit.

DO NOT rewrite the entire optimizer.

DO NOT add caching, model routing, semantic caching, orchestration, or unrelated features during Phase 5.

DO NOT rename the project during the research phase.

DO NOT publish projected metrics as measured results.
```

---

# 20. Recommended Commit Sequence

The Git history should tell the research story.

Suggested sequence:

```text
research: add v0.4 context failure taxonomy

research: add constraint-sensitive context benchmark

feat: add context preservation contracts

feat: enforce relational context constraints

feat: add contract-validated context transformations

feat: add omission-risk-aware allocation

feat: add constraint-aware optimization traces

research: add phase5 ablation and budget sweep

research: add phase5 model-backed evaluation

docs: document phase5 findings and limitations
```

Each commit should be independently testable.

Avoid a single `implement phase 5` commit.

---

# 21. Versioning Strategy

Do not automatically call the first Phase 5 change `v0.5.0`.

Use Phase 5 milestones internally during research.

Possible future public versioning:

```text
v0.4.0 — frozen heuristic optimization baseline
v0.5.0 — research preview, if useful
v1.0.0 — stable constraint-aware runtime
```

Only decide public version numbers after the research design stabilizes.

---

# 22. Expected End-State Architecture

```text
                     ContextItem
                         │
              ┌──────────┴──────────┐
              │                     │
     PreservationContract       Relation Graph
              │                     │
              └──────────┬──────────┘
                         ↓
                 Constraint Resolver
                         ↓
                  Legal Context State
                         ↓
             Utility + Omission Risk
                         ↓
                  Budget Allocation
                         ↓
             Type-Aware Transformation
                         ↓
                 Contract Validators
                    │             │
                  PASS          FAIL
                    │             │
                    │          fallback
                    └──────┬──────┘
                           ↓
                       Layout
                           ↓
                  Optimized Context
                           +
                  Constraint-Aware Trace
```

---

# 23. Expected Research Contributions

If Phase 5 succeeds, the project may support a paper around the following contributions:

1. **A formalization of heterogeneous context optimization under preservation constraints.**
2. **A constraint-aware optimizer with dependency closure, supersession, contradiction handling, and explicit failure semantics.**
3. **Type-aware context transformations validated against preservation contracts.**
4. **Risk-aware context allocation that separates hard safety/consistency constraints from soft utility.**
5. **An auditable optimization trace exposing why every context item was retained, transformed, or removed.**
6. **A constraint-sensitive benchmark measuring failure modes not exposed by ordinary task-score or semantic-similarity metrics.**
7. **A reproducible empirical comparison against full context, heuristic context selection, and simple baselines across multiple token budgets.**

Do not advertise these as contributions until experiments support them.

---

# 24. Likely Resume Evolution if Phase 5 Succeeds

Current v0.4 bullet:

> Reduced average input context by 54.3% with approximately 2.9 ms p95 optimizer latency; controlled ablations showed dependency-aware scoring improved task performance by 8 percentage points.

A future Phase 5 bullet should be written only after actual measurements.

The desired type of result would be:

> Reduced LLM input context by approximately X% while retaining Y% of task-critical information and cutting dependency/state-consistency failures by Z% versus relevance-only selection through constraint-aware context construction.

Where:

```text
X
Y
Z
```

must come from immutable benchmark artifacts.

Projected values must never be inserted into the final resume as measured results.

---

# 25. Execution Rule for the AI Agent

The AI coding agent should **not implement all of Phase 5 in one run**.

The required workflow is:

```text
Implement Phase 5A only
        ↓
produce failure taxonomy
        ↓
review evidence
        ↓
decide whether Phase 5B remains justified
        ↓
continue one sub-phase at a time
```

At the end of every sub-phase, the agent must provide:

1. files added;
2. files modified;
3. architectural decisions;
4. tests added;
5. commands executed;
6. results;
7. benchmark results if applicable;
8. limitations;
9. whether the next phase is still justified by evidence.

The agent must stop after each sub-phase unless explicitly instructed to continue.

---

# 26. Phase 5 Success Definition

Phase 5 is successful if it answers the research question honestly.

Success does not require ContextOS to win every benchmark.

Possible successful outcomes include:

- hard constraints materially reduce catastrophic context failures;
- type-aware validation reveals failure modes invisible to semantic metrics;
- some constraint classes are valuable while others add unnecessary complexity;
- relevance-only remains stronger for ordinary QA, while constraint-aware optimization is substantially safer for stateful/tool-using workflows;
- the new benchmark itself reveals a reproducible gap in current context-management strategies.

Any of these may support useful research if demonstrated rigorously.

The project must prioritize truthful measurement over predetermined conclusions.
