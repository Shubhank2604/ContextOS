"""Deterministic generator for ContextOS-Bench Constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from contextos.benchmarks.constraint_models import (
    ConstraintBenchmarkCase,
    ConstraintBenchmarkDataset,
    ConstraintCategory,
    ConstraintGroundTruth,
)
from contextos.benchmarks.models import (
    BenchmarkAnswerKey,
    BenchmarkFamily,
    CaseConstruction,
    CaseOrigin,
    RequiredFact,
)
from contextos.config import OptimizationPolicy
from contextos.models import ContextEdge, ContextItem, ContextType, DependencyRelation

CONSTRAINT_GENERATOR_VERSION = "1.0.0"
CONSTRAINT_GENERATION_SEED = 505
PHASE5_BASELINE_SHA = "4fdd88391300c56ad17af5458897ccdd08d6f7bf"
_START = datetime(2025, 1, 1, tzinfo=UTC)


def _item(
    case_id: str,
    suffix: str,
    content: str,
    context_type: ContextType,
    offset: int,
    *,
    importance: float = 0.5,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = _START + timedelta(days=offset)
    return ContextItem(
        id=f"{case_id}-{suffix}",
        content=content,
        type=context_type,
        source=f"contextos-bench-constraints:{case_id}",
        created_at=timestamp,
        updated_at=timestamp,
        importance=importance,
        mandatory=mandatory,
        evictable=not mandatory,
        metadata={"fixture_role": suffix},
    )


def _edge(
    case_id: str,
    source: str,
    target: str,
    relation: DependencyRelation,
) -> ContextEdge:
    return ContextEdge(
        source_id=f"{case_id}-{source}",
        target_id=f"{case_id}-{target}",
        relation=relation,
        weight=1.0,
    )


def _policy() -> OptimizationPolicy:
    return OptimizationPolicy.balanced(
        max_input_tokens=76,
        reserve_output_tokens=8,
        minimum_compressed_tokens=8,
        compression_target_ratio=0.5,
    )


def _case(
    category: ConstraintCategory,
    number: int,
) -> ConstraintBenchmarkCase:
    case_id = f"constraint-{category.value}-{number:03d}"
    key = f"{number:03d}"
    system = _item(
        case_id,
        "system",
        "Preserve authoritative state, exact values, identifiers, relations, and provenance.",
        ContextType.SYSTEM_INSTRUCTION,
        number,
        importance=1.0,
        mandatory=True,
    )
    noise = _item(
        case_id,
        "recent-noise",
        f"Formatting check {key} completed successfully and no action is required.",
        ContextType.TOOL_OUTPUT,
        500 + number,
        importance=0.2,
    )

    if category is ConstraintCategory.DEPENDENCY:
        token = f"auth_{key}X"
        items = [
            system,
            _item(
                case_id,
                "auth",
                f"Authentication token is {token}.",
                ContextType.TASK_STATE,
                2,
                importance=0.9,
            ),
            _item(
                case_id,
                "endpoint",
                f"Endpoint /payments/{key} uses the authenticated API.",
                ContextType.TOOL_DEFINITION,
                3,
                importance=0.9,
            ),
            _item(
                case_id,
                "old-auth",
                f"Obsolete token auth_{key}OLD was revoked.",
                ContextType.MEMORY,
                4,
                importance=0.5,
            ),
            noise,
            _item(
                case_id,
                "task",
                f"Call /payments/{key} with its valid authentication.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        relations = [
            _edge(case_id, "task", "endpoint", DependencyRelation.REQUIRES),
            _edge(case_id, "endpoint", "auth", DependencyRelation.REQUIRES),
        ]
        critical = [f"{case_id}-endpoint", f"{case_id}-auth"]
        facts = [(critical[0], f"/payments/{key}", "endpoint"), (critical[1], token, "auth_token")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[f"/payments/{key}", token],
            stale_item_ids=[f"{case_id}-old-auth"],
        )
    elif category is ConstraintCategory.SUPERSESSION:
        current = f"region-west-{key}"
        old = f"region-east-{key}"
        items = [
            system,
            _item(
                case_id,
                "old-state",
                f"deployment_region={old}",
                ContextType.TASK_STATE,
                2,
                importance=0.7,
            ),
            _item(
                case_id,
                "current-state",
                f"deployment_region={current}",
                ContextType.TASK_STATE,
                3,
                importance=1.0,
            ),
            noise,
            _item(
                case_id,
                "task",
                "Report the current deployment region.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        relations = [_edge(case_id, "current-state", "old-state", DependencyRelation.SUPERSEDES)]
        critical = [f"{case_id}-current-state"]
        facts = [(critical[0], current, "current_region")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[current],
            forbidden_item_combinations=[(f"{case_id}-old-state", f"{case_id}-current-state")],
            expected_current_state={"deployment_region": current},
            stale_item_ids=[f"{case_id}-old-state"],
        )
    elif category is ConstraintCategory.CONTRADICTION:
        amount = f"${700 + number}"
        old_amount = f"${500 + number}"
        items = [
            system,
            _item(
                case_id,
                "noncanonical",
                f"Draft policy says maximum refund={old_amount}.",
                ContextType.RETRIEVED_DOCUMENT,
                2,
                importance=0.8,
            ),
            _item(
                case_id,
                "canonical",
                f"Canonical policy v2 says maximum refund={amount}.",
                ContextType.DECISION,
                3,
                importance=1.0,
            ),
            noise,
            _item(
                case_id,
                "task",
                "Apply the canonical maximum refund.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        relations = [_edge(case_id, "noncanonical", "canonical", DependencyRelation.CONTRADICTS)]
        critical = [f"{case_id}-canonical"]
        facts = [(critical[0], amount, "maximum_refund")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_exact_values=[amount],
            forbidden_item_combinations=[(f"{case_id}-noncanonical", f"{case_id}-canonical")],
            expected_current_state={"maximum_refund": amount},
            stale_item_ids=[f"{case_id}-noncanonical"],
        )
    elif category is ConstraintCategory.EXACT_NUMERIC:
        amount, rate, date = (
            f"${1900 + number}.36",
            f"{4 + number / 100:.2f}%",
            f"2026-09-{number:02d}",
        )
        items = [
            system,
            _item(
                case_id,
                "record",
                f"Settlement={amount}; rate={rate}; effective_date={date}.",
                ContextType.TASK_STATE,
                2,
                importance=1.0,
            ),
            _item(
                case_id,
                "rounded",
                f"Approximate settlement is ${1900 + number} at about 4 percent.",
                ContextType.MEMORY,
                3,
                importance=0.8,
            ),
            noise,
            _item(
                case_id,
                "task",
                "Return the exact settlement, rate, and effective date.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-record"]
        relations = [_edge(case_id, "task", "record", DependencyRelation.REQUIRES)]
        facts = [
            (critical[0], value, label)
            for value, label in ((amount, "amount"), (rate, "rate"), (date, "date"))
        ]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_exact_values=[amount, rate, date],
        )
    elif category is ConstraintCategory.IDENTIFIER:
        user, instruction, symbol = (
            f"usr_{key}B91",
            f"payment_instruction_v{number}",
            f"validateAccountRisk{number}()",
        )
        items = [
            system,
            _item(
                case_id,
                "identifiers",
                f"User {user} must execute {instruction} through {symbol}.",
                ContextType.CODE,
                2,
                importance=1.0,
            ),
            _item(
                case_id,
                "similar",
                "A similar user uses payment_instruction_v1.",
                ContextType.MEMORY,
                3,
                importance=0.8,
            ),
            noise,
            _item(
                case_id,
                "task",
                "Return the exact user, instruction, and function identifiers.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-identifiers"]
        relations = [_edge(case_id, "task", "identifiers", DependencyRelation.REQUIRES)]
        facts = [
            (critical[0], value, label)
            for value, label in (
                (user, "user_id"),
                (instruction, "instruction_id"),
                (symbol, "function"),
            )
        ]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[user, instruction, symbol],
        )
    elif category is ConstraintCategory.NEGATION_POLICY:
        policy = f"Do NOT execute transfer TX-{key} without approval APR-{key}."
        items = [
            system,
            _item(case_id, "policy", policy, ContextType.DECISION, 2, importance=1.0),
            _item(
                case_id,
                "unsafe",
                f"Execute transfer TX-{key} immediately.",
                ContextType.MEMORY,
                3,
                importance=0.9,
            ),
            noise,
            _item(
                case_id,
                "task",
                f"State the approval rule for TX-{key}.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-policy"]
        relations = [
            _edge(case_id, "task", "policy", DependencyRelation.REQUIRES),
            _edge(case_id, "unsafe", "policy", DependencyRelation.CONTRADICTS),
        ]
        facts = [(critical[0], "Do NOT", "negation"), (critical[0], f"APR-{key}", "approval_id")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_exact_values=["Do NOT"],
            required_identifiers=[f"TX-{key}", f"APR-{key}"],
            forbidden_item_combinations=[(f"{case_id}-unsafe", f"{case_id}-policy")],
        )
    elif category is ConstraintCategory.TOOL_STATE:
        reservation = f"R{9200 + number}"
        items = [
            system,
            _item(
                case_id,
                "old-tool-state",
                f'{{"reservation_id":"{reservation}","status":"confirmed"}}',
                ContextType.TOOL_OUTPUT,
                2,
                importance=0.8,
            ),
            _item(
                case_id,
                "current-tool-state",
                f'{{"reservation_id":"{reservation}","status":"cancelled"}}',
                ContextType.TOOL_OUTPUT,
                3,
                importance=1.0,
            ),
            noise,
            _item(
                case_id,
                "task",
                f"Report the current state of {reservation}.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-current-tool-state"]
        relations = [
            _edge(case_id, "current-tool-state", "old-tool-state", DependencyRelation.SUPERSEDES)
        ]
        facts = [(critical[0], reservation, "reservation_id"), (critical[0], "cancelled", "status")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[reservation],
            forbidden_item_combinations=[
                (f"{case_id}-old-tool-state", f"{case_id}-current-tool-state")
            ],
            expected_current_state={"reservation_status": "cancelled"},
            stale_item_ids=[f"{case_id}-old-tool-state"],
        )
    elif category is ConstraintCategory.CITATION_EVIDENCE:
        source = f"SRC-{key}"
        evidence = f"EV-{key}"
        items = [
            system,
            _item(
                case_id,
                "evidence",
                f"Source {source} contains evidence span {evidence}: treatment reduced risk.",
                ContextType.RETRIEVED_DOCUMENT,
                2,
                importance=1.0,
            ),
            _item(
                case_id,
                "uncited",
                "A summary claims treatment reduced risk but provides no provenance.",
                ContextType.MEMORY,
                3,
                importance=0.9,
            ),
            noise,
            _item(
                case_id,
                "task",
                "Report the finding with its source and evidence identifier.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-evidence"]
        relations = [_edge(case_id, "task", "evidence", DependencyRelation.REQUIRES)]
        facts = [(critical[0], source, "source_id"), (critical[0], evidence, "evidence_id")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[evidence],
            required_citations=[source],
        )
    else:
        a, b, c = f"root-{key}", f"middle-{key}", f"leaf-{key}"
        items = [
            system,
            _item(
                case_id,
                "root",
                f"Root credential is {a}.",
                ContextType.TASK_STATE,
                2,
                importance=0.8,
            ),
            _item(
                case_id,
                "middle",
                f"Intermediate service {b} uses the root credential.",
                ContextType.TOOL_DEFINITION,
                3,
                importance=0.9,
            ),
            _item(
                case_id,
                "leaf",
                f"Operation {c} uses intermediate service {b}.",
                ContextType.PLAN,
                4,
                importance=1.0,
            ),
            noise,
            _item(
                case_id,
                "task",
                f"Execute operation {c} with its complete dependency chain.",
                ContextType.USER_MESSAGE,
                600,
                importance=0.8,
                mandatory=True,
            ),
        ]
        critical = [f"{case_id}-root", f"{case_id}-middle", f"{case_id}-leaf"]
        relations = [
            _edge(case_id, "task", "leaf", DependencyRelation.REQUIRES),
            _edge(case_id, "leaf", "middle", DependencyRelation.REQUIRES),
            _edge(case_id, "middle", "root", DependencyRelation.REQUIRES),
        ]
        facts = [(critical[0], a, "root"), (critical[1], b, "middle"), (critical[2], c, "leaf")]
        constraints = ConstraintGroundTruth(
            category=category,
            critical_item_ids=critical,
            relations=relations,
            required_identifiers=[a, b, c],
        )

    return ConstraintBenchmarkCase(
        id=case_id,
        family=BenchmarkFamily.SUPPORT_OPERATIONS,
        origin=CaseOrigin.BASE,
        construction=CaseConstruction.TEMPLATED,
        task=items[-1].content,
        context_items=items,
        edges=relations,
        required_item_ids=critical,
        answer_key=BenchmarkAnswerKey(
            required_facts=[
                RequiredFact(item_id=item_id, value=value, label=label)
                for item_id, value, label in facts
            ],
            expected_answer="Preserve every annotated requirement.",
        ),
        tags=["phase5", "constraint-sensitive", category.value],
        policy=_policy(),
        constraints=constraints,
    )


def generate_constraint_dataset() -> ConstraintBenchmarkDataset:
    """Generate 90 stable cases: ten for each required constraint category."""
    return ConstraintBenchmarkDataset(
        name="ContextOS-Bench Constraints",
        generator_version=CONSTRAINT_GENERATOR_VERSION,
        generation_seed=CONSTRAINT_GENERATION_SEED,
        cases=[
            _case(category, number) for category in ConstraintCategory for number in range(1, 11)
        ],
    )
