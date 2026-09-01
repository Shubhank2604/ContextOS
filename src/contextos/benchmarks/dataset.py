"""Deterministic construction of the project-owned ContextOS-Bench dataset."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contextos.benchmarks.models import (
    BenchmarkAnswerKey,
    BenchmarkFamily,
    CaseConstruction,
    CaseOrigin,
    ContextOSBenchCase,
    ContextOSBenchDataset,
    RequiredFact,
)
from contextos.config import OptimizationPolicy
from contextos.models import (
    ContextEdge,
    ContextItem,
    ContextType,
    DependencyRelation,
)

GENERATOR_VERSION = "1.0.0"
GENERATION_SEED = 404
_START = datetime(2025, 1, 1, tzinfo=UTC)
_REQUIRED_TAGS = [
    "old_critical_fact",
    "recent_irrelevant_fact",
    "duplicate_paraphrase",
    "changed_number_trap",
    "negation_trap",
    "superseded_decision",
    "contradiction",
    "one_hop_dependency",
    "two_hop_dependency",
    "exact_values",
]


def _item(
    case_id: str,
    suffix: str,
    content: str,
    context_type: ContextType,
    offset: int,
    *,
    importance: float | None = None,
    mandatory: bool = False,
) -> ContextItem:
    timestamp = _START + timedelta(days=offset)
    values: dict[str, object] = {
        "id": f"{case_id}-{suffix}",
        "content": content,
        "type": context_type,
        "source": f"contextos-bench:{case_id}",
        "created_at": timestamp,
        "updated_at": timestamp,
        "mandatory": mandatory,
        "evictable": not mandatory,
        "metadata": {"fixture_role": suffix},
    }
    if importance is not None:
        values["importance"] = importance
    return ContextItem.model_validate(values)


def _policy() -> OptimizationPolicy:
    return OptimizationPolicy.balanced(
        max_input_tokens=82,
        reserve_output_tokens=8,
        minimum_compressed_tokens=8,
        compression_target_ratio=0.5,
    )


def _edge(
    case_id: str,
    source: str,
    target: str,
    relation: DependencyRelation,
    weight: float,
) -> ContextEdge:
    return ContextEdge(
        source_id=f"{case_id}-{source}",
        target_id=f"{case_id}-{target}",
        relation=relation,
        weight=weight,
    )


def _coding_case(number: int) -> ContextOSBenchCase:
    case_id = f"coding-{number:03d}"
    service_id = f"svc-C{number:03d}"
    timeout = 30 + number
    old_timeout = timeout + 20
    error_code = f"E{4100 + number}"
    items = [
        _item(
            case_id,
            "system",
            "Preserve exact requirements, identifiers, code, and failure evidence.",
            ContextType.SYSTEM_INSTRUCTION,
            number,
            mandatory=True,
        ),
        _item(
            case_id,
            "old-decision",
            f"Original decision: {service_id} used a {old_timeout} second timeout.",
            ContextType.DECISION,
            number + 1,
            importance=0.4,
        ),
        _item(
            case_id,
            "critical-decision",
            f"Approved architecture: {service_id} must use exactly {timeout} seconds.",
            ContextType.DECISION,
            number + 2,
            importance=1.0,
        ),
        _item(
            case_id,
            "changed-number",
            f"Unapproved suggestion: {service_id} should use {timeout + 1} seconds.",
            ContextType.DECISION,
            number + 3,
            importance=0.2,
        ),
        _item(
            case_id,
            "negation",
            f"Constraint: do not disable timeout enforcement for {service_id}.",
            ContextType.MEMORY,
            number + 4,
            importance=0.8,
        ),
        _item(
            case_id,
            "code",
            f"def timeout_for_service():\n    return {timeout}  # {service_id}",
            ContextType.CODE,
            number + 5,
            importance=0.8,
        ),
        _item(
            case_id,
            "error",
            f"pytest failed with ERROR {error_code}: {service_id} timed out after "
            f"{timeout} seconds.",
            ContextType.ERROR,
            number + 6,
            importance=1.0,
        ),
        _item(
            case_id,
            "duplicate",
            f"The approved timeout for {service_id} is exactly {timeout} seconds.",
            ContextType.DECISION,
            number + 7,
            importance=0.5,
        ),
        _item(
            case_id,
            "repeated-output",
            f"Repeated test output: ERROR {error_code} occurred for {service_id}.",
            ContextType.TOOL_OUTPUT,
            number + 8,
            importance=0.3,
        ),
        _item(
            case_id,
            "recent-noise",
            "The latest formatting and spelling checks completed successfully.",
            ContextType.TOOL_OUTPUT,
            300 + number,
            importance=0.1,
        ),
        _item(
            case_id,
            "task",
            f"Fix {error_code} for {service_id} without violating the approved timeout.",
            ContextType.USER_MESSAGE,
            400 + number,
            mandatory=True,
        ),
    ]
    return ContextOSBenchCase(
        id=case_id,
        family=BenchmarkFamily.CODING_AGENT,
        origin=CaseOrigin.BASE,
        construction=CaseConstruction.TEMPLATED,
        task=items[-1].content,
        context_items=items,
        edges=[
            _edge(case_id, "task", "error", DependencyRelation.REQUIRES, 1.0),
            _edge(case_id, "error", "critical-decision", DependencyRelation.REQUIRES, 1.0),
            _edge(
                case_id,
                "critical-decision",
                "old-decision",
                DependencyRelation.SUPERSEDES,
                0.9,
            ),
            _edge(
                case_id,
                "changed-number",
                "critical-decision",
                DependencyRelation.CONTRADICTS,
                0.9,
            ),
            _edge(
                case_id,
                "duplicate",
                "critical-decision",
                DependencyRelation.RELATED_TO,
                0.8,
            ),
        ],
        required_item_ids=[f"{case_id}-critical-decision", f"{case_id}-error"],
        answer_key=BenchmarkAnswerKey(
            required_facts=[
                RequiredFact(
                    item_id=f"{case_id}-critical-decision",
                    value=service_id,
                    label="service_identifier",
                ),
                RequiredFact(
                    item_id=f"{case_id}-critical-decision",
                    value=f"{timeout} seconds",
                    label="approved_timeout",
                ),
                RequiredFact(
                    item_id=f"{case_id}-error",
                    value=error_code,
                    label="failure_code",
                ),
            ],
            expected_answer=f"Set {service_id} to {timeout} seconds and resolve {error_code}.",
        ),
        tags=["coding", "compile_test_failure", *_REQUIRED_TAGS],
        policy=_policy(),
    )


def _research_case(number: int) -> ContextOSBenchCase:
    case_id = f"research-{number:03d}"
    study_id = f"DOI:10.7000/R{number:03d}"
    sample_size = 700 + number * 11
    effect = 12 + number
    publication_date = f"2024-{(number % 12) + 1:02d}-{(number % 27) + 1:02d}"
    items = [
        _item(
            case_id,
            "system",
            "Cite exact source identifiers, dates, values, and conflicting evidence.",
            ContextType.SYSTEM_INSTRUCTION,
            number,
            mandatory=True,
        ),
        _item(
            case_id,
            "old-source",
            f"Study {study_id}, published {publication_date}, reports n={sample_size} and an "
            f"effect of {effect} percent.",
            ContextType.RETRIEVED_DOCUMENT,
            number + 1,
            importance=1.0,
        ),
        _item(
            case_id,
            "changed-number",
            f"A secondary note incorrectly reports n={sample_size + 1} and {effect + 1} percent.",
            ContextType.RETRIEVED_DOCUMENT,
            number + 2,
            importance=0.2,
        ),
        _item(
            case_id,
            "contradiction",
            f"A later commentary claims {study_id} found no measurable effect.",
            ContextType.RETRIEVED_DOCUMENT,
            number + 3,
            importance=0.7,
        ),
        _item(
            case_id,
            "negation",
            f"The protocol states that participants were not unblinded in {study_id}.",
            ContextType.MEMORY,
            number + 4,
            importance=0.7,
        ),
        _item(
            case_id,
            "duplicate",
            f"The primary paper enrolled {sample_size} participants and measured {effect} percent.",
            ContextType.RETRIEVED_DOCUMENT,
            number + 5,
            importance=0.4,
        ),
        _item(
            case_id,
            "synthesis",
            f"Synthesis record links the current question to primary study {study_id}.",
            ContextType.PLAN,
            number + 6,
            importance=0.9,
        ),
        _item(
            case_id,
            "superseded-summary",
            "An early draft summary was superseded after source verification.",
            ContextType.DECISION,
            number + 7,
            importance=0.2,
        ),
        _item(
            case_id,
            "recent-noise",
            "The bibliography export completed and the citation style was updated.",
            ContextType.TOOL_OUTPUT,
            300 + number,
            importance=0.1,
        ),
        _item(
            case_id,
            "task",
            f"Report the exact sample, effect, date, and identifier for study R{number:03d}.",
            ContextType.USER_MESSAGE,
            400 + number,
            mandatory=True,
        ),
    ]
    return ContextOSBenchCase(
        id=case_id,
        family=BenchmarkFamily.RESEARCH_AGENT,
        origin=CaseOrigin.BASE,
        construction=CaseConstruction.TEMPLATED,
        task=items[-1].content,
        context_items=items,
        edges=[
            _edge(case_id, "task", "synthesis", DependencyRelation.REQUIRES, 1.0),
            _edge(
                case_id,
                "synthesis",
                "old-source",
                DependencyRelation.DERIVED_FROM,
                1.0,
            ),
            _edge(
                case_id,
                "contradiction",
                "old-source",
                DependencyRelation.CONTRADICTS,
                0.9,
            ),
            _edge(
                case_id,
                "duplicate",
                "old-source",
                DependencyRelation.RELATED_TO,
                0.8,
            ),
            _edge(
                case_id,
                "synthesis",
                "superseded-summary",
                DependencyRelation.SUPERSEDES,
                0.7,
            ),
        ],
        required_item_ids=[f"{case_id}-old-source"],
        answer_key=BenchmarkAnswerKey(
            required_facts=[
                RequiredFact(
                    item_id=f"{case_id}-old-source",
                    value=study_id,
                    label="source_identifier",
                ),
                RequiredFact(
                    item_id=f"{case_id}-old-source",
                    value=publication_date,
                    label="publication_date",
                ),
                RequiredFact(
                    item_id=f"{case_id}-old-source",
                    value=f"n={sample_size}",
                    label="sample_size",
                ),
                RequiredFact(
                    item_id=f"{case_id}-old-source",
                    value=f"{effect} percent",
                    label="reported_effect",
                ),
            ],
            expected_answer=(f"{study_id}; {publication_date}; n={sample_size}; {effect} percent."),
        ),
        tags=["research", "source_metadata", *_REQUIRED_TAGS],
        policy=_policy(),
    )


def _support_case(number: int) -> ContextOSBenchCase:
    case_id = f"support-{number:03d}"
    ticket_id = f"CASE-S{number:04d}"
    commitment_date = f"2025-{(number % 12) + 1:02d}-{(number % 27) + 1:02d}"
    credit = 40 + number
    policy_days = 14 + (number % 5)
    items = [
        _item(
            case_id,
            "system",
            "Honor exact customer commitments and the latest approved policy.",
            ContextType.SYSTEM_INSTRUCTION,
            number,
            mandatory=True,
        ),
        _item(
            case_id,
            "old-policy",
            f"Superseded policy allowed resolution within {policy_days + 7} days.",
            ContextType.DECISION,
            number + 1,
            importance=0.3,
        ),
        _item(
            case_id,
            "current-policy",
            f"Approved policy for {ticket_id}: resolve within exactly {policy_days} days.",
            ContextType.DECISION,
            number + 2,
            importance=1.0,
        ),
        _item(
            case_id,
            "commitment",
            f"On {commitment_date}, support promised {ticket_id} a ${credit} account credit.",
            ContextType.MEMORY,
            number + 3,
            importance=1.0,
        ),
        _item(
            case_id,
            "changed-number",
            f"An unapproved draft lists a ${credit + 5} credit and {policy_days + 1} days.",
            ContextType.MEMORY,
            number + 4,
            importance=0.2,
        ),
        _item(
            case_id,
            "negation",
            f"Do not close {ticket_id} before the promised credit is applied.",
            ContextType.DECISION,
            number + 5,
            importance=0.8,
        ),
        _item(
            case_id,
            "duplicate",
            f"The customer commitment for {ticket_id} was a credit of ${credit}.",
            ContextType.MEMORY,
            number + 6,
            importance=0.5,
        ),
        _item(
            case_id,
            "history",
            f"Customer-history index links {ticket_id} to the original commitment.",
            ContextType.TASK_STATE,
            number + 7,
            importance=0.9,
        ),
        _item(
            case_id,
            "old-status",
            f"Earlier automated status update for {ticket_id} was delivered.",
            ContextType.TOOL_OUTPUT,
            number + 8,
            importance=0.2,
        ),
        _item(
            case_id,
            "recent-noise",
            "The most recent automated status email was delivered successfully.",
            ContextType.TOOL_OUTPUT,
            300 + number,
            importance=0.1,
        ),
        _item(
            case_id,
            "task",
            f"Resolve {ticket_id} using the approved policy and original commitment.",
            ContextType.USER_MESSAGE,
            400 + number,
            mandatory=True,
        ),
    ]
    return ContextOSBenchCase(
        id=case_id,
        family=BenchmarkFamily.SUPPORT_OPERATIONS,
        origin=CaseOrigin.BASE,
        construction=CaseConstruction.TEMPLATED,
        task=items[-1].content,
        context_items=items,
        edges=[
            _edge(case_id, "task", "history", DependencyRelation.REQUIRES, 1.0),
            _edge(case_id, "history", "commitment", DependencyRelation.REQUIRES, 1.0),
            _edge(
                case_id,
                "current-policy",
                "old-policy",
                DependencyRelation.SUPERSEDES,
                0.9,
            ),
            _edge(
                case_id,
                "changed-number",
                "commitment",
                DependencyRelation.CONTRADICTS,
                0.9,
            ),
            _edge(
                case_id,
                "duplicate",
                "commitment",
                DependencyRelation.RELATED_TO,
                0.8,
            ),
        ],
        required_item_ids=[f"{case_id}-current-policy", f"{case_id}-commitment"],
        answer_key=BenchmarkAnswerKey(
            required_facts=[
                RequiredFact(
                    item_id=f"{case_id}-current-policy",
                    value=f"{policy_days} days",
                    label="resolution_window",
                ),
                RequiredFact(
                    item_id=f"{case_id}-commitment",
                    value=commitment_date,
                    label="commitment_date",
                ),
                RequiredFact(
                    item_id=f"{case_id}-commitment",
                    value=ticket_id,
                    label="ticket_identifier",
                ),
                RequiredFact(
                    item_id=f"{case_id}-commitment",
                    value=f"${credit}",
                    label="credit_amount",
                ),
            ],
            expected_answer=(
                f"Apply ${credit} for {ticket_id} from {commitment_date} and resolve within "
                f"{policy_days} days."
            ),
        ),
        tags=["support_operations", "customer_commitment", *_REQUIRED_TAGS],
        policy=_policy(),
    )


def build_contextos_bench_dataset() -> ContextOSBenchDataset:
    """Return the deterministic Phase 4A dataset with exactly 50 base cases."""
    cases = [
        *(_coding_case(number) for number in range(1, 19)),
        *(_research_case(number) for number in range(1, 17)),
        *(_support_case(number) for number in range(1, 17)),
    ]
    return ContextOSBenchDataset(
        name="contextos-bench-phase4a",
        generator_version=GENERATOR_VERSION,
        generation_seed=GENERATION_SEED,
        cases=cases,
    )


def write_contextos_bench_dataset(output_path: Path) -> None:
    """Write the canonical deterministic JSON fixture."""
    dataset = build_contextos_bench_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def main() -> None:
    """Regenerate the canonical dataset from versioned templates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/datasets/contextos_bench.json"),
    )
    arguments = parser.parse_args()
    write_contextos_bench_dataset(arguments.output)


if __name__ == "__main__":
    main()
