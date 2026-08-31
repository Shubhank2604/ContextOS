"""Tests for semantic duplicate precedence, guards, and tie breaking."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from contextos.dedup.semantic import semantic_deduplicate
from contextos.embeddings.base import EmbeddingMatrix
from contextos.models import ContextItem, ContextType


class ConstantProvider:
    """Make every pair maximally similar so safety rules decide."""

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        return np.ones((len(texts), 2), dtype=np.float64)


def _item(
    item_id: str,
    content: str,
    *,
    context_type: ContextType = ContextType.DECISION,
    mandatory: bool = False,
    importance: float = 0.5,
    updated_offset: int = 0,
) -> ContextItem:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return ContextItem(
        id=item_id,
        content=content,
        type=context_type,
        created_at=start,
        updated_at=start + timedelta(seconds=updated_offset),
        importance=importance,
        mandatory=mandatory,
        evictable=not mandatory,
    )


def test_optional_semantic_duplicate_of_mandatory_is_removed() -> None:
    result = semantic_deduplicate(
        [
            _item("mandatory", "Deploy after tests pass", mandatory=True),
            _item("optional", "After tests pass deploy", importance=1.0),
        ],
        provider=ConstantProvider(),
    )
    assert [item.id for item in result.items] == ["mandatory"]
    match = result.matches_by_item_id["optional"]
    assert match.duplicate_of == "mandatory"
    assert match.reason == "semantic_duplicate_of_mandatory"
    assert match.similarity == pytest.approx(1.0)


def test_mandatory_semantic_duplicates_survive_with_warning() -> None:
    result = semantic_deduplicate(
        [
            _item("mandatory-a", "Deploy after tests pass", mandatory=True),
            _item("mandatory-b", "After tests pass deploy", mandatory=True),
        ],
        provider=ConstantProvider(),
    )
    assert len(result.items) == 2
    assert result.warnings == ["mandatory_semantic_duplicate:mandatory-a,mandatory-b"]


def test_semantic_dedup_is_type_scoped_and_uses_priority() -> None:
    result = semantic_deduplicate(
        [
            _item("lower", "Deploy after checks", importance=0.4),
            _item("higher", "After checks deploy", importance=0.9),
            _item(
                "other-type",
                "After checks deploy",
                context_type=ContextType.PLAN,
                importance=1.0,
            ),
        ],
        provider=ConstantProvider(),
    )
    assert [item.id for item in result.items] == ["higher", "other-type"]
    assert result.matches_by_item_id["lower"].duplicate_of == "higher"


@pytest.mark.parametrize(
    ("context_type", "left", "right"),
    [
        (ContextType.DECISION, "timeout is 10 seconds", "timeout is 100 seconds"),
        (ContextType.PLAN, "release on 2026-09-01", "release on 2026-09-10"),
        (ContextType.DECISION, "feature is enabled", "feature is not enabled"),
        (ContextType.TASK_STATE, "read /srv/dev/app.py", "read /srv/prod/app.py"),
        (ContextType.TASK_STATE, r"read C:\dev\app.py", r"read C:\prod\app.py"),
        (ContextType.TASK_STATE, "open BUG-1042", "open BUG-1402"),
        (ContextType.TASK_STATE, "see https://a.test/v1", "see https://a.test/v2"),
        (ContextType.CODE, "return enabled", "return not enabled"),
    ],
)
def test_material_differences_survive_semantic_similarity(
    context_type: ContextType,
    left: str,
    right: str,
) -> None:
    result = semantic_deduplicate(
        [
            _item("left", left, context_type=context_type),
            _item("right", right, context_type=context_type),
        ],
        provider=ConstantProvider(),
    )
    assert [item.id for item in result.items] == ["left", "right"]


def test_invalid_threshold_is_rejected_before_embedding() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        semantic_deduplicate([], provider=ConstantProvider(), threshold=1.1)
