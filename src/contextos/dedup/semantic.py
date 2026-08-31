"""Provider-driven semantic deduplication with conservative safety rules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contextos.dedup.base import DeduplicationResult, DuplicateMatch
from contextos.dedup.safety import is_safe_semantic_duplicate
from contextos.embeddings.base import EmbeddingProvider, validate_embedding_matrix
from contextos.models import ContextItem, validate_unique_item_ids


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return a clipped cosine similarity, rejecting zero vectors upstream."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _optional_priority(item: ContextItem) -> tuple[float, float, float, str]:
    return (
        -item.importance,
        -item.updated_at.timestamp(),
        -item.created_at.timestamp(),
        item.id,
    )


def _best_match(
    candidate_index: int,
    comparison_indices: Sequence[int],
    *,
    items: Sequence[ContextItem],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[int, float] | None:
    eligible: list[tuple[float, str, int]] = []
    candidate = items[candidate_index]
    for comparison_index in comparison_indices:
        comparison = items[comparison_index]
        similarity = cosine_similarity(embeddings[candidate_index], embeddings[comparison_index])
        if similarity < threshold:
            continue
        if not is_safe_semantic_duplicate(
            candidate.content,
            comparison.content,
            context_type=candidate.type,
        ):
            continue
        eligible.append((-similarity, comparison.id, comparison_index))
    if not eligible:
        return None
    negative_similarity, _, comparison_index = sorted(eligible)[0]
    return comparison_index, -negative_similarity


def semantic_deduplicate(
    items: Sequence[ContextItem],
    *,
    provider: EmbeddingProvider,
    threshold: float = 0.92,
) -> DeduplicationResult:
    """Remove safe optional semantic duplicates within the same context type."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    validate_unique_item_ids(items)
    copied = [item.model_copy(deep=True) for item in items]
    if not copied:
        return DeduplicationResult(items=[])
    embeddings = validate_embedding_matrix(
        provider.embed([item.content for item in copied]),
        expected_rows=len(copied),
    )

    by_type: dict[object, list[int]] = {}
    for index, item in enumerate(copied):
        by_type.setdefault(item.type, []).append(index)

    matches: list[DuplicateMatch] = []
    warnings: list[str] = []
    removed_indices: set[int] = set()
    for indices in by_type.values():
        mandatory_indices = sorted(
            (index for index in indices if copied[index].mandatory),
            key=lambda index: copied[index].id,
        )
        for offset, left_index in enumerate(mandatory_indices):
            for right_index in mandatory_indices[offset + 1 :]:
                similarity = cosine_similarity(embeddings[left_index], embeddings[right_index])
                if similarity >= threshold and is_safe_semantic_duplicate(
                    copied[left_index].content,
                    copied[right_index].content,
                    context_type=copied[left_index].type,
                ):
                    warnings.append(
                        "mandatory_semantic_duplicate:"
                        f"{copied[left_index].id},{copied[right_index].id}"
                    )

        optional_indices = sorted(
            (index for index in indices if not copied[index].mandatory),
            key=lambda index: _optional_priority(copied[index]),
        )
        surviving_optional: list[int] = []
        for candidate_index in optional_indices:
            mandatory_match = _best_match(
                candidate_index,
                mandatory_indices,
                items=copied,
                embeddings=embeddings,
                threshold=threshold,
            )
            if mandatory_match is not None:
                target_index, similarity = mandatory_match
                removed_indices.add(candidate_index)
                matches.append(
                    DuplicateMatch(
                        item_id=copied[candidate_index].id,
                        duplicate_of=copied[target_index].id,
                        reason="semantic_duplicate_of_mandatory",
                        similarity=similarity,
                    )
                )
                continue
            optional_match = _best_match(
                candidate_index,
                surviving_optional,
                items=copied,
                embeddings=embeddings,
                threshold=threshold,
            )
            if optional_match is not None:
                target_index, similarity = optional_match
                removed_indices.add(candidate_index)
                matches.append(
                    DuplicateMatch(
                        item_id=copied[candidate_index].id,
                        duplicate_of=copied[target_index].id,
                        reason="semantic_duplicate",
                        similarity=similarity,
                    )
                )
                continue
            surviving_optional.append(candidate_index)

    return DeduplicationResult(
        items=[item for index, item in enumerate(copied) if index not in removed_indices],
        matches=sorted(matches, key=lambda match: match.item_id),
        warnings=sorted(warnings),
    )
