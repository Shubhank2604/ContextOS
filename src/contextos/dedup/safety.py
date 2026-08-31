"""Conservative guards against destructive semantic deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextos.dedup.exact import normalize_content
from contextos.models import ContextType

_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?:%|[a-zA-Z]+)?")
_URL = re.compile(r"\b(?:https?|ftp)://[^\s<>'\"]+", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\[^\s<>'\"]+")
_UNIX_PATH = re.compile(r"(?<!\w)/(?:[^\s/<>'\"]+/)*[^\s/<>'\"]+")
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_STRUCTURED_IDENTIFIER = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]+\d+[A-Za-z0-9_-]*|"
    r"[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][A-Za-z0-9]*)+)\b"
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|none|without|cannot|can't|won't|isn't|aren't|"
    r"disabled|false|except|unless)\b",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
_CODE_SIGNAL = re.compile(
    r"(?:```|\b(?:def|class|return|import|SELECT|INSERT|UPDATE|DELETE)\b|"
    r"[{};]|(?:==|!=|<=|>=|=>|:=)|\w\s*=\s*[^=])"
)


@dataclass(frozen=True)
class SafetyDifference:
    """A protected feature category that differs between two items."""

    category: str
    left: tuple[str, ...]
    right: tuple[str, ...]


def _matches(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(sorted({value.casefold() for value in pattern.findall(text)}))


def semantic_safety_differences(
    left: str,
    right: str,
    *,
    context_type: ContextType,
) -> list[SafetyDifference]:
    """Return material protected-feature differences between two contents."""
    left_normalized = normalize_content(left)
    right_normalized = normalize_content(right)
    if left_normalized == right_normalized:
        return []
    if context_type is ContextType.CODE:
        return [SafetyDifference("code", (left_normalized,), (right_normalized,))]

    differences: list[SafetyDifference] = []
    patterns = {
        "number": _NUMBER,
        "date": _DATE,
        "url": _URL,
        "windows_path": _WINDOWS_PATH,
        "unix_path": _UNIX_PATH,
        "uuid": _UUID,
        "identifier": _STRUCTURED_IDENTIFIER,
        "negation": _NEGATION,
    }
    for category, pattern in patterns.items():
        left_values = _matches(pattern, left_normalized)
        right_values = _matches(pattern, right_normalized)
        if left_values != right_values and (left_values or right_values):
            differences.append(SafetyDifference(category, left_values, right_values))

    if _CODE_SIGNAL.search(left_normalized) or _CODE_SIGNAL.search(right_normalized):
        differences.append(SafetyDifference("code", (left_normalized,), (right_normalized,)))
    return differences


def is_safe_semantic_duplicate(
    left: str,
    right: str,
    *,
    context_type: ContextType,
) -> bool:
    """Return true only when no protected semantic difference is visible."""
    return not semantic_safety_differences(left, right, context_type=context_type)
