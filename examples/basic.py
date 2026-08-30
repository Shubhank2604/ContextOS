"""Construct and tokenize a minimal ContextOS item."""

from datetime import UTC, datetime

from contextos import ContextItem, ContextType
from contextos.tokenization import TiktokenTokenizer

item = ContextItem(
    id="task-1",
    content="Fix the authentication timeout without changing the stateless design.",
    type=ContextType.USER_MESSAGE,
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC),
    importance=1.0,
    mandatory=True,
    evictable=False,
)

tokenizer = TiktokenTokenizer()
print(f"{item.id}: {tokenizer.count_tokens(item.content)} tokens")
