"""Typed errors shared by ContextOS components."""


class ContextOSError(Exception):
    """Base exception for ContextOS failures."""


class TokenizerError(ContextOSError):
    """Raised when text cannot be tokenized safely."""


class DuplicateContextItemError(ContextOSError):
    """Raised when a collection contains duplicate context item IDs."""


class ContextItemNotFoundError(ContextOSError):
    """Raised when a requested context item is not present in a store."""
