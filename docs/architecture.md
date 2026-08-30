# Architecture

ContextOS sits between application state and an LLM provider. Its staged optimizer will validate and tokenize candidate context, enforce mandatory retention, remove safe redundancy, score optional items, allocate budget, compress where allowed, arrange the final layout, and emit a complete decision trace.

Detailed component boundaries will be documented as each sequential milestone is implemented and verified.
