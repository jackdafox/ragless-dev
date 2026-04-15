"""TUI state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class Message:
    """A single message in the conversation history."""
    role: str  # "user" or "agent"
    content: str
    timestamp: Optional[str] = None


# Using plain dict/list to avoid coupling TUI to the dev module's internals
@dataclass
class TUIState:
    """Mutable state shared across the TUI."""
    query: str = ""
    discovered_files: list[str] = field(default_factory=list)
    extracted_signatures: list[dict] = field(default_factory=list)  # plain dicts, not FunctionSignature
    full_files: dict[str, str] = field(default_factory=dict)
    needs_more_files: bool = False
    replan_reason: str = ""
    step: int = 0
    streaming: bool = False
    messages: list[Message] = field(default_factory=list)
    error: str = ""
