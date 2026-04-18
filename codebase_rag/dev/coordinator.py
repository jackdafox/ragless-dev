"""Coordinator — delegates to LangGraph compiled_graph."""

from __future__ import annotations

import hashlib
from .graph import compiled_graph
from .state import RagDevState


class DevCoordinator:
    """Runs rag dev queries through the LangGraph pipeline."""

    def __init__(self, root: str | None = None):
        self.root = root
        self._cache: dict[str, str] = {}

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def handle_query(self, query: str, explicit_files: list[str] | None = None) -> str:
        """Handle a rag dev query, return the LLM prompt string."""
        key = self._cache_key(query)
        if key in self._cache:
            return self._cache[key]
        result = self.get_context(query, explicit_files)
        output = result.get("retrieval_context", "")
        self._cache[key] = output
        return output

    def get_context(self, query: str, explicit_files: list[str] | None = None):
        """Run the graph and return final state."""
        initial_state: RagDevState = {
            "query": query,
            "discovered_files": explicit_files if explicit_files else [],
            "extracted_signatures": [],
            "full_files": {},
            "retrieval_context": "",
            "needs_more_files": False,
            "replan_reason": "",
            "step": 0,
            "messages": [],
        }
        result = compiled_graph.invoke(initial_state)
        return result
