"""Coordinator — delegates to LangGraph compiled_graph."""

from __future__ import annotations

import hashlib
from .graph import compiled_graph
from .state import RagDevState


class DevCoordinator:
    """Runs rag dev queries through the LangGraph pipeline."""

    def __init__(self, root: str | None = None):
        self.root = root
        self._cache: dict[str, dict] = {}

    def _cache_key(self, query: str, explicit_files: list[str] | None = None, skip_final_response: bool = False) -> str:
        key_parts = [query, str(skip_final_response)]
        if explicit_files:
            key_parts.extend(explicit_files)
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def get_context(self, query: str, explicit_files: list[str] | None = None, skip_final_response: bool = False):
        """Run the graph and return final state. Caches result for repeated queries."""
        key = self._cache_key(query, explicit_files)
        if key in self._cache:
            return self._cache[key]
        initial_state: RagDevState = {
            "query": query,
            "discovered_files": explicit_files if explicit_files else [],
            "extracted_signatures": [],
            "full_files": {},
            "retrieval_context": "",
            "final_response": "",
            "skip_final_response": skip_final_response,
            "needs_more_files": False,
            "replan_reason": "",
            "step": 0,
            "messages": [],
        }
        result = compiled_graph.invoke(initial_state)
        self._cache[key] = result
        return result

    def handle_query(self, query: str, explicit_files: list[str] | None = None) -> str:
        """Handle a rag dev query, return the LLM prompt string."""
        result = self.get_context(query, explicit_files)
        return result.get("retrieval_context", "")
