"""Coordinator — delegates to LangGraph compiled_graph."""

from __future__ import annotations

from .graph import compiled_graph
from .state import RagDevState


class DevCoordinator:
    """Runs rag dev queries through the LangGraph pipeline."""

    def __init__(self, root: str | None = None):
        self.root = root

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

    def handle_query(self, query: str, explicit_files: list[str] | None = None) -> str:
        """Handle a rag dev query, return the LLM prompt string."""
        result = self.get_context(query, explicit_files)
        return result.get("retrieval_context", "")
