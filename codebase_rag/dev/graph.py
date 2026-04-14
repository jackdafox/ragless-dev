"""LangGraph StateGraph for rag dev pipeline."""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import RagDevState
from .nodes import (
    file_discover_node,
    extract_signatures_node,
    build_retrieval_context_node,
    agent_node,
    replan_node,
    final_response_node,
)


def _should_replan(state: RagDevState) -> str:
    """Routing: loop back to file_discover_node or proceed to final_response."""
    if state.get("needs_more_files") and state.get("step", 0) < 3:
        return "file_discover_node"
    return "final_response_node"


builder = StateGraph(RagDevState)

builder.add_node("file_discover_node", file_discover_node)
builder.add_node("extract_signatures_node", extract_signatures_node)
builder.add_node("build_retrieval_context_node", build_retrieval_context_node)
builder.add_node("agent_node", agent_node)
builder.add_node("replan_node", replan_node)
builder.add_node("final_response_node", final_response_node)

builder.add_edge(START, "file_discover_node")
builder.add_edge("file_discover_node", "extract_signatures_node")
builder.add_edge("extract_signatures_node", "build_retrieval_context_node")
builder.add_edge("build_retrieval_context_node", "agent_node")
builder.add_edge("agent_node", "replan_node")
builder.add_conditional_edges(
    "replan_node",
    _should_replan,
    {
        "file_discover_node": "file_discover_node",
        "final_response_node": "final_response_node",
    },
)
builder.add_edge("final_response_node", END)

compiled_graph = builder.compile()
