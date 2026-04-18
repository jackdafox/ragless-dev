"""LangGraph nodes for rag dev pipeline."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage

from .file_finder import discover_files, discover_files_explicit
from .signature_extractor import extract_signatures
from .context_builder import build_context, format_llm_prompt
from .llm import build_agent
from .state import RagDevState


MAX_STEPS = 3

# Cached agent — built once per process
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def file_discover_node(state: RagDevState) -> dict:
    """Discover files based on the query."""
    query = state["query"]
    existing = set(state.get("discovered_files", []))
    step = state.get("step", 0)

    files = discover_files(query)
    # Merge with already-discovered files
    merged = list(existing) + [f for f in files if f not in existing]

    return {
        "discovered_files": merged,
        "step": step + 1,
    }


def extract_signatures_node(state: RagDevState) -> dict:
    """Extract function signatures from discovered files."""
    files = state.get("discovered_files", [])
    if not files:
        return {"extracted_signatures": []}

    sigs = extract_signatures(files)
    return {"extracted_signatures": sigs}


def build_retrieval_context_node(state: RagDevState) -> dict:
    """Build the LLM context string and append to messages."""
    query = state["query"]
    files = state.get("discovered_files", [])
    sigs = state.get("extracted_signatures", [])

    if not files:
        retrieval_context = (
            "No files matched the query. Answer using your own knowledge "
            f"to help with: {query}"
        )
    else:
        ctx = build_context(query, files, sigs)
        retrieval_context = format_llm_prompt(ctx)

    messages = [HumanMessage(content=retrieval_context)]
    return {
        "retrieval_context": retrieval_context,
        "messages": messages,
    }


def agent_node(state: RagDevState) -> dict:
    """Run the ReAct agent with current context."""
    agent = _get_agent()

    systemPrompt = (
        "You are a codebase reasoning assistant. The user wants help with a coding task. "
        "Review the provided file signatures and content. If you need more files, call "
        "request_file_discovery. If you have enough context, give a direct answer."
    )

    response = agent.invoke({
        "messages": [
            HumanMessage(content=systemPrompt),
            *state["messages"],
        ]
    })

    return {"messages": response["messages"]}


def replan_node(state: RagDevState) -> dict:
    """Inspect agent output, decide if more files needed."""
    messages = state.get("messages", [])
    step = state.get("step", 0)

    needs_more_files = False
    replan_reason = ""

    # Scan messages for tool calls
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call["name"] == "request_file_discovery":
                    needs_more_files = True
                    args = call["args"]
                    replan_reason = args.get("reason", "Agent requested more files")
                    break
            if needs_more_files:
                break

    # Cap loop
    if step >= MAX_STEPS:
        needs_more_files = False

    return {
        "needs_more_files": needs_more_files,
        "replan_reason": replan_reason,
    }


def final_response_node(state: RagDevState) -> dict:
    """Return the final output string."""
    retrieval_context = state.get("retrieval_context", "")
    return {"retrieval_context": retrieval_context}
