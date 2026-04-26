"""LangGraph nodes for rag dev pipeline."""

from __future__ import annotations

import os
import time
from langchain_core.messages import HumanMessage, AIMessage

from .file_finder import discover_files, discover_files_explicit
from .signature_extractor import extract_signatures
from .context_builder import build_context, format_llm_prompt
from .llm import build_agent
from .state import RagDevState

DEBUG_TIMING = os.environ.get("DEBUG_TIMING", "0") == "1"

MAX_STEPS = 3

# Cached agent — built once per process
_agent = None

# Streaming callback — prints chunks to stderr as they arrive
_stream_callback = os.environ.get("STREAM_OUTPUT", "1") == "1"


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def _timeit(name: str, fn, state: RagDevState):
    """Time a node function and print result if DEBUG_TIMING=1."""
    t0 = time.perf_counter()
    result = fn(state)
    elapsed = (time.perf_counter() - t0) * 1000
    if DEBUG_TIMING:
        print(f"[TIMING] {name}: {elapsed:.1f}ms", file=os.sys.stderr)
    return result


def file_discover_node(state: RagDevState) -> dict:
    """Discover files based on the query."""
    def _run(state):
        query = state["query"]
        existing = set(state.get("discovered_files", []))
        step = state.get("step", 0)

        files = discover_files(query)
        merged = list(existing) + [f for f in files if f not in existing]

        return {
            "discovered_files": merged,
            "step": step + 1,
        }
    return _timeit("file_discover_node", _run, state)


def extract_signatures_node(state: RagDevState) -> dict:
    def _run(state):
        files = state.get("discovered_files", [])
        if not files:
            return {"extracted_signatures": []}
        sigs = extract_signatures(files)
        return {"extracted_signatures": sigs}
    return _timeit("extract_signatures_node", _run, state)


def build_retrieval_context_node(state: RagDevState) -> dict:
    def _run(state):
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
    return _timeit("build_retrieval_context_node", _run, state)


def agent_node(state: RagDevState) -> dict:
    """Run the ReAct agent with current context, streaming output to stderr."""
    def _run(state):
        agent = _get_agent()

        systemPrompt = (
            "You are a codebase reasoning assistant. The user wants help with a coding task. "
            "Review the provided file signatures and content. If you need more files, call "
            "request_file_discovery. If you have enough context, give a direct answer."
        )

        input_messages = [
            HumanMessage(content=systemPrompt),
            *state["messages"],
        ]

        if _stream_callback:
            # Stream agent output as it generates
            chunks = []
            for event in agent.stream({"messages": input_messages}):
                for node_name, node_output in event.items():
                    if hasattr(node_output, "messages"):
                        for msg in node_output.messages:
                            if hasattr(msg, "content") and msg.content:
                                text = str(msg.content)
                                if text.strip():
                                    print(f"[agent] {text}", end="", flush=True, file=os.sys.stderr)
                                    chunks.append(text)
            print(flush=True, file=os.sys.stderr)
            # Get final result
            response = agent.invoke({"messages": input_messages})
        else:
            response = agent.invoke({"messages": input_messages})

        return {"messages": response["messages"]}
    return _timeit("agent_node", _run, state)


def replan_node(state: RagDevState) -> dict:
    def _run(state):
        messages = state.get("messages", [])
        step = state.get("step", 0)

        needs_more_files = False
        replan_reason = ""

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

        if step >= MAX_STEPS:
            needs_more_files = False

        return {
            "needs_more_files": needs_more_files,
            "replan_reason": replan_reason,
        }
    return _timeit("replan_node", _run, state)


def final_response_node(state: RagDevState) -> dict:
    """Generate a natural language response from the retrieval context using streaming."""
    def _run(state):
        # Skip if TUI is handling its own streaming
        if state.get("skip_final_response"):
            return {"final_response": ""}

        from .llm import get_llm
        retrieval_context = state.get("retrieval_context", "")
        query = state.get("query", "")

        llm = get_llm()
        prompt = (
            "You are a helpful coding assistant. Based on the following "
            "codebase information, answer the user's query in a friendly, "
            "concise way. If no relevant code was found, say so.\n\n"
            f"Query: {query}\n\n"
            f"Codebase context:\n{retrieval_context}\n\n"
            "Answer:"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            answer = " ".join(b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text")
        else:
            answer = str(raw)

        return {"final_response": answer}

    return _timeit("final_response_node", _run, state)
