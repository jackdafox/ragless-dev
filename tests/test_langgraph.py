"""Integration tests for LangGraph rag dev pipeline."""

import os
import tempfile

from codebase_rag.dev.state import RagDevState
from codebase_rag.dev.nodes import (
    file_discover_node,
    extract_signatures_node,
    build_retrieval_context_node,
    replan_node,
    MAX_STEPS,
)


def test_file_discover_node(tmp_path):
    # Create dummy files
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\ndef logout(user): pass\n")

    token = tmp_path / "token.py"
    token.write_text("def create_token(): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        state: RagDevState = {
            "query": "auth token",
            "discovered_files": [],
            "extracted_signatures": [],
            "full_files": {},
            "retrieval_context": "",
            "needs_more_files": False,
            "replan_reason": "",
            "step": 0,
            "messages": [],
        }
        result = file_discover_node(state)
        assert len(result["discovered_files"]) == 2
        assert result["step"] == 1
    finally:
        os.chdir(old_cwd)


def test_extract_signatures_node(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\ndef logout(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        state: RagDevState = {
            "query": "auth",
            "discovered_files": [str(auth)],
            "extracted_signatures": [],
            "full_files": {},
            "retrieval_context": "",
            "needs_more_files": False,
            "replan_reason": "",
            "step": 1,
            "messages": [],
        }
        result = extract_signatures_node(state)
        assert len(result["extracted_signatures"]) == 2
        names = {s.name for s in result["extracted_signatures"]}
        assert names == {"login", "logout"}
    finally:
        os.chdir(old_cwd)


def test_build_retrieval_context_node(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        state: RagDevState = {
            "query": "login",
            "discovered_files": [str(auth)],
            "extracted_signatures": [],
            "full_files": {},
            "retrieval_context": "",
            "needs_more_files": False,
            "replan_reason": "",
            "step": 1,
            "messages": [],
        }
        result = build_retrieval_context_node(state)
        assert "retrieval_context" in result
        assert len(result["messages"]) == 1
    finally:
        os.chdir(old_cwd)


def test_replan_node_no_tool_call():
    from langchain_core.messages import AIMessage, HumanMessage

    state: RagDevState = {
        "query": "auth",
        "discovered_files": [],
        "extracted_signatures": [],
        "full_files": {},
        "retrieval_context": "no files found",
        "needs_more_files": False,
        "replan_reason": "",
        "step": 1,
        "messages": [
            HumanMessage(content="help"),
            AIMessage(content="Here is the answer."),
        ],
    }
    result = replan_node(state)
    assert result["needs_more_files"] is False


def test_replan_node_with_tool_call():
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.messages.content import ToolCall

    state: RagDevState = {
        "query": "auth",
        "discovered_files": [],
        "extracted_signatures": [],
        "full_files": {},
        "retrieval_context": "",
        "needs_more_files": False,
        "replan_reason": "",
        "step": 1,
        "messages": [
            HumanMessage(content="need more files"),
            AIMessage(
                content="Let me find more files.",
                tool_calls=[
                    ToolCall(
                        name="request_file_discovery",
                        args={
                            "refined_query": "token jwt auth",
                            "reason": "Need JWT auth files",
                        },
                        id="call_abc123",
                    )
                ],
            ),
        ],
    }
    result = replan_node(state)
    assert result["needs_more_files"] is True
    assert "JWT auth" in result["replan_reason"]


def test_replan_node_caps_at_max_steps():
    from langchain_core.messages import AIMessage
    from langchain_core.messages.content import ToolCall

    state: RagDevState = {
        "query": "auth",
        "discovered_files": [],
        "extracted_signatures": [],
        "full_files": {},
        "retrieval_context": "",
        "needs_more_files": True,
        "replan_reason": "always need more",
        "step": MAX_STEPS,  # hit cap
        "messages": [
            AIMessage(
                content="need more",
                tool_calls=[
                    ToolCall(
                        name="request_file_discovery",
                        args={"refined_query": "x", "reason": "y"},
                        id="call_xyz",
                    )
                ],
            ),
        ],
    }
    result = replan_node(state)
    assert result["needs_more_files"] is False  # capped


def test_graph_state_schema():
    """Verify RagDevState has all required keys."""
    state: RagDevState = {
        "query": "",
        "discovered_files": [],
        "extracted_signatures": [],
        "full_files": {},
        "retrieval_context": "",
        "final_response": "",
        "needs_more_files": False,
        "replan_reason": "",
        "step": 0,
        "messages": [],
    }
    assert all(k in state for k in RagDevState.__annotations__)
