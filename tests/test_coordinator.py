"""Tests for coordinator module."""

import os
import tempfile
import pytest

from codebase_rag.dev.coordinator import DevCoordinator


pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


def test_get_context_returns_state(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        coord = DevCoordinator()
        state = coord.get_context("login")
        assert state is not None
        assert "query" in state
        assert "retrieval_context" in state
        assert "extracted_signatures" in state
    finally:
        os.chdir(old_cwd)


def test_handle_query_returns_string(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        coord = DevCoordinator()
        result = coord.handle_query("login")
        assert isinstance(result, str)
    finally:
        os.chdir(old_cwd)


def test_get_context_with_explicit_files(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        coord = DevCoordinator()
        state = coord.get_context("query", explicit_files=[str(auth)])
        assert len(state["discovered_files"]) == 1
        assert state["discovered_files"][0] == str(auth)
    finally:
        os.chdir(old_cwd)


def test_handle_query_no_files():
    coord = DevCoordinator()
    result = coord.handle_query("zzzz_no_match_xxxxx")
    assert "No files found" in result
