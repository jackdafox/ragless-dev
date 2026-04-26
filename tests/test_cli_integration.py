"""Tests for the CLI end-to-end."""

from __future__ import annotations

import os
import sys
import pytest

from codebase_rag.dev.coordinator import DevCoordinator

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


def test_rag_dev_cli_with_minimax_api_key_succeeds(tmp_path, monkeypatch, capsys):
    """CLI with MINIMAX_API_KEY set runs and outputs something."""
    # The coordinator calls get_context which runs the full pipeline
    # We just verify it runs without crashing and produces output
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(tmp_path)
        sys.argv = ["rag", "login"]
        from codebase_rag.dev.__main__ import main
        ret = main()
        # Should complete without error
        assert ret == 0
        captured = capsys.readouterr()
        # Should have printed something
        assert len(captured.out) > 0
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def test_get_context_returns_final_response(tmp_path):
    """get_context returns a final_response key with LLM-generated text."""
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        coord = DevCoordinator()
        ctx = coord.get_context("login")
        assert "final_response" in ctx
        # final_response is a non-empty string (or a string representation of a list)
        assert isinstance(ctx["final_response"], str)
        assert len(ctx["final_response"]) > 0
    finally:
        os.chdir(old_cwd)


def test_handle_query_returns_natural_language_answer(tmp_path):
    """handle_query returns a non-trivial answer."""
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        coord = DevCoordinator()
        result = coord.handle_query("login")
        # Should not be empty
        assert len(result) > 0
    finally:
        os.chdir(old_cwd)
