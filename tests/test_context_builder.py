"""Tests for context_builder module."""

import os
import tempfile

from codebase_rag.dev.signature_extractor import FunctionSignature
from codebase_rag.dev.context_builder import (
    format_signature,
    format_signatures,
    build_context,
    format_llm_prompt,
)


def test_format_signature():
    sig = FunctionSignature(
        name="create_token",
        params=["user_id: int", "expires_delta: timedelta"],
        return_type="str",
        docstring="Create a JWT token.",
        file_path="src/auth.py",
        start_line=10,
    )
    formatted = format_signature(sig)
    assert "create_token" in formatted
    assert "user_id" in formatted
    assert "str" in formatted
    assert "src/auth.py:10" in formatted


def test_format_signatures():
    sigs = [
        FunctionSignature("foo", ["x"], "int", None, "a.py", 1),
        FunctionSignature("bar", ["y"], "str", "Bar doc.", "a.py", 5),
        FunctionSignature("baz", [], None, None, "b.py", 20),
    ]
    formatted = format_signatures(sigs)
    assert "## a.py" in formatted
    assert "## b.py" in formatted
    assert "foo" in formatted
    assert "bar" in formatted


def test_build_context(tmp_path):
    # Create dummy files
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\ndef logout(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ctx = build_context("auth login logout", [str(auth)])
        assert ctx.query == "auth login logout"
        assert len(ctx.signatures) == 2
        assert str(auth) in ctx.full_files
    finally:
        os.chdir(old_cwd)


def test_format_llm_prompt(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ctx = build_context("login", [str(auth)])
        prompt = format_llm_prompt(ctx)
        assert "## Query" in prompt
        assert "## Signatures" in prompt
        assert "## Full file" in prompt
    finally:
        os.chdir(old_cwd)
