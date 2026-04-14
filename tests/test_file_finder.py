"""Tests for file_finder module."""

import os
import tempfile
from pathlib import Path

from codebase_rag.dev.file_finder import (
    extract_keywords,
    build_glob_patterns,
    discover_files,
    discover_files_explicit,
)


def test_extract_keywords():
    assert extract_keywords("add JWT refresh token support to auth service") == [
        "jwt", "refresh", "token", "auth", "service"
    ]
    assert extract_keywords("fix null pointer in config loader") == [
        "fix", "null", "pointer", "config", "loader"
    ]
    assert extract_keywords("the") == []


def test_build_glob_patterns():
    patterns = build_glob_patterns(["auth", "token"], [".py", ".js"])
    assert "**/*auth*.py" in patterns
    assert "**/*auth*.js" in patterns
    assert "**/*token*.py" in patterns


def test_discover_files_explicit(tmp_path):
    # Create dummy files
    (tmp_path / "auth.py").write_text("# auth")
    (tmp_path / "token.py").write_text("# token")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        found = discover_files_explicit(["auth.py", "token.py", "missing.py"])
        assert len(found) == 2
    finally:
        os.chdir(old_cwd)
