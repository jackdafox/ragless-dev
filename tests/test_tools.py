"""Tests for tools module."""

import os
import tempfile

from codebase_rag.dev.tools import request_file_discovery, get_file_signatures


def test_request_file_discovery_finds_files(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = request_file_discovery.invoke({"refined_query": "auth", "reason": "testing"})
        assert "auth.py" in result
        assert "Found 1 files" in result
    finally:
        os.chdir(old_cwd)


def test_request_file_discovery_no_match():
    result = request_file_discovery.invoke({"refined_query": "zzzz_no_match_xxxx", "reason": "testing"})
    assert "No files found" in result


def test_get_file_signatures(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text("def login(user): pass\ndef logout(user): pass\n")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = get_file_signatures.invoke({"file_paths": [str(auth)]})
        assert "login" in result
        assert "logout" in result
    finally:
        os.chdir(old_cwd)


def test_get_file_signatures_empty():
    result = get_file_signatures.invoke({"file_paths": []})
    assert "No signatures found" in result
