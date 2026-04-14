"""Tests for signature_extractor module."""

import os
import tempfile

from codebase_rag.dev.signature_extractor import (
    FunctionSignature,
    parse_python_file,
    extract_signatures,
)


def test_parse_python_file_regex(tmp_path):
    # Create a test Python file
    test_file = tmp_path / "sample.py"
    test_file.write_text('''
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}"

async def fetch_data(url: str, timeout: int = 30) -> bytes:
    """Fetch data from URL."""
    return b"data"
''')

    sigs = list(parse_python_file(str(test_file)))
    assert len(sigs) == 2

    greet = next(s for s in sigs if s.name == "greet")
    assert greet.params == ["name"]
    assert greet.return_type == "str"
    # docstring requires tree-sitter; regex fallback returns None
    assert greet.docstring is None
    assert greet.start_line == 2

    fetch = next(s for s in sigs if s.name == "fetch_data")
    assert fetch.params == ["url", "timeout"]


def test_extract_signatures(tmp_path):
    auth = tmp_path / "auth.py"
    auth.write_text('''
def create_token(user_id: int) -> str:
    pass

def verify_token(token: str) -> bool:
    pass
''')

    sigs = extract_signatures([str(auth)])
    assert len(sigs) == 2
    names = {s.name for s in sigs}
    assert names == {"create_token", "verify_token"}
