"""Tools for the ReAct agent in the rag dev pipeline."""

from __future__ import annotations

from langchain_core.tools import tool

from .file_finder import discover_files
from .signature_extractor import extract_signatures
from .context_builder import format_signature


@tool
def request_file_discovery(refined_query: str, reason: str) -> str:
    """Request additional files to be discovered based on a refined query.

    Args:
        refined_query: More specific query string for finding additional files.
        reason: Explanation of why these files are needed.
    """
    files = discover_files(refined_query)
    if not files:
        return f"No files found for refined query: {refined_query}"
    return f"Found {len(files)} files:\n" + "\n".join(files)


@tool
def get_file_signatures(file_paths: list[str]) -> str:
    """Get function signatures for specified Python files.

    Args:
        file_paths: List of Python file paths to extract signatures from.
    """
    sigs = extract_signatures(file_paths)
    if not sigs:
        return "No signatures found."
    return "\n".join(format_signature(s) for s in sigs)
