"""Glob-based file discovery from query keywords."""

from __future__ import annotations

import os
from glob import glob
from typing import Iterator


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a natural language query."""
    stopwords = {"a", "an", "the", "to", "for", "in", "on", "at", "add", "support", "use", "using", "with"}
    tokens = query.lower().split()
    return [t.strip(".,!?;:()[]{}") for t in tokens if t not in stopwords and len(t) > 2]


def build_glob_patterns(keywords: list[str], extensions: list[str] | None = None) -> list[str]:
    """Build glob patterns from keywords and file extensions."""
    if extensions is None:
        extensions = [".py"]

    patterns = []
    for kw in keywords:
        for ext in extensions:
            patterns.append(f"**/*{kw}*{ext}")
    return patterns


def find_files(patterns: list[str], root: str | None = None) -> Iterator[str]:
    """Yield file paths matching any of the given glob patterns."""
    if root is None:
        root = os.getcwd()

    seen: set[str] = set()
    for pattern in patterns:
        for path in glob(os.path.join(root, pattern), recursive=True):
            if path not in seen:
                seen.add(path)
                yield path


def discover_files(query: str, root: str | None = None) -> list[str]:
    """Find relevant files for a query using keyword-based glob.

    Falls back to listing all .py files in the project if no keywords
    match or glob returns nothing.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return _list_all_py_files(root)

    patterns = build_glob_patterns(keywords)
    results = list(find_files(patterns, root=root))

    # Fallback: if glob matched nothing, list all .py files in the project
    if not results:
        return _list_all_py_files(root)

    return results


def _list_all_py_files(root: str | None = None) -> list[str]:
    """Return all .py files in the project root (non-test, non-cache)."""
    if root is None:
        root = os.getcwd()

    skip_dirs = {"__pycache__", ".git", ".github", "tests", ".pytest_cache", ".claude"}
    skip_prefixes = ("test_", "_")

    results = []
    for path, _, filenames in os.walk(root):
        # Skip hidden dirs and test dirs
        parts = path.split(os.sep)
        if any(s in parts for s in skip_dirs):
            continue
        for fn in filenames:
            if fn.startswith("."):
                continue
            if fn.startswith(skip_prefixes):
                continue
            if fn.endswith(".py"):
                results.append(os.path.join(path, fn))

    return results


def discover_files_explicit(file_paths: list[str], root: str | None = None) -> list[str]:
    """Return explicitly provided file paths that exist."""
    if root is None:
        root = os.getcwd()

    found = []
    for fp in file_paths:
        full = os.path.join(root, fp) if not os.path.isabs(fp) else fp
        if os.path.isfile(full):
            found.append(full)
    return found
