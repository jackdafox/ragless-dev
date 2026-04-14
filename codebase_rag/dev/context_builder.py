"""Build LLM context from signatures and file content."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from .signature_extractor import FunctionSignature, extract_signatures


@dataclass
class LLMContext:
    """Formatted context ready for LLM consumption."""
    query: str
    signatures: list[FunctionSignature]
    full_files: dict[str, str]  # path -> content


def format_signature(sig: FunctionSignature) -> str:
    """Format a single signature as a readable string."""
    params = ", ".join(sig.params)
    sig_str = f"{sig.name}({params})"
    if sig.return_type:
        sig_str += f" -> {sig.return_type}"
    if sig.docstring:
        sig_str += f"\n    \"\"\"{sig.docstring}\"\"\""
    sig_str += f"\n    # defined at {sig.file_path}:{sig.start_line}"
    return sig_str


def format_signatures(sigs: list[FunctionSignature]) -> str:
    """Format all signatures grouped by file."""
    by_file: dict[str, list[FunctionSignature]] = {}
    for sig in sigs:
        by_file.setdefault(sig.file_path, []).append(sig)

    lines = []
    for fp, file_sigs in by_file.items():
        lines.append(f"\n## {fp}")
        for sig in file_sigs:
            lines.append(f"  {format_signature(sig)}")
        lines.append("")

    return "\n".join(lines)


def read_file_lines(file_path: str, start: int, end: int | None = None) -> str:
    """Read specific line range from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if end is None:
            end = len(lines)
        return "".join(lines[start - 1 : end])
    except Exception:
        return ""


def read_full_file(file_path: str) -> str:
    """Read entire file content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def build_context(
    query: str,
    file_paths: list[str],
    signatures: list[FunctionSignature] | None = None,
    top_n_files: int = 3,
) -> LLMContext:
    """Build LLM context: signatures for all files + full content for top candidates."""
    if signatures is None:
        signatures = extract_signatures(file_paths)

    # Select top N files for full content (by signature count as proxy relevance)
    sig_count_by_file: dict[str, int] = {}
    for sig in signatures:
        sig_count_by_file[sig.file_path] = sig_count_by_file.get(sig.file_path, 0) + 1

    sorted_files = sorted(sig_count_by_file, key=lambda f: sig_count_by_file[f], reverse=True)
    top_files = sorted_files[:top_n_files]

    full_files = {fp: read_full_file(fp) for fp in top_files}

    return LLMContext(
        query=query,
        signatures=signatures,
        full_files=full_files,
    )


def format_llm_prompt(ctx: LLMContext) -> str:
    """Format LLMContext as a full prompt string."""
    sigs_section = format_signatures(ctx.signatures)

    full_section = ""
    for fp, content in ctx.full_files.items():
        full_section += f"\n\n## Full file: {fp}\n```python\n{content}\n```"

    return f"""## Query
{ctx.query}

## Signatures
{sigs_section}

## Full file content
{full_section}
"""


def context_to_messages(ctx: LLMContext) -> list[dict]:
    """Convert LLMContext to a messages list for API calls."""
    prompt = format_llm_prompt(ctx)
    return [
        {"role": "user", "content": prompt},
    ]
