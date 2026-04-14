"""Coordinator agent — routes rag dev queries using file-based context."""

from __future__ import annotations

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebase_rag.dev.file_finder import discover_files, discover_files_explicit
from codebase_rag.dev.context_builder import build_context, format_llm_prompt


class CoordinatorAgent:
    """Main coordinator — skips RAG retrieval, uses file-based context directly."""

    def __init__(self, root: str | None = None):
        self.root = root or os.getcwd()

    def process(self, query: str, explicit_files: list[str] | None = None) -> dict:
        """Process a developer query and return context dict."""
        if explicit_files:
            file_paths = discover_files_explicit(explicit_files, root=self.root)
        else:
            file_paths = discover_files(query, root=self.root)

        if not file_paths:
            return {
                "query": query,
                "signatures": [],
                "full_files": {},
                "prompt": f"No files found for query: {query}",
                "file_paths": [],
            }

        ctx = build_context(query, file_paths)
        prompt = format_llm_prompt(ctx)

        return {
            "query": query,
            "signatures": ctx.signatures,
            "full_files": ctx.full_files,
            "prompt": prompt,
            "file_paths": file_paths,
        }

    def run(self, query: str, explicit_files: list[str] | None = None) -> str:
        """Run coordinator and return the LLM prompt."""
        result = self.process(query, explicit_files)
        return result["prompt"]
