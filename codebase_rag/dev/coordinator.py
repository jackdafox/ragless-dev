"""Coordinator agent — no ChromaDB, file-based context."""

from __future__ import annotations

from .file_finder import discover_files, discover_files_explicit
from .context_builder import build_context, format_llm_prompt


class DevCoordinator:
    """Coordinates rag dev using on-demand file discovery."""

    def __init__(self, root: str | None = None):
        self.root = root

    def handle_query(
        self,
        query: str,
        explicit_files: list[str] | None = None,
    ) -> str:
        """Handle a rag dev query without ChromaDB."""
        if explicit_files:
            file_paths = discover_files_explicit(explicit_files, root=self.root)
        else:
            file_paths = discover_files(query, root=self.root)

        if not file_paths:
            return f"No files found for query: {query}"

        ctx = build_context(query, file_paths)
        prompt = format_llm_prompt(ctx)

        return prompt  # Caller invokes LLM with this prompt

    def get_context(self, query: str, explicit_files: list[str] | None = None):
        """Return structured context for external LLM invocation."""
        if explicit_files:
            file_paths = discover_files_explicit(explicit_files, root=self.root)
        else:
            file_paths = discover_files(query, root=self.root)

        if not file_paths:
            return None

        return build_context(query, file_paths)
