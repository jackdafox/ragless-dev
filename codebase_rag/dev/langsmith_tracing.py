"""LangSmith tracing integration for rag dev pipeline.

Enable via LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY env vars,
or call `setup_tracing(project_name: str | None = None)` programmatically.
"""

from __future__ import annotations

import os
from typing import Any

from langsmith import Client as LangSmithClient
from langchain_core.runnables import RunnableConfig

# Module-level client (lazily initialized)
_client: LangSmithClient | None = None


def get_tracing_uri() -> str:
    """Return the LangSmith endpoint from environment or default."""
    return os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def setup_tracing(project_name: str | None = None) -> None:
    """Initialize LangSmith tracing globally.

    Reads LANGCHAIN_API_KEY from environment. Does nothing if the key is absent.
    Call once at application startup.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        return

    os.environ.setdefault("LANGCHAIN_PROJECT", project_name or "rag-dev")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def get_client() -> LangSmithClient | None:
    """Return the LangSmith client, or None if no API key is configured."""
    global _client
    if _client is None:
        api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        if not api_key:
            return None
        _client = LangSmithClient(
            api_key=api_key,
            endpoint=get_tracing_uri(),
        )
    return _client


def trace_node(name: str, tags: list[str] | None = None):
    """Decorator / context manager that wraps a node function in a LangSmith span.

    Usage as decorator:
        @trace_node("file_discover")
        def file_discover_node(state):
            ...

    Usage as context manager:
        with trace_node("my_step"):
            do_work()
    """
    from langsmith.trace import get_current_span
    from contextlib import contextmanager

    @contextmanager
    def _span():
        span_tags = tags or []
        # Use the low-level span API when tracing is enabled
        client = get_client()
        if client is None:
            yield None
            return

        with client.start_span(name=name, tags=span_tags) as span:
            try:
                yield span
            except Exception as e:
                span.add_event({"name": "error", "properties": {"error": str(e)}})
                raise

    return _span