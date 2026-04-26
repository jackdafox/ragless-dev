"""LLM setup — MiniMax via Anthropic SDK compatible with LangChain."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load .env from project root
_dotenv = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_dotenv)

# Cached LLM instance — built once per process
_llm = None


def get_llm() -> ChatAnthropic:
    """Return cached ChatAnthropic instance pointing at MiniMax."""
    global _llm
    if _llm is None:
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is not set")
        _llm = ChatAnthropic(
            model="MiniMax-M2.7",
            api_key=api_key,
            base_url="https://api.minimax.io/anthropic",
            temperature=0.2,
            max_tokens_to_sample=4096,
        )
    return _llm


def build_agent():
    """Build a ReAct agent with rag dev tools."""
    from langgraph.prebuilt import create_react_agent
    from .tools import request_file_discovery, get_file_signatures
    return create_react_agent(get_llm(), [request_file_discovery, get_file_signatures])