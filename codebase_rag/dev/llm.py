"""LLM setup — MiniMax via Anthropic SDK compatible with LangChain."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load .env from project root
_dotenv = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_dotenv)


def get_llm() -> ChatAnthropic:
    """Build ChatAnthropic instance pointing at MiniMax."""
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY environment variable is not set")
    return ChatAnthropic(
        model="MiniMax-M2.7",
        anthropic_api_key=api_key,
        temperature=0.2,
        max_tokens=4096,
    )


def build_agent():
    """Build a ReAct agent with rag dev tools."""
    from langgraph.prebuilt import create_react_agent
    from .tools import request_file_discovery, get_file_signatures
    llm = get_llm()
    return create_react_agent(llm, [request_file_discovery, get_file_signatures])