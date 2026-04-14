"""LLM setup — MiniMax via ChatOpenAI-compatible endpoint + ReAct agent."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .tools import request_file_discovery, get_file_signatures


def get_llm():
    """Build ChatOpenAI instance pointing at MiniMax API."""
    return ChatOpenAI(
        model="minimax-2.7-flash",
        openai_api_key=os.environ.get("MINIMAX_API_KEY", ""),
        openai_api_base="https://api.minimaxi.com/v1",
        temperature=0.2,
        max_tokens=4096,
    )


def build_agent():
    """Build a ReAct agent with rag dev tools."""
    llm = get_llm()
    tools = [request_file_discovery, get_file_signatures]
    return create_react_agent(llm, tools)
