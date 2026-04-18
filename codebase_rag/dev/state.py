"""State schema for LangGraph rag dev pipeline."""

from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from .signature_extractor import FunctionSignature


class RagDevState(TypedDict):
    """State passed between nodes in the rag dev LangGraph."""

    query: str
    discovered_files: list[str]
    extracted_signatures: list[FunctionSignature]
    full_files: dict[str, str]
    retrieval_context: str
    final_response: str
    needs_more_files: bool
    replan_reason: str
    step: int
    messages: Annotated[list[BaseMessage], add_messages]
