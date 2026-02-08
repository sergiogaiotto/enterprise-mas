"""Shared application state for the LangGraph DAG.

Each node reads and mutates this state as it flows through the graph.
Modeled after the enterprise MAS architecture: plan → search → act → review → respond.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


class SubTask(TypedDict, total=False):
    id: int
    description: str
    tool: str
    status: Literal["pending", "done", "failed"]
    result: str


class AgentState(TypedDict, total=False):
    """Unified state passed through every node in the graph."""

    # --- input ---
    query: str
    chat_history: list[dict[str, str]]

    # --- planning ---
    plan: list[SubTask]
    current_task_index: int

    # --- retrieval ---
    context_documents: Annotated[list[str], operator.add]
    search_queries: list[str]

    # --- execution ---
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    # --- review ---
    review_passed: bool
    review_feedback: str
    revision_count: int

    # --- response ---
    draft_response: str
    final_response: str

    # --- routing ---
    next_node: str
    error: str
