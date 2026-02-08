"""Router: conditional edge functions that determine graph traversal.

Implements the DAG branching logic — decides whether to continue
executing sub-tasks, loop back for revision, or finalize the response.
"""

from __future__ import annotations

from app.core.state import AgentState


def route_after_executor(state: AgentState) -> str:
    """After executing a sub-task, decide: execute more or generate response."""
    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)

    pending = [t for t in plan if t.get("status") == "pending"]

    if pending and idx < len(plan):
        # More tasks to process — loop back to search + execute
        return "search"

    return "respond"


def route_after_review(state: AgentState) -> str:
    """After review, decide: finalize or revise."""
    if state.get("review_passed", False):
        return "finalize"
    return "revise"
