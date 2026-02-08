"""Enterprise Multi-Agent System Graph.

Defines the LangGraph StateGraph (DAG) that orchestrates:
  Plan → Search → Execute → (loop if more tasks) → Respond → Review → Finalize/Revise

This is the central artifact of the system — a directed acyclic graph
with conditional edges implementing the full Plan-Retrieve-Execute pattern.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core.state import AgentState
from app.agents.planner import planning_node
from app.agents.searcher import search_node
from app.agents.executor import executor_node
from app.agents.reviewer import review_node
from app.agents.responder import respond_node
from app.agents.router import route_after_executor, route_after_review


def _finalize_node(state: AgentState) -> dict:
    """Terminal node: copies draft into final_response."""
    return {"final_response": state.get("draft_response", "")}


def build_graph() -> StateGraph:
    """Construct and compile the enterprise MAS graph."""

    graph = StateGraph(AgentState)

    # --- nodes (prefixed to avoid collision with state keys) ---
    graph.add_node("agent_plan", planning_node)
    graph.add_node("agent_search", search_node)
    graph.add_node("agent_execute", executor_node)
    graph.add_node("agent_respond", respond_node)
    graph.add_node("agent_review", review_node)
    graph.add_node("agent_finalize", _finalize_node)

    # --- edges ---
    graph.add_edge(START, "agent_plan")
    graph.add_edge("agent_plan", "agent_search")
    graph.add_edge("agent_search", "agent_execute")

    # After execution: loop back to search (more tasks) or generate response
    graph.add_conditional_edges(
        "agent_execute",
        route_after_executor,
        {"search": "agent_search", "respond": "agent_respond"},
    )

    graph.add_edge("agent_respond", "agent_review")

    # After review: finalize or revise
    graph.add_conditional_edges(
        "agent_review",
        route_after_review,
        {"finalize": "agent_finalize", "revise": "agent_respond"},
    )

    graph.add_edge("agent_finalize", END)

    return graph.compile()


# Singleton compiled graph
mas_graph = build_graph()
