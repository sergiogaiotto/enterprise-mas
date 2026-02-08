"""Planning Agent: decomposes the user query into an ordered list of sub-tasks.

Corresponds to the 'Planning Agent' node in the enterprise MAS architecture.
Uses an LLM call to produce a structured plan that downstream agents execute.
"""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import AgentState


PLAN_SYSTEM = """You are a planning agent inside an enterprise multi-agent system.
Given a user query, decompose it into 1-5 ordered sub-tasks.

Each sub-task must specify:
- "id": sequential integer starting at 1
- "description": what needs to be done
- "tool": one of "search", "calculate", "code", "general"
- "status": always "pending"

Respond ONLY with a JSON array. No markdown, no explanation."""


def planning_node(state: AgentState) -> dict:
    """LangGraph node: produces a plan from the user query."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )

    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": state["query"]},
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Robust JSON extraction
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = [
            {
                "id": 1,
                "description": state["query"],
                "tool": "general",
                "status": "pending",
            }
        ]

    return {
        "plan": plan,
        "current_task_index": 0,
        "search_queries": [],
        "context_documents": [],
        "tool_results": [],
    }
