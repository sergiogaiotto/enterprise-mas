"""Executor Agent: carries out actions defined in the plan.

Corresponds to the 'Action Agent' — handles tool-calling capabilities
such as code execution, calculations, and general reasoning over context.
"""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import AgentState


EXECUTOR_SYSTEM = """You are an execution agent in an enterprise multi-agent system.
You receive a sub-task description and supporting context documents.
Execute the task and produce a clear, concise result.

If the task involves calculation, show the computation.
If the task involves code, produce working code.
If the task involves analysis, provide structured analysis.

Respond with a JSON object:
{{"result": "your output", "status": "done"}}

If you cannot complete the task:
{{"result": "explanation of failure", "status": "failed"}}
"""


def executor_node(state: AgentState) -> dict:
    """LangGraph node: executes the current sub-task using context."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
    )

    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)
    task = plan[idx] if idx < len(plan) else {"description": state["query"], "tool": "general"}
    context = state.get("context_documents", [])

    context_block = "\n---\n".join(context[:5]) if context else "No context available."

    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Sub-task: {task['description']}\n"
                f"Tool hint: {task.get('tool', 'general')}\n\n"
                f"Context:\n{context_block}"
            ),
        },
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"result": response.content.strip(), "status": "done"}

    # Update the task status in the plan
    updated_plan = list(plan)
    if idx < len(updated_plan):
        updated_plan[idx] = {**updated_plan[idx], "status": result.get("status", "done"), "result": result.get("result", "")}

    next_idx = idx + 1

    return {
        "plan": updated_plan,
        "current_task_index": next_idx,
        "tool_results": [{"task_id": task.get("id", idx), "output": result.get("result", "")}],
    }
