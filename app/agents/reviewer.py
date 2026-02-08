"""Review Agent: quality assurance, guardrails, and hallucination check.

Corresponds to the 'Review Agent' — inspects the draft response for
accuracy, completeness, and coherence before final delivery.
"""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import AgentState


REVIEW_SYSTEM = """You are a review agent in an enterprise multi-agent system.
You receive the original query, the plan, execution results, and a draft response.

Evaluate the draft on:
1. Accuracy — does it align with the context and results?
2. Completeness — does it address the full query?
3. Coherence — is it well-structured and clear?

Respond ONLY with a JSON object:
{{"passed": true/false, "feedback": "specific feedback if not passed"}}

Be strict but fair. Pass if the response is adequate, even if imperfect."""


def review_node(state: AgentState) -> dict:
    """LangGraph node: reviews the draft response."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )

    revision_count = state.get("revision_count", 0)

    # Avoid infinite loops — auto-pass after 2 revisions
    if revision_count >= 2:
        return {
            "review_passed": True,
            "review_feedback": "Auto-approved after maximum revisions.",
            "revision_count": revision_count,
        }

    tool_outputs = "\n".join(
        f"- Task {r.get('task_id', '?')}: {str(r.get('output', ''))[:500]}"
        for r in state.get("tool_results", [])
    )

    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Original query: {state['query']}\n\n"
                f"Execution results:\n{tool_outputs}\n\n"
                f"Draft response:\n{state.get('draft_response', 'N/A')}"
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
        review = json.loads(raw)
    except json.JSONDecodeError:
        review = {"passed": True, "feedback": "Unable to parse review; auto-passing."}

    return {
        "review_passed": review.get("passed", True),
        "review_feedback": review.get("feedback", ""),
        "revision_count": revision_count + 1,
    }
