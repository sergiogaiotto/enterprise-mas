"""Response Generation Agent: synthesizes all gathered information into a final answer.

Corresponds to the 'Response Generation Agent' — combines context documents,
tool results, and the plan into a coherent, well-structured response.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import AgentState


RESPOND_SYSTEM = """You are the response generation agent in an enterprise multi-agent system.
You synthesize information from planning, retrieval, and execution into a final answer.

Guidelines:
- Be accurate and cite the context when relevant.
- Structure the response clearly.
- If the review agent provided feedback, incorporate it.
- Keep the response focused and professional.
- Use the user's language (detect from the query).
"""


def respond_node(state: AgentState) -> dict:
    """LangGraph node: generates or revises the final response."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3,
    )

    context_block = "\n---\n".join(state.get("context_documents", [])[:5])

    tool_outputs = "\n".join(
        f"- Task {r.get('task_id', '?')}: {str(r.get('output', ''))[:800]}"
        for r in state.get("tool_results", [])
    )

    plan_summary = "\n".join(
        f"  {t.get('id', '?')}. [{t.get('status', '?')}] {t.get('description', '')}"
        for t in state.get("plan", [])
    )

    feedback = state.get("review_feedback", "")
    prior_draft = state.get("draft_response", "")

    revision_instruction = ""
    if feedback and prior_draft:
        revision_instruction = (
            f"\n\nPREVIOUS DRAFT:\n{prior_draft}\n\n"
            f"REVIEW FEEDBACK (address this):\n{feedback}"
        )

    messages = [
        {"role": "system", "content": RESPOND_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User query: {state['query']}\n\n"
                f"Plan:\n{plan_summary}\n\n"
                f"Retrieved context:\n{context_block}\n\n"
                f"Execution results:\n{tool_outputs}"
                f"{revision_instruction}"
            ),
        },
    ]

    response = llm.invoke(messages)
    draft = response.content.strip()

    return {"draft_response": draft}
