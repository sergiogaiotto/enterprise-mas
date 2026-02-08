"""Search Agent: multi-hop retrieval across web and local knowledge base.

Implements the 'Search Agent' from the architecture — performs iterative
retrieval, grades relevance, and rephrases queries when documents are irrelevant.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import AgentState
from app.tools.web_search import web_search
from app.tools.knowledge_base import query_knowledge_base


GRADER_SYSTEM = """You are a relevance grader. Given a user query and a document,
respond with ONLY 'yes' or 'no' — is the document relevant to answering the query?"""

REPHRASE_SYSTEM = """You are a query optimizer. Given a user query, rephrase it
to be more effective for web search. Respond with ONLY the rephrased query."""


def _grade_document(llm: ChatOpenAI, query: str, doc: str) -> bool:
    """Return True if the document is relevant to the query."""
    response = llm.invoke([
        {"role": "system", "content": GRADER_SYSTEM},
        {"role": "user", "content": f"Query: {query}\n\nDocument: {doc[:1500]}"},
    ])
    return response.content.strip().lower().startswith("yes")


def _rephrase_query(llm: ChatOpenAI, query: str) -> str:
    """Rephrase a query for better web search results."""
    response = llm.invoke([
        {"role": "system", "content": REPHRASE_SYSTEM},
        {"role": "user", "content": query},
    ])
    return response.content.strip()


def search_node(state: AgentState) -> dict:
    """LangGraph node: performs multi-hop retrieval for the current sub-task."""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )

    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)
    task_desc = plan[idx]["description"] if idx < len(plan) else state["query"]

    collected_docs: list[str] = []
    queries_used: list[str] = [task_desc]

    # Hop 1: local knowledge base
    kb_results = query_knowledge_base(task_desc)
    for doc in kb_results:
        if _grade_document(llm, task_desc, doc):
            collected_docs.append(doc)

    # Hop 2: web search (original or rephrased query)
    if len(collected_docs) < 2:
        rephrased = _rephrase_query(llm, task_desc)
        queries_used.append(rephrased)
        web_results = web_search(rephrased, max_results=3)
        for doc in web_results:
            if _grade_document(llm, task_desc, doc):
                collected_docs.append(doc)

    # Hop 3: fallback broader search
    if not collected_docs:
        broader = _rephrase_query(llm, f"broader context: {task_desc}")
        queries_used.append(broader)
        web_results = web_search(broader, max_results=3)
        collected_docs.extend(web_results[:2])

    return {
        "context_documents": collected_docs,
        "search_queries": queries_used,
    }
