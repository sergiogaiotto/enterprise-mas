"""API Routes: FastAPI endpoints for the enterprise MAS.

Exposes:
  POST /api/chat       — run a query through the multi-agent graph
  POST /api/ingest     — add documents to the knowledge base
  GET  /api/health     — health check
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph import mas_graph
from app.tools.knowledge_base import ingest_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    query: str
    chat_history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    response: str
    plan: list[dict[str, Any]]
    search_queries: list[str]
    tool_results: list[dict[str, Any]]
    review_passed: bool
    review_feedback: str
    elapsed_ms: int


class IngestRequest(BaseModel):
    documents: list[str]


class IngestResponse(BaseModel):
    ingested: int


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run a query through the full multi-agent pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.perf_counter()

    try:
        initial_state = {
            "query": req.query,
            "chat_history": req.chat_history,
            "revision_count": 0,
        }
        result = mas_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        response=result.get("final_response", result.get("draft_response", "No response generated.")),
        plan=result.get("plan", []),
        search_queries=result.get("search_queries", []),
        tool_results=result.get("tool_results", []),
        review_passed=result.get("review_passed", False),
        review_feedback=result.get("review_feedback", ""),
        elapsed_ms=elapsed,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest documents into the knowledge base."""
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    count = ingest_documents(req.documents)
    return IngestResponse(ingested=count)


@router.get("/health")
async def health():
    return {"status": "ok", "service": "Enterprise MAS"}
