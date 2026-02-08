"""Knowledge Base Tool: local vector store for enterprise document retrieval.

Uses ChromaDB with OpenAI embeddings for semantic search.
Supports ingestion and querying of documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "enterprise_kb"


def _get_collection() -> chromadb.Collection:
    global _client, _collection

    if _collection is not None:
        return _collection

    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=str(persist_dir))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    return _collection


def ingest_documents(texts: list[str], metadatas: list[dict] | None = None) -> int:
    """Add documents to the knowledge base. Returns count of added docs."""
    collection = _get_collection()
    ids = [f"doc_{collection.count() + i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas or [{}] * len(texts),
    )
    return len(ids)


def query_knowledge_base(query: str, n_results: int = 3) -> list[str]:
    """Query the knowledge base and return relevant document texts."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
        documents = results.get("documents", [[]])[0]
        return documents

    except Exception as exc:
        logger.warning("Knowledge base query failed: %s", exc)
        return []
