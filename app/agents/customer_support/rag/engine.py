"""Hybrid (vector + BM25) query-engine construction."""

from __future__ import annotations

import logging

from app.core.prompt_rules import CONCISE_ANSWER_RULE
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)

# Production-grade QA prompt that discourages hallucination.
QA_PROMPT_TEMPLATE = (
    "Context information is provided strictly below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and NOT prior knowledge, "
    "answer the user query accurately and professionally.\n"
    "If the answer cannot be found or inferred directly from the provided context, "
    "reply with exactly this sentence and nothing else: "
    "The documentation does not contain information about this topic.\n"
    + CONCISE_ANSWER_RULE.replace("{", "{{").replace("}", "}}")
    + "\n\n"
    "Query: {query_str}\n"
    "Answer: "
)


class RAGEngineBuilder:
    """Builds a query engine using Weaviate hybrid search when available."""

    def __init__(self, index: VectorStoreIndex) -> None:
        if index is None:
            raise ValueError("Cannot build query engine without a valid VectorStoreIndex.")
        self.index = index

    def build_hybrid_query_engine(self, top_k: int = 4, alpha: float = 0.5) -> BaseQueryEngine:
        """Return a query engine for the index.

        Args:
            top_k: number of top relevant chunks to retrieve.
            alpha: hybrid weighting — 0.0 pure keyword (BM25), 1.0 pure vector.
                Ignored for non-Weaviate (in-memory) stores.
        """
        logger.info("Configuring Hybrid Query Engine parameters (Top-K: %s, Alpha: %s)...", top_k, alpha)
        qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)

        if isinstance(self.index.storage_context.vector_store, WeaviateVectorStore):
            engine = self.index.as_query_engine(
                vector_store_query_mode="hybrid",
                alpha=alpha,
                similarity_top_k=top_k,
                text_qa_template=qa_prompt,
            )
            logger.info("✅ Hybrid Query Engine built successfully and wired to Groq LLM.")
        else:
            logger.info("Using standard vector search for in-memory mode.")
            engine = self.index.as_query_engine(
                vector_store_query_mode="default",
                similarity_top_k=top_k,
                text_qa_template=qa_prompt,
            )
            logger.info("✅ Standard Query Engine built successfully and wired to Groq LLM.")
        return engine
