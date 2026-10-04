"""Interactive CLI for local RAG experiments.

Run from the repository root::

    python -m app.agents.customer_support.rag.cli
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from .bootstrap import configure_llama_index
from .chunking import ChunkingFactory
from .engine import RAGEngineBuilder
from .indexing import VectorIndexManager

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
INDEX_NAME = "AdmetIndex"


def _load_or_build_index(manager: VectorIndexManager, rebuild: bool = False) -> Optional[VectorStoreIndex]:
    """Load the persisted index, or build it from DATA_DIR markdown files."""
    if not rebuild:
        try:
            return manager.load_persisted_index()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Could not load existing index, preparing to build a new one. Details: {exc}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Loading documents from '{DATA_DIR}'...")
    documents = SimpleDirectoryReader(str(DATA_DIR), required_exts=[".md"]).load_data()
    if not documents:
        print("❌ No markdown (.md) documents found in the data directory. Exiting.")
        return None

    nodes = ChunkingFactory.get_strategy("markdown").chunk(documents)
    return manager.create_and_save_index(nodes)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("Starting Professional Groq + Weaviate RAG System")
    print("=" * 50)

    configure_llama_index()
    manager = VectorIndexManager(index_name=INDEX_NAME)
    try:
        index = _load_or_build_index(manager)
        if index is None:
            return

        query_engine = RAGEngineBuilder(index=index).build_hybrid_query_engine(top_k=4, alpha=0.5)
        print("\n🤖 System is ready! Type 'exit' to quit.\n")

        while True:
            question = input("❓ Enter your question: ").strip()
            if question.lower() == "exit":
                print("Shutting down RAG system. Goodbye!")
                break
            if not question:
                continue
            print("\n🔍 Searching and generating response...")
            print("\n✨ Answer:")
            print(query_engine.query(question))
            print("\n" + "-" * 50 + "\n")
    finally:
        manager.close_connection()


if __name__ == "__main__":
    main()
