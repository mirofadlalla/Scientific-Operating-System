"""Concrete chunking strategies built on llama-index node parsers."""

from __future__ import annotations

import logging
from typing import List

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode, Document

from .base import BaseChunkingStrategy

logger = logging.getLogger(__name__)


class MarkdownStrategy(BaseChunkingStrategy):
    """Chunk Markdown by headers (``##``); ideal for structured documents."""

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        logger.info("Executing MarkdownChunkingStrategy...")
        return MarkdownNodeParser().get_nodes_from_documents(documents)


class SentenceStrategy(BaseChunkingStrategy):
    """Chunk by sentences; useful for text without strict formatting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        logger.info(
            "Executing SentenceStrategy (Size: %d, Overlap: %d)...",
            self.chunk_size, self.chunk_overlap,
        )
        parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return parser.get_nodes_from_documents(documents)


class TokenStrategy(BaseChunkingStrategy):
    """Chunk purely by tokens; useful when strict LLM token limits apply."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        logger.info(
            "Executing TokenStrategy (Size: %d, Overlap: %d)...",
            self.chunk_size, self.chunk_overlap,
        )
        parser = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return parser.get_nodes_from_documents(documents)
