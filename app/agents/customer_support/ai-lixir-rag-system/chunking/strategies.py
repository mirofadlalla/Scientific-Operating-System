from typing import List
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, TokenTextSplitter
from .base import BaseChunkingStrategy

class MarkdownStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking Markdown files based on headers (##).
    Perfect for structured documents.
    """
    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        print("Executing MarkdownChunkingStrategy...")
        parser = MarkdownNodeParser()
        return parser.get_nodes_from_documents(documents)

class SentenceStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text by sentences.
    Useful for general text documents without strict formatting.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        print(f"Executing SentenceStrategy (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return parser.get_nodes_from_documents(documents)

class TokenStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text purely by tokens.
    Useful when strict token limits are required by the LLM.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        print(f"Executing TokenStrategy (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        parser = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return parser.get_nodes_from_documents(documents)