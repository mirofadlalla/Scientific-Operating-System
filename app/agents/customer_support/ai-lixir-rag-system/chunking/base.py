from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document, BaseNode

class BaseChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.
    Any new strategy must implement the 'chunk' method.
    """
    
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        """
        Takes a list of documents and returns a list of parsed nodes.
        """
        pass