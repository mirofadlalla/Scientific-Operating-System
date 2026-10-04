"""Abstract chunking strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from llama_index.core.schema import BaseNode, Document


class BaseChunkingStrategy(ABC):
    """Base class for all chunking strategies.

    Any new strategy must implement :meth:`chunk`.
    """

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        """Parse ``documents`` into nodes."""
