from .strategies import MarkdownStrategy, SentenceStrategy, TokenStrategy
from .base import BaseChunkingStrategy

class ChunkingFactory:
    """
    Factory class to instantiate the appropriate chunking strategy.
    """
    
    @staticmethod
    def get_strategy(strategy_type: str, **kwargs) -> BaseChunkingStrategy:
        """
        Returns the requested chunking strategy.
        
        Args:
            strategy_type (str): 'markdown', 'sentence', or 'token'
            **kwargs: Additional parameters like chunk_size and chunk_overlap
        """
        strategy_type = strategy_type.strip().lower()
        
        if strategy_type == "markdown":
            return MarkdownStrategy()
        elif strategy_type == "sentence":
            return SentenceStrategy(**kwargs)
        elif strategy_type == "token":
            return TokenStrategy(**kwargs)
        else:
            raise ValueError(f"Unsupported chunking strategy type: {strategy_type}")