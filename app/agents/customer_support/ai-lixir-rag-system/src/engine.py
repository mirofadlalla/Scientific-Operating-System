from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import PromptTemplate

class RAGEngineBuilder:
    """
    Builder class to configure and construct an advanced Query Engine
    utilizing Hybrid Search (Vector + BM25) and custom system prompts.
    """
    
    def __init__(self, index: VectorStoreIndex):
        """
        Initializes the builder with a prepared VectorStoreIndex.
        """
        if index is None:
            raise ValueError("Cannot build query engine without a valid VectorStoreIndex.")
        self.index = index

    def build_hybrid_query_engine(self, top_k: int = 4, alpha: float = 0.5) -> BaseQueryEngine:
        """
        Configures and returns a Query Engine optimized for Weaviate Hybrid Search.
        
        Args:
            top_k (int): The number of top relevant text chunks to retrieve.
            alpha (float): Hybrid search weight balance. 
                           0.0 = Pure Keyword (BM25) Search.
                           1.0 = Pure Vector (Semantic) Search.
                           0.5 = Equal combination of both.
        """
        print(f"Configuring Hybrid Query Engine parameters (Top-K: {top_k}, Alpha: {alpha})...")
        
        # 1. Design a production-grade custom QA prompt template to eliminate hallucinations
        custom_qa_prompt_str = (
            "Context information is provided strictly below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and NOT prior knowledge, "
            "answer the user query accurately, structurally, and professionally.\n"
            "If the answer cannot be found or inferred directly from the provided context, "
            "honestly state that the information is not available in the documentation.\n\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        custom_qa_prompt = PromptTemplate(custom_qa_prompt_str)

        # 2. Build the query engine leveraging Weaviate's hybrid search mode
        query_engine = self.index.as_query_engine(
            vector_store_query_mode="hybrid",
            alpha=alpha,
            similarity_top_k=top_k,
            text_qa_template=custom_qa_prompt
        )
        
        print("✅ Hybrid Query Engine built successfully and wired to Groq LLM.")
        return query_engine