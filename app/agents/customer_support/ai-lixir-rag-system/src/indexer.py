import weaviate
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import BaseNode

class VectorIndexManager:
    """
    Manager class responsible for connecting to Weaviate, ingesting data, 
    and managing the Hybrid Vector Store Index.
    """
    
    def __init__(self, index_name: str = "AdmetIndex"):
        """
        Initializes the Weaviate client and sets the target index name.
        """
        self.index_name = index_name
        print("Connecting to Weaviate instance...")
        
        # Connect to a local embedded Weaviate instance.
        # Note: In production, use weaviate.connect_to_weaviate_cloud() or connect_to_custom()
        self.client = weaviate.connect_to_local()
        
    def _get_vector_store(self) -> WeaviateVectorStore:
        """
        Helper method to initialize the WeaviateVectorStore.
        """

        # "استخدم Weaviate كـ storage layer بدل الـ SimpleVectorStore"
        # ربط LlamaIndex بـ Weaviate
        return WeaviateVectorStore(
            weaviate_client=self.client, 
            index_name=self.index_name
        )

    def create_and_save_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        Builds a VectorStoreIndex from the provided nodes and ingests them into Weaviate.
        """
        print(f"Initializing Weaviate Hybrid Index with {len(nodes)} nodes...")
        print("Generating embeddings via Groq API and storing in Weaviate...")
        
        vector_store = self._get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index and ingest data into Weaviate
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context
        )
        
        print("✅ Data successfully ingested into Weaviate Vector Database.")
        return index

    def load_persisted_index(self) -> VectorStoreIndex:
        """
        Loads an existing index directly from Weaviate.
        """
        print(f"Connecting to existing Weaviate index: '{self.index_name}'...")
        vector_store = self._get_vector_store()
        
        # Reconstruct the index from the existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store)
        print("✅ Weaviate index successfully loaded.")
        return index

    def close_connection(self):
        """
        Closes the connection to Weaviate. Should be called when shutting down.
        """
        self.client.close()
        print("Weaviate connection closed.")