import weaviate
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import BaseNode

class VectorIndexManager:
    _GLOBAL_IN_MEMORY_INDEX = None

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
        
        # Connect to a local or remote Weaviate instance based on config.
        from src.config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
        try:
            self.client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
        except Exception as e:
            print(f"[RAGAgent] Initialisation failed: Connection to Weaviate failed. Details: {e}")
            self.client = None
        
    def _get_vector_store(self) -> WeaviateVectorStore:
        """
        Helper method to initialize the WeaviateVectorStore.
        """

        # "استخدم Weaviate كـ storage layer بدل الـ SimpleVectorStore"
        # ربط LlamaIndex بـ Weaviate
        if self.client is None:
            return None

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
        if vector_store:
            index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context
            )
        else:
            index = VectorStoreIndex(nodes)
            VectorIndexManager._GLOBAL_IN_MEMORY_INDEX = index
        
        print("✅ Data successfully ingested into Weaviate Vector Database.")
        return index

    def load_persisted_index(self) -> VectorStoreIndex:
        """
        Loads an existing index directly from Weaviate.
        """
        if self.client is None:
            raise Exception("Weaviate is not connected. Cannot load persisted index.")

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
        if self.client:
            self.client.close()
            print("Weaviate connection closed.")