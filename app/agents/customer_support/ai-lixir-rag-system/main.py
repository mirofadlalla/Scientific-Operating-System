import os
from src.config import setup_groq_environment
from chunking import factory as ChunkingFactory
from llama_index.core import SimpleDirectoryReader
from src.indexer import VectorIndexManager
from src.engine import RAGEngineBuilder

def main():
    print("==================================================")
    print("Starting Professional Groq + Weaviate RAG System")
    print("==================================================")
    
    # Step 1: Setup Groq Environment (LLM & Embeddings)
    setup_groq_environment()
    
    # Step 2: Initialize the Vector Index Manager for Weaviate
    # Change 'AdmetIndex' to whatever name you want for your schema
    index_manager = VectorIndexManager(index_name="AdmetIndex")
    
    index = None
    
    # Check if index already exists in Weaviate to avoid redundant embedding generation
    # LlamaIndex will safely handle this or you can dynamically choose to rebuild
    rebuild_index = False # Set to True if you added new data files
    
    if not rebuild_index:
        try:
            index = index_manager.load_persisted_index()
        except Exception as e:
            print(f"⚠️ Could not load existing index, preparing to build a new one. Details: {e}")
    
    if index is None:
        # Step 3: Load raw files from data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Created empty '{data_dir}' folder. Please place your markdown files there and rerun.")
            return
            
        print(f"📥 Loading documents from '{data_dir}'...")
        documents = SimpleDirectoryReader(data_dir, required_exts=[".md"]).load_data()
        
        if not documents:
            print("❌ No markdown (.md) documents found in the 'data' directory. Exiting.")
            return

        # Step 4: Chunking using Strategy & Factory Patterns
        # Using the specific 'markdown' strategy we created for your "##" structure
        chunking_strategy = ChunkingFactory.get_strategy("markdown")
        nodes = chunking_strategy.chunk(documents)
        
        # Step 5: Build and Save Index into Weaviate
        index = index_manager.create_and_save_index(nodes)

    # Step 6: Build the Advanced Hybrid Query Engine
    engine_builder = RAGEngineBuilder(index=index)
    # alpha=0.5 guarantees 50% semantic vector search and 50% keyword search
    query_engine = engine_builder.build_hybrid_query_engine(top_k=4, alpha=0.5)
    
    print("\n==================================================")
    print("🤖 System is ready! Type 'exit' to quit.")
    print("==================================================\n")
    
    # Interactive CLI loop for testing
    while True:
        user_query = input("❓ Enter your question: ")
        if user_query.strip().lower() == 'exit':
            print("Shutting down RAG system. Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        print("\n🔍 Searching and generating response...")
        response = query_engine.query(user_query)
        
        print("\n✨ Answer:")
        print(response)
        print("\n" + "-"*50 + "\n")
        
    # Clean up connections
    index_manager.close_connection()

if __name__ == "__main__":
    main()