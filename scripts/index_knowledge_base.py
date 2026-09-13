"""
Script: Index Knowledge Base
Scans documents, extracts text, chunks, computes embeddings, and persists the local vector store.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from services.rag_service import RAGService


def main():
    print("========================================")
    print("RAG KNOWLEDGE BASE INDEXING")
    print("========================================")

    try:
        service = RAGService(
            knowledge_base_dir=PROJECT_ROOT / "knowledge_base",
            vector_store_path=PROJECT_ROOT / "data" / "vector_store",
        )

        status = service.get_status()
        if not status["is_configured"]:
            print(f"\n[ERROR] Embedding provider '{status['provider_name']}' is not configured.")
            print("Please ensure GEMINI_API_KEY is set in your .env file.")
            sys.exit(1)

        print(f"\nUsing Embedding Provider: {status['provider_name']} ({status['model_name']})")
        print(f"Scanning Knowledge Base: {status['knowledge_base_dir']}")

        result = service.index_knowledge_base()

        print("\nIndexing Summary:")
        print(f"Documents found: {result['documents_found']}")
        print(f"Documents processed: {result['documents_processed']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Embeddings generated: {result['embeddings_generated']}")
        print(f"Vector store: {result['vector_store_status']}")
        print(f"Index storage path: {result['storage_path']}")
        print("========================================")

        if not result["success"]:
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Indexing failed: {e}")
        print("========================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
