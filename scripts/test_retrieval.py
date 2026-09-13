"""
Script: Test Retrieval
Tests semantic search and retriever functionality independently from Gemini LLM generation.
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


def run_retrieval_test(query: str, top_k: int = 2):
    print("========================================")
    print("RAG RETRIEVAL TEST")
    print("========================================")
    print("\nQuery:")
    print(query)

    try:
        service = RAGService(
            knowledge_base_dir=PROJECT_ROOT / "knowledge_base",
            vector_store_path=PROJECT_ROOT / "data" / "vector_store",
        )

        status = service.get_status()
        if not status["index_loaded"] or status["chunk_count"] == 0:
            print("\n[ERROR] Vector store is empty or index not found.")
            print("Please run scripts/index_knowledge_base.py first.")
            sys.exit(1)

        results = service.search(query=query, top_k=top_k)

        if not results:
            print("\nNo matching chunks found.")
        else:
            for idx, res in enumerate(results, 1):
                page_str = str(res.get("page_number")) if res.get("page_number") is not None else "N/A"
                print(f"\nResult {idx}:")
                print(f"Source: {res.get('source_filename', 'N/A')}")
                print(f"Topic: {res.get('topic', 'N/A')}")
                print(f"Page: {page_str}")
                print(f"Score: {res.get('score', 0.0):.4f}")
                print(f"Text:\n{res.get('text', '').strip()}")

        print("\n========================================")

    except Exception as e:
        print(f"\n[ERROR] Retrieval test failed: {e}")
        print("========================================")
        sys.exit(1)


def main():
    # Support custom query via CLI argument, defaulting to "What is supervised learning?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is supervised learning?"

    run_retrieval_test(query=test_query, top_k=2)


if __name__ == "__main__":
    main()
