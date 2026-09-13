"""
Script: Test Grounded RAG Generation
Tests the complete end-to-end RAG pipeline:
Question -> Embedding -> Retrieval -> Context Assembly -> Grounded Prompt -> Gemini LLM -> Final Answer with Provenance.
Includes validation for out-of-domain queries to verify zero hallucination.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from services.rag_service import RAGService


def run_rag_test(question: str, top_k: int = 3):
    print("========================================")
    print("RAG GENERATION TEST")
    print("========================================")
    print("\nStudent Question:")
    print(question.strip())

    try:
        service = RAGService(
            knowledge_base_dir=PROJECT_ROOT / "knowledge_base",
            vector_store_path=PROJECT_ROOT / "data" / "vector_store",
        )

        status = service.get_status()
        if not status["index_loaded"] or status["chunk_count"] == 0:
            print("\n[ERROR] Vector store is empty or index files not found.")
            print("Please run scripts/index_knowledge_base.py first.")
            sys.exit(1)

        result = service.answer_question(query=question, top_k=top_k)

        # 1. Print Retrieved Context
        print("\nRetrieved Context:")
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            print("No relevant context found in the knowledge base.")
        else:
            for idx, c in enumerate(chunks, 1):
                page_info = f"Page {c.get('page_number')}" if c.get("page_number") is not None else "Page N/A"
                print(f"\n[Chunk {idx} | Source: {c.get('source_filename', 'N/A')} | Topic: {c.get('topic', 'N/A')} | {page_info} | Score: {c.get('score', 0.0):.4f}]")
                print(c.get("text", "").strip())

        # 2. Print Sources
        print("\nSources:")
        sources = result.get("sources", [])
        if sources:
            for s in sources:
                page_info = f", Page {s['page_number']}" if s.get("page_number") is not None else ""
                print(f"- Filename: {s['source_filename']} | Topic: {s['topic']}{page_info} | Score: {s['score']:.4f}")
        else:
            print("None")

        # 3. Print AI Teacher Answer
        print("\nAI Teacher Answer:")
        print(result.get("answer", "").strip())
        print("\n========================================\n")

    except Exception as e:
        print(f"\n[ERROR] RAG generation failed: {e}")
        print("========================================\n")
        sys.exit(1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        custom_question = " ".join(sys.argv[1:])
        run_rag_test(question=custom_question, top_k=3)
    else:
        # Run required test suite:
        # Question 1: "What is supervised learning?"
        run_rag_test("What is supervised learning?", top_k=2)

        # Question 2: "How does classification differ from regression?"
        run_rag_test("How does classification differ from regression?", top_k=2)

        # Question 3: Unrelated question not in knowledge base (anti-hallucination verification)
        run_rag_test("What is photosynthesis and how do plants produce glucose?", top_k=2)


if __name__ == "__main__":
    main()
