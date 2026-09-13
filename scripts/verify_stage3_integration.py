"""
Stage 3 End-to-End Application Integration Verification Script.
Validates that MemoryService, RAGService, Zep, and Gemini 3.7 Flash
are unified properly in the AI Teacher pipeline.
"""
import sys
import os
from pathlib import Path

# Ensure utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from services import ZepService, MemoryService, get_ai_service, RAGService
from prompts.teacher_prompt import format_unified_context

def main():
    print("=" * 60)
    print("STAGE 3: FULL AI TEACHER APPLICATION INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize Services
    print("\n[1/4] Initializing AI Teacher Services...")
    zep_service = ZepService()
    ai_service = get_ai_service()
    rag_service = RAGService()
    memory_service = MemoryService(
        zep_service=zep_service,
        ai_service=ai_service,
        rag_service=rag_service,
    )

    print(f"  • AI Provider: {ai_service.provider_name} (Model: {ai_service.model})")
    print(f"  • Zep Memory Configured: {zep_service.is_configured()}")
    print(f"  • RAG Knowledge Base Chunks: {rag_service.vector_store.count()}")

    assert rag_service.vector_store.count() > 0, "Vector store has no indexed chunks!"
    expected_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    assert ai_service.model == expected_model, f"Unexpected model: {ai_service.model}"

    # 2. Test RAG Retrieval Directly
    print("\n[2/4] Testing RAG Semantic Retrieval...")
    query = "What is supervised learning and what are some examples?"
    retrieved_chunks = rag_service.retrieve(query, top_k=3)
    print(f"  Query: '{query}'")
    print(f"  Retrieved Chunks: {len(retrieved_chunks)}")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"    [{i}] Score: {chunk.score:.4f} | Source: {chunk.metadata.get('source_filename')} | Topic: {chunk.metadata.get('topic')}")

    assert len(retrieved_chunks) > 0, "No chunks retrieved!"
    assert retrieved_chunks[0].score > 0.4, f"Top chunk score too low: {retrieved_chunks[0].score}"

    # 3. Test Unified Pipeline (MemoryService + RAG + Gemini Streaming)
    print("\n[3/4] Testing Unified Context Construction & Streaming Execution...")
    test_user_id = "test-learner-stage3"
    test_thread_id = "test-thread-stage3"
    learner_name = "Alex"

    # Initialize Zep user & thread if configured
    if zep_service.is_configured():
        zep_service.get_or_create_user(user_id=test_user_id, first_name=learner_name)
        zep_service.create_thread(user_id=test_user_id, thread_id=test_thread_id)

    raw_context, stream, sources = memory_service.retrieve_context_and_stream_response(
        thread_id=test_thread_id,
        user_message=query,
        learner_name=learner_name,
        conversation_history=[],
    )

    print(f"  • Unified context constructed successfully. (Length: {len(raw_context) if raw_context else 0} chars)")
    print(f"  • Source attribution items returned: {len(sources)}")
    assert len(sources) > 0, "Expected sources to be populated!"

    for s in sources:
        print(f"    - Source: {s['source_filename']} (Topic: {s['topic']}, Score: {s['score']:.2f})")

    # 4. Stream Response
    print(f"\n[4/4] Streaming AI Teacher Response ({ai_service.model}):")
    print("-" * 50)
    full_response = ""
    for chunk in stream:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        full_response += chunk
    print("\n" + "-" * 50)

    assert len(full_response) > 50, "Generated response is too short!"
    print("\n✅ Verification Successful: Grounded response received with source attribution and pedagogical formatting.")
    print("=" * 60)

if __name__ == "__main__":
    main()
