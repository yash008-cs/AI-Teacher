"""
Memory and Workflow Coordinator.
Connects Zep long-term memory retrieval, OpenAI pedagogical generation,
and persistent interaction storage.
"""
import logging
from typing import Generator, List, Dict, Tuple, Optional, Any

from prompts.teacher_prompt import get_teacher_system_prompt, format_memory_context, format_unified_context
from prompts.rag_teacher_prompt import format_rag_context
from .zep_service import ZepService
from .base_provider import BaseAIService
from .rag_service import RAGService

logger = logging.getLogger("ai_teacher.memory_service")


class MemoryService:
    """
    Coordinates the pedagogical interaction cycle:
    User Message -> Zep Ingestion -> Zep Memory Retrieval + RAG Retrieval -> Gemini Streaming -> Assistant Ingestion.
    """

    def __init__(
        self,
        zep_service: ZepService,
        ai_service: BaseAIService,
        rag_service: Optional[RAGService] = None,
    ):
        self.zep_service = zep_service
        self.ai_service = ai_service
        self.rag_service = rag_service

    def initialize_learner_session(self, learner_name: str, user_id: str, thread_id: str) -> bool:
        """
        Initializes the learner profile and session thread in Zep.
        """
        user_res = self.zep_service.get_or_create_user(user_id=user_id, first_name=learner_name)
        thread_res = self.zep_service.create_thread(user_id=user_id, thread_id=thread_id)
        logger.info(f"Learner session initialized. User: {user_id}, Thread: {thread_id}")
        return True

    def retrieve_context_and_stream_response(
        self,
        thread_id: str,
        user_message: str,
        learner_name: str,
        conversation_history: List[Dict[str, str]],
        voice_mode: bool = False,
    ) -> Tuple[Optional[str], Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Executes Steps 4 through 7 of the pedagogical interaction flow:
        1. Ingests user message into Zep.
        2. Retrieves relevant long-term memory context from Zep.
        3. Retrieves relevant curriculum context from RAG Knowledge Base.
        4. Formats unified prompt with system guidelines + RAG curriculum + Zep memory context.
        5. Yields streamed tokens from Gemini/OpenAI.
        6. Returns (raw_zep_context, stream, retrieved_sources).
        """
        import concurrent.futures

        raw_zep_context: Optional[str] = None
        rag_context: Optional[str] = None
        retrieved_sources: List[Dict[str, Any]] = []

        # High-Speed Parallel Execution: Ingestion, Zep context retrieval, and RAG search run concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Dispatch Zep user turn ingestion asynchronously (does not block retrieval)
            executor.submit(
                self.zep_service.add_message,
                thread_id=thread_id,
                role="user",
                content=user_message,
                name=learner_name,
            )

            # Concurrently fetch Zep learner context
            fut_zep = executor.submit(self.zep_service.get_user_context, thread_id=thread_id)

            # Concurrently fetch RAG knowledge chunks
            fut_rag = None
            if self.rag_service and self.rag_service.vector_store.count() > 0:
                fut_rag = executor.submit(self.rag_service.retriever.retrieve, query=user_message, top_k=3)

            try:
                raw_zep_context = fut_zep.result(timeout=4.0)
            except Exception as e:
                logger.warning(f"Zep context fetch error: {e}")

            if fut_rag:
                try:
                    chunks = fut_rag.result(timeout=4.0)
                    if chunks:
                        rag_context = format_rag_context(chunks)
                        for c in chunks:
                            retrieved_sources.append({
                                "source_filename": c.source_filename,
                                "topic": c.topic,
                                "page_number": c.page_number,
                                "score": round(float(c.score), 4),
                                "chunk_id": c.chunk_id,
                                "snippet": c.text.strip()[:160].replace("\n", " "),
                            })
                        logger.info(f"RAG retrieved {len(chunks)} chunks in parallel.")
                except Exception as e:
                    logger.warning(f"RAG parallel retrieval error: {e}")

        # Step 6: Prepare unified context and system prompt
        unified_context = format_unified_context(
            memory_context=raw_zep_context,
            rag_context=rag_context,
        )
        system_prompt = get_teacher_system_prompt(learner_name=learner_name)
        if voice_mode:
            system_prompt += (
                "\n\n🎙️ [VOICE CONVERSATION MODE ACTIVE]:\n"
                "You are speaking aloud to the student over audio.\n"
                "- Speak warmly, conversationally, and concisely in 2 to 3 clear sentences (around 40-55 words).\n"
                "- Do NOT use markdown tables, bulleted lists, LaTeX math formulas, or bold headers, as these do not sound natural when spoken aloud.\n"
                "- Provide a clear, intuitive answer and naturally invite the student to explore further if they wish."
            )

        # Step 7: Stream response from active AI provider (Gemini)
        stream = self.ai_service.stream_response(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            memory_context=unified_context,
        )

        return unified_context, stream, retrieved_sources

    def record_assistant_response(self, thread_id: str, content: str) -> bool:
        """
        Step 9: Stores completed AI response in Zep for temporal memory building.
        """
        return self.zep_service.add_message(
            thread_id=thread_id,
            role="assistant",
            content=content,
            name="AI Teacher",
        )
