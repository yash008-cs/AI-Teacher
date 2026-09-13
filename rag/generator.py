"""
RAG Generator Module for AI Teacher.
Assembles grounded context, formats the pedagogical RAG prompt,
calls the Gemini LLM (gemini-3.8-flash), and returns structured answers with source attribution.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from prompts.rag_teacher_prompt import (
    get_rag_teacher_system_prompt,
    format_rag_context,
    build_rag_user_prompt,
)
from services.base_provider import BaseAIService
from services.ai_factory import get_ai_service
from .retriever import RetrievedChunk

logger = logging.getLogger("ai_teacher.rag.generator")


@dataclass
class RAGResponse:
    """Represents a grounded educational response from the RAG pipeline."""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context_used: str = ""
    top_similarity_score: float = 0.0
    is_grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_chunks": self.retrieved_chunks,
            "top_similarity_score": self.top_similarity_score,
            "is_grounded": self.is_grounded,
            "context_used": self.context_used,
        }


class RAGGenerator:
    """
    Grounded RAG Generator:
    Takes a student question and retrieved chunks -> generates a factual pedagogical response.
    """

    def __init__(
        self,
        ai_service: Optional[BaseAIService] = None,
        min_similarity_threshold: float = 0.40,
    ):
        self.ai_service = ai_service or get_ai_service()
        self.min_similarity_threshold = min_similarity_threshold

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        learner_name: Optional[str] = None,
    ) -> RAGResponse:
        """
        Executes grounded generation:
        1. Extracts source metadata from retrieved chunks.
        2. Formats the educational knowledge base context.
        3. Invokes Gemini with strict grounding instructions.
        4. Packages response with full provenance.
        """
        cleaned_question = question.strip()

        # 1. Extract source metadata & chunk summaries
        sources: List[Dict[str, Any]] = []
        chunk_dicts: List[Dict[str, Any]] = []
        top_score = 0.0

        for c in retrieved_chunks:
            chunk_score = float(c.score)
            if chunk_score > top_score:
                top_score = chunk_score

            sources.append({
                "source_filename": c.source_filename,
                "topic": c.topic,
                "page_number": c.page_number,
                "score": round(chunk_score, 4),
                "chunk_id": c.chunk_id,
            })
            chunk_dicts.append(c.to_dict())

        # 2. Check for empty retrieval
        if not retrieved_chunks:
            logger.info("No chunks retrieved. Generating direct out-of-context response.")
            no_info_answer = (
                "Based on the course materials currently in my knowledge base, "
                "I do not have enough information to answer that question. "
                "Please add relevant study notes or documents to the knowledge base so I can teach you this topic!"
            )
            return RAGResponse(
                question=cleaned_question,
                answer=no_info_answer,
                sources=[],
                retrieved_chunks=[],
                context_used="",
                top_similarity_score=0.0,
                is_grounded=True,
            )

        # 3. Format structured context block
        context_block = format_rag_context(retrieved_chunks)

        # 4. Prepare system prompt and user prompt
        system_instruction = get_rag_teacher_system_prompt(learner_name=learner_name)
        user_prompt = build_rag_user_prompt(question=cleaned_question, context_block=context_block)

        logger.info(
            f"Invoking {self.ai_service.provider_name} ({self.ai_service.model}) "
            f"for grounded answer (top similarity score: {top_score:.4f})..."
        )

        # 5. Call LLM for direct grounded response
        raw_answer = self.ai_service.generate_response(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.2,
        )

        return RAGResponse(
            question=cleaned_question,
            answer=raw_answer.strip(),
            sources=sources,
            retrieved_chunks=chunk_dicts,
            context_used=context_block,
            top_similarity_score=round(top_score, 4),
            is_grounded=True,
        )
