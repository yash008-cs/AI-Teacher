"""Prompts package for AI Teacher."""
from .teacher_prompt import (
    get_teacher_system_prompt,
    format_memory_context,
    format_unified_context,
)
from .rag_teacher_prompt import (
    RAG_TEACHER_SYSTEM_PROMPT,
    get_rag_teacher_system_prompt,
    format_rag_context,
    build_rag_user_prompt,
)

__all__ = [
    "get_teacher_system_prompt",
    "format_memory_context",
    "format_unified_context",
    "RAG_TEACHER_SYSTEM_PROMPT",
    "get_rag_teacher_system_prompt",
    "format_rag_context",
    "build_rag_user_prompt",
]
