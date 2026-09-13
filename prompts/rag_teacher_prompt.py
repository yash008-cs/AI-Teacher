"""
Dedicated RAG Teacher Prompt & Context Formatter.
Enforces strict grounding on retrieved knowledge base materials,
prevents hallucinations, and maintains pedagogical clarity.
"""
from typing import List, Dict, Any, Optional

RAG_TEACHER_SYSTEM_PROMPT = """You are a warm, articulate, and encouraging female AI Teacher and personal tutor.
Your mission is to help the learner understand concepts with clarity, intuition, and intellectual depth.

CRITICAL GROUNDING & FACTUALITY RULES:
1. Ground your answer strictly and primarily in the provided [KNOWLEDGE BASE CONTEXT].
2. Treat the provided context as your authoritative curriculum and reference material.
3. If the [KNOWLEDGE BASE CONTEXT] does NOT contain sufficient information to answer the student's question, or if the question is on a topic outside the provided context, you MUST explicitly and politely state:
   "Based on the course materials currently in my knowledge base, I do not have enough information to answer that question."
4. NEVER fabricate, assume, or hallucinate facts that are not grounded in the provided context or core academic principles directly supported by it.
5. Do not pretend to know details that are missing from the reference documents.

PEDAGOGICAL GUIDELINES:
1. Start with a crystal-clear, simple definition of the concept.
2. Provide an intuitive explanation of why the concept exists and what problem it solves.
3. Use concrete details from the course materials (such as algorithm names, formulas, or comparison tables).
4. If appropriate, highlight key contrasts (e.g., discrete vs. continuous targets).
5. Maintain an encouraging, mentorship tone suitable for a student.
6. When referencing facts, attribute them naturally to the course materials or topics (e.g., "According to our Machine Learning notes...").
"""


def format_rag_context(chunks: List[Any]) -> str:
    """
    Formats a list of retrieved chunks (RetrievedChunk objects or dicts)
    into a clean, structured context block with clear provenance headers.
    """
    if not chunks:
        return "No relevant context found in the knowledge base."

    context_blocks: List[str] = []
    for idx, c in enumerate(chunks, 1):
        if hasattr(c, "text"):
            text = c.text
            source = c.source_filename
            topic = c.topic
            page = c.page_number
            score = c.score
        elif isinstance(c, dict):
            text = c.get("text", "")
            source = c.get("source_filename", "unknown")
            topic = c.get("topic", "General")
            page = c.get("page_number", None)
            score = c.get("score", 0.0)
        else:
            continue

        page_info = f"Page {page}" if page is not None else "Page N/A"
        header = f"[Excerpt {idx} | Source: {source} | Topic: {topic} | {page_info} | Relevance: {score:.4f}]"
        context_blocks.append(f"{header}\n{text.strip()}")

    joined = "\n\n---\n\n".join(context_blocks)
    return f"=== KNOWLEDGE BASE CONTEXT START ===\n\n{joined}\n\n=== KNOWLEDGE BASE CONTEXT END ==="


def build_rag_user_prompt(question: str, context_block: str) -> str:
    """
    Combines the student question and formatted context block
    into a single user prompt for the grounded AI Teacher.
    """
    return (
        f"{context_block}\n\n"
        f"STUDENT QUESTION:\n{question.strip()}\n\n"
        f"Please provide an educational, clear, and grounded response to the student's question "
        f"based strictly on the context provided above."
    )


def get_rag_teacher_system_prompt(learner_name: Optional[str] = None) -> str:
    """Returns the RAG teacher system prompt, optionally addressing the learner."""
    prompt = RAG_TEACHER_SYSTEM_PROMPT
    if learner_name and learner_name.strip():
        prompt += f"\n\nCurrent Student Name: {learner_name.strip()}\nAddress the student warmly."
    return prompt
