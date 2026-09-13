"""
Chunking Module for AI Teacher RAG Pipeline.
Performs boundary-aware text splitting preserving paragraphs, sentences, and document metadata.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .ingestion import Document

logger = logging.getLogger("ai_teacher.rag.chunking")


@dataclass
class TextChunk:
    """Represents an individual text chunk with associated document metadata and chunk index."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    @property
    def source_filename(self) -> str:
        return self.metadata.get("source_filename", "")

    @property
    def topic(self) -> str:
        return self.metadata.get("topic", "General")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number", None)

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)


class TextChunker:
    """
    Boundary-aware recursive text chunker.
    Splits along markdown headers, paragraphs, sentences, and word boundaries
    rather than blindly slicing text mid-token.
    
    Default configuration:
      chunk_size: 600 characters (~100-150 words)
      chunk_overlap: 100 characters (~15-25 words)
    """

    DEFAULT_SEPARATORS = [
        "\n## ",     # Markdown H2
        "\n### ",    # Markdown H3
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence end
        "? ",        # Question end
        "! ",        # Exclamation end
        "; ",        # Semicolon clause
        " ",         # Word boundary
        "",          # Character fallback
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        # Allow environment overrides or sensible defaults
        env_size = os.getenv("RAG_CHUNK_SIZE")
        env_overlap = os.getenv("RAG_CHUNK_OVERLAP")

        self.chunk_size = chunk_size or (int(env_size) if env_size and env_size.isdigit() else 600)
        self.chunk_overlap = chunk_overlap or (int(env_overlap) if env_overlap and env_overlap.isdigit() else 100)
        self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller than chunk_size ({self.chunk_size})"
            )

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Splits text on a specific separator while preserving delimiter cues where appropriate."""
        if not separator:
            return list(text)
        return text.split(separator)

    def _split_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Hierarchically splits text down to acceptable chunk sizes respecting semantic boundaries."""
        text = text.strip()
        if len(text) <= self.chunk_size or not separators:
            return [text] if text else []

        sep = separators[0]
        remaining_seps = separators[1:]
        splits = self._split_text_with_separator(text, sep)

        result: List[str] = []
        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                sub_splits = self._split_recursively(piece, remaining_seps)
                result.extend(sub_splits)

        return result

    def _merge_splits_with_overlap(self, pieces: List[str]) -> List[str]:
        """
        Merges atomic pieces into chunks up to chunk_size, maintaining chunk_overlap
        between consecutive chunks.
        """
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in pieces:
            piece_len = len(piece)
            # Add 2 for joining spacing if needed
            prospective_len = current_length + piece_len + (2 if current_chunk else 0)

            if prospective_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length = prospective_len
            else:
                if current_chunk:
                    merged = "\n\n".join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)

                    # Build overlap from tail pieces
                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for prev_piece in reversed(current_chunk):
                        if overlap_len + len(prev_piece) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_piece)
                            overlap_len += len(prev_piece) + 2
                        else:
                            break

                    current_chunk = overlap_chunk
                    current_length = sum(len(p) + 2 for p in current_chunk)

                current_chunk.append(piece)
                current_length += piece_len + (2 if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = "\n\n".join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """Splits raw text string into chunks."""
        if not text or not text.strip():
            return []
        pieces = self._split_recursively(text, self.separators)
        return self._merge_splits_with_overlap(pieces)

    def split_document(self, document: Document) -> List[TextChunk]:
        """Splits a Document into TextChunk instances, propagating all metadata and assigning IDs."""
        raw_chunks = self.split_text(document.content)
        chunks: List[TextChunk] = []

        filename = document.source_filename or "doc"
        page_suffix = f"_p{document.page_number}" if document.page_number else ""

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_metadata = dict(document.metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["chunk_size_chars"] = len(chunk_text)

            chunk_id = f"{filename}{page_suffix}#chunk_{idx:03d}"
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
            )

        return chunks

    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Batch splits multiple documents into a flat list of TextChunk instances."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            doc_chunks = self.split_document(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks
