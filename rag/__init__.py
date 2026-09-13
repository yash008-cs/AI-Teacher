"""
RAG (Retrieval-Augmented Generation) package for AI Teacher.
Provides independent document ingestion, chunking, embeddings, local vector storage, and retrieval.
"""
from .ingestion import Document, DocumentIngestor
from .chunking import TextChunk, TextChunker
from .embeddings import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from .vector_store import (
    BaseVectorStore,
    LocalVectorStore,
    SearchResult,
)
from .retriever import (
    RAGRetriever,
    RetrievedChunk,
)
from .generator import (
    RAGGenerator,
    RAGResponse,
)

__all__ = [
    "Document",
    "DocumentIngestor",
    "TextChunk",
    "TextChunker",
    "BaseEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
    "BaseVectorStore",
    "LocalVectorStore",
    "SearchResult",
    "RAGRetriever",
    "RetrievedChunk",
    "RAGGenerator",
    "RAGResponse",
]
