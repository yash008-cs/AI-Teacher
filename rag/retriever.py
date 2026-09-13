"""
Retriever Module for AI Teacher RAG Pipeline.
Coordinates query embedding generation, similarity search, and structured retrieval results.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .embeddings import BaseEmbeddingProvider, get_embedding_provider
from .vector_store import BaseVectorStore, LocalVectorStore, SearchResult

logger = logging.getLogger("ai_teacher.rag.retriever")


@dataclass
class RetrievedChunk:
    """Structured result returned by the retriever."""
    text: str
    score: float
    source_filename: str
    topic: str
    page_number: Optional[int] = None
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a dictionary."""
        return {
            "text": self.text,
            "score": self.score,
            "source_filename": self.source_filename,
            "topic": self.topic,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


class RAGRetriever:
    """
    Orchestrates semantic retrieval:
    Query -> Query Embedding -> Vector Store Search -> Ranked Relevant Chunks.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        default_top_k: int = 3,
        min_score_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store or LocalVectorStore()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.default_top_k = default_top_k
        self.min_score_threshold = min_score_threshold

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Takes a natural language query, generates its embedding,
        queries the vector store, and returns structured relevant chunks.
        """
        k = top_k or self.default_top_k
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            logger.warning("Empty query passed to retriever.")
            return []

        logger.info(f"Generating query embedding for: '{cleaned_query[:60]}...'")
        query_vector = self.embedding_provider.embed_query(cleaned_query)

        search_results: List[SearchResult] = self.vector_store.similarity_search(
            query_embedding=query_vector,
            top_k=k,
        )

        retrieved: List[RetrievedChunk] = []
        for res in search_results:
            if self.min_score_threshold is not None and res.score < self.min_score_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    text=res.chunk_text,
                    score=res.score,
                    source_filename=res.source_filename,
                    topic=res.topic,
                    page_number=res.page_number,
                    chunk_id=res.chunk_id,
                    metadata=res.metadata,
                )
            )

        logger.info(f"Retrieved {len(retrieved)} relevant chunks for query.")
        return retrieved
