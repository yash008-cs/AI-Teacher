"""
Local Vector Store for AI Teacher RAG Pipeline.
Persists normalized embeddings (NumPy) and chunk metadata (JSON) locally.
Performs cosine similarity search with score ranking.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np

from .chunking import TextChunk

logger = logging.getLogger("ai_teacher.rag.vector_store")


@dataclass
class SearchResult:
    """Represents a matched chunk from similarity search with its relevance score and metadata."""
    chunk_text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    @property
    def source_filename(self) -> str:
        return self.metadata.get("source_filename", "unknown")

    @property
    def topic(self) -> str:
        return self.metadata.get("topic", "General")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number", None)


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Stores."""

    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """Adds text chunks and their corresponding embedding vectors to the store."""
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[SearchResult]:
        """Searches the vector store for the top_k most similar chunks."""
        pass

    @abstractmethod
    def save(self, directory: Optional[str | Path] = None) -> bool:
        """Persists the vector index and metadata to local storage."""
        pass

    @abstractmethod
    def load(self, directory: Optional[str | Path] = None) -> bool:
        """Loads a persisted vector index from local storage."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        pass

    @abstractmethod
    def clear(self):
        """Clears all in-memory vectors and chunks."""
        pass


class LocalVectorStore(BaseVectorStore):
    """
    Lightweight, high-performance local vector store.
    Stores embeddings in normalized NumPy float32 arrays.
    Persists vectors to embeddings.npz and chunk metadata to metadata.json.
    """

    def __init__(self, storage_dir: Optional[str | Path] = None):
        configured_dir = storage_dir or os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        self.storage_dir = Path(configured_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
        self._chunks_data: List[Dict[str, Any]] = []    # List of chunk dicts
        self._dimension: Optional[int] = None
        self._last_saved_at: Optional[str] = None

    @property
    def vectors_path(self) -> Path:
        return self.storage_dir / "embeddings.npz"

    @property
    def metadata_path(self) -> Path:
        return self.storage_dir / "metadata.json"

    def count(self) -> int:
        return len(self._chunks_data)

    def clear(self):
        """Clears all stored vectors and metadata from memory."""
        self._embeddings = None
        self._chunks_data = []
        self._dimension = None
        logger.info("LocalVectorStore in-memory index cleared.")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes vectors along axis 1 to unit length for fast cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """
        Adds chunks and corresponding embedding vectors to the local store.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: Received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        if not chunks:
            return

        new_vectors = np.array(embeddings, dtype=np.float32)
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)

        dim = new_vectors.shape[1]
        if self._dimension is not None and self._dimension != dim:
            raise ValueError(
                f"Embedding dimension mismatch: store has dim {self._dimension}, but received dim {dim}"
            )
        self._dimension = dim

        # Unit-normalize vectors
        normalized_new = self._normalize(new_vectors)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = normalized_new
        else:
            self._embeddings = np.vstack([self._embeddings, normalized_new])

        for c in chunks:
            self._chunks_data.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
            })

        logger.info(f"Added {len(chunks)} chunks. Total indexed: {self.count()} chunks (dim={dim}).")

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[SearchResult]:
        """
        Performs cosine similarity search using dot product on unit-normalized vectors.
        Returns top_k SearchResults sorted by descending score.
        """
        if self._embeddings is None or len(self._chunks_data) == 0:
            logger.warning("Vector store is empty. No results found.")
            return []

        q_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self._embeddings, q_vec.T).flatten()

        actual_k = min(top_k, len(scores))
        # Top-k indices
        if actual_k == len(scores):
            top_indices = np.argsort(-scores)
        else:
            # Fast partial sort for large sets
            partitioned = np.argpartition(-scores, actual_k)[:actual_k]
            top_indices = partitioned[np.argsort(-scores[partitioned])]

        results: List[SearchResult] = []
        for idx in top_indices:
            chunk_info = self._chunks_data[idx]
            score_val = float(scores[idx])
            results.append(
                SearchResult(
                    chunk_text=chunk_info["text"],
                    score=round(score_val, 4),
                    metadata=chunk_info.get("metadata", {}),
                    chunk_id=chunk_info.get("chunk_id", ""),
                )
            )

        return results

    def save(self, directory: Optional[str | Path] = None) -> bool:
        """
        Persists the index (embeddings.npz and metadata.json) to disk.
        """
        save_dir = Path(directory).resolve() if directory else self.storage_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._embeddings is None or len(self._chunks_data) == 0:
            logger.warning("Attempted to save empty vector store.")
            return False

        try:
            # 1. Save normalized numpy embeddings
            vec_path = save_dir / "embeddings.npz"
            np.savez_compressed(vec_path, embeddings=self._embeddings)

            # 2. Save chunk texts and metadata
            meta_path = save_dir / "metadata.json"
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_chunks": len(self._chunks_data),
                "dimension": self._dimension,
                "chunks": self._chunks_data,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self._last_saved_at = payload["created_at"]
            logger.info(f"Vector store successfully persisted to: {save_dir} ({len(self._chunks_data)} chunks).")
            return True
        except Exception as e:
            logger.error(f"Failed to persist vector store: {e}")
            return False

    def load(self, directory: Optional[str | Path] = None) -> bool:
        """
        Loads persisted index from disk.
        """
        target_dir = Path(directory).resolve() if directory else self.storage_dir
        vec_path = target_dir / "embeddings.npz"
        meta_path = target_dir / "metadata.json"

        if not vec_path.exists() or not meta_path.exists():
            logger.warning(f"Vector store files not found in: {target_dir}")
            return False

        try:
            # 1. Load vectors
            with np.load(vec_path) as data:
                self._embeddings = data["embeddings"].astype(np.float32)

            # 2. Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self._chunks_data = payload.get("chunks", [])
            self._dimension = payload.get("dimension")
            self._last_saved_at = payload.get("created_at")

            logger.info(
                f"Vector store loaded successfully from {target_dir}: "
                f"{len(self._chunks_data)} chunks, dim={self._dimension}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {target_dir}: {e}")
            return False
