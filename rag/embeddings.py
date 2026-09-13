"""
Embeddings Service for AI Teacher RAG Pipeline.
Provides an isolated, provider-agnostic embedding interface with official Google Gemini support.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger("ai_teacher.rag.embeddings")

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False
    logger.warning("google-genai SDK not available.")

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    logger.warning("openai SDK not available.")


class BaseEmbeddingProvider(ABC):
    """Abstract Base Class for embedding providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the provider has valid credentials configured."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of text chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single search query."""
        pass


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google Gemini Embedding Provider using the official google-genai SDK.
    Generates high-dimensional semantic embeddings (default: gemini-embedding-001).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        self._model = (
            model
            or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
        )
        self.client: Optional[genai.Client] = None
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "Gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _initialize_client(self):
        if not GEMINI_SDK_AVAILABLE:
            logger.warning("google-genai package is unavailable.")
            return

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini embedding client initialized with model '{self._model}'.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini embedding client: {e}")
                self.client = None
        else:
            logger.warning("GEMINI_API_KEY not configured. Operating in unconfigured mode.")

    def is_configured(self) -> bool:
        return self.client is not None

    def embed_documents(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        Embeds a list of document chunks, batching requests to respect API limits.
        """
        if not self.is_configured():
            raise RuntimeError(
                "Gemini embedding provider is not configured. Please ensure GEMINI_API_KEY is set in .env."
            )

        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Process in batches to prevent payload size limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                res = self.client.models.embed_content(
                    model=self._model,
                    contents=batch,
                )
                if not res.embeddings:
                    raise ValueError("Gemini API returned empty embeddings list.")

                for emb in res.embeddings:
                    all_embeddings.append(list(emb.values))

            except Exception as e:
                logger.error(f"Error generating Gemini embeddings for batch {i//batch_size}: {e}")
                raise

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string for vector search."""
        if not self.is_configured():
            raise RuntimeError(
                "Gemini embedding provider is not configured. Please ensure GEMINI_API_KEY is set in .env."
            )

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        try:
            res = self.client.models.embed_content(
                model=self._model,
                contents=cleaned_query,
            )
            if not res.embeddings:
                raise ValueError("Gemini API returned empty query embedding.")

            return list(res.embeddings[0].values)
        except Exception as e:
            logger.error(f"Error generating Gemini query embedding: {e}")
            raise


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI Embedding Provider using the official openai SDK.
    Serves as an interchangeable alternative provider (default: text-embedding-3-small).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        self.client: Optional[OpenAI] = None
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_name(self) -> str:
        return self._model

    def _initialize_client(self):
        if not OPENAI_SDK_AVAILABLE:
            logger.warning("openai package is unavailable.")
            return

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI embedding client initialized with model '{self._model}'.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding client: {e}")
                self.client = None
        else:
            logger.warning("OPENAI_API_KEY not configured.")

    def is_configured(self) -> bool:
        return self.client is not None

    def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if not self.is_configured():
            raise RuntimeError("OpenAI embedding provider is not configured.")

        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self._model,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(list(item.embedding))

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        if not self.is_configured():
            raise RuntimeError("OpenAI embedding provider is not configured.")

        response = self.client.embeddings.create(
            model=self._model,
            input=query.strip(),
        )
        return list(response.data[0].embedding)


def get_embedding_provider(provider: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory function to instantiate the configured embedding provider.
    Defaults to Gemini for our Gemini-based architecture.
    Configurable via EMBEDDING_PROVIDER in .env ('gemini' or 'openai').
    """
    chosen = (provider or os.getenv("EMBEDDING_PROVIDER", "gemini")).strip().lower()

    if chosen == "openai":
        logger.info("Instantiating OpenAI embedding provider.")
        return OpenAIEmbeddingProvider()

    logger.info("Instantiating Google Gemini embedding provider.")
    return GeminiEmbeddingProvider()
