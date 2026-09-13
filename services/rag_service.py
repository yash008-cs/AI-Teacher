"""
RAG Service.
High-level service orchestrator for document indexing and semantic retrieval.
Operates independently from the Streamlit UI and LLM response generation.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag.ingestion import DocumentIngestor, Document
from rag.chunking import TextChunker, TextChunk
from rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from rag.vector_store import LocalVectorStore
from rag.retriever import RAGRetriever, RetrievedChunk
from rag.generator import RAGGenerator, RAGResponse
from services.base_provider import BaseAIService

logger = logging.getLogger("ai_teacher.services.rag")


class RAGService:
    """
    High-level RAG service for the AI Teacher.
    Coordinates document ingestion, chunking, vector embedding generation,
    local storage persistence, semantic retrieval, and grounded response generation.
    """

    def __init__(
        self,
        knowledge_base_dir: Optional[str | Path] = None,
        vector_store_path: Optional[str | Path] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        ai_service: Optional[BaseAIService] = None,
    ):
        self.kb_dir = Path(knowledge_base_dir or "knowledge_base").resolve()
        self.vector_store_dir = Path(
            vector_store_path or os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        ).resolve()

        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = LocalVectorStore(storage_dir=self.vector_store_dir)
        self.chunker = TextChunker()
        self.ingestor = DocumentIngestor(base_dir=self.kb_dir)

        # Attempt to load existing index
        self._index_loaded = self.vector_store.load()
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )
        self.generator = RAGGenerator(ai_service=ai_service)

    def index_knowledge_base(self, custom_kb_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        """
        Executes full indexing pipeline:
        1. Scan directory & extract documents.
        2. Chunk documents.
        3. Generate embeddings.
        4. Store in vector store.
        5. Persist to disk.
        """
        target_dir = Path(custom_kb_dir).resolve() if custom_kb_dir else self.kb_dir
        logger.info(f"Starting knowledge base indexing from: {target_dir}")

        # 1. Ingest documents
        documents = self.ingestor.load_directory(target_dir)
        total_docs = len(documents)

        # Count unique filenames
        unique_files = len({doc.source_filename for doc in documents if doc.source_filename})

        if not documents:
            logger.warning("No extractable documents found in knowledge base.")
            return {
                "success": False,
                "documents_found": 0,
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "vector_store_status": "EMPTY",
            }

        # 2. Chunk documents
        chunks = self.chunker.split_documents(documents)
        total_chunks = len(chunks)

        if not chunks:
            logger.warning("Chunking produced 0 chunks.")
            return {
                "success": False,
                "documents_found": unique_files,
                "documents_processed": total_docs,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "vector_store_status": "EMPTY",
            }

        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {total_chunks} chunks using {self.embedding_provider.provider_name}...")
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_documents(chunk_texts)

        # 4. Clear and populate vector store
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        # 5. Persist index
        save_success = self.vector_store.save()
        self._index_loaded = save_success

        return {
            "success": save_success,
            "documents_found": unique_files,
            "documents_processed": total_docs,
            "chunks_created": total_chunks,
            "embeddings_generated": len(embeddings),
            "vector_store_status": "SUCCESS" if save_success else "SAVE_FAILED",
            "provider": self.embedding_provider.provider_name,
            "model": self.embedding_provider.model_name,
            "storage_path": str(self.vector_store_dir),
        }

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Retrieves top-k relevant chunks as RetrievedChunk objects.
        """
        return self.retriever.retrieve(query=query, top_k=top_k)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for a query and returns structured results as dicts.
        """
        retrieved_chunks = self.retrieve(query=query, top_k=top_k)
        return [chunk.to_dict() for chunk in retrieved_chunks]

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        learner_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end grounded RAG generation:
        1. Retrieves relevant chunks via RAGRetriever.
        2. Formats context and executes grounded generation via RAGGenerator.
        3. Returns structured answer with source attribution and retrieval scores.
        """
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k)
        rag_response = self.generator.generate(
            question=query,
            retrieved_chunks=retrieved_chunks,
            learner_name=learner_name,
        )
        return rag_response.to_dict()

    def get_status(self) -> Dict[str, Any]:
        """
        Returns diagnostic status information about the RAG system.
        """
        return {
            "is_configured": self.embedding_provider.is_configured(),
            "provider_name": self.embedding_provider.provider_name,
            "model_name": self.embedding_provider.model_name,
            "chunk_count": self.vector_store.count(),
            "storage_path": str(self.vector_store_dir),
            "index_loaded": self.vector_store.count() > 0,
            "knowledge_base_dir": str(self.kb_dir),
        }
