"""
Document Ingestion System for AI Teacher RAG Pipeline.
Supports PDF, TXT, and Markdown (.md) documents with metadata preservation.
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ai_teacher.rag.ingestion")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf is not installed. PDF ingestion will be unavailable.")


@dataclass
class Document:
    """Represents an extracted document or document page with associated metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_filename(self) -> str:
        return self.metadata.get("source_filename", "")

    @property
    def source_path(self) -> str:
        return self.metadata.get("source_path", "")

    @property
    def file_type(self) -> str:
        return self.metadata.get("file_type", "")

    @property
    def topic(self) -> str:
        return self.metadata.get("topic", "General")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number", None)


class DocumentIngestor:
    """
    Scans and ingests documents from the knowledge base directory.
    Extracts text, preserves metadata, and logs clear warnings for unsupported/corrupted files.
    """
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

    def __init__(self, base_dir: Optional[str | Path] = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def _determine_topic(self, file_path: Path) -> str:
        """
        Infers the topic category from the immediate parent directory name
        relative to the knowledge base root.
        """
        parent_name = file_path.parent.name
        if self.base_dir and file_path.is_relative_to(self.base_dir):
            relative_parts = file_path.relative_to(self.base_dir).parts
            if len(relative_parts) > 1:
                return relative_parts[0].replace("_", " ")
        if parent_name and parent_name not in ("knowledge_base", "data", ".", ""):
            return parent_name.replace("_", " ")
        return "General"

    def _load_text_file(self, file_path: Path, file_type: str) -> List[Document]:
        """Loads and extracts text from UTF-8 Markdown or plain text files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"File is empty: {file_path}")
                return []

            topic = self._determine_topic(file_path)
            doc = Document(
                content=content,
                metadata={
                    "source_filename": file_path.name,
                    "source_path": str(file_path),
                    "file_type": file_type,
                    "topic": topic,
                    "page_number": None,
                },
            )
            return [doc]
        except Exception as e:
            logger.error(f"Failed to read text file '{file_path}': {e}")
            return []

    def _load_pdf_file(self, file_path: Path) -> List[Document]:
        """Loads and extracts text page-by-page from a PDF document using pypdf."""
        if not PYPDF_AVAILABLE:
            logger.warning(f"Skipping PDF '{file_path.name}': pypdf library is not installed.")
            return []

        documents: List[Document] = []
        topic = self._determine_topic(file_path)

        try:
            reader = pypdf.PdfReader(str(file_path))
            num_pages = len(reader.pages)
            if num_pages == 0:
                logger.warning(f"PDF contains 0 pages: {file_path}")
                return []

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    doc = Document(
                        content=page_text,
                        metadata={
                            "source_filename": file_path.name,
                            "source_path": str(file_path),
                            "file_type": "pdf",
                            "topic": topic,
                            "page_number": page_idx + 1,
                            "total_pages": num_pages,
                        },
                    )
                    documents.append(doc)

            if not documents:
                logger.warning(f"No extractable text found in PDF: {file_path}")

            return documents
        except Exception as e:
            logger.error(f"Corrupted or unreadable PDF document '{file_path}': {e}")
            return []

    def load_file(self, file_path: str | Path) -> List[Document]:
        """Loads a single document file if supported."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            logger.warning(f"Target path does not exist or is not a file: {path}")
            return []

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"Unsupported document format '{suffix}' for file: {path.name}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
            return []

        if suffix == ".md":
            return self._load_text_file(path, file_type="markdown")
        elif suffix == ".txt":
            return self._load_text_file(path, file_type="text")
        elif suffix == ".pdf":
            return self._load_pdf_file(path)
        return []

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """
        Recursively scans directory for documents.
        Processes supported files (.pdf, .txt, .md) and warns for unsupported files.
        """
        target_dir = Path(dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            logger.error(f"Knowledge base directory does not exist: {target_dir}")
            return []

        self.base_dir = target_dir
        all_documents: List[Document] = []
        ignored_files = {".gitkeep", ".ds_store", "thumbs.db"}

        for item in sorted(target_dir.rglob("*")):
            if item.is_file():
                if item.name.lower() in ignored_files or item.name.startswith("."):
                    continue

                suffix = item.suffix.lower()
                if suffix in self.SUPPORTED_EXTENSIONS:
                    docs = self.load_file(item)
                    all_documents.extend(docs)
                else:
                    logger.warning(
                        f"Unsupported file format '{suffix}' detected at: {item.relative_to(target_dir)}. "
                        f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
                    )

        return all_documents
