"""
Abstract Base Provider for AI/LLM Services.
Enables pluggable model providers (Google Gemini, OpenAI, etc.).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional


class BaseAIService(ABC):
    """
    Abstract interface for AI model providers.
    All providers must implement this contract to coordinate with MemoryService.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'Gemini', 'OpenAI')."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Active model identifier (e.g., 'gemini-3.8-flash', 'gpt-4o')."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the provider is properly configured with an API key."""
        pass

    @abstractmethod
    def stream_response(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        memory_context: Optional[str] = None,
        temperature: float = 0.6,
    ) -> Generator[str, None, None]:
        """
        Streams generated response chunks from the model.
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generates a direct, non-streaming text response from the model.
        Used for grounded RAG generation and deterministic evaluations.
        """
        pass

    def transcribe_audio(self, audio_file: any) -> str:
        """
        Optional multimodal audio transcription. Default returns empty string.
        """
        return ""
