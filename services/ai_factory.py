"""
AI Provider Factory.
Instantiates the appropriate AI service (Google Gemini or OpenAI) based on environment configuration.
"""
import os
import logging
from typing import Optional

from .base_provider import BaseAIService
from .gemini_service import GeminiService
from .openai_service import OpenAIService

logger = logging.getLogger("ai_teacher.factory")


def get_ai_service(provider: Optional[str] = None) -> BaseAIService:
    """
    Returns an instance of the configured AI provider.
    Defaults to Gemini for zero-cost development.
    Configurable via LLM_PROVIDER in .env ('gemini' or 'openai').
    """
    chosen_provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()

    if chosen_provider == "openai":
        logger.info("Initializing OpenAI service as active AI provider.")
        return OpenAIService()

    # Default to Gemini for zero API cost development
    logger.info("Initializing Google Gemini service as active AI provider.")
    return GeminiService()
