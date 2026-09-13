"""
OpenAI LLM Service.
Integrates with the official OpenAI Python SDK.
Handles pedagogical completions with streaming support.
"""
import os
import logging
from typing import List, Dict, Generator, Optional, Any

from .base_provider import BaseAIService

logger = logging.getLogger("ai_teacher.openai")

try:
    from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    logger.warning("openai SDK not installed.")


class OpenAIService(BaseAIService):
    """
    Manages interactions with OpenAI Chat Completion models.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o").strip()
        self.client: Optional[OpenAI] = None
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model(self) -> str:
        return self._model

    def _initialize_client(self):
        if not OPENAI_SDK_AVAILABLE:
            logger.warning("openai package is unavailable.")
            return

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model '{self.model}'.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.info("OPENAI_API_KEY not provided. Operating in mock mode.")

    def is_configured(self) -> bool:
        """Returns True if a valid OpenAI client is initialized."""
        return self.client is not None

    def build_chat_payload(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        memory_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Assembles the payload for OpenAI Chat Completions API.
        Enforces system instruction authority and frames Zep memory as untrusted context.
        """
        payload = [
            {"role": "system", "content": system_prompt}
        ]

        if memory_context and memory_context.strip():
            # Ingest retrieved memory safely as untrusted user-level context
            payload.append({
                "role": "user",
                "content": memory_context.strip()
            })
            payload.append({
                "role": "assistant",
                "content": "Noted. I have internalized this learner context and will personalize my explanations accordingly while strictly adhering to my pedagogical guidelines."
            })

        # Append recent thread conversation history
        for msg in conversation_history:
            payload.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return payload

    def stream_response(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        memory_context: Optional[str] = None,
        temperature: float = 0.6,
    ) -> Generator[str, None, None]:
        """
        Streams response chunks from OpenAI.
        Yields text chunks as they arrive for responsive UI rendering.
        """
        if not self.is_configured():
            yield (
                "⚠️ **OpenAI API Key Not Configured**\n\n"
                "Please configure your `OPENAI_API_KEY` in the `.env` file to enable the AI Teacher."
            )
            return

        messages = self.build_chat_payload(system_prompt, conversation_history, memory_context)

        try:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in response_stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content

        except AuthenticationError:
            yield "\n\n❌ **Authentication Error**: The provided OpenAI API key is invalid. Please check your `.env` configuration."
        except RateLimitError:
            yield "\n\n⚠️ **Rate Limit Exceeded**: OpenAI rate limit or quota exceeded. Please check your account balance or try again shortly."
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            yield f"\n\n⚠️ **OpenAI Service Notice**: Unable to complete request at this moment. ({type(e).__name__})"
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI completion: {e}")
            yield f"\n\n⚠️ **Connection Notice**: An unexpected error occurred while communicating with the AI service."

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generates a direct, non-streaming text response from OpenAI.
        Used for grounded RAG generation and deterministic pedagogical responses.
        """
        if not self.is_configured():
            return "⚠️ OpenAI API Key Not Configured. Please add OPENAI_API_KEY to your .env file."

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generate_response error: {e}")
            return f"❌ OpenAI Error: {e}"
