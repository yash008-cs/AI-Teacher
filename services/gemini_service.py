"""
Google Gemini AI Provider Service.
Uses the official google-genai Python SDK.
Supports streaming responses, untrusted user context injection, and system instructions.
"""
import os
import time
import logging
from typing import List, Dict, Generator, Optional
from dotenv import load_dotenv

from .base_provider import BaseAIService

logger = logging.getLogger("ai_teacher.gemini")

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError, ServerError
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False
    logger.warning("google-genai SDK not installed.")


class GeminiService(BaseAIService):
    """
    Manages interactions with Google Gemini models using the official google-genai SDK.
    Provides free-tier development support with zero API cost.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        load_dotenv(override=True)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        self._fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip()
        self._active_model = self._model
        self._primary_quota_exhausted_until: float = 0.0
        self.client: Optional[genai.Client] = None
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        return "Gemini"

    @property
    def model(self) -> str:
        return self._active_model

    def _initialize_client(self):
        if not GEMINI_SDK_AVAILABLE:
            logger.warning("google-genai package is unavailable.")
            return

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized with model '{self._model}'.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            logger.info("GEMINI_API_KEY not provided. Operating in unconfigured mode.")

    def is_configured(self) -> bool:
        """Returns True if a valid Gemini client is initialized."""
        return self.client is not None

    def build_contents(
        self,
        conversation_history: List[Dict[str, str]],
        memory_context: Optional[str] = None,
    ) -> List[types.Content]:
        """
        Constructs the contents list for Gemini API.
        Enforces system instruction authority and frames Zep memory as untrusted user-level context.
        """
        contents: List[types.Content] = []

        # If retrieved memory context exists, inject as untrusted user-level context turn
        if memory_context and memory_context.strip():
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=memory_context.strip())]
                )
            )
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Noted. I have internalized this learner background and will personalize my explanations accordingly while strictly adhering to my pedagogical guidelines.")]
                )
            )

        # Append recent conversation turns
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )

        # Gemini requires that contents is not empty and ends with a user turn
        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Hello")]
                )
            )
        elif contents[-1].role == "model":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Please proceed with the teaching session based on the above context.")]
                )
            )

        return contents

    def stream_response(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        memory_context: Optional[str] = None,
        temperature: float = 0.6,
    ) -> Generator[str, None, None]:
        """
        Streams response chunks from Google Gemini.
        Yields text chunks as they arrive for responsive UI rendering.
        """
        if not self.is_configured():
            yield (
                "⚠️ **Gemini API Key Not Configured**\n\n"
                "To run the AI Teacher on Google Gemini Free Tier, please add your `GEMINI_API_KEY` "
                "to the `.env` file.\n\n"
                "You can get a free key from [Google AI Studio](https://aistudio.google.com/apikey)."
            )
            return

        contents = self.build_contents(
            conversation_history=conversation_history,
            memory_context=memory_context,
        )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        models_to_try = []
        if time.time() < self._primary_quota_exhausted_until and self._fallback_model:
            models_to_try.append(self._fallback_model)
            if self._model not in models_to_try:
                models_to_try.append(self._model)
        else:
            models_to_try.append(self._model)
            if self._fallback_model and self._fallback_model != self._model:
                models_to_try.append(self._fallback_model)

        last_error = None
        for current_model in models_to_try:
            max_retries = 2 if current_model == self._model and len(models_to_try) > 1 else 3
            for attempt in range(max_retries):
                try:
                    response_stream = self.client.models.generate_content_stream(
                        model=current_model,
                        contents=contents,
                        config=config,
                    )

                    has_yielded = False
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                            has_yielded = True

                    if has_yielded:
                        self._active_model = current_model
                        return

                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    is_quota = "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg
                    is_transient = "UNAVAILABLE" in err_msg or "503" in err_msg

                    # If primary model hits quota, seamlessly fall back to the backup model
                    if is_quota:
                        self._primary_quota_exhausted_until = time.time() + 60.0
                        if current_model != self._fallback_model and self._fallback_model:
                            logger.warning(
                                f"Primary model '{current_model}' quota exhausted. Seamlessly falling back to '{self._fallback_model}'..."
                            )
                            break

                    if is_transient and attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 2.0
                        logger.warning(
                            f"Gemini API streaming transient error ({err_msg[:60]}...). Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        continue
                    break

        if last_error:
            err_msg = str(last_error)
            logger.error(f"Gemini API streaming error: {err_msg}")
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                yield "\n\n⚠️ **Gemini Rate Limit Reached**: The free-tier request quota was momentarily exceeded. Please wait a few moments and try again."
            elif "UNAVAILABLE" in err_msg or "503" in err_msg:
                yield "\n\n⚠️ **Gemini High Demand**: The model is momentarily experiencing high demand. Please try again in a moment."
            elif "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg or "PERMISSION_DENIED" in err_msg:
                yield "\n\n❌ **Invalid Gemini API Key**: Please check that `GEMINI_API_KEY` in your `.env` file is valid."
            else:
                yield f"\n\n❌ **Gemini Error**: {err_msg}"

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generates a direct, non-streaming text response from Google Gemini.
        Used for grounded RAG generation and deterministic pedagogical responses.
        """
        if not self.is_configured():
            return (
                "⚠️ Gemini API Key Not Configured. Please add GEMINI_API_KEY to your .env file."
            )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        models_to_try = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            models_to_try.append(self._fallback_model)

        last_error = None
        for current_model in models_to_try:
            max_retries = 2 if current_model == self._model and len(models_to_try) > 1 else 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config,
                    )
                    if response.text:
                        self._active_model = current_model
                        return response.text
                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    is_quota = "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg
                    is_transient = "UNAVAILABLE" in err_msg or "503" in err_msg

                    if is_quota and current_model != self._fallback_model and self._fallback_model:
                        logger.warning(
                            f"Primary model '{current_model}' quota exhausted. Seamlessly falling back to '{self._fallback_model}'..."
                        )
                        break

                    if is_transient and attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 2.5
                        logger.warning(
                            f"Gemini API transient spike ({err_msg[:60]}...). Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        continue
                    break

        if last_error:
            err_msg = str(last_error)
            logger.error(f"Gemini generate_response error: {err_msg}")
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                return "Gemini Rate Limit Reached: The request quota was momentarily exceeded. Please try again shortly."
            elif "UNAVAILABLE" in err_msg or "503" in err_msg:
                return "Gemini High Demand: The model is momentarily experiencing high demand. Please try again shortly."
            elif "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg or "PERMISSION_DENIED" in err_msg:
                return "Invalid Gemini API Key: Please check that GEMINI_API_KEY in your .env file is valid."
            return f"Gemini Error: {err_msg}"
        return ""

    def transcribe_audio(self, audio_file: any) -> str:
        """
        Transcribes speech audio directly using Gemini's native multimodal audio capabilities.
        Free, fast, highly accurate, and has zero dependency on ElevenLabs character limits.
        """
        if not self.is_configured():
            logger.warning("GeminiService is not configured for audio transcription.")
            return ""

        raw_bytes = None
        mime_type = "audio/wav"

        if hasattr(audio_file, "getvalue"):
            raw_bytes = audio_file.getvalue()
            name = getattr(audio_file, "name", "")
            if name.endswith(".mp3"):
                mime_type = "audio/mp3"
            elif name.endswith(".webm"):
                mime_type = "audio/webm"
            elif name.endswith(".ogg"):
                mime_type = "audio/ogg"
        elif isinstance(audio_file, bytes):
            raw_bytes = audio_file
        elif hasattr(audio_file, "read"):
            raw_bytes = audio_file.read()

        if not raw_bytes:
            return ""

        models_to_try = []
        if self._fallback_model:
            models_to_try.append(self._fallback_model)
        if self._model not in models_to_try:
            models_to_try.append(self._model)

        prompt = (
            "Transcribe this spoken audio accurately. Output ONLY the transcribed speech verbatim, "
            "with no introductory text, no explanations, and no quotes. If completely inaudible or silent, return an empty string."
        )

        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=raw_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                if response and response.text:
                    cleaned = response.text.strip()
                    # Strip any accidental wrapping quotes
                    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
                        cleaned = cleaned[1:-1].strip()
                    return cleaned
            except Exception as e:
                logger.warning(f"Gemini audio transcription with model '{model_name}' failed: {e}")
                continue

        return ""


