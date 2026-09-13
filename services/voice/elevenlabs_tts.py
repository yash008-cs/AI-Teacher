"""
ElevenLabs Text-to-Speech (TTS) Service.
Encapsulates client initialization, standard audio synthesis,
and low-latency real-time streaming TTS using the official ElevenLabs SDK.
"""
import os
import re
import queue
import threading
import time
import logging
from typing import Iterator, Optional, Dict, Any, List
from dotenv import load_dotenv

logger = logging.getLogger("ai_teacher.voice.tts")

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.core.api_error import ApiError
    ELEVENLABS_SDK_AVAILABLE = True
except ImportError:
    ELEVENLABS_SDK_AVAILABLE = False
    ApiError = Exception  # type: ignore
    logger.warning("elevenlabs package is not installed in the active environment.")


def _format_api_error(err: Exception) -> str:
    """Extracts a clean, informative error message without exposing credentials."""
    if isinstance(err, ApiError):
        status = getattr(err, "status_code", "Unknown")
        body = getattr(err, "body", {})
        if isinstance(body, dict):
            detail = body.get("detail", {})
            if isinstance(detail, dict):
                msg = detail.get("message") or detail.get("status") or str(detail)
                return f"ElevenLabs API Error [{status}]: {msg}"
            return f"ElevenLabs API Error [{status}]: {detail}"
        return f"ElevenLabs API Error [{status}]: {body}"
    return f"TTS Error: {str(err)}"


class ElevenLabsTTSService:
    """
    Reusable Text-to-Speech service powered by ElevenLabs.
    Provides isolated standard synthesis and real-time streaming audio generation.
    """

    DEFAULT_MODEL = "eleven_flash_v2_5"
    DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George (premade conversational voice)

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        load_dotenv()
        if api_key is not None:
            raw_key = api_key.strip()
        else:
            raw_key = (
                os.getenv("ELEVENLABS_API_KEY", "").strip()
                or os.getenv("ELEVNLABS_API_KEY", "").strip()
            )
        self.api_key = raw_key if raw_key and not raw_key.startswith("your_") else ""

        if voice_id is not None:
            self.voice_id = voice_id.strip() or self.DEFAULT_VOICE_ID
        else:
            self.voice_id = (
                os.getenv("ELEVENLABS_VOICE_ID", "").strip()
                or self.DEFAULT_VOICE_ID
            )

        if model is not None:
            self.model = model.strip() or self.DEFAULT_MODEL
        else:
            self.model = (
                os.getenv("ELEVENLABS_MODEL", "").strip()
                or self.DEFAULT_MODEL
            )

        self.client: Optional[ElevenLabs] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the official ElevenLabs SDK client."""
        if not ELEVENLABS_SDK_AVAILABLE:
            logger.warning("elevenlabs package is not available.")
            return

        if self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                logger.info(
                    f"ElevenLabs client initialized (Model: {self.model}, Voice: {self.voice_id})."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs client: {_format_api_error(e)}")
                self.client = None
        else:
            logger.info("ELEVENLABS_API_KEY not configured. Operating in unconfigured mode.")

    def is_configured(self) -> bool:
        """Returns True if a valid ElevenLabs client is initialized."""
        return self.client is not None

    def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        output_format: str = "mp3_44100_128",
    ) -> bytes:
        """
        Synthesizes text into audio bytes using standard ElevenLabs TTS.

        Args:
            text: Text to convert to speech.
            voice_id: Optional voice ID override.
            model: Optional model ID override.
            output_format: Desired audio format (default: mp3_44100_128).

        Returns:
            bytes: Complete audio binary data.

        Raises:
            ValueError: If input is empty or service is unconfigured.
            RuntimeError: If synthesis fails or API returns an error.
        """
        if not self.is_configured():
            raise ValueError(
                "ElevenLabs TTS service is not configured. Please ensure ELEVENLABS_API_KEY is set in .env"
            )

        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("Text cannot be empty.")

        target_voice = voice_id or self.voice_id
        target_model = model or self.model

        try:
            audio_stream = self.client.text_to_speech.convert(
                voice_id=target_voice,
                text=cleaned_text,
                model_id=target_model,
                output_format=output_format,
            )
            audio_bytes = b"".join(audio_stream)
            if not audio_bytes:
                raise RuntimeError("ElevenLabs TTS returned empty audio data.")
            return audio_bytes

        except Exception as e:
            formatted_err = _format_api_error(e)
            logger.error(f"TTS conversion failed: {formatted_err}")
            raise RuntimeError(formatted_err) from e

    def text_to_speech_stream(
        self,
        text_stream: Iterator[str],
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        output_format: str = "mp3_44100_128",
    ) -> Iterator[bytes]:
        """
        Converts an incoming text stream (such as tokens from Gemini LLM)
        into a real-time stream of audio byte chunks using ElevenLabs WebSocket streaming.

        Args:
            text_stream: Iterator yielding text fragments.
            voice_id: Optional voice ID override.
            model: Optional model ID override.
            output_format: Desired audio format (default: mp3_44100_128).

        Yields:
            bytes: Audio chunks as they stream in from ElevenLabs.

        Raises:
            ValueError: If service is unconfigured.
            RuntimeError: If streaming fails or API returns an error.
        """
        if not self.is_configured():
            raise ValueError(
                "ElevenLabs TTS service is not configured. Please ensure ELEVENLABS_API_KEY is set in .env"
            )

        target_voice = voice_id or self.voice_id
        target_model = model or self.model

        try:
            audio_stream = self.client.text_to_speech.convert_realtime(
                voice_id=target_voice,
                text=text_stream,
                model_id=target_model,
                output_format=output_format,
            )
            for chunk in audio_stream:
                if chunk:
                    yield chunk

        except Exception as e:
            formatted_err = _format_api_error(e)
            logger.error(f"Streaming TTS failed: {formatted_err}")
            raise RuntimeError(formatted_err) from e


def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, emojis, math markers, and code blocks
    so the synthesized speech sounds natural, fluent, and pedagogical.
    """
    if not text:
        return ""
    # Strip markdown headers
    t = re.sub(r"#+\s*", "", text)
    # Strip bold/italic markers
    t = re.sub(r"\*+", "", t)
    # Strip inline backticks
    t = re.sub(r"`+", "", t)
    # Strip LaTeX math delimiters
    t = re.sub(r"\$\$|\$", "", t)
    # Strip emojis
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    # Strip horizontal rules
    t = re.sub(r"---+", "", t)
    return t


class VoiceStreamPipe:
    """
    Concurrently pipes a single LLM token stream into both:
    1. A UI display stream (yielding raw tokens with markdown formatting).
    2. A background ElevenLabs TTS streaming pipeline (generating audio chunks).

    Guarantees single LLM generator consumption, zero duplicate requests,
    and graceful error fallback if TTS encounters an issue.
    """

    def __init__(
        self,
        tts_service: Optional[ElevenLabsTTSService] = None,
        enabled: bool = True,
    ):
        self.tts_service = tts_service
        self.enabled = bool(enabled and tts_service and tts_service.is_configured())
        self.queue: queue.Queue[Optional[str]] = queue.Queue()
        self.audio_chunks: List[bytes] = []
        self.tts_thread: Optional[threading.Thread] = None
        self.first_chunk_latency: Optional[float] = None
        self.total_tts_time: Optional[float] = None
        self.error: Optional[Exception] = None

    def _tts_worker(self):
        """Worker thread that consumes cleaned tokens and streams them to ElevenLabs."""
        try:
            def _tts_token_gen():
                while True:
                    tok = self.queue.get()
                    if tok is None:
                        break
                    cleaned = clean_text_for_speech(tok)
                    if cleaned:
                        yield cleaned

            t0 = time.perf_counter()
            for chunk in self.tts_service.text_to_speech_stream(_tts_token_gen()):
                if self.first_chunk_latency is None:
                    self.first_chunk_latency = time.perf_counter() - t0
                self.audio_chunks.append(chunk)

            self.total_tts_time = time.perf_counter() - t0
        except Exception as e:
            logger.warning(f"VoiceStreamPipe TTS worker error: {e}")
            self.error = e

    def pipe(self, source_generator: Iterator[str]) -> Iterator[str]:
        """
        Wraps the source generator:
        - Yields each raw token for the text display stream (e.g. st.write_stream).
        - Concurrently feeds tokens into the background TTS pipeline.
        """
        if self.enabled:
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

        try:
            for token in source_generator:
                if self.enabled:
                    self.queue.put(token)
                yield token
        finally:
            if self.enabled:
                self.queue.put(None)

    def get_audio(self, timeout: float = 12.0) -> Optional[bytes]:
        """
        Waits for background TTS streaming to complete and returns assembled audio bytes.
        Returns None if TTS is disabled or encountered an error.
        """
        if not self.enabled or not self.tts_thread:
            return None

        self.tts_thread.join(timeout=timeout)
        if self.audio_chunks:
            return b"".join(self.audio_chunks)
        return None

