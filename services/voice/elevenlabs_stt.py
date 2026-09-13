"""
ElevenLabs Real-Time Speech-to-Text (STT) Service.
Integrates with ElevenLabs Scribe Realtime (scribe_v2_realtime)
over Secure WebSockets for low-latency streaming transcription,
voice activity detection (VAD), partial transcripts, and committed utterances.
"""
import io
import os
import json
import base64
import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List
from dotenv import load_dotenv

logger = logging.getLogger("ai_teacher.voice.stt")

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.realtime import (
        AudioFormat,
        CommitStrategy,
        RealtimeEvents,
        RealtimeConnection,
        RealtimeAudioOptions,
    )
    ELEVENLABS_SDK_AVAILABLE = True
except ImportError:
    ELEVENLABS_SDK_AVAILABLE = False
    AudioFormat = None  # type: ignore
    CommitStrategy = None  # type: ignore
    RealtimeEvents = None  # type: ignore
    RealtimeConnection = None  # type: ignore
    logger.warning("elevenlabs SDK or realtime module is unavailable.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.info("sounddevice library not installed. Microphone capture will be unavailable.")


class ElevenLabsSTTService:
    """
    Service for real-time speech transcription using ElevenLabs Scribe Realtime.
    Supports audio streaming via WebSocket, server-side VAD, partial updates,
    and committed final transcripts.
    """

    DEFAULT_MODEL = "scribe_v2_realtime"
    DEFAULT_SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 100  # 100ms per audio chunk for low latency

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        vad_silence_threshold_secs: float = 0.8,
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

        self.model_id = (
            model_id
            or os.getenv("ELEVENLABS_STT_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
        self.sample_rate = sample_rate
        self.vad_silence_threshold_secs = vad_silence_threshold_secs

        self.client: Optional[ElevenLabs] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the ElevenLabs client if SDK and key are available."""
        if not ELEVENLABS_SDK_AVAILABLE:
            logger.warning("elevenlabs package is not available.")
            return

        if self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                logger.info(f"ElevenLabs STT client ready with model '{self.model_id}'.")
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs STT client: {e}")
                self.client = None
        else:
            logger.info("ELEVENLABS_API_KEY not configured for STT.")

    def is_configured(self) -> bool:
        """Returns True if the client is configured with a valid key."""
        return self.client is not None

    def transcribe_file(self, audio_file: Any, model_id: Optional[str] = "scribe_v1") -> str:
        """
        Transcribes an uploaded audio file, BytesIO, raw bytes, or audio file path using ElevenLabs Scribe STT.

        Args:
            audio_file: File-like object (e.g. Streamlit UploadedFile, io.BytesIO), raw audio bytes, or path.
            model_id: Model ID, defaults to 'scribe_v1'.

        Returns:
            The transcribed text string.
        """
        if not self.is_configured():
            raise ValueError(
                "ElevenLabs STT service is not configured. Please ensure ELEVENLABS_API_KEY is set in .env"
            )

        target_model = model_id or "scribe_v1"
        try:
            if hasattr(audio_file, "getvalue"):
                b = io.BytesIO(audio_file.getvalue())
                b.name = getattr(audio_file, "name", "recording.wav") or "recording.wav"
                b.seek(0)
                file_to_send = b
            elif isinstance(audio_file, bytes):
                b = io.BytesIO(audio_file)
                b.name = "recording.wav"
                b.seek(0)
                file_to_send = b
            else:
                file_to_send = audio_file

            response = self.client.speech_to_text.convert(
                file=file_to_send,
                model_id=target_model,
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            logger.error(f"Failed to transcribe audio file with ElevenLabs Scribe: {e}")
            raise

    async def connect(
        self,
        commit_strategy: str = "vad",
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> RealtimeConnection:
        """
        Establishes an asynchronous WebSocket connection with ElevenLabs Scribe Realtime.

        Args:
            commit_strategy: 'vad' (server-side Voice Activity Detection) or 'manual'.
            on_partial: Callback for intermediate/unstable partial transcripts.
            on_final: Callback for committed, finalized utterance transcripts.
            on_error: Callback for connection or transcription error events.

        Returns:
            RealtimeConnection instance.

        Raises:
            ValueError: If unconfigured or unsupported options.
            RuntimeError: If connection cannot be established.
        """
        if not self.is_configured():
            raise ValueError(
                "ElevenLabs STT service is not configured. Please ensure ELEVENLABS_API_KEY is set in .env"
            )

        strategy = CommitStrategy.VAD if commit_strategy.lower() == "vad" else CommitStrategy.MANUAL

        options = {
            "model_id": self.model_id,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": self.sample_rate,
            "commit_strategy": strategy,
        }

        if strategy == CommitStrategy.VAD:
            options["vad_silence_threshold_secs"] = self.vad_silence_threshold_secs

        try:
            connection = await self.client.speech_to_text.realtime.connect(options)
        except Exception as e:
            logger.error(f"Failed to connect to Scribe Realtime: {e}")
            raise RuntimeError(f"ElevenLabs Real-time STT connection failed: {e}") from e

        # Register event handlers
        if on_partial:
            connection.on(
                RealtimeEvents.PARTIAL_TRANSCRIPT,
                lambda data: on_partial(data.get("text", "")),
            )

        if on_final:
            connection.on(
                RealtimeEvents.COMMITTED_TRANSCRIPT,
                lambda data: on_final(data.get("text", "")),
            )

        if on_error:
            connection.on(RealtimeEvents.ERROR, on_error)

        return connection

    async def send_pcm_chunk(self, connection: RealtimeConnection, pcm_chunk: bytes):
        """
        Sends a raw 16-bit 16kHz PCM audio chunk to the active connection.
        Encodes bytes to base64 before sending.
        """
        if not pcm_chunk:
            return
        b64_audio = base64.b64encode(pcm_chunk).decode("utf-8")
        await connection.send({"audio_base_64": b64_audio})

    async def transcribe_audio_stream(
        self,
        audio_stream,
        commit_strategy: str = "vad",
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """
        Streams PCM audio chunks through Scribe Realtime and gathers committed transcripts.

        Args:
            audio_stream: Async iterable yielding raw PCM bytes chunks.
            commit_strategy: 'vad' or 'manual'.
            on_partial: Callback for partial transcripts.
            on_final: Callback for committed transcripts.

        Returns:
            List of committed final transcript strings.
        """
        committed_transcripts: List[str] = []

        def _handle_final(text: str):
            if text and text.strip():
                committed_transcripts.append(text.strip())
                if on_final:
                    on_final(text.strip())

        connection = await self.connect(
            commit_strategy=commit_strategy,
            on_partial=on_partial,
            on_final=_handle_final,
        )

        try:
            async for chunk in audio_stream:
                await self.send_pcm_chunk(connection, chunk)

            if commit_strategy == "manual":
                await connection.commit()

            # Brief grace period for server to finalize processing
            await asyncio.sleep(0.5)

        finally:
            await connection.close()

        return committed_transcripts

    def record_and_transcribe_microphone(
        self,
        max_duration_seconds: float = 10.0,
        silence_timeout_seconds: float = 2.0,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Synchronous helper: Records from system microphone, streams PCM chunks
        in real-time to ElevenLabs Scribe, and returns the committed utterance.

        Args:
            max_duration_seconds: Maximum recording duration.
            silence_timeout_seconds: Auto-stops recording after this much silence following speech.
            on_partial: Callback for live partial transcripts.
            on_final: Callback when final transcript is committed.

        Returns:
            The committed final transcript string.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Microphone recording requires the 'sounddevice' package. "
                "Please run: pip install sounddevice"
            )

        return asyncio.run(
            self._async_record_and_transcribe(
                max_duration_seconds=max_duration_seconds,
                silence_timeout_seconds=silence_timeout_seconds,
                on_partial=on_partial,
                on_final=on_final,
            )
        )

    async def _async_record_and_transcribe(
        self,
        max_duration_seconds: float = 10.0,
        silence_timeout_seconds: float = 2.0,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Internal asynchronous microphone streaming implementation."""
        final_result: List[str] = []
        speech_detected_event = asyncio.Event()
        done_event = asyncio.Event()

        def _handle_partial(text: str):
            if text.strip():
                speech_detected_event.set()
                if on_partial:
                    on_partial(text.strip())

        def _handle_final(text: str):
            if text.strip():
                final_result.append(text.strip())
                if on_final:
                    on_final(text.strip())
                done_event.set()

        connection = await self.connect(
            commit_strategy="vad",
            on_partial=_handle_partial,
            on_final=_handle_final,
        )

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        chunk_samples = int(self.sample_rate * (self.CHUNK_DURATION_MS / 1000.0))

        def _sd_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"SoundDevice status: {status}")
            # indata is numpy float32 or int16. Using dtype='int16' yields raw 16-bit PCM bytes
            raw_pcm = indata.tobytes()
            loop.call_soon_threadsafe(audio_queue.put_nowait, raw_pcm)

        # Open system microphone input stream (16kHz mono int16)
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            callback=_sd_callback,
        )

        async def _stream_sender():
            start_time = asyncio.get_event_loop().time()
            last_speech_time = None

            while not done_event.is_set():
                now = asyncio.get_event_loop().time()
                if now - start_time > max_duration_seconds:
                    done_event.set()
                    break

                if speech_detected_event.is_set():
                    if last_speech_time is None:
                        last_speech_time = now

                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    await self.send_pcm_chunk(connection, chunk)
                except asyncio.TimeoutError:
                    pass

        try:
            with stream:
                sender_task = asyncio.create_task(_stream_sender())
                # Wait for commit event or maximum timeout
                await asyncio.wait(
                    [sender_task, asyncio.create_task(done_event.wait())],
                    timeout=max_duration_seconds + 1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                done_event.set()
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass

                # Brief wait for pending network responses
                await asyncio.sleep(0.4)

        finally:
            await connection.close()

        return " ".join(final_result).strip()
