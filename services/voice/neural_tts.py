"""
Neural Text-to-Speech (TTS) Service using Edge Neural Speech.
Provides high-fidelity, natural female AI Teacher voice synthesis
with zero monthly quota limits and no API key requirements.
Serves as an automatic, seamless fallback when ElevenLabs quota is exhausted.
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("ai_teacher.voice.neural_tts")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts package is not installed.")

from .elevenlabs_tts import clean_text_for_speech

DEFAULT_NEURAL_VOICE = "en-US-JennyNeural"  # Natural, articulate female tutor voice


async def _async_generate_audio(text: str, voice: str) -> bytes:
    """Async generation of MP3 audio bytes using edge-tts."""
    cleaned = clean_text_for_speech(text)
    if not cleaned or not cleaned.strip():
        return b""

    communicate = edge_tts.Communicate(text=cleaned.strip(), voice=voice, rate="+10%")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def synthesize_speech_neural(
    text: str,
    voice: Optional[str] = None,
) -> Optional[bytes]:
    """
    Synthesizes speech audio from text using Microsoft Edge Neural TTS.
    Returns MP3 audio bytes ready for browser playback (st.audio).

    Args:
        text: Response text to speak.
        voice: Optional voice name (defaults to 'en-US-JennyNeural').

    Returns:
        bytes of MP3 audio, or None if synthesis fails.
    """
    if not EDGE_TTS_AVAILABLE:
        logger.warning("edge-tts is unavailable.")
        return None

    if not text or not text.strip():
        return None

    target_voice = voice or DEFAULT_NEURAL_VOICE

    try:
        audio_bytes = asyncio.run(_async_generate_audio(text, target_voice))
        if audio_bytes:
            logger.info(f"Synthesized {len(audio_bytes):,} bytes of speech with {target_voice}.")
            return audio_bytes
    except Exception as e:
        logger.error(f"Neural TTS synthesis error: {e}")

    return None
