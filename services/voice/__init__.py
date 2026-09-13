"""
Voice Services Module for AI Teacher.
Provides text-to-speech, real-time speech-to-text, and voice streaming pipeline integrations using ElevenLabs.
"""
from .elevenlabs_tts import (
    ElevenLabsTTSService,
    VoiceStreamPipe,
    clean_text_for_speech,
)
from .elevenlabs_stt import ElevenLabsSTTService
from .neural_tts import synthesize_speech_neural

__all__ = [
    "ElevenLabsTTSService",
    "ElevenLabsSTTService",
    "VoiceStreamPipe",
    "clean_text_for_speech",
    "synthesize_speech_neural",
]

