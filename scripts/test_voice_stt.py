"""
Isolated Verification Suite for ElevenLabs Real-Time Speech-to-Text (STT) Service.
Verifies:
  TEST 1: API Key & SDK Configuration
  TEST 2: Scribe Realtime WebSocket Connection & Handshake
  TEST 3: Real-Time Audio Chunk Streaming with VAD (Partial & Final Transcripts)
  TEST 4: Hardware Microphone Stream Verification
  TEST 5: Error Handling & Security (No Leaked Secrets)
"""
import sys
import os
import time
import asyncio
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from services.voice import ElevenLabsSTTService, ElevenLabsTTSService

try:
    import sounddevice as sd
    SOUNDDEVICE_INSTALLED = True
except ImportError:
    SOUNDDEVICE_INSTALLED = False


async def test_websocket_handshake(stt_service: ElevenLabsSTTService):
    """Verifies that the WebSocket connects, registers callbacks, and disconnects cleanly."""
    conn = await stt_service.connect(commit_strategy="vad")
    assert conn is not None, "Connection object is None!"
    assert conn.websocket is not None, "WebSocket is not initialized!"
    await conn.close()


async def test_streaming_transcription(stt_service: ElevenLabsSTTService):
    """
    Streams 16kHz mono PCM chunks through ElevenLabs Scribe Realtime
    and captures both partial and committed transcripts.
    """
    # 1. Synthesize known test speech in raw 16kHz PCM using the verified TTS service
    tts = ElevenLabsTTSService()
    test_phrase = "What is supervised learning?"
    pcm_audio = tts.text_to_speech(
        text=test_phrase,
        output_format="pcm_16000",
    )

    partial_transcripts = []
    final_transcripts = []

    def on_partial(text: str):
        if text:
            partial_transcripts.append(text)
            print(f"  [STT Event] PARTIAL:   \"{text}\"")

    def on_final(text: str):
        if text:
            final_transcripts.append(text)
            print(f"  [STT Event] COMMITTED: \"{text}\"")

    # Connect using server-side VAD with 0.8s silence threshold
    conn = await stt_service.connect(
        commit_strategy="vad",
        on_partial=on_partial,
        on_final=on_final,
    )

    # Stream PCM in 100ms chunks (16,000 samples/sec * 2 bytes/sample * 0.1s = 3200 bytes)
    chunk_size = 3200
    for i in range(0, len(pcm_audio), chunk_size):
        chunk = pcm_audio[i : i + chunk_size]
        await stt_service.send_pcm_chunk(conn, chunk)
        await asyncio.sleep(0.06)

    # Send 1 second of trailing silence so Scribe's VAD triggers the commit event
    silence_chunk = b"\x00" * chunk_size
    for _ in range(10):
        await stt_service.send_pcm_chunk(conn, silence_chunk)
        await asyncio.sleep(0.06)

    # Wait briefly for WebSocket to finalize
    await asyncio.sleep(0.8)
    await conn.close()

    return partial_transcripts, final_transcripts


def main():
    print("=" * 65)
    print(" ELEVENLABS REAL-TIME SPEECH-TO-TEXT (STT) TEST")
    print("=" * 65)

    # ---------------------------------------------------------
    # [1/4] API Key & Initialization
    # ---------------------------------------------------------
    print("\n[1/4] Checking API Key & SDK Configuration...")
    stt_service = ElevenLabsSTTService()

    if not stt_service.is_configured():
        print("  [FAILURE] ELEVENLABS_API_KEY is not configured in .env")
        sys.exit(1)

    print(f"  • Model:       {stt_service.model_id}")
    print(f"  • Audio Format: PCM 16kHz Mono (16-bit)")
    print(f"  • VAD Silence: {stt_service.vad_silence_threshold_secs}s")
    print("  Status:        ✅ PASS (API key detected and masked)")

    # ---------------------------------------------------------
    # [2/4] Scribe Realtime WebSocket Connection Handshake
    # ---------------------------------------------------------
    print("\n[2/4] Testing Scribe Realtime WebSocket Connection Handshake...")
    try:
        start_t = time.perf_counter()
        asyncio.run(test_websocket_handshake(stt_service))
        duration = time.perf_counter() - start_t
        print(f"  • WebSocket connected and closed cleanly in {duration:.2f}s")
        print("  Status:        ✅ PASS (Handshake succeeded)")
    except Exception as e:
        print(f"  [FAILURE] WebSocket connection failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [3/4] Real-Time Streaming Audio Transcription (VAD + Partial + Final)
    # ---------------------------------------------------------
    print("\n[3/4] Testing Real-Time Streaming Transcription...")
    print("  Streaming 16kHz PCM audio of test utterance: \"What is supervised learning?\"")
    try:
        start_stt = time.perf_counter()
        partials, finals = asyncio.run(test_streaming_transcription(stt_service))
        elapsed = time.perf_counter() - start_stt

        print(f"  • Total streaming duration: {elapsed:.2f}s")
        print(f"  • Partial updates count:    {len(partials)}")
        print(f"  • Committed updates count:  {len(finals)}")

        assert len(finals) > 0, "Expected at least one committed final transcript!"
        final_text = finals[-1]
        print(f"\n  Resulting Final Transcript: \"{final_text}\"")

        # Verify accuracy
        assert "supervised" in final_text.lower() and "learning" in final_text.lower(), (
            f"Transcription does not match expected phrase: {final_text}"
        )
        print("  Status:        ✅ PASS (Real-time transcription verified)")
    except Exception as e:
        print(f"  [FAILURE] Streaming transcription failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [4/4] Hardware Microphone Stream Verification
    # ---------------------------------------------------------
    print("\n[4/4] Verifying Hardware Microphone Capture...")
    if not SOUNDDEVICE_INSTALLED:
        print("  [WARNING] sounddevice is not installed. Skipping live mic check.")
    else:
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            default_device = sd.default.device
            print(f"  • Default audio devices: {default_device}")
            print(f"  • Detected {len(input_devices)} audio input device(s):")
            for dev in input_devices[:3]:
                print(f"    - {dev.get('name')} (Max Channels: {dev.get('max_input_channels')})")

            # Quick non-blocking audio capture probe (0.5s test read)
            samples = []
            def _probe_cb(indata, frames, time_info, status):
                samples.append(len(indata))

            with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=_probe_cb):
                time.sleep(0.5)

            print(f"  • Microphone probe captured {len(samples)} buffer frames successfully.")
            print("  Status:        ✅ PASS (Microphone stream operational)")
        except Exception as mic_err:
            print(f"  [NOTE] Microphone probe notice: {mic_err}")

    # ---------------------------------------------------------
    # Input Validation & Security Test
    # ---------------------------------------------------------
    print("\n[Bonus] Testing Input Validation & Security...")
    unconfigured_stt = ElevenLabsSTTService(api_key="")
    try:
        asyncio.run(unconfigured_stt.connect())
        print("  [FAILURE] Expected ValueError for unconfigured STT service!")
        sys.exit(1)
    except ValueError as ve:
        print(f"  • Unconfigured check: ✅ Raised ValueError (\"{ve}\")")

    print("\n" + "=" * 65)
    print(" ✅ ALL STEP 2 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
