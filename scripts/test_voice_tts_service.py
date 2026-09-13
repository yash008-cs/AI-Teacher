"""
Isolated Test Suite for Reusable ElevenLabs TTS Service.
Verifies:
  TEST A: Standard Text-to-Speech (Full audio generation)
  TEST B: Real-Time Streaming Text-to-Speech (Streaming iterator input -> Audio chunk stream)
  TEST C: Input Validation & Edge Case Error Handling
"""
import sys
import os
import time
from pathlib import Path
from typing import Iterator

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from services.voice import ElevenLabsTTSService


def simulated_text_generator() -> Iterator[str]:
    """Simulates text arriving in chunks as would happen from an LLM token stream."""
    sentences = [
        "This is the first sentence. ",
        "Here is another sentence. ",
        "Streaming allows audio generation to begin earlier.",
    ]
    for s in sentences:
        yield s


def main():
    print("=" * 65)
    print(" STEP 1: ELEVENLABS TTS REUSABLE SERVICE VERIFICATION")
    print("=" * 65)

    # 1. Initialize Service
    print("\n[1/4] Initializing ElevenLabsTTSService...")
    tts_service = ElevenLabsTTSService()

    print(f"  • Configured: {tts_service.is_configured()}")
    print(f"  • Model:      {tts_service.model}")
    print(f"  • Voice ID:   {tts_service.voice_id}")

    if not tts_service.is_configured():
        print("\n[FAILURE] ElevenLabsTTSService is not configured.")
        print("Please check that ELEVENLABS_API_KEY is properly set in your .env file.")
        sys.exit(1)

    audio_dir = PROJECT_ROOT / "data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # TEST A: Standard TTS
    # ==========================================================
    print("\n[2/4] Running TEST A: Standard Text-to-Speech...")
    test_a_text = (
        "Hello! I am your AI Teacher. Today we are going to learn supervised machine learning."
    )
    print(f"  Input text: \"{test_a_text}\"")

    start_a = time.perf_counter()
    try:
        audio_bytes = tts_service.text_to_speech(test_a_text)
        duration_a = time.perf_counter() - start_a
        print(f"  Synthesized in: {duration_a:.2f}s")
        print(f"  Received bytes: {len(audio_bytes):,} bytes")

        output_path_a = audio_dir / "test_service_tts.mp3"
        with open(output_path_a, "wb") as f:
            f.write(audio_bytes)

        assert output_path_a.exists(), f"Output file does not exist: {output_path_a}"
        assert output_path_a.stat().st_size > 0, "Output file is empty!"
        print(f"  Saved file:     {output_path_a.relative_to(PROJECT_ROOT)} ({output_path_a.stat().st_size:,} bytes)")
        print("  Status:         ✅ TEST A PASSED")
    except Exception as e:
        print(f"  [FAILURE] TEST A failed: {e}")
        sys.exit(1)

    # ==========================================================
    # TEST B: Streaming TTS
    # ==========================================================
    print("\n[3/4] Running TEST B: Real-Time Streaming Text-to-Speech...")
    print("  Feeding simulated text generator into text_to_speech_stream()...")

    start_b = time.perf_counter()
    chunks_received = []
    first_chunk_time = None

    try:
        stream_generator = simulated_text_generator()
        audio_chunk_iter = tts_service.text_to_speech_stream(stream_generator)

        for chunk in audio_chunk_iter:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start_b
            chunks_received.append(chunk)

        total_duration_b = time.perf_counter() - start_b
        total_audio_bytes = sum(len(c) for c in chunks_received)

        print(f"  First audio chunk latency: {first_chunk_time:.2f}s" if first_chunk_time else "  No chunks received")
        print(f"  Total streaming time:      {total_duration_b:.2f}s")
        print(f"  Total audio chunks:        {len(chunks_received)}")
        print(f"  Total streamed bytes:      {total_audio_bytes:,} bytes")

        assert len(chunks_received) > 0, "No audio chunks were streamed!"
        assert total_audio_bytes > 0, "Streamed audio bytes must be greater than 0!"

        output_path_b = audio_dir / "test_service_tts_stream.mp3"
        with open(output_path_b, "wb") as f:
            for chunk in chunks_received:
                f.write(chunk)

        assert output_path_b.exists(), f"Output file does not exist: {output_path_b}"
        assert output_path_b.stat().st_size > 0, "Streamed output file is empty!"
        print(f"  Saved file:                {output_path_b.relative_to(PROJECT_ROOT)} ({output_path_b.stat().st_size:,} bytes)")
        print("  Status:                    ✅ TEST B PASSED")
    except Exception as e:
        print(f"  [FAILURE] TEST B failed: {e}")
        sys.exit(1)

    # ==========================================================
    # TEST C: Input Validation & Error Handling
    # ==========================================================
    print("\n[4/4] Running TEST C: Input Validation & Edge Case Handling...")
    try:
        # Test 1: Empty text string should raise ValueError
        try:
            tts_service.text_to_speech("")
            print("  [FAILURE] Expected ValueError for empty text, but none was raised.")
            sys.exit(1)
        except ValueError as ve:
            print(f"  • Empty text validation:        ✅ Correctly raised ValueError (\"{ve}\")")

        # Test 2: Unconfigured service should raise ValueError
        unconfigured = ElevenLabsTTSService(api_key="")
        try:
            unconfigured.text_to_speech("Test")
            print("  [FAILURE] Expected ValueError for unconfigured service, but none was raised.")
            sys.exit(1)
        except ValueError as ve:
            print(f"  • Unconfigured service check:   ✅ Correctly raised ValueError (\"{ve}\")")

        print("  Status:                         ✅ TEST C PASSED")
    except Exception as e:
        print(f"  [FAILURE] TEST C failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" ✅ ALL STEP 1 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
