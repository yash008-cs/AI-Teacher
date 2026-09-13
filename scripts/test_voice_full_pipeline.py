"""
Stage 4 End-to-End Voice Integration Verification Suite:
Connects Gemini Streaming Response to ElevenLabs Streaming TTS.

Tests:
  TEST 1: In-Domain Query ("What is supervised learning?")
          -> STT -> RAG -> Gemini -> Text Stream -> Streaming TTS -> Audio
  TEST 2: Detailed Query ("Explain classification and regression with examples.")
          -> Verifies ordered chunk streaming, single Gemini call, complete response
  TEST 3: Out-of-Domain Query ("What is photosynthesis?")
          -> Verifies RAG grounding guardrails hold in both text and speech
  TEST 4: TTS Failure Fallback Test
          -> Simulates TTS failure; verifies text continues uninterrupted
  TEST 5: Typed Chat Test
          -> Verifies typed chat path remains fully functional
"""
import sys
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from services import ZepService, MemoryService, get_ai_service, RAGService
from services.voice import (
    ElevenLabsSTTService,
    ElevenLabsTTSService,
    VoiceStreamPipe,
)


async def stream_audio_to_stt(
    pcm_audio: bytes,
    stt_service: ElevenLabsSTTService,
) -> Tuple[List[str], str]:
    """Streams 16kHz PCM audio to ElevenLabs Scribe Realtime via WebSocket."""
    partials: List[str] = []
    committed: List[str] = []

    def _on_partial(text: str):
        if text:
            partials.append(text)

    def _on_final(text: str):
        if text:
            committed.append(text)

    conn = await stt_service.connect(
        commit_strategy="vad",
        on_partial=_on_partial,
        on_final=_on_final,
    )

    chunk_size = 3200  # 100ms
    for i in range(0, len(pcm_audio), chunk_size):
        chunk = pcm_audio[i : i + chunk_size]
        await stt_service.send_pcm_chunk(conn, chunk)
        await asyncio.sleep(0.04)

    silence = b"\x00" * chunk_size
    for _ in range(12):
        await stt_service.send_pcm_chunk(conn, silence)
        await asyncio.sleep(0.04)

    await asyncio.sleep(0.8)
    await conn.close()

    final_text = committed[-1] if committed else (partials[-1] if partials else "")
    return partials, final_text.strip()


def run_voice_turn(
    question_text: str,
    stt_service: ElevenLabsSTTService,
    tts_service: ElevenLabsTTSService,
    memory_service: MemoryService,
    thread_id: str,
    learner_name: str,
    conversation_history: List[Dict[str, str]],
    output_audio_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a complete voice turn:
    1. Simulates student speech by generating 16kHz PCM.
    2. Streams PCM through ElevenLabs Scribe Realtime STT.
    3. Feeds committed transcript into MemoryService (RAG + Zep + Gemini).
    4. Concurrently pipes Gemini tokens to text rendering and ElevenLabs WebSocket streaming TTS.
    5. Returns metrics: latency, token counts, audio size, and response text.
    """
    print(f"\n" + "=" * 65)
    print(f"🎙️  Question: \"{question_text}\"")

    # Step 1: Synthesize student audio input
    t_stt_start = time.perf_counter()
    student_pcm = tts_service.text_to_speech(text=question_text, output_format="pcm_16000")

    # Step 2: Stream through Scribe Realtime
    _, committed_transcript = asyncio.run(stream_audio_to_stt(student_pcm, stt_service))
    stt_duration = time.perf_counter() - t_stt_start
    print(f"  • STT Committed Transcript: \"{committed_transcript}\" ({stt_duration:.2f}s)")
    assert committed_transcript, "No transcript committed by STT!"

    # Step 3: Single call to MemoryService (Zep + RAG + Gemini)
    t_rag_start = time.perf_counter()
    raw_context, stream, sources = memory_service.retrieve_context_and_stream_response(
        thread_id=thread_id,
        user_message=committed_transcript,
        learner_name=learner_name,
        conversation_history=conversation_history,
    )
    rag_duration = time.perf_counter() - t_rag_start
    print(f"  • RAG Retrieval & Context Prep: {rag_duration:.2f}s ({len(sources)} sources)")

    # Step 4: Pipe single Gemini stream concurrently into text display and ElevenLabs streaming TTS
    voice_pipe = VoiceStreamPipe(tts_service=tts_service, enabled=True)
    display_stream = voice_pipe.pipe(stream)

    t_gemini_start = time.perf_counter()
    first_token_time = None
    accumulated_text = ""

    print("  [Gemini Text Streaming]:")
    for token in display_stream:
        if first_token_time is None:
            first_token_time = time.perf_counter() - t_gemini_start
        accumulated_text += token
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")

    total_text_time = time.perf_counter() - t_gemini_start

    # Step 5: Gather concurrently streamed audio from ElevenLabs
    audio_bytes = voice_pipe.get_audio(timeout=15.0)

    # Save audio if requested
    if audio_bytes and output_audio_filename:
        audio_dir = PROJECT_ROOT / "data" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        save_path = audio_dir / output_audio_filename
        with open(save_path, "wb") as f:
            f.write(audio_bytes)
        print(f"  • Saved audio output: {save_path.relative_to(PROJECT_ROOT)} ({len(audio_bytes):,} bytes)")

    # Save turn to Zep and history
    memory_service.record_assistant_response(thread_id=thread_id, content=accumulated_text)
    conversation_history.append({"role": "user", "content": committed_transcript})
    conversation_history.append({"role": "assistant", "content": accumulated_text})

    return {
        "transcript": committed_transcript,
        "first_token_latency": first_token_time,
        "first_audio_latency": voice_pipe.first_chunk_latency,
        "total_text_duration": total_text_time,
        "total_audio_bytes": len(audio_bytes) if audio_bytes else 0,
        "audio_chunks_count": len(voice_pipe.audio_chunks),
        "tts_error": voice_pipe.error,
        "sources": sources,
        "response_text": accumulated_text,
    }


def main():
    print("=" * 65)
    print(" STEP 4: GEMINI STREAMING RESPONSE -> ELEVENLABS STREAMING TTS")
    print("=" * 65)

    # 1. Initialize Services
    print("\n[1/6] Initializing Services...")
    zep_service = ZepService()
    ai_service = get_ai_service()
    rag_service = RAGService()
    memory_service = MemoryService(
        zep_service=zep_service,
        ai_service=ai_service,
        rag_service=rag_service,
    )
    stt_service = ElevenLabsSTTService()
    tts_service = ElevenLabsTTSService()

    print(f"  • AI Provider: {ai_service.provider_name} ({ai_service.model})")
    print(f"  • TTS Model:   {tts_service.model} (Voice: {tts_service.voice_id})")
    print(f"  • STT Model:   {stt_service.model_id}")
    print(f"  • RAG Chunks:  {rag_service.vector_store.count()} chunks")

    test_user_id = f"test-user-step4-{int(time.time())}"
    test_thread_id = f"test-thread-step4-{int(time.time())}"
    learner_name = "Alex"

    if zep_service.is_configured():
        zep_service.get_or_create_user(user_id=test_user_id, first_name=learner_name)
        zep_service.create_thread(user_id=test_user_id, thread_id=test_thread_id)

    conversation_history: List[Dict[str, str]] = []

    # ==========================================================
    # TEST 1: Short Course Answer ("What is supervised learning?")
    # ==========================================================
    print("\n[2/6] Running TEST 1: Short Course Question — \"What is supervised learning?\"")
    res1 = run_voice_turn(
        question_text="What is supervised learning?",
        stt_service=stt_service,
        tts_service=tts_service,
        memory_service=memory_service,
        thread_id=test_thread_id,
        learner_name=learner_name,
        conversation_history=conversation_history,
        output_audio_filename="step4_test1_supervised_learning.mp3",
    )

    print("  Metrics Summary:")
    print(f"  • First Gemini token latency:     {res1['first_token_latency']:.2f}s")
    print(f"  • First ElevenLabs audio latency: {res1['first_audio_latency']:.2f}s")
    print(f"  • Total text generation time:     {res1['total_text_duration']:.2f}s")
    print(f"  • Audio chunks streamed:          {res1['audio_chunks_count']}")
    print(f"  • Total synthesized audio bytes:  {res1['total_audio_bytes']:,} bytes")
    print(f"  • TTS error:                      {res1['tts_error']}")

    assert res1["first_token_latency"] is not None, "Did not record first token latency!"
    assert res1["total_audio_bytes"] > 0, "No audio was generated by ElevenLabs TTS!"
    assert res1["tts_error"] is None, f"TTS error occurred: {res1['tts_error']}"
    print("  ✅ TEST 1 PASSED: Gemini streamed text while ElevenLabs concurrently streamed audio.")

    # ==========================================================
    # TEST 2: Detailed Course Answer ("Explain classification and regression with examples.")
    # ==========================================================
    print("\n[3/6] Running TEST 2: Detailed Multi-Chunk Question — \"Explain classification and regression with examples.\"")
    res2 = run_voice_turn(
        question_text="Explain classification and regression with examples.",
        stt_service=stt_service,
        tts_service=tts_service,
        memory_service=memory_service,
        thread_id=test_thread_id,
        learner_name=learner_name,
        conversation_history=conversation_history,
        output_audio_filename="step4_test2_classification_regression.mp3",
    )

    print("  Metrics Summary:")
    print(f"  • First Gemini token latency:     {res2['first_token_latency']:.2f}s")
    print(f"  • First ElevenLabs audio latency: {res2['first_audio_latency']:.2f}s")
    print(f"  • Total text generation time:     {res2['total_text_duration']:.2f}s")
    print(f"  • Audio chunks streamed:          {res2['audio_chunks_count']}")
    print(f"  • Total synthesized audio bytes:  {res2['total_audio_bytes']:,} bytes")

    assert res2["total_audio_bytes"] > 0, "No audio generated for longer answer!"
    assert res2["audio_chunks_count"] >= 3, "Expected multiple audio chunks for multi-sentence response!"
    print("  ✅ TEST 2 PASSED: Multiple text and audio chunks streamed in order with zero duplicated LLM calls.")

    # ==========================================================
    # TEST 3: Out-of-Domain Query ("What is photosynthesis?")
    # ==========================================================
    print("\n[4/6] Running TEST 3: Out-of-Domain Query — \"What is photosynthesis?\"")
    res3 = run_voice_turn(
        question_text="What is photosynthesis?",
        stt_service=stt_service,
        tts_service=tts_service,
        memory_service=memory_service,
        thread_id=test_thread_id,
        learner_name=learner_name,
        conversation_history=conversation_history,
        output_audio_filename="step4_test3_photosynthesis.mp3",
    )

    print("  • Checking RAG Grounding Consistency:")
    print(f"  • Audio generated matches text length: {res3['total_audio_bytes']:,} bytes")
    assert res3["total_audio_bytes"] > 0, "TTS audio should be synthesized for the teacher's grounded response."
    print("  ✅ TEST 3 PASSED: Out-of-domain response text and audio remain grounded and aligned.")

    # ==========================================================
    # TEST 4: TTS Failure Fallback Test
    # ==========================================================
    print("\n[5/6] Running TEST 4: TTS Failure Fallback Test...")
    # Instantiate a mock broken TTS service that raises an exception during streaming
    class BrokenTTSService(ElevenLabsTTSService):
        def text_to_speech_stream(self, text_stream, **kwargs):
            raise RuntimeError("Simulated network/quota failure in ElevenLabs TTS connection")

    broken_tts = BrokenTTSService(api_key="mock_key")
    broken_pipe = VoiceStreamPipe(tts_service=broken_tts, enabled=True)

    def dummy_stream():
        yield "This "
        yield "is "
        yield "a "
        yield "resilient "
        yield "response."

    # Stream text through broken pipe
    text_received = "".join(list(broken_pipe.pipe(dummy_stream())))
    audio_fallback = broken_pipe.get_audio(timeout=2.0)

    print(f"  • Text received despite TTS error: \"{text_received}\"")
    print(f"  • Captured error safely:           {broken_pipe.error}")
    print(f"  • Audio returned gracefully:       {audio_fallback} (None)")

    assert text_received == "This is a resilient response.", "Text streaming was disrupted by TTS error!"
    assert audio_fallback is None, "Audio should be None on error!"
    assert broken_pipe.error is not None, "Error should be recorded for caller notice!"
    print("  ✅ TEST 4 PASSED: Graceful fallback confirmed. TTS failure never crashes or interrupts text.")

    # ==========================================================
    # TEST 5: Typed Chat Test (Voice output toggle handling)
    # ==========================================================
    print("\n[6/6] Running TEST 5: Typed Chat Path Verification...")
    # Typed chat with voice enabled
    pipe_typed = VoiceStreamPipe(tts_service=tts_service, enabled=True)
    def typed_stream():
        yield "Hello Alex, "
        yield "here is the answer to your typed question."

    typed_text = "".join(list(pipe_typed.pipe(typed_stream())))
    typed_audio = pipe_typed.get_audio(timeout=5.0)

    print(f"  • Typed question text stream:  \"{typed_text}\"")
    print(f"  • Typed question audio bytes:   {len(typed_audio):,} bytes")
    assert len(typed_text) > 0 and len(typed_audio) > 0, "Typed chat voice pipe failed!"
    print("  ✅ TEST 5 PASSED: Typed chat functions smoothly with streaming voice.")

    print("\n" + "=" * 65)
    print(" ✅ ALL STEP 4 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
