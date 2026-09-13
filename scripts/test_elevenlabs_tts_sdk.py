"""
ElevenLabs Python SDK Text-to-Speech (TTS) Verification Script.
Tests synthesis using the official `elevenlabs` SDK client (ElevenLabs).
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main():
    print("=" * 60)
    print(" ElevenLabs Official Python SDK - TTS Verification")
    print("=" * 60)

    # 1. Load environment variables
    load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    key_var_name = "ELEVENLABS_API_KEY"

    if not api_key:
        fallback_key = os.getenv("ELEVNLABS_API_KEY")
        if fallback_key:
            api_key = fallback_key
            key_var_name = "ELEVNLABS_API_KEY (Note: missing 'e' in ELEVEN)"

    if not api_key or not api_key.strip():
        print("\n[FAILURE] ElevenLabs API Key not found in .env")
        print("Please ensure your .env file contains: ELEVENLABS_API_KEY=your_key_here")
        sys.exit(1)

    print(f"\n[1/4] Key detected in .env under: {key_var_name}")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "***"
    print(f"      Key Format: Valid ({len(api_key)} chars, {masked_key})")

    # 2. Import official SDK
    print("\n[2/4] Initializing official ElevenLabs client...")
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        print("[FAILURE] The 'elevenlabs' package is not installed in the active environment.")
        print("Please run: pip install elevenlabs")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key.strip())
    print("      Client initialized successfully.")

    # 3. Test text-to-speech synthesis using official SDK
    # Voice: George (premade voice ID: JBFqnCBsd6RMkjVDRZzb)
    # Model: eleven_flash_v2_5 (ultra low-latency model optimized for real-time conversation)
    voice_id = "JBFqnCBsd6RMkjVDRZzb"
    model_id = "eleven_flash_v2_5"
    test_text = "Hello! I am your AI Teacher. Welcome to our personalized learning session."

    print(f"\n[3/4] Generating speech using SDK (Model: {model_id}, Voice ID: {voice_id})...")
    print(f"      Text: \"{test_text}\"")

    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=test_text,
            model_id=model_id,
            output_format="mp3_44100_128",
        )

        # Collect the audio stream chunks into a single byte buffer
        audio_bytes = b"".join(audio_stream)
        print(f"      Speech synthesis succeeded! Total audio size: {len(audio_bytes)} bytes")

    except Exception as e:
        print(f"\n[FAILURE] TTS synthesis failed via SDK: {e}")
        return False

    # 4. Save test audio to verify file playback
    output_dir = Path("data") / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_teacher_speech.mp3"

    try:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"\n[4/4] Saved verification audio file:")
        print(f"      Path: {output_path.resolve()}")
    except Exception as e:
        print(f"\n[WARNING] Could not save audio file: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] ElevenLabs Official SDK TTS test completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
