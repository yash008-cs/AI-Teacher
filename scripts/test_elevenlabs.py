"""
Standalone Verification Script for ElevenLabs API Integration.
Tests authentication and connectivity without modifying any existing application code.
"""
import os
import sys
import requests
from dotenv import load_dotenv

def main():
    print("=" * 60)
    print(" ElevenLabs API Authentication & Connectivity Test")
    print("=" * 60)

    # 1. Load environment variables from .env
    load_dotenv()

    # Support standard ELEVENLABS_API_KEY as well as common typo ELEVNLABS_API_KEY
    api_key = os.getenv("ELEVENLABS_API_KEY")
    key_var_name = "ELEVENLABS_API_KEY"

    if not api_key:
        fallback_key = os.getenv("ELEVNLABS_API_KEY")
        if fallback_key:
            api_key = fallback_key
            key_var_name = "ELEVNLABS_API_KEY (Note: missing 'e' in ELEVEN)"

    # 2. Verify key availability
    if not api_key or not api_key.strip():
        print("\n[FAILURE] ElevenLabs API Key not found in .env")
        print("Please ensure your .env file contains: ELEVENLABS_API_KEY=your_key_here")
        sys.exit(1)

    print(f"\n[1/3] Key detected in .env under: {key_var_name}")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "***"
    print(f"      Key Format: Valid ({len(api_key)} characters, {masked_key})")

    # 3. Make minimal ElevenLabs API request to verify authentication
    print("\n[2/3] Contacting ElevenLabs API (GET https://api.elevenlabs.io/v1/models)...")
    headers = {
        "xi-api-key": api_key.strip(),
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/models",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            models = response.json()
            model_names = [m.get("name") for m in models[:4] if isinstance(m, dict)]
            print("      Status Code: 200 OK")
            print(f"      Available Models: {', '.join(model_names)}...")
            print("\n[3/3] Testing minimal speech synthesis capability (model: eleven_flash_v2_5)...")
            
            # Test default premade voice (George) with minimal text
            tts_url = "https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb"
            tts_payload = {
                "text": "Hello!",
                "model_id": "eleven_flash_v2_5"
            }
            tts_response = requests.post(
                tts_url,
                headers={"xi-api-key": api_key.strip(), "Content-Type": "application/json"},
                json=tts_payload,
                timeout=15,
            )
            
            if tts_response.status_code == 200:
                print(f"      Speech synthesis succeeded! (Received {len(tts_response.content)} audio bytes)")
            else:
                print(f"      Note on TTS endpoint: HTTP {tts_response.status_code}")

            print("\n" + "=" * 60)
            print("[SUCCESS] ElevenLabs API Key is VALID and authenticated!")
            print("=" * 60)
            return True

        elif response.status_code in (401, 403):
            print(f"      Status Code: {response.status_code} Unauthorized / Forbidden")
            print(f"      Details: {response.text}")
            print("\n" + "=" * 60)
            print("[FAILURE] ElevenLabs API Key was rejected by the server.")
            print("=" * 60)
            return False

        else:
            print(f"      Status Code: {response.status_code}")
            print(f"      Details: {response.text}")
            print("\n" + "=" * 60)
            print(f"[WARNING] Server returned unexpected status: {response.status_code}")
            print("=" * 60)
            return False

    except requests.exceptions.RequestException as e:
        print(f"\n[FAILURE] Network or connection error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
