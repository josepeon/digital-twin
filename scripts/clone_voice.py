#!/usr/bin/env python3
"""
Voice cloning script — create ElevenLabs IVC clones for EN and ES.

Prerequisites:
  - ELEVENLABS_API_KEY set in .env
  - Voice samples recorded in:
      data/personal/voice/en/  (English samples)
      data/personal/voice/es/  (Spanish samples)

Usage:
    python scripts/clone_voice.py [--language en|es|both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ELEVENLABS_API_KEY, VOICE_DIR, VOICE_ID_EN_PATH, VOICE_ID_ES_PATH


def clone_voice(
    language: str,
    samples_dir: Path,
    voice_name: str,
    description: str,
    output_path: Path,
) -> str:
    """Create an ElevenLabs IVC voice clone.

    Args:
        language: "en" or "es".
        samples_dir: Directory containing audio samples (.mp3, .wav, .m4a).
        voice_name: Display name for the voice on ElevenLabs.
        description: Description for the voice.
        output_path: Path to save the voice ID.

    Returns:
        The voice ID.
    """
    from elevenlabs.client import ElevenLabs

    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY not set. Add it to .env")
        sys.exit(1)

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Find audio files
    extensions = ["*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(sorted(samples_dir.glob(ext)))

    if not audio_files:
        print(f"❌ No audio samples found in {samples_dir}")
        print(f"   Supported formats: {', '.join(e.replace('*', '') for e in extensions)}")
        sys.exit(1)

    print(f"🎤 Creating IVC clone: {voice_name}")
    print(f"   Samples ({len(audio_files)}):")
    for f in audio_files:
        print(f"     - {f.name}")

    # Open files for upload
    file_handles = [open(f, "rb") for f in audio_files]

    try:
        voice = client.voices.ivc.create(
            name=voice_name,
            description=description,
            files=file_handles,
            labels=json.dumps({"language": language, "accent": "cuban", "gender": "male"}),
            remove_background_noise=True,
        )
    finally:
        for fh in file_handles:
            fh.close()

    voice_id = voice.voice_id
    print(f"✅ Voice created: {voice_id}")

    # Save voice ID
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(voice_id)
    print(f"   Saved to {output_path}")

    return voice_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ElevenLabs IVC voice clones")
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="both",
        choices=["en", "es", "both"],
        help="Which language voice to clone",
    )
    args = parser.parse_args()

    languages = ["en", "es"] if args.language == "both" else [args.language]

    for lang in languages:
        samples_dir = VOICE_DIR / lang
        if not samples_dir.exists():
            print(f"⚠ Samples directory not found: {samples_dir}")
            print(f"  Record voice samples and place them there.")
            continue

        if lang == "en":
            clone_voice(
                language="en",
                samples_dir=samples_dir,
                voice_name="Jose Peon (English)",
                description="Digital twin voice - English. Professional, composed, daily tone.",
                output_path=VOICE_ID_EN_PATH,
            )
        else:
            clone_voice(
                language="es",
                samples_dir=samples_dir,
                voice_name="Jose Peon (Español)",
                description="Digital twin voice - Spanish. Warm, animated, Cuban family energy.",
                output_path=VOICE_ID_ES_PATH,
            )

    print("\n🎉 Voice cloning complete!")
    print("   Test with: python scripts/chat_with_jose.py --voice")


if __name__ == "__main__":
    main()
