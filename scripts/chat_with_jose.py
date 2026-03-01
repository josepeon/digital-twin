#!/usr/bin/env python3
"""
Interactive chat with Jose — terminal interface for testing the digital twin.

Usage:
    python scripts/chat_with_jose.py           # Text only
    python scripts/chat_with_jose.py --voice    # Text + voice output
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Jose's digital twin")
    parser.add_argument("--voice", action="store_true", help="Enable voice output")
    parser.add_argument("--model", type=str, default=None, help="Override Groq model")
    args = parser.parse_args()

    from src.personality_engine import PersonalityEngine
    from src.voice import VoiceEngine

    print("=" * 60)
    print("  🧬 Digital Twin — Jose Peon")
    print("  Type your message or 'quit' to exit")
    if args.voice:
        print("  🔊 Voice output: ON")
    print("=" * 60)

    engine = PersonalityEngine(model=args.model)
    voice = VoiceEngine() if args.voice else None

    while True:
        try:
            query = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Adiós!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "bye", "q"}:
            print("\n👋 Adiós!")
            break

        try:
            result = engine.ask(query)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

        response = result["response"]
        lang = result["language"]
        register = result["register"]
        model = result["model"]
        examples = result["examples_used"]

        print(f"\n🧬 Jose ({lang.upper()}, {register}): {response}")
        print(f"   [model={model}, rag_examples={examples}]")

        # Voice output
        if voice and response:
            try:
                audio_bytes = voice.speak(response, language=lang)
                # Play audio using afplay (macOS)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                import subprocess
                subprocess.Popen(
                    ["afplay", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"   🔇 Voice error: {e}")


if __name__ == "__main__":
    main()
