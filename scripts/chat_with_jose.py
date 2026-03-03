#!/usr/bin/env python3
"""
Interactive chat with Jose — rich terminal interface.

Features:
  - Conversation history (context-aware follow-ups)
  - Voice output with auto-play (macOS afplay)
  - Streaming voice mode for low-latency audio
  - Colored/formatted terminal output
  - Chat history saved to file
  - Works locally or against the deployed API

Usage:
    python scripts/chat_with_jose.py                    # text only, local engine
    python scripts/chat_with_jose.py --voice             # with voice output
    python scripts/chat_with_jose.py --voice --stream    # streaming voice
    python scripts/chat_with_jose.py --api               # use deployed Railway API
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── ANSI Colors ────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
BLUE = "\033[34m"


# ── Chat History ───────────────────────────────────────────────────────────────

class ChatHistory:
    """Maintains conversation history for context-aware responses."""

    def __init__(self, max_turns: int = 20):
        self.turns: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str, metadata: dict | None = None):
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def get_context_summary(self, last_n: int = 6) -> str:
        """Build a context string from recent conversation."""
        recent = self.turns[-last_n * 2:] if len(self.turns) > last_n * 2 else self.turns
        if not recent:
            return ""
        lines = []
        for turn in recent:
            prefix = "User" if turn["role"] == "user" else "Jose"
            lines.append(f"{prefix}: {turn['content'][:200]}")
        return "\n".join(lines)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.turns, f, indent=2, ensure_ascii=False)

    @property
    def turn_count(self) -> int:
        return len([t for t in self.turns if t["role"] == "user"])


# ── Audio Playback ─────────────────────────────────────────────────────────────

def play_audio(audio_bytes: bytes, block: bool = False) -> subprocess.Popen | None:
    """Play MP3 audio on macOS using afplay."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        proc = subprocess.Popen(
            ["afplay", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if block:
            proc.wait()
            os.unlink(tmp_path)
        return proc
    except FileNotFoundError:
        print(f"   {DIM}(afplay not found — voice requires macOS){RESET}")
        return None
    except Exception as e:
        print(f"   {RED}Audio error: {e}{RESET}")
        return None


def play_audio_stream(audio_chunks, block: bool = True) -> subprocess.Popen | None:
    """Play streaming audio by writing chunks to a temp file, then playing."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            for chunk in audio_chunks:
                tmp.write(chunk)
            tmp_path = tmp.name
        proc = subprocess.Popen(
            ["afplay", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if block:
            proc.wait()
            os.unlink(tmp_path)
        return proc
    except Exception as e:
        print(f"   {RED}Stream audio error: {e}{RESET}")
        return None


# ── API Client Mode ────────────────────────────────────────────────────────────

def ask_api(
    query: str,
    api_url: str,
    api_key: str,
    voice: bool = False,
    stream: bool = False,
) -> dict:
    """Ask the deployed API."""
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if voice and not stream:
        resp = requests.post(
            f"{api_url.rstrip('/')}/ask-jose/voice",
            json={"query": query},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "response": resp.headers.get("X-Response-Text", ""),
            "language": resp.headers.get("X-Response-Language", "en"),
            "register": resp.headers.get("X-Response-Register", "neutral"),
            "model": "remote",
            "examples_used": -1,
            "audio_bytes": resp.content,
        }
    elif voice and stream:
        resp = requests.post(
            f"{api_url.rstrip('/')}/ask-jose/stream",
            json={"query": query},
            headers=headers,
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()
        audio_chunks = list(resp.iter_content(chunk_size=4096))
        return {
            "response": resp.headers.get("X-Response-Text", ""),
            "language": resp.headers.get("X-Response-Language", "en"),
            "register": "casual",
            "model": "remote",
            "examples_used": -1,
            "audio_chunks": audio_chunks,
        }
    else:
        resp = requests.post(
            f"{api_url.rstrip('/')}/ask-jose",
            json={"query": query},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "response": data["response"],
            "language": data["language"],
            "register": data.get("style_register", "neutral"),
            "model": data.get("model", "remote"),
            "examples_used": data.get("examples_used", -1),
        }


# ── Main Chat Loop ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Jose's digital twin")
    parser.add_argument("--voice", action="store_true", help="Enable voice output")
    parser.add_argument("--stream", action="store_true", help="Streaming voice (lower latency)")
    parser.add_argument("--model", type=str, default=None, help="Override Groq model")
    parser.add_argument("--api", nargs="?",
                        const="https://digital-twin-production-c355.up.railway.app",
                        default=None, help="Use deployed API (default: Railway URL)")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--save", type=str, default=None, help="Save chat history to file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    if args.no_color:
        global BOLD, DIM, CYAN, GREEN, YELLOW, RED, MAGENTA, RESET, BLUE
        BOLD = DIM = CYAN = GREEN = YELLOW = RED = MAGENTA = RESET = BLUE = ""

    # Load API key
    api_key = args.api_key
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("TWIN_API_KEY", "")

    # Initialize engines
    engine = None
    voice_engine = None

    if args.api:
        mode = f"API ({args.api})"
    else:
        mode = "Local"
        from src.personality_engine import PersonalityEngine
        engine = PersonalityEngine(model=args.model)
        if args.voice or args.stream:
            from src.voice import VoiceEngine
            voice_engine = VoiceEngine()

    history = ChatHistory()

    # Banner
    print()
    print(f"{CYAN}{'═' * 60}{RESET}")
    print(f"{CYAN}║{RESET}  {BOLD}🧬 Digital Twin — Jose Peon{RESET}")
    print(f"{CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  {DIM}Mode:  {mode}{RESET}")
    if args.voice or args.stream:
        print(f"{CYAN}║{RESET}  {DIM}Voice: {'Streaming' if args.stream else 'Standard'}{RESET}")
    print(f"{CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  {DIM}Commands: /quit  /voice  /history  /clear  /help{RESET}")
    print(f"{CYAN}{'═' * 60}{RESET}")

    voice_on = args.voice or args.stream
    stream_on = args.stream

    while True:
        try:
            query = input(f"\n{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{YELLOW}👋 Adiós!{RESET}")
            break

        if not query:
            continue

        # ── Slash commands ─────────────────────────────────────────────
        if query.startswith("/"):
            cmd = query.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                print(f"\n{YELLOW}👋 Adiós!{RESET}")
                break
            elif cmd == "/voice":
                voice_on = not voice_on
                print(f"  {DIM}Voice output: {'ON' if voice_on else 'OFF'}{RESET}")
                if voice_on and not voice_engine and not args.api:
                    from src.voice import VoiceEngine
                    voice_engine = VoiceEngine()
                continue
            elif cmd == "/stream":
                stream_on = not stream_on
                print(f"  {DIM}Voice mode: {'streaming' if stream_on else 'standard'}{RESET}")
                continue
            elif cmd == "/history":
                print(f"  {DIM}Conversation: {history.turn_count} turns{RESET}")
                for turn in history.turns[-10:]:
                    prefix = f"{GREEN}You{RESET}" if turn["role"] == "user" else f"{CYAN}Jose{RESET}"
                    print(f"  {prefix}: {turn['content'][:100]}")
                continue
            elif cmd == "/clear":
                history = ChatHistory()
                print(f"  {DIM}Conversation cleared.{RESET}")
                continue
            elif cmd == "/help":
                print(f"  {DIM}/voice   — toggle voice output{RESET}")
                print(f"  {DIM}/stream  — toggle streaming voice{RESET}")
                print(f"  {DIM}/history — show recent conversation{RESET}")
                print(f"  {DIM}/clear   — clear conversation history{RESET}")
                print(f"  {DIM}/quit    — exit{RESET}")
                continue
            else:
                print(f"  {DIM}Unknown command. Type /help{RESET}")
                continue

        # ── Build context-enriched query ───────────────────────────────
        context_summary = history.get_context_summary()
        enriched_query = query
        if context_summary:
            enriched_query = (
                f"[Previous conversation for context:\n{context_summary}]\n\n"
                f"User's latest message: {query}"
            )

        history.add("user", query)

        # ── Get response ───────────────────────────────────────────────
        start = time.time()
        try:
            if args.api:
                result = ask_api(
                    enriched_query, args.api, api_key,
                    voice=voice_on, stream=stream_on,
                )
            else:
                result = engine.ask(enriched_query)
        except KeyboardInterrupt:
            print(f"\n  {DIM}(interrupted){RESET}")
            continue
        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}")
            continue

        elapsed = time.time() - start
        response = result["response"]
        lang = result["language"]
        register = result.get("register", "neutral")
        model = result.get("model", "?")
        examples = result.get("examples_used", 0)

        history.add("assistant", response, {
            "language": lang, "register": register, "model": model,
        })

        # ── Display ────────────────────────────────────────────────────
        lang_badge = f"{MAGENTA}ES{RESET}" if lang == "es" else f"{BLUE}EN{RESET}"
        print(f"\n{CYAN}{BOLD}Jose{RESET} [{lang_badge}]: {response}")
        print(f"  {DIM}[{register} | {model} | {examples} examples | {elapsed:.1f}s]{RESET}")

        # ── Voice output ───────────────────────────────────────────────
        if voice_on and response:
            if args.api:
                audio_bytes = result.get("audio_bytes")
                audio_chunks = result.get("audio_chunks")
                if audio_chunks:
                    play_audio_stream(audio_chunks, block=True)
                elif audio_bytes:
                    play_audio(audio_bytes, block=True)
            elif voice_engine:
                try:
                    if stream_on:
                        chunks = voice_engine.speak_stream(response, language=lang)
                        play_audio_stream(chunks, block=True)
                    else:
                        audio_bytes = voice_engine.speak(response, language=lang)
                        play_audio(audio_bytes, block=True)
                except Exception as e:
                    print(f"  {RED}Voice error: {e}{RESET}")

    # ── Save history ───────────────────────────────────────────────────────
    save_path = args.save
    if not save_path and history.turn_count > 0:
        save_path = f"data/personal/processed/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if save_path and history.turn_count > 0:
        history.save(Path(save_path))
        print(f"\n{DIM}Chat saved to {save_path}{RESET}")


if __name__ == "__main__":
    main()
