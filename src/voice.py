"""
Voice module — ElevenLabs TTS with dual-language voice selection.

Handles:
  - Loading voice IDs (English + Spanish clones)
  - Language detection for auto voice selection
  - Text-to-speech (standard and streaming)
  - Audio format conversion
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from elevenlabs.types import VoiceSettings

from src.config import (
    ELEVENLABS_API_KEY,
    TTS_MODEL,
    TTS_MODEL_REALTIME,
    TTS_OUTPUT_FORMAT,
    VOICE_DIR,
    VOICE_ID_EN_PATH,
    VOICE_ID_ES_PATH,
)

# ── Language Detection ─────────────────────────────────────────────────────────

SPANISH_COMMON = {
    "que", "de", "en", "es", "un", "una", "por", "para", "como",
    "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
    "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    "estoy", "vamos", "bueno", "claro", "entonces", "porque",
}


def detect_language(text: str) -> str:
    """Detect EN or ES from text content."""
    words = set(re.findall(r"\w+", text.lower()))
    spanish_count = len(words & SPANISH_COMMON)
    return "es" if spanish_count >= 3 else "en"


# ── Voice Module ───────────────────────────────────────────────────────────────

class VoiceEngine:
    """ElevenLabs TTS engine with dual-language Jose voice clones.

    Usage:
        voice = VoiceEngine()
        audio_bytes = voice.speak("Hey, let me tell you about this project")
        audio_bytes_es = voice.speak("Oye, déjame contarte algo increíble")
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id_en: str | None = None,
        voice_id_es: str | None = None,
    ):
        self._api_key = api_key or ELEVENLABS_API_KEY
        self._voice_id_en = voice_id_en
        self._voice_id_es = voice_id_es
        self._client = None

    # ── Initialization ─────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY not set. "
                    "Get one at https://elevenlabs.io and add it to .env"
                )
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def _load_voice_id(self, path: Path) -> str | None:
        """Load a voice ID from a file."""
        if path.exists():
            return path.read_text().strip()
        return None

    @property
    def voice_id_en(self) -> str | None:
        if self._voice_id_en is None:
            self._voice_id_en = self._load_voice_id(VOICE_ID_EN_PATH)
        return self._voice_id_en

    @property
    def voice_id_es(self) -> str | None:
        if self._voice_id_es is None:
            self._voice_id_es = self._load_voice_id(VOICE_ID_ES_PATH)
        return self._voice_id_es

    def get_voice_id(self, language: str) -> str:
        """Get the voice ID for a language, with fallback."""
        if language == "es" and self.voice_id_es:
            return self.voice_id_es
        if self.voice_id_en:
            return self.voice_id_en
        # If no cloned voices, use a default ElevenLabs voice
        return "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George" — placeholder

    @property
    def has_cloned_voices(self) -> bool:
        return bool(self.voice_id_en or self.voice_id_es)

    # ── Text-to-Speech ─────────────────────────────────────────────────────

    def speak(
        self,
        text: str,
        language: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
    ) -> bytes:
        """Convert text to speech using Jose's cloned voice.

        Args:
            text: The text to speak.
            language: Force language ("en" or "es"). Auto-detects if None.
            model_id: ElevenLabs model ID override.
            output_format: Audio format override.

        Returns:
            Audio bytes (MP3 by default).
        """
        lang = language or detect_language(text)
        voice_id = self.get_voice_id(lang)
        client = self._get_client()

        # Use high similarity + speaker boost for better clone fidelity
        voice_settings = VoiceSettings(
            stability=0.3,
            similarity_boost=1.0,
            style=0.5,
            use_speaker_boost=True,
        )

        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id or TTS_MODEL,
            output_format=output_format or TTS_OUTPUT_FORMAT,
            voice_settings=voice_settings,
        )

        # Collect all chunks into bytes
        audio_buffer = io.BytesIO()
        for chunk in audio_generator:
            audio_buffer.write(chunk)
        return audio_buffer.getvalue()

    def speak_stream(
        self,
        text: str,
        language: str | None = None,
        model_id: str | None = None,
    ):
        """Stream TTS audio for low-latency applications.

        Args:
            text: The text to speak.
            language: Force language ("en" or "es"). Auto-detects if None.
            model_id: ElevenLabs model for streaming (default: Flash v2.5).

        Yields:
            Audio chunks (bytes).
        """
        lang = language or detect_language(text)
        voice_id = self.get_voice_id(lang)
        client = self._get_client()

        voice_settings = VoiceSettings(
            stability=0.3,
            similarity_boost=1.0,
            style=0.5,
            use_speaker_boost=True,
        )

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id or TTS_MODEL_REALTIME,
            output_format=TTS_OUTPUT_FORMAT,
            voice_settings=voice_settings,
        )

        yield from audio_stream

    # ── Utility ────────────────────────────────────────────────────────────

    def list_voices(self) -> list[dict]:
        """List all available ElevenLabs voices (for debugging)."""
        client = self._get_client()
        response = client.voices.get_all()
        return [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "labels": v.labels,
                "category": v.category,
            }
            for v in response.voices
        ]
