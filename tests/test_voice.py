"""T2.3 part 2: VoiceEngine — language detection, voice routing, cost recording.

We mock the ElevenLabs client so tests don't make real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.voice import (
    SPANISH_COMMON,
    VoiceEngine,
    _record_elevenlabs_usage,
    detect_language,
)


# ---------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------


class TestDetectLanguage:
    def test_obvious_english(self):
        assert detect_language("Hey what are you working on today") == "en"

    def test_obvious_spanish(self):
        assert detect_language(
            "Hola, cómo estás? La familia está bien por allá?"
        ) == "es"

    def test_one_or_two_spanish_words_stays_english(self):
        # Threshold is 3 — "de" alone doesn't flip
        assert detect_language("From de la la-la song") == "en"

    def test_empty_string_defaults_english(self):
        assert detect_language("") == "en"

    def test_cuban_slang_recognized(self):
        # "qué bolá" + family terms
        assert detect_language("Asere qué bolá, todo bien con la familia?") == "es"

    def test_spanish_common_dictionary_non_empty(self):
        # Smoke-check the shared dictionary so it doesn't accidentally empty
        assert len(SPANISH_COMMON) >= 30


# ---------------------------------------------------------------------
# Voice routing
# ---------------------------------------------------------------------


class TestVoiceRouting:
    def test_explicit_en_voice_returned(self):
        engine = VoiceEngine(voice_id_en="EN-ID", voice_id_es="ES-ID")
        assert engine.get_voice_id("en") == "EN-ID"

    def test_explicit_es_voice_returned(self):
        engine = VoiceEngine(voice_id_en="EN-ID", voice_id_es="ES-ID")
        assert engine.get_voice_id("es") == "ES-ID"

    def test_falls_back_when_es_missing(self):
        # If Spanish clone hasn't been set up, fall back to English voice
        engine = VoiceEngine(voice_id_en="EN-ID", voice_id_es=None)
        # The fallback path uses property accessors; both should yield EN
        with patch.object(VoiceEngine, "voice_id_es", new=None):
            assert engine.get_voice_id("es") == "EN-ID"

    def test_no_voices_at_all_falls_back_to_placeholder(self):
        engine = VoiceEngine(voice_id_en=None, voice_id_es=None)
        with patch.object(VoiceEngine, "voice_id_en", new=None):
            with patch.object(VoiceEngine, "voice_id_es", new=None):
                # Falls back to ElevenLabs "George" — non-empty string
                fallback = engine.get_voice_id("en")
                assert isinstance(fallback, str) and len(fallback) > 0
                # And reports has_cloned_voices=False
                assert engine.has_cloned_voices is False


# ---------------------------------------------------------------------
# Cost recording helper
# ---------------------------------------------------------------------


class TestRecordElevenLabsUsage:
    def test_records_character_count(self):
        from src.observability import begin_run, get_tracker

        begin_run()
        _record_elevenlabs_usage("hello world!", "voice-en", "en")
        totals = get_tracker().total()
        assert totals.calls == 1
        assert totals.completion_tokens == len("hello world!")

    def test_empty_text_is_no_op(self):
        from src.observability import begin_run, get_tracker

        begin_run()
        _record_elevenlabs_usage("", "voice-en", "en")
        _record_elevenlabs_usage(None, "voice-en", "en")
        assert get_tracker().total().calls == 0


# ---------------------------------------------------------------------
# Speak with mocked client
# ---------------------------------------------------------------------


class TestSpeakMocked:
    def _engine_with_mock(self, audio_chunks=None):
        engine = VoiceEngine(voice_id_en="VOICE-EN", voice_id_es="VOICE-ES")
        mock_client = MagicMock()
        # The convert() call returns an iterable of byte chunks
        mock_client.text_to_speech.convert.return_value = iter(
            audio_chunks or [b"chunk1", b"chunk2"]
        )
        engine._client = mock_client
        # API-key check is gated behind _get_client; setting _client bypasses it
        return engine, mock_client

    def test_speak_returns_concatenated_bytes(self):
        engine, _ = self._engine_with_mock(
            audio_chunks=[b"hello", b"world"]
        )
        result = engine.speak("hi there")
        assert result == b"helloworld"

    def test_speak_uses_es_voice_for_spanish_text(self):
        engine, mock_client = self._engine_with_mock()
        engine.speak("Hola, cómo estás? La familia está bien?")
        call = mock_client.text_to_speech.convert.call_args
        assert call.kwargs["voice_id"] == "VOICE-ES"

    def test_speak_uses_en_voice_for_english_text(self):
        engine, mock_client = self._engine_with_mock()
        engine.speak("Hey what's up")
        call = mock_client.text_to_speech.convert.call_args
        assert call.kwargs["voice_id"] == "VOICE-EN"

    def test_explicit_language_overrides_detection(self):
        engine, mock_client = self._engine_with_mock()
        # English-looking text, but forced Spanish
        engine.speak("Hello there friend", language="es")
        call = mock_client.text_to_speech.convert.call_args
        assert call.kwargs["voice_id"] == "VOICE-ES"

    def test_speak_records_cost(self):
        from src.observability import begin_run, get_tracker

        engine, _ = self._engine_with_mock()
        begin_run()
        engine.speak("Hello there!")
        totals = get_tracker().total()
        assert totals.calls == 1
        assert totals.completion_tokens == len("Hello there!")


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------


class TestSpeakStream:
    def test_speak_stream_yields_chunks(self):
        engine = VoiceEngine(voice_id_en="VOICE-EN", voice_id_es="VOICE-ES")
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter(
            [b"frag1", b"frag2", b"frag3"]
        )
        engine._client = mock_client
        chunks = list(engine.speak_stream("Hi"))
        assert chunks == [b"frag1", b"frag2", b"frag3"]

    def test_speak_stream_records_cost_upfront(self):
        from src.observability import begin_run, get_tracker

        engine = VoiceEngine(voice_id_en="VOICE-EN", voice_id_es="VOICE-ES")
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"x"])
        engine._client = mock_client

        begin_run()
        list(engine.speak_stream("12345"))
        totals = get_tracker().total()
        assert totals.completion_tokens == 5
