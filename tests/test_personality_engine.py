"""Tests for the personality engine — context detection and prompt assembly."""

import pytest

from src.personality_engine import (
    JOSE_SYSTEM_PROMPT,
    PersonalityEngine,
    detect_context,
    detect_language,
    detect_register,
)


class TestDetectLanguage:
    def test_english_text(self):
        assert detect_language("Hey, what are you working on?") == "en"

    def test_spanish_text(self):
        assert detect_language("Oye, cómo está la familia? Todo bien?") == "es"

    def test_mixed_defaults_english(self):
        assert detect_language("Hello world") == "en"

    def test_cuban_spanish(self):
        assert detect_language("Asere qué bolá? Dale vamos para la casa") == "es"


class TestDetectRegister:
    def test_casual(self):
        assert detect_register("hey bro what's up lol") == "casual"

    def test_professional(self):
        assert detect_register("Regarding the meeting deadline") == "professional"

    def test_family(self):
        assert detect_register("Cómo está mamá y la familia?") == "family"

    def test_neutral(self):
        assert detect_register("Tell me about the weather") == "neutral"


class TestDetectContext:
    def test_family_forces_spanish(self):
        ctx = detect_context("Cómo está mamá?")
        assert ctx["language"] == "es"
        assert ctx["register"] == "family"

    def test_professional_forces_english(self):
        ctx = detect_context("Regarding the meeting deadline please")
        assert ctx["language"] == "en"
        assert ctx["register"] == "professional"

    def test_casual_preserves_detected_language(self):
        ctx = detect_context("hey dude what's up lol")
        assert ctx["register"] == "casual"
        assert ctx["language"] == "en"


class TestPersonalityEngine:
    def test_build_prompt_basic(self):
        engine = PersonalityEngine()
        messages = engine.build_prompt("What are you working on?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Jose Peon" in messages[0]["content"]

    def test_build_prompt_with_examples(self):
        engine = PersonalityEngine()
        examples = [
            {"text": "Working on OH, it's going great", "context": "What's up?", "source": "whatsapp"},
            {"text": "Building AI tools for our platform", "context": "Any projects?", "source": "chatgpt"},
        ]
        messages = engine.build_prompt("What are you working on?", examples=examples)
        system_content = messages[0]["content"]
        assert "Working on OH" in system_content
        assert "Building AI tools" in system_content

    def test_build_prompt_spanish_context(self):
        engine = PersonalityEngine()
        ctx = {"language": "es", "register": "family"}
        messages = engine.build_prompt("Cómo estás?", context=ctx)
        system_content = messages[0]["content"]
        assert "Spanish" in system_content or "warmth" in system_content

    def test_system_prompt_has_identity(self):
        assert "Jose Peon" in JOSE_SYSTEM_PROMPT
        assert "Cuba" in JOSE_SYSTEM_PROMPT
        assert "English" in JOSE_SYSTEM_PROMPT
        assert "Spanish" in JOSE_SYSTEM_PROMPT
