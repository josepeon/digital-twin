"""T2.4: multi-persona mode — addendum routing + API plumbing."""

from __future__ import annotations

import pytest

from src.personality_engine import (
    JOSE_SYSTEM_PROMPT,
    PERSONA_ADDENDA,
    PERSONA_REGISTER_FILTER,
    VALID_PERSONAS,
    system_prompt_for,
)


class TestPersonaAddenda:
    def test_default_addendum_is_empty(self):
        assert PERSONA_ADDENDA["default"] == ""

    def test_known_personas_documented(self):
        # Roadmap promised four modes
        assert set(VALID_PERSONAS) == {
            "default", "professional", "casual", "creative",
        }

    def test_each_addendum_non_default_has_text(self):
        for mode in VALID_PERSONAS:
            if mode == "default":
                continue
            assert len(PERSONA_ADDENDA[mode]) > 50

    def test_addenda_mention_their_mode(self):
        for mode in ("professional", "casual", "creative"):
            assert mode.upper() in PERSONA_ADDENDA[mode]


class TestSystemPromptFor:
    def test_default_returns_base_prompt(self):
        assert system_prompt_for("default") == JOSE_SYSTEM_PROMPT

    def test_default_when_no_arg(self):
        assert system_prompt_for() == JOSE_SYSTEM_PROMPT

    def test_professional_extends_base(self):
        prompt = system_prompt_for("professional")
        # Base prompt intact at the top
        assert prompt.startswith(JOSE_SYSTEM_PROMPT)
        # Addendum present
        assert "PROFESSIONAL" in prompt

    def test_casual_extends_base(self):
        prompt = system_prompt_for("casual")
        assert prompt.startswith(JOSE_SYSTEM_PROMPT)
        assert "CASUAL" in prompt

    def test_creative_extends_base(self):
        prompt = system_prompt_for("creative")
        assert prompt.startswith(JOSE_SYSTEM_PROMPT)
        assert "CREATIVE" in prompt

    def test_unknown_falls_back_to_default(self, caplog):
        # Should not raise — falls back with a warning
        prompt = system_prompt_for("madeup-mode")
        assert prompt == JOSE_SYSTEM_PROMPT


class TestRegisterFilter:
    def test_professional_pins_register(self):
        assert PERSONA_REGISTER_FILTER["professional"] == "professional"

    def test_casual_pins_register(self):
        assert PERSONA_REGISTER_FILTER["casual"] == "casual"

    def test_default_leaves_register_open(self):
        # None = "let detect_register decide"
        assert PERSONA_REGISTER_FILTER["default"] is None

    def test_creative_pulls_from_both(self):
        # Creative doesn't pin a single register
        assert PERSONA_REGISTER_FILTER["creative"] is None


# ---------------------------------------------------------------------
# API plumbing — uses the existing test_twin_api fixture-style mock
# ---------------------------------------------------------------------


class TestApiPlumbing:
    """The mode kwarg propagates from the HTTP request through to the engine."""

    @pytest.fixture
    def client(self, monkeypatch):
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from src import twin_api as api_module

        fake_engine = MagicMock()
        fake_engine.ask.return_value = {
            "response": "ok", "language": "en", "register": "casual",
            "model": "llama", "examples_used": 0,
        }
        fake_voice = MagicMock()
        fake_voice.has_cloned_voices = True
        fake_voice.speak.return_value = b""

        monkeypatch.setattr(api_module, "PersonalityEngine", lambda: fake_engine)
        monkeypatch.setattr(api_module, "VoiceEngine", lambda: fake_voice)
        monkeypatch.setattr(api_module, "TWIN_API_KEY", "")

        with TestClient(api_module.app) as c:
            c._fake_engine = fake_engine  # type: ignore
            yield c

    def test_mode_propagated_to_engine(self, client):
        client.post(
            "/ask-jose",
            json={"query": "hi", "mode": "professional"},
        )
        call = client._fake_engine.ask.call_args  # type: ignore
        assert call.kwargs["mode"] == "professional"

    def test_mode_defaults_to_default_when_omitted(self, client):
        client.post("/ask-jose", json={"query": "hi"})
        call = client._fake_engine.ask.call_args  # type: ignore
        assert call.kwargs["mode"] == "default"

    def test_mode_propagates_to_voice_endpoint(self, client):
        client.post(
            "/ask-jose/voice",
            json={"query": "hi", "mode": "casual"},
        )
        call = client._fake_engine.ask.call_args  # type: ignore
        assert call.kwargs["mode"] == "casual"
