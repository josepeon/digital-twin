"""Tests for the data models."""

from datetime import datetime, timezone

from src.models import (
    ConversationTurn,
    Language,
    ParsedMessage,
    Register,
    Source,
    TrainingExample,
)


class TestParsedMessage:
    def test_create_basic(self):
        msg = ParsedMessage(
            text="Working on this AI project",
            source=Source.CHATGPT,
        )
        assert msg.text == "Working on this AI project"
        assert msg.source == Source.CHATGPT
        assert msg.language == Language.EN
        assert msg.register == Register.NEUTRAL

    def test_word_count(self):
        msg = ParsedMessage(text="one two three four five", source=Source.WHATSAPP)
        assert msg.word_count == 5

    def test_to_dict_roundtrip(self):
        msg = ParsedMessage(
            text="Oye, esto está buenísimo",
            source=Source.INSTAGRAM,
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
            context="What are you working on?",
            register=Register.CASUAL,
            language=Language.ES,
            metadata={"type": "dm"},
        )
        d = msg.to_dict()
        restored = ParsedMessage.from_dict(d)
        assert restored.text == msg.text
        assert restored.source == msg.source
        assert restored.language == msg.language
        assert restored.register == msg.register
        assert restored.metadata == msg.metadata

    def test_from_dict_defaults(self):
        d = {"text": "Hello", "source": "email"}
        msg = ParsedMessage.from_dict(d)
        assert msg.language == Language.EN
        assert msg.register == Register.NEUTRAL
        assert msg.timestamp is None


class TestTrainingExample:
    def test_create_and_roundtrip(self):
        ex = TrainingExample(
            id="chatgpt_000001",
            messages=[
                ConversationTurn(role="user", content="What do you think about AI?"),
                ConversationTurn(role="assistant", content="I think it's transformative"),
            ],
            source=Source.CHATGPT,
            quality_score=0.85,
        )
        d = ex.to_dict()
        restored = TrainingExample.from_dict(d)
        assert restored.id == ex.id
        assert len(restored.messages) == 2
        assert restored.quality_score == 0.85


class TestEnums:
    def test_source_values(self):
        assert Source.CHATGPT.value == "chatgpt"
        assert Source.INSTAGRAM.value == "instagram"
        assert Source.WHATSAPP.value == "whatsapp"
        assert Source.IMESSAGE.value == "imessage"
        assert Source.EMAIL.value == "email"

    def test_language_values(self):
        assert Language.EN.value == "en"
        assert Language.ES.value == "es"

    def test_register_values(self):
        assert Register.CASUAL.value == "casual"
        assert Register.PROFESSIONAL.value == "professional"
        assert Register.FAMILY.value == "family"
