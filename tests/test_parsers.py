"""Tests for parsers — using synthetic test data."""

import json
import tempfile
from pathlib import Path

import pytest

from src.models import Source


class TestChatGPTParser:
    def test_parse_basic_conversation(self, tmp_path):
        """Test parsing a minimal ChatGPT export."""
        from scripts.parse_chatgpt import parse_chatgpt_export

        conversation = [
            {
                "title": "Test conversation",
                "mapping": {
                    "root": {
                        "message": None,
                        "parent": None,
                        "children": ["msg1"],
                    },
                    "msg1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["What is machine learning?"]},
                            "create_time": 1705000000.0,
                        },
                        "parent": "root",
                        "children": ["msg2"],
                    },
                    "msg2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Machine learning is a subset of AI..."]},
                            "create_time": 1705000010.0,
                        },
                        "parent": "msg1",
                        "children": ["msg3"],
                    },
                    "msg3": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["That makes sense, I've been thinking about using it for my project at work"]},
                            "create_time": 1705000020.0,
                        },
                        "parent": "msg2",
                        "children": [],
                    },
                },
            }
        ]

        filepath = tmp_path / "conversations.json"
        filepath.write_text(json.dumps(conversation))

        messages = parse_chatgpt_export(filepath, scrub_pii=False)
        assert len(messages) == 2  # Two user messages
        assert messages[0].source == Source.CHATGPT
        assert "machine learning" in messages[0].text.lower()

    def test_empty_export(self, tmp_path):
        filepath = tmp_path / "conversations.json"
        filepath.write_text("[]")

        from scripts.parse_chatgpt import parse_chatgpt_export
        messages = parse_chatgpt_export(filepath)
        assert messages == []


class TestWhatsAppParser:
    def test_parse_bracket_format(self, tmp_path):
        """Test WhatsApp bracket timestamp format."""
        from scripts.parse_whatsapp import parse_whatsapp_file

        content = """[1/15/24, 2:30:15 PM] Other Person: Hey, how are you?
[1/15/24, 2:31:00 PM] Jose Peon: Doing great, working on this really cool AI project right now
[1/15/24, 2:32:00 PM] Other Person: Nice! Tell me more about it
[1/15/24, 2:33:00 PM] Jose Peon: It's a digital twin that can replicate how I think and speak
"""
        filepath = tmp_path / "chat.txt"
        filepath.write_text(content)

        messages = parse_whatsapp_file(filepath, scrub_pii=False)
        assert len(messages) == 2
        assert messages[0].source == Source.WHATSAPP
        assert "AI project" in messages[0].text

    def test_parse_dash_format(self, tmp_path):
        from scripts.parse_whatsapp import parse_whatsapp_file

        content = """15/01/2024, 14:30 - Other Person: Oye que tal?
15/01/2024, 14:31 - Jose Peon: Todo bien hermano, aquí trabajando en el proyecto este
"""
        filepath = tmp_path / "chat.txt"
        filepath.write_text(content)

        messages = parse_whatsapp_file(filepath, scrub_pii=False)
        assert len(messages) == 1
        assert messages[0].context  # Should have the previous message as context

    def test_skips_system_messages(self, tmp_path):
        from scripts.parse_whatsapp import parse_whatsapp_file

        content = """[1/15/24, 2:30:00 PM] Jose Peon: Messages and calls are end-to-end encrypted
[1/15/24, 2:30:15 PM] Jose Peon: <Media omitted>
[1/15/24, 2:31:00 PM] Jose Peon: Hey this is a real message that should definitely pass the filter
"""
        filepath = tmp_path / "chat.txt"
        filepath.write_text(content)

        messages = parse_whatsapp_file(filepath, scrub_pii=False)
        assert len(messages) == 1
        assert "real message" in messages[0].text


class TestParseFileWrappers:
    """parse_file(path) is the dispatch contract for ingest_new_data.py."""

    WHATSAPP_CHAT = """[1/15/24, 2:30:15 PM] Other Person: Hey, how are you?
[1/15/24, 2:31:00 PM] Jose Peon: Doing great, working on this really cool AI project right now
"""

    def test_all_parsers_expose_parse_file(self):
        import importlib

        for name in (
            "parse_chatgpt", "parse_whatsapp", "parse_instagram",
            "parse_imessage", "parse_email",
        ):
            module = importlib.import_module(f"scripts.{name}")
            assert callable(getattr(module, "parse_file", None)), name

    def test_whatsapp_parse_file_appends_processed(self, tmp_path, monkeypatch):
        import json

        import src.config as config
        from scripts.parse_whatsapp import parse_file

        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")

        chat = tmp_path / "chat.txt"
        chat.write_text(self.WHATSAPP_CHAT)

        messages = parse_file(chat)
        assert len(messages) == 1

        out = tmp_path / "processed" / "whatsapp_parsed.jsonl"
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 1
        assert "AI project" in rows[0]["text"]

        # Second ingest appends rather than overwriting
        parse_file(chat)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 2

    def test_email_parse_file_requires_addresses(self, tmp_path, monkeypatch):
        import pytest

        from scripts.parse_email import parse_file

        monkeypatch.delenv("JOSE_EMAILS", raising=False)
        with pytest.raises(ValueError, match="JOSE_EMAILS"):
            parse_file(tmp_path)
