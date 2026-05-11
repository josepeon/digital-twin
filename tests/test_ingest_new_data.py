"""T2.2: real-time learning — incremental data ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ingest_new_data as ingest  # noqa: E402


class TestSourceDetection:
    @pytest.mark.parametrize("name,expected", [
        ("instagram_messages.html", "instagram"),
        ("messages_instagram.html", "instagram"),
        ("chatgpt_export_2026.json", "chatgpt"),
        ("chat.db", "imessage"),
        ("imessage_dump.json", "imessage"),
        ("WhatsApp Chat with Mom.txt", "whatsapp"),
        ("wa_export.txt", "whatsapp"),
        ("email_export.eml", "email"),
        ("mbox_archive.txt", "email"),
        ("random_file.csv", None),
        ("README.md", None),
    ])
    def test_detects_source(self, tmp_path, name, expected):
        path = tmp_path / name
        path.touch()
        assert ingest._detect_source(path) == expected


class TestHashFile:
    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_text("hello world")
        b.write_text("hello world")
        assert ingest.hash_file(a) == ingest.hash_file(b)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_text("hello")
        b.write_text("world")
        assert ingest.hash_file(a) != ingest.hash_file(b)


class TestIngestState:
    def test_unseen_at_start(self, tmp_path):
        state = ingest.IngestState(tmp_path / "state.json")
        assert state.seen("hash-1") is False

    def test_mark_then_seen(self, tmp_path):
        state = ingest.IngestState(tmp_path / "state.json")
        path = tmp_path / "fake.txt"
        path.touch()
        state.mark("hash-1", path, "instagram", 42)
        assert state.seen("hash-1") is True

    def test_total_ingested(self, tmp_path):
        state = ingest.IngestState(tmp_path / "state.json")
        path = tmp_path / "fake.txt"
        path.touch()
        state.mark("a", path, "instagram", 10)
        state.mark("b", path, "chatgpt", 5)
        assert state.total_ingested == 15

    def test_state_persists_across_instances(self, tmp_path):
        state_path = tmp_path / "state.json"
        first = ingest.IngestState(state_path)
        path = tmp_path / "fake.txt"
        path.touch()
        first.mark("hash-1", path, "whatsapp", 100)

        # Fresh instance reads the on-disk state
        second = ingest.IngestState(state_path)
        assert second.seen("hash-1") is True
        assert second.total_ingested == 100

    def test_corrupt_state_recovers(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("not json")
        # Should not raise — falls back to empty
        state = ingest.IngestState(state_path)
        assert state.total_ingested == 0


class TestDiscoverNewFiles:
    def test_finds_recognised_files(self, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        (incoming / "instagram_messages.html").write_text("<html></html>")
        (incoming / "chatgpt_export.json").write_text("{}")
        (incoming / "ignored.csv").write_text("a,b,c")

        state = ingest.IngestState(tmp_path / "state.json")
        new = ingest.discover_new_files(incoming, state)
        sources = {item[1] for item in new}
        assert sources == {"instagram", "chatgpt"}

    def test_skips_already_seen(self, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        f = incoming / "instagram_messages.html"
        f.write_text("<html></html>")

        state = ingest.IngestState(tmp_path / "state.json")
        first = ingest.discover_new_files(incoming, state)
        assert len(first) == 1

        # Mark + retry — should be empty
        _, source, file_hash = first[0]
        state.mark(file_hash, f, source, 0)
        second = ingest.discover_new_files(incoming, state)
        assert second == []

    def test_missing_incoming_dir(self, tmp_path):
        state = ingest.IngestState(tmp_path / "state.json")
        new = ingest.discover_new_files(tmp_path / "no-such-dir", state)
        assert new == []


class TestRunOnce:
    def test_no_files_returns_zero(self, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        added = ingest.run_once(
            incoming=incoming,
            state_path=tmp_path / "state.json",
        )
        assert added == 0

    def test_unknown_files_ignored(self, tmp_path):
        incoming = tmp_path / "incoming"
        incoming.mkdir()
        (incoming / "totally_random.xyz").write_text("data")
        added = ingest.run_once(
            incoming=incoming,
            state_path=tmp_path / "state.json",
        )
        assert added == 0
