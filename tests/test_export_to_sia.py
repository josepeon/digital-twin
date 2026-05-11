"""T1.2: twin → SIA distillation bridge — pair extraction + filters."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Project root + scripts/ on path so the test can import the script module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.memory import MemoryStore  # noqa: E402

import export_to_sia  # noqa: E402


def _build_memory(db_path: Path, conversation: list[tuple[str, str, str]]):
    """Seed a memory.db with a list of (session_id, role, content) rows."""
    store = MemoryStore(db_path)
    for session_id, role, content in conversation:
        store.add_message(session_id, role, content)
    store.close()


class TestExtractPairs:
    def test_extracts_basic_pair(self, tmp_path):
        db = tmp_path / "memory.db"
        _build_memory(db, [
            ("s1", "user", "What's the best way to ship spatial commerce?"),
            ("s1", "assistant",
             "I'd lean on Arcware for streaming and keep the catalog in "
             "Shopify so we don't duplicate inventory state. The trick is "
             "session affinity per visitor."),
        ])
        pairs = export_to_sia.extract_pairs(db)
        assert len(pairs) == 1
        msgs = pairs[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert pairs[0]["metadata"]["source"] == "twin-memory"

    def test_filters_short_user(self, tmp_path):
        db = tmp_path / "memory.db"
        _build_memory(db, [
            ("s1", "user", "hi"),  # too short
            ("s1", "assistant", "Hey, what's going on with you today?"),
        ])
        assert export_to_sia.extract_pairs(db) == []

    def test_filters_short_assistant(self, tmp_path):
        db = tmp_path / "memory.db"
        _build_memory(db, [
            ("s1", "user", "What do you think about OH's Spline integration?"),
            ("s1", "assistant", "Good."),  # too short
        ])
        assert export_to_sia.extract_pairs(db) == []

    def test_dedupes_repeated_pairs(self, tmp_path):
        db = tmp_path / "memory.db"
        # Add the same pair twice in different sessions
        for session in ("a", "b"):
            _build_memory(db, [
                (session, "user", "What's your take on Unreal for retail?"),
                (session, "assistant",
                 "Unreal gives us the visual fidelity we need; the win is "
                 "spatial UI that feels native to the store."),
            ])
        pairs = export_to_sia.extract_pairs(db)
        assert len(pairs) == 1

    def test_multiturn_conversation(self, tmp_path):
        db = tmp_path / "memory.db"
        _build_memory(db, [
            ("s1", "user", "How would you handle session continuity?"),
            ("s1", "assistant",
             "I'd ladder it: signed JWT in a HttpOnly cookie, with a sliding "
             "TTL bumped on every API hit so casual visitors don't churn."),
            ("s1", "user", "What if they switch devices mid-session?"),
            ("s1", "assistant",
             "QR-handoff with a one-time code is the cleanest pattern — "
             "the new device claims the session on first use, old one "
             "gets booted gracefully."),
        ])
        pairs = export_to_sia.extract_pairs(db)
        assert len(pairs) == 2

    def test_pending_user_without_reply_dropped(self, tmp_path):
        db = tmp_path / "memory.db"
        _build_memory(db, [
            ("s1", "user", "Should we switch DBs?"),
            # No assistant reply — user message in flight
            ("s1", "user", "Actually never mind. What about caching?"),
            ("s1", "assistant",
             "Redis with TTL per key is the right default; reach for "
             "Cloudflare cache only when the data is truly public."),
        ])
        pairs = export_to_sia.extract_pairs(db)
        # Only the second pair (the answered one) should survive
        assert len(pairs) == 1
        assert "caching" in pairs[0]["messages"][0]["content"]

    def test_missing_db_returns_empty(self, tmp_path):
        assert export_to_sia.extract_pairs(tmp_path / "no-such.db") == []

    def test_oversize_filtered(self, tmp_path):
        db = tmp_path / "memory.db"
        long_text = "x" * 10000
        _build_memory(db, [
            ("s1", "user", long_text),
            ("s1", "assistant", "ok " * 50),
        ])
        assert export_to_sia.extract_pairs(db) == []


class TestWriteJsonl:
    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "pairs.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": "hi"}], "metadata": {}},
        ]
        count = export_to_sia.write_jsonl(out, rows)
        assert count == 1
        assert out.exists()

    def test_round_trip(self, tmp_path):
        out = tmp_path / "pairs.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": str(i)}]} for i in range(5)
        ]
        export_to_sia.write_jsonl(out, rows)
        loaded = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(loaded) == 5
        assert loaded[3]["messages"][0]["content"] == "3"
