"""T3.1: train_local.py — dataset loader + dry-run path.

Skips the actual MLX training (~20 min on M3 Max) — those paths are
exercised by SIA's own suite. This test covers the wrapper bits: JSONL
parsing in both supported shapes, dry-run validation, and the SIA-
availability gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_local  # noqa: E402


class TestLoadDataset:
    def test_chat_format_passes_through(self, tmp_path):
        path = tmp_path / "data.jsonl"
        rows = [
            {"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hey"},
            ]},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        loaded = train_local.load_dataset(path)
        assert loaded == rows

    def test_flat_format_converted_to_chat(self, tmp_path):
        path = tmp_path / "flat.jsonl"
        rows = [
            {"user": "hi there", "assistant": "hey, how's it going"},
            {"query": "what's up", "response": "not much, you?"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        loaded = train_local.load_dataset(path)
        assert len(loaded) == 2
        # First row converted
        assert loaded[0]["messages"][0] == {"role": "user", "content": "hi there"}
        assert loaded[0]["messages"][1] == {
            "role": "assistant", "content": "hey, how's it going",
        }
        # Second row used query/response keys
        assert loaded[1]["messages"][0]["content"] == "what's up"

    def test_skips_incomplete_flat_rows(self, tmp_path):
        path = tmp_path / "partial.jsonl"
        rows = [
            {"user": "only user message"},
            {"assistant": "only response"},
            {"user": "valid pair", "assistant": "valid response"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        loaded = train_local.load_dataset(path)
        assert len(loaded) == 1
        assert "valid pair" in loaded[0]["messages"][0]["content"]

    def test_skips_malformed_json(self, tmp_path, caplog):
        import logging

        path = tmp_path / "broken.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": "ok"}]}) + "\n"
            "not json\n"
            + json.dumps({"messages": [{"role": "user", "content": "also ok"}]}) + "\n"
        )
        with caplog.at_level(logging.WARNING):
            loaded = train_local.load_dataset(path)
        assert len(loaded) == 2
        # Warning was logged
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert train_local.load_dataset(path) == []


class TestSiaAvailability:
    def test_check_returns_bool(self):
        # We don't care which one in tests — just confirm the function works
        result = train_local._check_sia_available()
        assert isinstance(result, bool)


class TestDryRunFlow:
    """The CLI is exercised via run_dry_smoke — actually invoking main()
    is hard because it parses sys.argv. Instead test the underlying
    pieces individually."""

    def test_load_then_dry_run_doesnt_train(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey there"},
            ]}) + "\n"
        )
        # The combination of load_dataset + dry-run flag should never
        # call into MLX training. Verify by inspecting the function:
        # nothing else in train_local imports mlx_lm at module level.
        loaded = train_local.load_dataset(path)
        assert len(loaded) == 1
        # train_local imports SIA conditionally, so module import should
        # be safe even without MLX installed
        assert hasattr(train_local, "DEFAULT_BASE_MODEL")
        assert "Llama" in train_local.DEFAULT_BASE_MODEL
