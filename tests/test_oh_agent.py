"""T3.2: OH agent adapter — protocol parsing + streaming."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src import oh_agent


class TestParseIncoming:
    def test_query_with_all_fields(self):
        parsed = oh_agent.parse_incoming({
            "type": "query",
            "content": "Tell me about Nooon",
            "mode": "creative",
            "session_id": "s-1",
            "visitor_id": "v-42",
        })
        assert parsed["type"] == "query"
        assert parsed["content"] == "Tell me about Nooon"
        assert parsed["mode"] == "creative"
        assert parsed["session_id"] == "s-1"
        assert parsed["visitor_id"] == "v-42"

    def test_query_defaults_to_professional_mode(self):
        parsed = oh_agent.parse_incoming({
            "type": "query",
            "content": "What's OH?",
        })
        # No mode specified → professional (commerce context)
        assert parsed["mode"] == "professional"

    def test_accepts_json_string(self):
        raw = json.dumps({"type": "query", "content": "Hi"})
        parsed = oh_agent.parse_incoming(raw)
        assert parsed["type"] == "query"
        assert parsed["content"] == "Hi"

    def test_presence_message(self):
        parsed = oh_agent.parse_incoming({
            "type": "presence",
            "session_id": "s-1",
            "visitor_id": "v-42",
        })
        assert parsed["type"] == "presence"
        assert parsed["session_id"] == "s-1"

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            oh_agent.parse_incoming("not json at all")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown type"):
            oh_agent.parse_incoming({"type": "fart"})

    def test_empty_query_content_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            oh_agent.parse_incoming({"type": "query", "content": ""})

    def test_whitespace_query_content_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            oh_agent.parse_incoming({"type": "query", "content": "   "})

    def test_non_string_payload_rejected(self):
        with pytest.raises(ValueError, match="unexpected payload"):
            oh_agent.parse_incoming(42)


class TestEncoders:
    def test_encode_delta(self):
        out = json.loads(oh_agent.encode_delta("Hello", "s-1"))
        assert out["type"] == "delta"
        assert out["content"] == "Hello"
        assert out["session_id"] == "s-1"
        assert "v" in out

    def test_encode_done(self):
        out = json.loads(oh_agent.encode_done("s-1", language="en"))
        assert out["type"] == "done"
        assert out["language"] == "en"

    def test_encode_error(self):
        out = json.loads(oh_agent.encode_error("something broke"))
        assert out["type"] == "error"
        assert out["content"] == "something broke"

    def test_encode_presence_ack_includes_agent_id(self):
        out = json.loads(oh_agent.encode_presence_ack("s-1"))
        assert out["type"] == "presence_ack"
        assert out["agent"] == "jose-twin"


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_emits_delta_frames_then_done(self):
        fake_engine = MagicMock()
        fake_engine.ask_stream.return_value = iter(["Hello", " there"])

        parsed = oh_agent.parse_incoming({
            "type": "query", "content": "hi", "session_id": "s-1",
        })

        frames = []
        async for frame in oh_agent.stream_response(fake_engine, parsed):
            frames.append(json.loads(frame))

        assert frames[0]["type"] == "delta"
        assert frames[0]["content"] == "Hello"
        assert frames[1]["type"] == "delta"
        assert frames[-1]["type"] == "done"
        # session_id is carried through every frame
        assert all(f.get("session_id") == "s-1" for f in frames)

    @pytest.mark.asyncio
    async def test_skips_empty_deltas(self):
        fake_engine = MagicMock()
        fake_engine.ask_stream.return_value = iter(["", "real", "", None])

        parsed = oh_agent.parse_incoming({"type": "query", "content": "hi"})

        frames = []
        async for frame in oh_agent.stream_response(fake_engine, parsed):
            frames.append(json.loads(frame))

        # Empty/None deltas skipped; "real" + "done" remain
        delta_frames = [f for f in frames if f["type"] == "delta"]
        assert len(delta_frames) == 1
        assert delta_frames[0]["content"] == "real"

    @pytest.mark.asyncio
    async def test_engine_exception_becomes_error_frame(self):
        fake_engine = MagicMock()
        fake_engine.ask_stream.side_effect = RuntimeError("model unavailable")

        parsed = oh_agent.parse_incoming({"type": "query", "content": "hi"})

        frames = []
        async for frame in oh_agent.stream_response(fake_engine, parsed):
            frames.append(json.loads(frame))

        # error frame + done frame (always)
        types = [f["type"] for f in frames]
        assert "error" in types
        assert types[-1] == "done"

    @pytest.mark.asyncio
    async def test_mode_passed_to_engine(self):
        fake_engine = MagicMock()
        fake_engine.ask_stream.return_value = iter(["ok"])

        parsed = oh_agent.parse_incoming({
            "type": "query", "content": "design ideas", "mode": "creative",
        })

        async for _ in oh_agent.stream_response(fake_engine, parsed):
            pass

        call = fake_engine.ask_stream.call_args
        assert call.kwargs["mode"] == "creative"
