"""T2.3 part 3: twin_api routes — auth, health, ask-jose, metrics, ws.

PersonalityEngine and VoiceEngine are mocked so tests don't load
ChromaDB or call ElevenLabs/Groq.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with the engines mocked out."""
    # Engines are constructed in the lifespan; replace constructors so the
    # tests don't load real ChromaDB / ElevenLabs.
    from src import twin_api as api_module

    fake_engine = MagicMock()
    fake_engine.ask.return_value = {
        "response": "hello there",
        "language": "en",
        "register": "casual",
        "model": "llama-3.3-70b-versatile",
        "examples_used": 3,
    }
    fake_engine._get_collection.return_value = MagicMock(count=lambda: 100)

    fake_voice = MagicMock()
    fake_voice.has_cloned_voices = True
    fake_voice.speak.return_value = b"fake-mp3-bytes"

    monkeypatch.setattr(api_module, "PersonalityEngine", lambda: fake_engine)
    monkeypatch.setattr(api_module, "VoiceEngine", lambda: fake_voice)

    # No API key in test env -> auth becomes a no-op (dev mode)
    monkeypatch.setattr(api_module, "TWIN_API_KEY", "")

    with TestClient(api_module.app) as c:
        c._fake_engine = fake_engine  # type: ignore
        c._fake_voice = fake_voice  # type: ignore
        yield c


# ---------------------------------------------------------------------
# Health + metrics
# ---------------------------------------------------------------------


class TestHealth:
    def test_returns_ok_with_engine_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["rag_index_loaded"] is True
        assert body["voice_clones_loaded"] is True


class TestMetrics:
    def test_endpoint_returns_prometheus_text(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        for metric in (
            "sia_requests_total",
            "twin_voice_characters_total",
            "twin_rag_hits_total",
            "twin_active_sessions",
        ):
            assert f"# HELP {metric}" in body


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------


class TestAuth:
    def test_open_when_no_api_key_set(self, client):
        # Fixture sets TWIN_API_KEY="" — endpoint should be accessible
        response = client.post(
            "/ask-jose",
            json={"query": "hi"},
        )
        assert response.status_code == 200

    def test_rejects_bad_key(self, client, monkeypatch):
        from src import twin_api as api_module

        # Re-enable auth for this test
        monkeypatch.setattr(api_module, "TWIN_API_KEY", "secret-key")

        response = client.post(
            "/ask-jose",
            json={"query": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    def test_accepts_correct_key_bearer(self, client, monkeypatch):
        from src import twin_api as api_module

        monkeypatch.setattr(api_module, "TWIN_API_KEY", "secret-key")
        response = client.post(
            "/ask-jose",
            json={"query": "hi"},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert response.status_code == 200

    def test_accepts_x_api_key_header(self, client, monkeypatch):
        from src import twin_api as api_module

        monkeypatch.setattr(api_module, "TWIN_API_KEY", "secret-key")
        response = client.post(
            "/ask-jose",
            json={"query": "hi"},
            headers={"X-API-Key": "secret-key"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------
# /ask-jose (text)
# ---------------------------------------------------------------------


class TestAskJose:
    def test_returns_engine_output(self, client):
        response = client.post("/ask-jose", json={"query": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "hello there"
        assert body["language"] == "en"

    def test_validates_empty_query(self, client):
        response = client.post("/ask-jose", json={"query": ""})
        # Pydantic min_length=1 — should reject
        assert response.status_code == 422

    def test_validates_query_length(self, client):
        response = client.post(
            "/ask-jose", json={"query": "x" * 6000}
        )
        assert response.status_code == 422

    def test_passes_session_id_to_engine(self, client):
        client.post(
            "/ask-jose",
            json={"query": "hi", "session_id": "sess-123"},
        )
        # Engine.ask should have been called with session_id="sess-123"
        call = client._fake_engine.ask.call_args  # type: ignore
        assert call.kwargs.get("session_id") == "sess-123"

    def test_engine_error_returns_5xx(self, client):
        """If the engine raises, the API surfaces a 5xx rather than crashing."""
        client._fake_engine.ask.side_effect = RuntimeError("backend down")  # type: ignore
        response = client.post("/ask-jose", json={"query": "hi"})
        assert response.status_code >= 500


# ---------------------------------------------------------------------
# /ask-jose/voice
# ---------------------------------------------------------------------


class TestAskJoseVoice:
    def test_returns_audio_bytes(self, client):
        response = client.post(
            "/ask-jose/voice",
            json={"query": "say hi"},
        )
        assert response.status_code == 200
        # Audio response: content is the bytes from voice.speak()
        assert response.headers["content-type"].startswith("audio/")
        assert response.content == b"fake-mp3-bytes"


class TestAskJoseStreamSSE:
    def test_sse_emits_data_frames_and_done(self, client):
        # Make ask_stream yield three deltas
        client._fake_engine.ask_stream.return_value = iter(  # type: ignore
            ["Hello", ", ", "world!"]
        )
        response = client.post(
            "/ask-jose/stream-sse",
            json={"query": "hi"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        body = response.text
        frames = [
            line[len("data: "):]
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        assert frames, "no SSE frames emitted"
        # OpenAI-style terminator
        assert frames[-1].strip() == "[DONE]"

        # Each non-DONE frame is JSON; at least one carries a delta
        import json as _json

        deltas = [
            _json.loads(f).get("delta")
            for f in frames[:-1]
            if f.strip() != "[DONE]"
        ]
        assert "Hello" in deltas

    def test_sse_surfaces_engine_error_as_frame(self, client):
        client._fake_engine.ask_stream.side_effect = RuntimeError("boom")  # type: ignore
        response = client.post(
            "/ask-jose/stream-sse",
            json={"query": "hi"},
        )
        assert response.status_code == 200
        assert "error" in response.text
        assert "boom" in response.text
        assert "[DONE]" in response.text


# ---------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------


class TestTranscribe:
    def test_rejects_empty_file(self, client):
        response = client.post(
            "/transcribe",
            files={"audio": ("empty.webm", b"", "audio/webm")},
        )
        assert response.status_code == 400

    def test_rejects_oversize_file(self, client):
        big = b"x" * (25 * 1024 * 1024 + 1)
        response = client.post(
            "/transcribe",
            files={"audio": ("big.webm", big, "audio/webm")},
        )
        assert response.status_code == 413


# ---------------------------------------------------------------------
# Root UI
# ---------------------------------------------------------------------


class TestRootUI:
    def test_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        # Either text/html or the static file
        assert "html" in response.headers["content-type"].lower()
