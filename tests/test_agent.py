"""T3.3: DraftQueue + /agent/* endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agent import DraftQueue, draft_reply


# ---------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------


@pytest.fixture
def queue(tmp_path):
    q = DraftQueue(tmp_path / "drafts.db")
    yield q
    q.close()


class TestEnqueue:
    def test_round_trip(self, queue):
        draft_id = queue.enqueue(
            channel="email",
            sender="ocean@example.com",
            original="Are you free Thursday?",
            draft="Yeah, around 3pm works.",
        )
        row = queue.get(draft_id)
        assert row["channel"] == "email"
        assert row["sender"] == "ocean@example.com"
        assert row["status"] == "pending"
        assert row["reviewed_at"] is None

    def test_invalid_channel_rejected(self, queue):
        with pytest.raises(ValueError, match="unknown channel"):
            queue.enqueue(
                channel="fax", sender=None, original="x", draft="y",
            )

    def test_empty_original_rejected(self, queue):
        with pytest.raises(ValueError, match="original"):
            queue.enqueue(channel="email", sender=None, original="   ", draft="y")

    def test_empty_draft_rejected(self, queue):
        with pytest.raises(ValueError, match="draft"):
            queue.enqueue(channel="email", sender=None, original="x", draft="  ")

    def test_unique_ids(self, queue):
        a = queue.enqueue(channel="email", sender=None, original="a", draft="aa")
        b = queue.enqueue(channel="email", sender=None, original="b", draft="bb")
        assert a != b


class TestApproveReject:
    def test_approve_sets_status(self, queue):
        draft_id = queue.enqueue(
            channel="email", sender=None, original="x", draft="reply",
        )
        assert queue.approve(draft_id) is True
        row = queue.get(draft_id)
        assert row["status"] == "approved"
        assert row["reviewed_at"] is not None

    def test_reject_sets_status(self, queue):
        draft_id = queue.enqueue(
            channel="email", sender=None, original="x", draft="reply",
        )
        assert queue.reject(draft_id) is True
        assert queue.get(draft_id)["status"] == "rejected"

    def test_approve_nonexistent_returns_false(self, queue):
        assert queue.approve("d-nonexistent") is False

    def test_delete_removes_row(self, queue):
        draft_id = queue.enqueue(
            channel="email", sender=None, original="x", draft="reply",
        )
        assert queue.delete(draft_id) is True
        assert queue.get(draft_id) is None


class TestListing:
    def test_list_pending_only(self, queue):
        a = queue.enqueue(channel="email", sender=None, original="a", draft="aa")
        b = queue.enqueue(channel="email", sender=None, original="b", draft="bb")
        c = queue.enqueue(channel="email", sender=None, original="c", draft="cc")
        queue.approve(a)
        queue.reject(c)

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == b

    def test_list_all_includes_every_status(self, queue):
        a = queue.enqueue(channel="email", sender=None, original="a", draft="aa")
        b = queue.enqueue(channel="email", sender=None, original="b", draft="bb")
        queue.approve(a)

        all_drafts = queue.list_all()
        assert len(all_drafts) == 2
        statuses = {d["status"] for d in all_drafts}
        assert statuses == {"approved", "pending"}

    def test_stats_groups_by_status(self, queue):
        for i in range(3):
            queue.enqueue(
                channel="email", sender=None, original=f"m{i}", draft=f"r{i}",
            )
        ids = [d["id"] for d in queue.list_all()]
        queue.approve(ids[0])
        queue.reject(ids[1])

        stats = queue.stats()
        assert stats == {"pending": 1, "approved": 1, "rejected": 1}


# ---------------------------------------------------------------------
# draft_reply helper
# ---------------------------------------------------------------------


class TestDraftReply:
    def test_returns_engine_response(self):
        engine = MagicMock()
        engine.ask.return_value = {"response": "Yeah, sounds good."}
        out = draft_reply(engine, "email", "Are you free?", sender="x@y.com")
        assert out == "Yeah, sounds good."

    def test_email_channel_uses_professional_mode(self):
        engine = MagicMock()
        engine.ask.return_value = {"response": "Sure."}
        draft_reply(engine, "email", "Quick q for you.")
        call = engine.ask.call_args
        assert call.kwargs["mode"] == "professional"

    def test_instagram_channel_uses_casual_mode(self):
        engine = MagicMock()
        engine.ask.return_value = {"response": "dale!"}
        draft_reply(engine, "instagram", "yo what's up")
        call = engine.ask.call_args
        assert call.kwargs["mode"] == "casual"

    def test_unknown_channel_uses_default_mode(self):
        engine = MagicMock()
        engine.ask.return_value = {"response": "ok"}
        draft_reply(engine, "test", "hi")
        call = engine.ask.call_args
        assert call.kwargs["mode"] == "default"


# ---------------------------------------------------------------------
# /agent/* API routes
# ---------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src import twin_api as api_module

    fake_engine = MagicMock()
    fake_engine.ask.return_value = {
        "response": "Yeah I'm around — text me when you're close.",
        "language": "en",
        "register": "casual",
        "model": "llama",
        "examples_used": 1,
    }
    fake_voice = MagicMock()
    fake_voice.has_cloned_voices = True

    monkeypatch.setattr(api_module, "PersonalityEngine", lambda: fake_engine)
    monkeypatch.setattr(api_module, "VoiceEngine", lambda: fake_voice)
    monkeypatch.setattr(api_module, "TWIN_API_KEY", "")
    # Make sure the queue uses a tmp DB
    monkeypatch.setattr(api_module, "_drafts", DraftQueue(tmp_path / "drafts.db"))

    with TestClient(api_module.app) as c:
        c._fake_engine = fake_engine  # type: ignore
        yield c


class TestApi:
    def test_create_draft(self, client):
        response = client.post(
            "/agent/draft",
            json={
                "channel": "instagram",
                "sender": "ocean",
                "original": "see you tonight?",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "pending"
        assert body["draft"]
        assert body["channel"] == "instagram"

    def test_create_draft_bad_channel(self, client):
        response = client.post(
            "/agent/draft",
            json={"channel": "fax", "original": "hi"},
        )
        # Caught by enqueue ValueError → 400
        assert response.status_code == 400

    def test_queue_endpoint(self, client):
        # Create two drafts, approve one
        first = client.post(
            "/agent/draft",
            json={"channel": "email", "original": "a"},
        ).json()
        client.post(
            "/agent/draft",
            json={"channel": "email", "original": "b"},
        )
        client.post(f"/agent/approve/{first['id']}")

        queue = client.get("/agent/queue").json()
        # Only one pending now
        assert len(queue["drafts"]) == 1
        assert queue["stats"]["approved"] == 1

    def test_approve_marks_status(self, client):
        created = client.post(
            "/agent/draft",
            json={"channel": "email", "original": "ping"},
        ).json()

        response = client.post(f"/agent/approve/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

    def test_reject_marks_status(self, client):
        created = client.post(
            "/agent/draft",
            json={"channel": "email", "original": "ping"},
        ).json()

        response = client.post(f"/agent/reject/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    def test_approve_unknown_returns_404(self, client):
        response = client.post("/agent/approve/d-doesnt-exist")
        assert response.status_code == 404

    def test_empty_engine_response_yields_502(self, client):
        client._fake_engine.ask.return_value = {"response": ""}  # type: ignore
        response = client.post(
            "/agent/draft",
            json={"channel": "email", "original": "ping"},
        )
        assert response.status_code == 502
