"""T2.3 part 1: MemoryStore — SQLite-backed conversation memory."""

from __future__ import annotations

import time

import pytest

from src.memory import MAX_HISTORY_MESSAGES, SESSION_TTL_SECS, MemoryStore


@pytest.fixture
def store(tmp_path):
    """Per-test SQLite file so tests are isolated."""
    db = tmp_path / "memory.db"
    s = MemoryStore(db)
    yield s
    s.close()


class TestAddAndGet:
    def test_round_trip(self, store):
        store.add_message("sess1", "user", "hi")
        store.add_message("sess1", "assistant", "hello")
        history = store.get_history("sess1")
        assert history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_empty_session_returns_empty(self, store):
        assert store.get_history("nonexistent") == []

    def test_sessions_isolated(self, store):
        store.add_message("a", "user", "a-message")
        store.add_message("b", "user", "b-message")
        assert store.get_history("a") == [{"role": "user", "content": "a-message"}]
        assert store.get_history("b") == [{"role": "user", "content": "b-message"}]


class TestSlidingWindow:
    def test_caps_at_max_history(self, store):
        # Insert MAX_HISTORY_MESSAGES + 5 — should keep only the last MAX
        for i in range(MAX_HISTORY_MESSAGES + 5):
            store.add_message("s", "user", f"msg-{i}")
        history = store.get_history("s")
        assert len(history) == MAX_HISTORY_MESSAGES
        # Most recent message should be present
        assert history[-1]["content"] == f"msg-{MAX_HISTORY_MESSAGES + 4}"
        # Earliest preserved is msg-5 (msgs 0-4 trimmed)
        assert history[0]["content"] == "msg-5"

    def test_order_preserved_chronologically(self, store):
        for i in range(5):
            store.add_message("s", "user", f"u{i}")
            store.add_message("s", "assistant", f"a{i}")
        contents = [m["content"] for m in store.get_history("s")]
        assert contents == ["u0", "a0", "u1", "a1", "u2", "a2", "u3", "a3", "u4", "a4"]


class TestClearSession:
    def test_clear_removes_all_messages(self, store):
        store.add_message("s", "user", "msg")
        store.add_message("s", "assistant", "reply")
        store.clear_session("s")
        assert store.get_history("s") == []

    def test_clear_does_not_affect_other_sessions(self, store):
        store.add_message("a", "user", "keep-me")
        store.add_message("b", "user", "delete-me")
        store.clear_session("b")
        assert store.get_history("a") == [{"role": "user", "content": "keep-me"}]
        assert store.get_history("b") == []


class TestCleanupExpired:
    def test_no_expired_returns_zero(self, store):
        store.add_message("fresh", "user", "msg")
        # Default TTL is 24h; fresh session shouldn't expire
        assert store.cleanup_expired() == 0
        assert store.get_history("fresh") == [{"role": "user", "content": "msg"}]

    def test_expired_session_purged(self, store, monkeypatch):
        store.add_message("old", "user", "stale")
        store.add_message("new", "user", "fresh")

        # Fast-forward time so 'old' becomes expired while keeping 'new'
        original_time = time.time
        future = original_time() + SESSION_TTL_SECS + 60

        # Mark 'new' as last-active recently by adding another message at "now+TTL"
        monkeypatch.setattr("src.memory.time.time", lambda: future)
        store.add_message("new", "user", "still-fresh-at-future")

        # Both sessions exist, but 'old' is now stale. cleanup_expired should
        # remove 'old' but keep 'new'.
        expired = store.cleanup_expired()
        assert expired == 1
        assert store.get_history("old") == []
        assert len(store.get_history("new")) == 2


class TestSessionCount:
    def test_count_tracks_unique_sessions(self, store):
        assert store.get_session_count() == 0
        store.add_message("a", "user", "x")
        store.add_message("b", "user", "y")
        store.add_message("a", "user", "z")  # same session, doesn't increment
        assert store.get_session_count() == 2


class TestConnectionFailure:
    def test_close_idempotent(self, store):
        store.close()
        store.close()  # should not raise

    def test_methods_safe_after_close(self, store):
        store.close()
        # Should not crash — defensive None checks
        store.add_message("s", "user", "after close")
        assert store.get_history("s") == []
        assert store.get_session_count() == 0
        assert store.cleanup_expired() == 0


class TestWalMode:
    def test_wal_mode_enabled(self, store, tmp_path):
        # WAL mode is set in _init_db; verify via PRAGMA
        cursor = store._conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].lower() == "wal"


class TestThreadSafety:
    def test_concurrent_writers_dont_corrupt(self, store):
        import threading

        errors: list[Exception] = []

        def writer(session: str) -> None:
            try:
                for i in range(50):
                    store.add_message(session, "user", f"msg {i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(f"s{n}",)) for n in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Each session keeps its own trimmed window, in insertion order
        for n in range(8):
            history = store.get_history(f"s{n}")
            assert len(history) == 20
            assert history[-1]["content"] == "msg 49"

    def test_history_preserves_same_tick_ordering(self, store):
        # user + assistant pairs land within the same clock tick; insertion
        # order (id) must win over timestamp ties
        for i in range(5):
            store.add_message("s", "user", f"q{i}")
            store.add_message("s", "assistant", f"a{i}")
        history = store.get_history("s")
        roles = [h["role"] for h in history]
        assert roles == ["user", "assistant"] * 5
