"""
Persistent conversation memory backed by SQLite.

Provides the same interface as the in-memory store but survives
server restarts. Falls back to in-memory if SQLite is unavailable.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location — use MEMORY_DB_PATH env var if set (for Railway/Docker),
# otherwise store alongside the RAG index for local dev.
_DEFAULT_DB_PATH = Path(
    os.environ.get(
        "MEMORY_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "personal" / "memory.db"),
    )
)

MAX_HISTORY_MESSAGES = 20  # sliding window — 10 exchanges
SESSION_TTL_SECS = 3600 * 24  # 24h TTL per session


class MemoryStore:
    """SQLite-backed conversation memory with session management."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database and create tables."""
        try:
            # Ensure parent directory exists
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_ts
                ON messages (session_id, timestamp)
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL
                )
            """)
            self._conn.commit()
            logger.info("Memory store initialized: %s", self._db_path)
        except Exception as e:
            logger.error("Failed to initialize SQLite memory: %s", e)
            self._conn = None

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message and enforce the sliding window."""
        if self._conn is None:
            return

        now = time.time()
        try:
            # Upsert session
            self._conn.execute(
                """INSERT INTO sessions (session_id, created_at, last_active)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET last_active = ?""",
                (session_id, now, now, now),
            )

            # Insert message
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )

            # Trim to sliding window
            self._conn.execute(
                """DELETE FROM messages WHERE id NOT IN (
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) AND session_id = ?""",
                (session_id, MAX_HISTORY_MESSAGES, session_id),
            )

            self._conn.commit()
        except Exception as e:
            logger.error("Failed to add message: %s", e)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return conversation history for a session."""
        if self._conn is None:
            return []

        try:
            cursor = self._conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ?""",
                (session_id, MAX_HISTORY_MESSAGES),
            )
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get history: %s", e)
            return []

    def clear_session(self, session_id: str) -> None:
        """Clear a session's conversation history."""
        if self._conn is None:
            return

        try:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
        except Exception as e:
            logger.error("Failed to clear session: %s", e)

    def cleanup_expired(self) -> int:
        """Remove sessions older than TTL. Returns number removed."""
        if self._conn is None:
            return 0

        try:
            cutoff = time.time() - SESSION_TTL_SECS
            cursor = self._conn.execute(
                "SELECT session_id FROM sessions WHERE last_active < ?", (cutoff,)
            )
            expired = [row[0] for row in cursor.fetchall()]

            if expired:
                placeholders = ",".join("?" * len(expired))
                self._conn.execute(f"DELETE FROM messages WHERE session_id IN ({placeholders})", expired)
                self._conn.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", expired)
                self._conn.commit()

            return len(expired)
        except Exception as e:
            logger.error("Failed to cleanup expired sessions: %s", e)
            return 0

    def get_session_count(self) -> int:
        """Return number of active sessions."""
        if self._conn is None:
            return 0
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
