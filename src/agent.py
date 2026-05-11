"""
Autonomous draft queue — read-only / no-send half of Phase 8d.

When the twin sees an incoming message (email, DM, etc) it can draft
a reply in Jose's voice. Sending is deliberately out of scope for this
phase — every send risks ElevenLabs charges, platform spam-flag billing,
and the obvious "this is the AI replying as me" trust problem.

The flow this module enables:

    1. Some external integration (Gmail, IG, Slack — separate code,
       not deployed yet) ingests a message and calls
       DraftQueue.draft(incoming).
    2. The twin generates a candidate reply via PersonalityEngine.
    3. The draft is persisted in SQLite alongside the original message.
    4. /agent/queue lists pending drafts.
    5. /agent/approve/<id> marks a draft "approved" and copies the
       text to a Jose-visible spot. **No automatic send happens.**
    6. /agent/reject/<id> deletes the draft.

Reads from Gmail/IG/Slack/etc are NOT implemented here — they're
deferred to a paid sprint when budget allows. This module is the
*queue + approval* half, which is pure code.

Storage: drafts live in their own SQLite DB (``data/personal/drafts.db``)
separate from conversation memory so we can apply different retention
policies. Schema is small:

    drafts(id, channel, sender, original, draft, status, created_at,
           reviewed_at)

    status ∈ {"pending", "approved", "rejected"}
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_DB = Path("data/personal/drafts.db")

VALID_STATUSES: tuple[str, ...] = ("pending", "approved", "rejected")
VALID_CHANNELS: tuple[str, ...] = ("email", "instagram", "whatsapp", "slack", "test")


class DraftQueue:
    """SQLite-backed queue of twin-drafted replies awaiting Jose's review."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    sender TEXT,
                    original TEXT NOT NULL,
                    draft TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    reviewed_at REAL
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status_created "
                "ON drafts (status, created_at)"
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize drafts DB: %s", exc)
            self._conn = None

    # ── Mutations ────────────────────────────────────────────────────

    def enqueue(
        self,
        *,
        channel: str,
        sender: str | None,
        original: str,
        draft: str,
    ) -> str:
        """Add a new pending draft. Returns its id.

        Raises ValueError on bad input — keep the queue small and
        well-shaped rather than accepting garbage.
        """
        if channel not in VALID_CHANNELS:
            raise ValueError(f"unknown channel {channel!r}")
        if not original.strip():
            raise ValueError("original message cannot be empty")
        if not draft.strip():
            raise ValueError("draft cannot be empty")
        if self._conn is None:
            raise RuntimeError("drafts DB unavailable")

        draft_id = f"d-{uuid.uuid4().hex[:12]}"
        now = time.time()
        self._conn.execute(
            "INSERT INTO drafts (id, channel, sender, original, draft, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (draft_id, channel, sender, original.strip(), draft.strip(), now),
        )
        self._conn.commit()
        return draft_id

    def approve(self, draft_id: str) -> bool:
        """Mark a draft as approved. Returns True if a row was updated.

        Approval does NOT trigger a send — it just marks the draft as
        ready for Jose to copy + paste. The actual send pathway is
        deferred until the budget gate opens.
        """
        return self._set_status(draft_id, "approved")

    def reject(self, draft_id: str) -> bool:
        return self._set_status(draft_id, "rejected")

    def delete(self, draft_id: str) -> bool:
        if self._conn is None:
            return False
        cursor = self._conn.execute(
            "DELETE FROM drafts WHERE id = ?", (draft_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def _set_status(self, draft_id: str, status: str) -> bool:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if self._conn is None:
            return False
        cursor = self._conn.execute(
            "UPDATE drafts SET status = ?, reviewed_at = ? WHERE id = ?",
            (status, time.time(), draft_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Queries ──────────────────────────────────────────────────────

    def get(self, draft_id: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT id, channel, sender, original, draft, status, "
            "created_at, reviewed_at FROM drafts WHERE id = ?",
            (draft_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_by_status("pending", limit)

    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT id, channel, sender, original, draft, status, "
            "created_at, reviewed_at FROM drafts "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _list_by_status(self, status: str, limit: int) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT id, channel, sender, original, draft, status, "
            "created_at, reviewed_at FROM drafts WHERE status = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """{status: count} — backs the dashboard 'how many drafts' card."""
        if self._conn is None:
            return {}
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM drafts GROUP BY status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "channel": row[1],
            "sender": row[2],
            "original": row[3],
            "draft": row[4],
            "status": row[5],
            "created_at": row[6],
            "reviewed_at": row[7],
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------
# Helper: produce a draft from an incoming message
# ---------------------------------------------------------------------


def draft_reply(engine, channel: str, original: str, sender: str | None = None) -> str:
    """Generate a Jose-voice reply for ``original`` via the PersonalityEngine.

    The persona mode is selected to match the channel:
      - email / slack → professional
      - instagram / whatsapp → casual
      - test → default (let auto-detect run)
    """
    mode_by_channel = {
        "email": "professional",
        "slack": "professional",
        "instagram": "casual",
        "whatsapp": "casual",
        "test": "default",
    }
    mode = mode_by_channel.get(channel, "default")

    # Build a query that gives the engine context, not just the raw message
    query_parts = []
    if sender:
        query_parts.append(f"Message from {sender}:")
    query_parts.append(original)
    query_parts.append("\n\nWrite back as you would.")
    query = "\n".join(query_parts)

    result = engine.ask(query=query, mode=mode)
    return result.get("response", "")
