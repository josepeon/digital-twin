#!/usr/bin/env python3
"""
iMessage parser.

Extracts Jose's iMessages from the macOS Messages database (chat.db).

Database location: ~/Library/Messages/chat.db (SQLite)
Requires Full Disk Access enabled for the terminal app:
  System Settings → Privacy & Security → Full Disk Access

The parser:
  1. Copies chat.db to data/personal/raw/imessage/ (so we don't hold a lock)
  2. Queries Jose's sent messages with the preceding received message as context
  3. Outputs ParsedMessage objects

Usage:
    python scripts/parse_imessage.py [--input ~/Library/Messages/chat.db]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Language, ParsedMessage, Register, Source
from src.pii import scrub_text

# Apple's Core Data epoch: 2001-01-01 00:00:00 UTC
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def _detect_language(text: str) -> Language:
    spanish_words = {
        "que", "de", "en", "es", "un", "una", "por", "para", "como",
        "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
        "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    }
    words = set(text.lower().split())
    return Language.ES if len(words & spanish_words) >= 3 else Language.EN


def _apple_ts_to_datetime(apple_ts: int | None) -> datetime | None:
    """Convert Apple Core Data timestamp (nanoseconds since 2001-01-01) to datetime."""
    if apple_ts is None or apple_ts == 0:
        return None
    try:
        # Apple timestamps are in nanoseconds since 2001-01-01
        seconds = apple_ts / 1_000_000_000
        return APPLE_EPOCH + timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        return None


def _detect_register(chat_identifier: str, text: str) -> Register:
    """Heuristic register detection based on contact + content."""
    text_lower = text.lower()
    family_signals = ["mamá", "papá", "familia", "hermano", "hermana", "mami", "papi", "cuba", "casa"]
    if any(s in text_lower for s in family_signals):
        return Register.FAMILY

    casual_signals = ["lol", "haha", "jaja", "bro", "dude", "lmao", "omg", "oye", "dale"]
    if any(s in text_lower for s in casual_signals):
        return Register.CASUAL

    return Register.CASUAL  # iMessage is generally casual


def parse_imessage_db(
    db_path: Path,
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse iMessages from a chat.db SQLite database.

    Args:
        db_path: Path to the chat.db file.
        scrub_pii: Whether to scrub PII.

    Returns:
        List of ParsedMessage objects for Jose's sent messages.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Query all messages with chat info, ordered by chat and date
    query = """
        SELECT
            m.rowid AS message_id,
            m.text,
            m.is_from_me,
            m.date AS apple_date,
            m.service,
            c.chat_identifier,
            c.display_name AS chat_display_name
        FROM message m
        JOIN chat_message_join cmj ON m.rowid = cmj.message_id
        JOIN chat c ON cmj.chat_id = c.rowid
        WHERE m.text IS NOT NULL
          AND m.text != ''
        ORDER BY c.chat_identifier, m.date ASC
    """

    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError as e:
        print(f"  ⚠ SQLite query error: {e}")
        print("    Ensure Full Disk Access is enabled for your terminal.")
        conn.close()
        return []

    # Group by chat and extract Jose's messages with context
    results: list[ParsedMessage] = []
    prev_received_text = ""
    prev_chat = None

    for row in rows:
        chat_id = row["chat_identifier"] or ""
        text = row["text"] or ""
        is_from_me = row["is_from_me"]

        # Reset context on new chat
        if chat_id != prev_chat:
            prev_received_text = ""
            prev_chat = chat_id

        if not text.strip():
            continue

        if not is_from_me:
            prev_received_text = text
            continue

        # Jose's sent message
        msg_text = text
        ctx = prev_received_text

        if scrub_pii:
            msg_text = scrub_text(msg_text).scrubbed
            ctx = scrub_text(ctx).scrubbed

        pm = ParsedMessage(
            text=msg_text,
            source=Source.IMESSAGE,
            timestamp=_apple_ts_to_datetime(row["apple_date"]),
            context=ctx,
            register=_detect_register(chat_id, msg_text),
            language=_detect_language(msg_text),
            metadata={
                "chat_identifier": chat_id,
                "service": row["service"] or "iMessage",
            },
        )
        results.append(pm)
        prev_received_text = ""

    conn.close()
    return results


def _append_processed(messages: "list[ParsedMessage]", filename: str) -> None:
    """Append parsed rows to the processed JSONL (incremental ingest)."""
    from src.config import PROCESSED_DIR

    out = PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")


def parse_file(path: Path) -> "list[ParsedMessage]":
    """Entry point for scripts/ingest_new_data.py dispatch.

    Accepts a chat.db SQLite file; appends results to the default
    processed JSONL.
    """
    messages = parse_imessage_db(Path(path))
    _append_processed(messages, "imessage_parsed.jsonl")
    return messages


def main() -> None:
    default_db = Path.home() / "Library" / "Messages" / "chat.db"
    local_copy = Path("data/personal/raw/imessage/chat.db")

    parser = argparse.ArgumentParser(description="Parse iMessages into training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help=f"Path to chat.db (default: {default_db})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/personal/processed/imessage_parsed.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy chat.db locally")
    args = parser.parse_args()

    # Determine source
    if args.input:
        db_path = args.input
    elif local_copy.exists():
        print(f"📂 Using existing local copy: {local_copy}")
        db_path = local_copy
    elif default_db.exists():
        db_path = default_db
    else:
        print(f"❌ chat.db not found at {default_db}")
        print("   Ensure Full Disk Access is enabled:")
        print("   System Settings → Privacy & Security → Full Disk Access")
        sys.exit(1)

    # Copy to local (safer than locking the live DB)
    if not args.no_copy and db_path == default_db:
        print(f"📋 Copying chat.db to {local_copy}...")
        local_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, local_copy)
        db_path = local_copy
        print("   Done.")

    print(f"📂 Parsing iMessages: {db_path}")
    messages = parse_imessage_db(db_path, scrub_pii=not args.no_scrub)
    print(f"   Found {len(messages)} Jose messages")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(messages)} messages to {args.output}")

    # Stats
    langs = {"en": 0, "es": 0}
    for m in messages:
        langs[m.language.value] += 1
    print(f"   Languages: EN={langs['en']}, ES={langs['es']}")


if __name__ == "__main__":
    main()
