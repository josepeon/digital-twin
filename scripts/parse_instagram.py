#!/usr/bin/env python3
"""
Instagram export parser.

Parses the Instagram data export (JSON format) into ParsedMessage objects.

Instagram export structure (JSON format, request from Accounts Center):
  your_instagram_activity/
    messages/
      inbox/
        <username>/
          message_1.json   ← DM conversations
    content/
      posts_1.json         ← Feed posts with captions
    comments/
      post_comments_1.json ← Comments Jose left on posts

We extract:
  - DM messages sent BY Jose (not received)
  - Post captions (Jose's public voice)
  - Comments Jose left on others' posts

Usage:
    python scripts/parse_instagram.py [--input data/personal/raw/instagram]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Language, ParsedMessage, Register, Source
from src.pii import scrub_text


def _detect_language(text: str) -> Language:
    spanish_words = {
        "que", "de", "en", "es", "un", "una", "por", "para", "como",
        "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
        "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    }
    words = set(text.lower().split())
    return Language.ES if len(words & spanish_words) >= 3 else Language.EN


def _decode_instagram_text(text: str) -> str:
    """Instagram exports encode UTF-8 as Latin-1 escape sequences. Fix that."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _ts_from_ms(timestamp_ms: int | None) -> datetime | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _ts_from_epoch(timestamp: int | float | None) -> datetime | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (ValueError, OSError):
        return None


# ── DM Parser ──────────────────────────────────────────────────────────────────

def parse_dms(instagram_dir: Path, jose_name: str, scrub_pii: bool = True) -> list[ParsedMessage]:
    """Parse Instagram DM conversations.

    Args:
        instagram_dir: Root of the Instagram export folder.
        jose_name: Jose's Instagram display name (used to identify his messages).
        scrub_pii: Whether to scrub PII.

    Returns:
        List of ParsedMessage objects for Jose's DM messages.
    """
    messages_dir = instagram_dir / "your_instagram_activity" / "messages" / "inbox"
    if not messages_dir.exists():
        # Try alternate structure
        messages_dir = instagram_dir / "messages" / "inbox"
    if not messages_dir.exists():
        print(f"  ⚠ DM directory not found: {messages_dir}")
        return []

    results: list[ParsedMessage] = []

    for convo_dir in sorted(messages_dir.iterdir()):
        if not convo_dir.is_dir():
            continue

        # Each conversation may have multiple message_N.json files
        for msg_file in sorted(convo_dir.glob("message_*.json")):
            with open(msg_file, encoding="utf-8") as f:
                data = json.load(f)

            participants = [_decode_instagram_text(p.get("name", "")) for p in data.get("participants", [])]
            chat_messages = data.get("messages", [])

            # Build context: previous message from other person
            prev_other_text = ""
            for msg in reversed(chat_messages):  # Instagram stores newest first
                sender = _decode_instagram_text(msg.get("sender_name", ""))
                content = _decode_instagram_text(msg.get("content", ""))

                if not content or content == "Liked a message" or content.startswith("Reacted"):
                    prev_other_text = ""
                    continue

                if sender.lower() != jose_name.lower():
                    prev_other_text = content
                    continue

                # This is Jose's message
                text = content
                if scrub_pii:
                    text = scrub_text(text).scrubbed
                    prev_other_text = scrub_text(prev_other_text).scrubbed

                pm = ParsedMessage(
                    text=text,
                    source=Source.INSTAGRAM,
                    timestamp=_ts_from_ms(msg.get("timestamp_ms")),
                    context=prev_other_text,
                    register=Register.CASUAL,
                    language=_detect_language(text),
                    metadata={
                        "type": "dm",
                        "conversation": convo_dir.name,
                        "participants": participants,
                    },
                )
                results.append(pm)
                prev_other_text = ""

    return results


# ── Post Captions Parser ──────────────────────────────────────────────────────

def parse_posts(instagram_dir: Path, scrub_pii: bool = True) -> list[ParsedMessage]:
    """Parse Instagram post captions."""
    # Try multiple known locations for posts JSON
    candidates = [
        instagram_dir / "your_instagram_activity" / "content",
        instagram_dir / "your_instagram_activity" / "media",
        instagram_dir / "content",
        instagram_dir / "media",
    ]
    posts_dir = None
    for c in candidates:
        if c.exists() and list(c.glob("posts_*.json")):
            posts_dir = c
            break
    if posts_dir is None:
        print(f"  ⚠ Posts directory not found (tried content/ and media/)")
        return []

    results: list[ParsedMessage] = []

    for posts_file in sorted(posts_dir.glob("posts_*.json")):
        with open(posts_file, encoding="utf-8") as f:
            data = json.load(f)

        posts = data if isinstance(data, list) else data.get("ig_fb_posts", data.get("posts", []))

        for post in posts:
            # Extract caption
            media_list = post.get("media", [post])  # Some formats nest under "media"
            for media in (media_list if isinstance(media_list, list) else [media_list]):
                title = _decode_instagram_text(media.get("title", ""))
                if not title:
                    continue

                text = title
                if scrub_pii:
                    text = scrub_text(text).scrubbed

                ts = media.get("creation_timestamp") or media.get("taken_at")

                pm = ParsedMessage(
                    text=text,
                    source=Source.INSTAGRAM,
                    timestamp=_ts_from_epoch(ts),
                    context="",
                    register=Register.CASUAL,
                    language=_detect_language(text),
                    metadata={"type": "post_caption"},
                )
                results.append(pm)

    return results


# ── Comments Parser ───────────────────────────────────────────────────────────

def parse_comments(instagram_dir: Path, scrub_pii: bool = True) -> list[ParsedMessage]:
    """Parse comments Jose left on posts."""
    comments_dir = instagram_dir / "your_instagram_activity" / "comments"
    if not comments_dir.exists():
        comments_dir = instagram_dir / "comments"
    if not comments_dir.exists():
        print(f"  ⚠ Comments directory not found: {comments_dir}")
        return []

    results: list[ParsedMessage] = []

    for comments_file in sorted(
        list(comments_dir.glob("post_comments*.json")) +
        list(comments_dir.glob("reels_comments*.json"))
    ):
        with open(comments_file, encoding="utf-8") as f:
            data = json.load(f)

        comments = data if isinstance(data, list) else data.get("comments_media_comments", [])

        for comment_block in comments:
            # Handle nested structure
            values = comment_block.get("string_map_data", {})
            comment_text = values.get("Comment", {}).get("value", "")
            if not comment_text:
                # Try alternate format
                comment_text = comment_block.get("value", comment_block.get("comment", ""))

            comment_text = _decode_instagram_text(str(comment_text))
            if not comment_text:
                continue

            text = comment_text
            if scrub_pii:
                text = scrub_text(text).scrubbed

            ts = (
                comment_block.get("string_map_data", {}).get("Time", {}).get("timestamp")
                or comment_block.get("timestamp")
            )

            pm = ParsedMessage(
                text=text,
                source=Source.INSTAGRAM,
                timestamp=_ts_from_epoch(ts),
                context="",
                register=Register.CASUAL,
                language=_detect_language(text),
                metadata={"type": "comment"},
            )
            results.append(pm)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_instagram_export(
    instagram_dir: Path,
    jose_name: str = "Jose Peon",
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse the full Instagram export.

    Args:
        instagram_dir: Root directory of the Instagram data export.
        jose_name: Jose's display name on Instagram (for DM attribution).
        scrub_pii: Whether to scrub PII.

    Returns:
        Combined list of ParsedMessage objects from DMs, posts, and comments.
    """
    print(f"  Parsing DMs (name={jose_name})...")
    dms = parse_dms(instagram_dir, jose_name, scrub_pii)
    print(f"    → {len(dms)} DM messages")

    print("  Parsing post captions...")
    posts = parse_posts(instagram_dir, scrub_pii)
    print(f"    → {len(posts)} post captions")

    print("  Parsing comments...")
    comments = parse_comments(instagram_dir, scrub_pii)
    print(f"    → {len(comments)} comments")

    return dms + posts + comments


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

    Instagram exports are directories; a file inside one resolves to
    its export root. Appends results to the default processed JSONL.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(
            f"Instagram ingest expects the export directory, got file: {path}"
        )
    messages = parse_instagram_export(path)
    _append_processed(messages, "instagram_parsed.jsonl")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Instagram export into training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/personal/raw/instagram"),
        help="Path to Instagram export directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/personal/processed/instagram_parsed.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--jose-name",
        type=str,
        default="Jose Peon",
        help="Jose's Instagram display name (for DM sender matching)",
    )
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input directory not found: {args.input}")
        print("   Export: Instagram → Accounts Center → Download your information → JSON")
        sys.exit(1)

    # Support multi-export: if the input dir contains sub-export folders, scan each
    sub_exports = sorted([
        d for d in args.input.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    has_activity = (args.input / "your_instagram_activity").exists()

    all_messages: list[ParsedMessage] = []
    if sub_exports and not has_activity:
        print(f"📂 Found {len(sub_exports)} Instagram sub-exports in {args.input}")
        for sub in sub_exports:
            print(f"\n── {sub.name} ──")
            msgs = parse_instagram_export(sub, jose_name=args.jose_name, scrub_pii=not args.no_scrub)
            all_messages.extend(msgs)
    else:
        print(f"📂 Parsing Instagram export: {args.input}")
        all_messages = parse_instagram_export(args.input, jose_name=args.jose_name, scrub_pii=not args.no_scrub)

    # Deduplicate by (text, timestamp) across sub-exports
    seen = set()
    messages = []
    for m in all_messages:
        key = (m.text, str(m.timestamp))
        if key not in seen:
            seen.add(key)
            messages.append(m)
    print(f"\n   Total: {len(messages)} messages (deduped from {len(all_messages)})")


    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(messages)} messages to {args.output}")


if __name__ == "__main__":
    main()
