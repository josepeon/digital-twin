#!/usr/bin/env python3
"""
WhatsApp export parser.

Parses WhatsApp chat exports (.txt) into ParsedMessage objects.

WhatsApp export format (via Chat → Export Chat → Without Media):
    [1/15/24, 2:30:15 PM] Jose Peon: Hey, how's it going?
    [1/15/24, 2:31:00 PM] Other Person: Good! What about you?
    [1/15/24, 2:32:10 PM] Jose Peon: Working on this AI project...

    Alternate formats:
    15/01/2024, 14:30 - Jose Peon: Message text
    2024-01-15, 14:30 - Jose Peon: Message text

We extract only Jose's messages, with the preceding message as context.

Usage:
    python scripts/parse_whatsapp.py [--input data/personal/raw/whatsapp]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Language, ParsedMessage, Register, Source
from src.pii import scrub_text

# ── WhatsApp message line patterns ────────────────────────────────────────────

# Pattern 1: [M/D/YY, H:MM:SS AM/PM] Name: Message
# Pattern 2: [DD/MM/YYYY, HH:MM:SS] Name: Message
PATTERN_BRACKET = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\]\s*"
    r"(.+?):\s*(.*)",
    re.IGNORECASE,
)

# Pattern 3: DD/MM/YYYY, HH:MM - Name: Message
# Pattern 4: M/D/YY, H:MM AM/PM - Name: Message
PATTERN_DASH = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\s*-\s*"
    r"(.+?):\s*(.*)",
    re.IGNORECASE,
)

# System messages (skip these)
SYSTEM_PATTERNS = [
    "Messages and calls are end-to-end encrypted",
    "created group",
    "added you",
    "left",
    "removed",
    "changed the subject",
    "changed this group",
    "changed the group description",
    "changed their phone number",
    "security code changed",
    "<Media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "document omitted",
    "GIF omitted",
    "Contact card omitted",
    "location:",
    "Live location shared",
    "You deleted this message",
    "This message was deleted",
    "Missed voice call",
    "Missed video call",
    "Voice call,",
    "Video call,",
]


def _detect_language(text: str) -> Language:
    spanish_words = {
        "que", "de", "en", "es", "un", "una", "por", "para", "como",
        "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
        "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    }
    words = set(text.lower().split())
    return Language.ES if len(words & spanish_words) >= 3 else Language.EN


def _is_system_message(text: str) -> bool:
    # Strip Unicode control chars (e.g. LTR mark U+200E) that WhatsApp inserts
    text_lower = text.replace("\u200e", "").lower().strip()
    return any(p.lower() in text_lower for p in SYSTEM_PATTERNS)


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Try multiple datetime formats for WhatsApp timestamps."""
    formats = [
        "%m/%d/%y, %I:%M:%S %p",    # US: 1/15/24, 2:30:15 PM
        "%m/%d/%y, %I:%M %p",        # US short: 1/15/24, 2:30 PM
        "%d/%m/%Y, %H:%M:%S",        # EU: 15/01/2024, 14:30:15
        "%d/%m/%Y, %H:%M",           # EU short: 15/01/2024, 14:30
        "%d/%m/%y, %H:%M:%S",        # EU 2-digit year
        "%d/%m/%y, %H:%M",           # EU 2-digit year short
        "%m/%d/%Y, %I:%M:%S %p",     # US 4-digit year
        "%m/%d/%Y, %I:%M %p",        # US 4-digit year short
        "%Y-%m-%d, %H:%M:%S",        # ISO
        "%Y-%m-%d, %H:%M",           # ISO short
    ]
    ts_clean = ts_str.strip().rstrip("]").lstrip("[")
    for fmt in formats:
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue
    return None


def _parse_line(line: str) -> tuple[str | None, str | None, str | None]:
    """Parse a WhatsApp message line.

    Returns:
        (timestamp_str, sender, message_text) or (None, None, None) if not a message line.
    """
    for pattern in [PATTERN_BRACKET, PATTERN_DASH]:
        m = pattern.match(line)
        if m:
            return m.group(1), m.group(2).strip(), m.group(3).strip()
    return None, None, None


def parse_whatsapp_file(
    filepath: Path,
    jose_names: list[str] | None = None,
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse a single WhatsApp chat export file.

    Args:
        filepath: Path to the .txt export file.
        jose_names: Possible display names for Jose in this chat.
                    Defaults to common variants.
        scrub_pii: Whether to scrub PII.

    Returns:
        List of ParsedMessage objects for Jose's messages.
    """
    if jose_names is None:
        jose_names = ["Jose Peon", "Jose", "Jose Peón", "JP", "You"]

    jose_names_lower = [n.lower() for n in jose_names]

    with open(filepath, encoding="utf-8") as f:
        raw_lines = f.readlines()

    # First pass: merge multi-line messages
    merged_messages: list[tuple[str, str, str]] = []
    current_ts, current_sender, current_text = None, None, None

    for line in raw_lines:
        line = line.rstrip("\n")
        ts, sender, text = _parse_line(line)

        if ts is not None:
            # New message — save previous
            if current_sender is not None and current_text is not None:
                merged_messages.append((current_ts, current_sender, current_text))
            current_ts, current_sender, current_text = ts, sender, text
        elif current_text is not None:
            # Continuation of previous message
            current_text += "\n" + line

    # Save last message
    if current_sender is not None and current_text is not None:
        merged_messages.append((current_ts, current_sender, current_text))

    # Second pass: extract Jose's messages with context
    results: list[ParsedMessage] = []
    prev_other_text = ""
    chat_name = filepath.stem

    for ts_str, sender, text in merged_messages:
        if _is_system_message(text):
            continue

        is_jose = sender.lower() in jose_names_lower

        if not is_jose:
            prev_other_text = text
            continue

        # This is Jose's message
        msg_text = text
        ctx = prev_other_text

        if scrub_pii:
            msg_text = scrub_text(msg_text).scrubbed
            ctx = scrub_text(ctx).scrubbed

        pm = ParsedMessage(
            text=msg_text,
            source=Source.WHATSAPP,
            timestamp=_parse_timestamp(ts_str) if ts_str else None,
            context=ctx,
            register=Register.CASUAL,
            language=_detect_language(msg_text),
            metadata={
                "chat": chat_name,
                "sender_name": sender,
            },
        )
        results.append(pm)
        prev_other_text = ""

    return results


def parse_whatsapp_export(
    whatsapp_dir: Path,
    jose_names: list[str] | None = None,
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse all WhatsApp .txt exports in a directory.

    Args:
        whatsapp_dir: Directory containing WhatsApp .txt chat exports.
        jose_names: Jose's display name variants.
        scrub_pii: Whether to scrub PII.

    Returns:
        Combined list of ParsedMessage objects.
    """
    txt_files = sorted(whatsapp_dir.glob("*.txt"))
    if not txt_files:
        print(f"  ⚠ No .txt files found in {whatsapp_dir}")
        return []

    all_messages: list[ParsedMessage] = []
    for txt_file in txt_files:
        print(f"  Parsing {txt_file.name}...")
        msgs = parse_whatsapp_file(txt_file, jose_names, scrub_pii)
        print(f"    → {len(msgs)} Jose messages")
        all_messages.extend(msgs)

    return all_messages


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

    Accepts a single WhatsApp .txt export or a directory of them;
    appends results to the default processed JSONL.
    """
    path = Path(path)
    if path.is_dir():
        messages = parse_whatsapp_export(path)
    else:
        messages = parse_whatsapp_file(path)
    _append_processed(messages, "whatsapp_parsed.jsonl")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse WhatsApp exports into training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/personal/raw/whatsapp"),
        help="Directory containing WhatsApp .txt exports",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/personal/processed/whatsapp_parsed.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--jose-names",
        type=str,
        nargs="+",
        default=None,
        help="Jose's display name(s) in WhatsApp",
    )
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input directory not found: {args.input}")
        print("   Export: WhatsApp → Chat → ⋮ → Export Chat → Without Media")
        sys.exit(1)

    print(f"📂 Parsing WhatsApp exports: {args.input}")
    messages = parse_whatsapp_export(args.input, args.jose_names, not args.no_scrub)
    print(f"   Total: {len(messages)} Jose messages")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(messages)} messages to {args.output}")


if __name__ == "__main__":
    main()
