#!/usr/bin/env python3
"""
ChatGPT export parser.

Parses the ChatGPT data export (JSON) into ParsedMessage objects.

ChatGPT export format (conversations.json):
[
  {
    "title": "...",
    "mapping": {
      "<node_id>": {
        "message": {
          "author": {"role": "user" | "assistant" | "system"},
          "content": {"parts": ["text..."]},
          "create_time": 1234567890.0
        },
        "parent": "<parent_node_id>",
        "children": ["<child_node_id>"]
      }
    }
  }
]

We extract Jose's user messages (what Jose typed) and pair them with the
context (assistant reply before Jose's message, or Jose's reply to his own
prior prompt). For RAG, we want Jose's actual words.

Usage:
    python scripts/parse_chatgpt.py [--input data/personal/raw/chatgpt/conversations.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Language, ParsedMessage, Register, Source
from src.pii import scrub_text


def _detect_language(text: str) -> Language:
    """Simple EN/ES language detection."""
    spanish_words = {
        "que", "de", "en", "es", "un", "una", "por", "para", "como",
        "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
        "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    }
    words = set(text.lower().split())
    spanish_count = len(words & spanish_words)
    return Language.ES if spanish_count >= 3 else Language.EN


def _extract_text(content: dict | None) -> str:
    """Extract text from a ChatGPT message content block."""
    if content is None:
        return ""
    parts = content.get("parts", [])
    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            # Could be an image or other rich content — skip
            text_parts.append(part.get("text", ""))
    return "\n".join(text_parts).strip()


def _walk_conversation(mapping: dict) -> list[dict]:
    """Linearize a conversation tree into an ordered list of messages."""
    # Find the root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        return []

    # Walk down the main branch (first child path)
    ordered: list[dict] = []
    current_id = root_id
    while current_id is not None:
        node = mapping.get(current_id)
        if node is None:
            break
        msg = node.get("message")
        if msg is not None:
            ordered.append(msg)
        children = node.get("children", [])
        current_id = children[0] if children else None

    return ordered


def _load_conversations(filepath: Path) -> list[dict]:
    """Load conversations from a single JSON file or a directory with split files."""
    if filepath.is_dir():
        # Scan for conversations*.json (split exports: conversations-000.json, etc.)
        conv_files = sorted(filepath.rglob("conversations*.json"))
        if not conv_files:
            print(f"❌ No conversations*.json files found in {filepath}")
            return []
        all_convos: list[dict] = []
        for cf in conv_files:
            with open(cf, encoding="utf-8") as f:
                data = json.load(f)
            print(f"   Loaded {len(data)} conversations from {cf.name}")
            all_convos.extend(data)
        return all_convos
    else:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


def parse_chatgpt_export(filepath: Path, scrub_pii: bool = True) -> list[ParsedMessage]:
    """Parse a ChatGPT conversations.json export.

    Args:
        filepath: Path to conversations.json or directory containing split files.
        scrub_pii: Whether to run the PII scrubber on extracted text.

    Returns:
        List of ParsedMessage objects — one per Jose user message.
    """
    conversations = _load_conversations(filepath)

    messages: list[ParsedMessage] = []

    for convo in conversations:
        mapping = convo.get("mapping", {})
        if not mapping:
            continue

        ordered = _walk_conversation(mapping)

        # Walk through the linearized conversation and extract Jose's turns
        prev_assistant_text = ""
        for msg in ordered:
            author_role = msg.get("author", {}).get("role", "")
            text = _extract_text(msg.get("content"))

            if not text:
                continue

            if author_role == "assistant":
                prev_assistant_text = text
                continue

            if author_role == "user":
                # This is Jose typing — this is what we want
                if scrub_pii:
                    text = scrub_text(text).scrubbed
                    prev_assistant_text = scrub_text(prev_assistant_text).scrubbed

                timestamp = None
                create_time = msg.get("create_time")
                if create_time:
                    try:
                        timestamp = datetime.fromtimestamp(create_time, tz=timezone.utc)
                    except (ValueError, OSError):
                        pass

                pm = ParsedMessage(
                    text=text,
                    source=Source.CHATGPT,
                    timestamp=timestamp,
                    context=prev_assistant_text,
                    register=Register.NEUTRAL,  # ChatGPT convos are mixed
                    language=_detect_language(text),
                    metadata={
                        "conversation_title": convo.get("title", ""),
                    },
                )
                messages.append(pm)
                prev_assistant_text = ""

    return messages


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

    Accepts a conversations.json file or a directory of split exports;
    appends results to the default processed JSONL.
    """
    messages = parse_chatgpt_export(Path(path))
    _append_processed(messages, "chatgpt_parsed.jsonl")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ChatGPT export into training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/personal/raw/chatgpt"),
        help="Path to conversations.json or directory containing split conversation files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/personal/processed/chatgpt_parsed.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        print("   Export your data: ChatGPT → Settings → Data Controls → Export")
        sys.exit(1)

    print(f"📂 Parsing {args.input}...")
    messages = parse_chatgpt_export(args.input, scrub_pii=not args.no_scrub)
    print(f"   Found {len(messages)} Jose messages")

    # Write JSONL
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
    avg_words = sum(m.word_count for m in messages) / max(len(messages), 1)
    print(f"   Avg words per message: {avg_words:.1f}")


if __name__ == "__main__":
    main()
