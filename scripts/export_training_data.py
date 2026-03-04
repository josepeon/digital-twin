"""
Export Jose's messaging data into chat-format JSONL for LoRA fine-tuning.

Builds conversation pairs from parsed data:
  - Instagram/WhatsApp/iMessage: context → Jose's response
  - ChatGPT: Jose's prompts (shows how he thinks/asks questions)

Output format (OpenAI chat format):
  {"messages": [
    {"role": "system", "content": "You are Jose Peon..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Usage:
    python scripts/export_training_data.py
    python scripts/export_training_data.py --min-length 10 --max-length 500
    python scripts/export_training_data.py --output data/training/jose_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, IDENTITY_DIR

SYSTEM_PROMPT = """\
You ARE Jose Peon — a 24-year-old Cuban fashion engineer, software engineer, and entrepreneur living in New York. You left Cuba at 20. Self-taught in code, fashion design, 3D, and AI. You're building OH (o-h.systems) — a runtime for spatial commerce, and Nooon (nooon.online) — your fashion brand.

Your energy: confident, enthusiastic, optimistic. Short responses (1-3 sentences). You say "jaja" not "haha", you say "dale" often. Cuban Spanish when speaking Spanish. Direct and real — never use AI phrases like "absolutely", "I appreciate that", "feel free to", "I'd love to"."""


def load_parsed_data(source: str) -> list[dict]:
    """Load a parsed JSONL file."""
    path = PROCESSED_DIR / f"{source}_parsed.jsonl"
    if not path.exists():
        print(f"  ⚠️  {path} not found, skipping")
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def is_quality_message(text: str, min_length: int = 5, max_length: int = 1000) -> bool:
    """Filter for quality training messages."""
    if not text or len(text.strip()) < min_length:
        return False
    if len(text) > max_length:
        return False

    # Skip emoji-only messages
    stripped = re.sub(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]", "", text).strip()
    if len(stripped) < 3:
        return False

    # Skip very generic messages
    generic = {"ok", "okay", "yes", "no", "yeah", "lol", "haha", "jaja", "😂", "👍", "🔥", "❤️"}
    if text.strip().lower() in generic:
        return False

    return True


def build_conversation_pairs(
    records: list[dict],
    min_length: int = 10,
    max_length: int = 500,
) -> list[dict]:
    """Build training examples from parsed records.

    Records with 'context' become user→assistant pairs.
    Records without context are grouped by conversation into multi-turn exchanges.
    """
    pairs = []

    # Group by conversation for multi-turn
    conversations: dict[str, list[dict]] = {}
    for r in records:
        conv_id = r.get("metadata", {}).get("conversation", "default")
        conversations.setdefault(conv_id, []).append(r)

    for conv_id, msgs in conversations.items():
        # Sort by timestamp
        msgs.sort(key=lambda x: x.get("timestamp", ""))

        for msg in msgs:
            text = msg.get("text", "").strip()
            context = msg.get("context", "").strip()

            if not is_quality_message(text, min_length, max_length):
                continue

            if context and is_quality_message(context, min_length=3, max_length=1000):
                # We have a user→assistant pair
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": context},
                        {"role": "assistant", "content": text},
                    ]
                })

    return pairs


def build_chatgpt_examples(
    records: list[dict],
    min_length: int = 10,
    max_length: int = 500,
) -> list[dict]:
    """Build examples from ChatGPT conversations.

    These show how Jose asks questions / thinks out loud.
    Group consecutive messages into mini-conversations.
    """
    pairs = []

    # Group by conversation
    conversations: dict[str, list[dict]] = {}
    for r in records:
        conv_title = r.get("metadata", {}).get("conversation_title", "default")
        conversations.setdefault(conv_title, []).append(r)

    for title, msgs in conversations.items():
        msgs.sort(key=lambda x: x.get("timestamp", ""))

        for msg in msgs:
            text = msg.get("text", "").strip()
            context = msg.get("context", "").strip()

            if not is_quality_message(text, min_length, max_length):
                continue

            # Jose's prompt to ChatGPT — this shows his voice when asking questions
            if context:
                # Context = ChatGPT's previous response, text = Jose's follow-up
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"How would you ask about: {title}?"},
                        {"role": "assistant", "content": text},
                    ]
                })
            else:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Start a conversation about {title}."},
                        {"role": "assistant", "content": text},
                    ]
                })

    return pairs


def export_training_data(
    output_path: Path,
    min_length: int = 10,
    max_length: int = 500,
    train_split: float = 0.9,
    seed: int = 42,
) -> dict:
    """Export all training data to JSONL files.

    Creates:
      - {output_path}_train.jsonl — training set
      - {output_path}_val.jsonl — validation set
    """
    all_pairs: list[dict] = []
    source_counts: dict[str, int] = {}

    # Process messaging data (Instagram, WhatsApp, iMessage)
    for source in ["instagram", "whatsapp", "imessage"]:
        records = load_parsed_data(source)
        if records:
            pairs = build_conversation_pairs(records, min_length, max_length)
            source_counts[source] = len(pairs)
            all_pairs.extend(pairs)
            print(f"  {source}: {len(records)} records → {len(pairs)} training pairs")

    # Process ChatGPT data
    records = load_parsed_data("chatgpt")
    if records:
        pairs = build_chatgpt_examples(records, min_length, max_length)
        source_counts["chatgpt"] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  chatgpt: {len(records)} records → {len(pairs)} training pairs")

    if not all_pairs:
        print("  ❌ No training pairs generated!")
        return {"total": 0}

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * train_split)
    train_set = all_pairs[:split_idx]
    val_set = all_pairs[split_idx:]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_path.parent / f"{output_path.stem}_train.jsonl"
    val_path = output_path.parent / f"{output_path.stem}_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_set:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for pair in val_set:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "total": len(all_pairs),
        "train": len(train_set),
        "val": len(val_set),
        "sources": source_counts,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    print(f"\n  Total: {len(all_pairs)} pairs")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"  Saved: {train_path}")
    print(f"         {val_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Export training data for LoRA fine-tuning")
    parser.add_argument("--output", type=str, default="data/training/jose",
                        help="Output path prefix (without extension)")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Minimum message length in chars")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum message length in chars")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Train/val split ratio")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Training Data Export for LoRA Fine-Tuning  ║")
    print("╚══════════════════════════════════════════════╝\n")

    output_path = PROJECT_ROOT / args.output
    export_training_data(
        output_path=output_path,
        min_length=args.min_length,
        max_length=args.max_length,
        train_split=args.split,
    )


if __name__ == "__main__":
    main()
