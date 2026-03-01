#!/usr/bin/env python3
"""
Dataset builder — quality scoring, deduplication, and versioned dataset creation.

Reads parsed JSONL files from all sources, applies:
  1. Minimum length filter
  2. Quality scoring (length, informativeness, uniqueness)
  3. Near-duplicate removal
  4. Versioned dataset output (v0.1 → v1.0)

Usage:
    python scripts/build_dataset.py [--version v0.1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_SOURCES,
    DATASET_VERSIONS,
    MIN_MESSAGE_LENGTH,
    MAX_MESSAGE_LENGTH,
    PROCESSED_DIR,
)
from src.models import ParsedMessage

# ── Quality Scoring ───────────────────────────────────────────────────────────


def score_length(word_count: int) -> float:
    """Score based on message length. Sweet spot is 15-150 words."""
    if word_count < MIN_MESSAGE_LENGTH:
        return 0.0
    if word_count > MAX_MESSAGE_LENGTH:
        return 0.2  # Probably a paste/dump, low personality signal
    if 15 <= word_count <= 150:
        return 1.0  # Ideal range
    if 10 <= word_count < 15:
        return 0.5
    if 150 < word_count <= 500:
        return 0.8
    return 0.6


def score_informativeness(text: str) -> float:
    """Heuristic: does this text reveal personality, opinions, or knowledge?"""
    score = 0.5  # Baseline

    # Personal pronouns → personal voice
    personal = ["i ", "i'm", "i've", "i'd", "my ", "me ", "myself",
                 "yo ", "mi ", "me ", "soy", "tengo", "estoy"]
    text_lower = text.lower()
    if any(p in text_lower for p in personal):
        score += 0.15

    # Opinions/stance
    opinion = ["think", "believe", "feel", "prefer", "love", "hate",
                "creo", "pienso", "prefiero", "me gusta", "odio",
                "honestly", "actually", "in my opinion", "imo"]
    if any(o in text_lower for o in opinion):
        score += 0.2

    # Questions (cognitive engagement)
    if "?" in text:
        score += 0.1

    # Code/technical (knowledge signal)
    technical = ["function", "class ", "import ", "const ", "def ",
                 "async", "await", "docker", "deploy", "api", "model"]
    if any(t in text_lower for t in technical):
        score += 0.1

    # Humor/exclamation (personality signal)
    if any(h in text_lower for h in ["haha", "lol", "jaja", "😂", "😅", "🤣"]):
        score += 0.1

    # Emotional depth
    emotional = ["grateful", "excited", "frustrated", "proud", "miss",
                 "agradecido", "emocionado", "frustrado", "orgulloso", "extraño"]
    if any(e in text_lower for e in emotional):
        score += 0.15

    return min(score, 1.0)


def score_has_context(context: str) -> float:
    """Bonus for having conversation context (more useful for RAG)."""
    if context and len(context.split()) >= 5:
        return 1.0
    if context:
        return 0.5
    return 0.0


def compute_quality_score(msg: ParsedMessage) -> float:
    """Compute an overall quality score (0-1) for a parsed message."""
    length_score = score_length(msg.word_count)
    info_score = score_informativeness(msg.text)
    context_score = score_has_context(msg.context)

    # Weighted combination
    return (
        0.30 * length_score
        + 0.50 * info_score
        + 0.20 * context_score
    )


# ── Deduplication ─────────────────────────────────────────────────────────────

def _text_fingerprint(text: str) -> str:
    """Create a normalized fingerprint for near-duplicate detection."""
    # Normalize: lowercase, strip whitespace, remove punctuation
    import re
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Remove near-duplicate messages. Keeps the first occurrence."""
    seen: set[str] = set()
    unique: list[ParsedMessage] = []

    for msg in messages:
        fp = _text_fingerprint(msg.text)
        if fp not in seen:
            seen.add(fp)
            unique.append(msg)

    return unique


# ── Dataset Builder ───────────────────────────────────────────────────────────

def load_parsed_source(source: str) -> list[ParsedMessage]:
    """Load parsed JSONL for a given source."""
    path = PROCESSED_DIR / f"{source}_parsed.jsonl"
    if not path.exists():
        print(f"  ⚠ {path} not found — skipping {source}")
        return []

    messages: list[ParsedMessage] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                messages.append(ParsedMessage.from_dict(data))

    return messages


def build_dataset(
    version: str,
    min_quality: float = 0.3,
) -> list[dict]:
    """Build a versioned dataset from parsed sources.

    Args:
        version: Dataset version (e.g., "v0.1"). Determines which sources to include.
        min_quality: Minimum quality score to include a message.

    Returns:
        List of scored message dicts ready for JSONL output.
    """
    sources = DATASET_VERSIONS.get(version)
    if sources is None:
        available = ", ".join(DATASET_VERSIONS.keys())
        raise ValueError(f"Unknown version '{version}'. Available: {available}")

    print(f"📦 Building dataset {version}")
    print(f"   Sources: {', '.join(sources)}")

    # Load all sources
    all_messages: list[ParsedMessage] = []
    for source in sources:
        msgs = load_parsed_source(source)
        print(f"   {source}: {len(msgs)} messages")
        all_messages.extend(msgs)

    if not all_messages:
        print("   ⚠ No messages found. Run the parsers first.")
        return []

    # Filter: minimum length
    before = len(all_messages)
    all_messages = [m for m in all_messages if m.word_count >= MIN_MESSAGE_LENGTH]
    print(f"   After min-length filter ({MIN_MESSAGE_LENGTH} words): {len(all_messages)} (removed {before - len(all_messages)})")

    # Deduplicate
    before = len(all_messages)
    all_messages = deduplicate(all_messages)
    print(f"   After dedup: {len(all_messages)} (removed {before - len(all_messages)} duplicates)")

    # Score and filter
    scored: list[dict] = []
    for i, msg in enumerate(all_messages):
        quality = compute_quality_score(msg)
        if quality < min_quality:
            continue
        d = msg.to_dict()
        d["quality_score"] = round(quality, 3)
        d["id"] = f"{msg.source.value}_{i:06d}"
        scored.append(d)

    print(f"   After quality filter (>={min_quality}): {len(scored)}")

    # Sort by quality (highest first)
    scored.sort(key=lambda x: x["quality_score"], reverse=True)

    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Build versioned training dataset")
    parser.add_argument(
        "--version", "-v",
        type=str,
        default="v0.1",
        choices=list(DATASET_VERSIONS.keys()),
        help="Dataset version to build",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.3,
        help="Minimum quality score (0-1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/personal/processed/{version}.jsonl)",
    )
    args = parser.parse_args()

    output = args.output or (PROCESSED_DIR / f"{args.version}.jsonl")

    dataset = build_dataset(args.version, args.min_quality)

    if not dataset:
        print("❌ No messages passed quality filters. Check your parsed data.")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset {args.version}: {len(dataset)} examples → {output}")

    # Stats
    by_source: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    total_quality = 0.0
    for entry in dataset:
        src = entry["source"]
        by_source[src] = by_source.get(src, 0) + 1
        lang = entry["language"]
        by_lang[lang] = by_lang.get(lang, 0) + 1
        total_quality += entry["quality_score"]

    print("\n📊 Stats:")
    print(f"   By source: {by_source}")
    print(f"   By language: {by_lang}")
    print(f"   Avg quality: {total_quality / len(dataset):.3f}")


if __name__ == "__main__":
    main()
