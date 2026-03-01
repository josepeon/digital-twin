#!/usr/bin/env python3
"""
PII scrubbing CLI — scrub a JSONL file of parsed messages.

Reads a parsed JSONL, scrubs PII from text and context fields,
reports statistics on what was redacted.

Usage:
    python scripts/scrub_pii.py --input data/personal/processed/chatgpt_parsed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pii import scrub_text


def scrub_jsonl(input_path: Path, output_path: Path | None = None) -> dict:
    """Scrub PII from a JSONL file.

    Args:
        input_path: Input JSONL path.
        output_path: Output path. If None, overwrites input.

    Returns:
        Stats dict with redaction counts per category.
    """
    out = output_path or input_path
    lines: list[str] = []
    stats: dict[str, int] = {}
    total_redactions = 0
    total_messages = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_messages += 1
            entry = json.loads(line)

            # Scrub text field
            if "text" in entry:
                result = scrub_text(entry["text"])
                entry["text"] = result.scrubbed
                for r in result.redactions:
                    cat = r["category"]
                    stats[cat] = stats.get(cat, 0) + 1
                    total_redactions += 1

            # Scrub context field
            if "context" in entry and entry["context"]:
                result = scrub_text(entry["context"])
                entry["context"] = result.scrubbed
                for r in result.redactions:
                    cat = r["category"]
                    stats[cat] = stats.get(cat, 0) + 1
                    total_redactions += 1

            lines.append(json.dumps(entry, ensure_ascii=False))

    with open(out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return {
        "total_messages": total_messages,
        "total_redactions": total_redactions,
        "by_category": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub PII from parsed JSONL")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL (default: overwrite input)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    print(f"🔒 Scrubbing PII: {args.input}")
    stats = scrub_jsonl(args.input, args.output)

    out = args.output or args.input
    print(f"✅ Wrote to: {out}")
    print(f"   Messages processed: {stats['total_messages']}")
    print(f"   Total redactions: {stats['total_redactions']}")
    if stats["by_category"]:
        print("   By category:")
        for cat, count in sorted(stats["by_category"].items()):
            print(f"     {cat}: {count}")
    else:
        print("   No PII found — data is clean.")


if __name__ == "__main__":
    main()
