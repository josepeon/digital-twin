#!/usr/bin/env python3
"""Export captured twin conversations to SIA training format.

Reads ``data/personal/memory.db`` (the twin's live SQLite memory) and
extracts user → assistant pairs, writing them in
``self-improving-agent``'s expected SFT JSONL schema. With the [sia]
extra installed, the script can optionally drive SIA's distillation
pipeline directly.

This closes the loop the OH AI stack is built around: real users
chatting with the twin → SFT pairs → smaller local model that mimics
the twin → cheap inference fallback when Groq rate-limits.

Without ``--run distill``: writes JSONL only, prints the SIA training
command to run by hand.

With ``--run distill``: invokes SIA's ``run_distillation()`` on the
exported rows (requires GROQ_API_KEY for the judge — uses Groq's free
tier).

Usage:
    pip install -e '.[sia]'   # one-time

    # Export only
    python scripts/export_to_sia.py --out training_data/twin_v1.jsonl

    # Export + drive SIA's distillation pipeline
    python scripts/export_to_sia.py --run distill --out training_data/twin_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Project root on path so `from src...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_DB = Path(
    os.environ.get(
        "MEMORY_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "personal" / "memory.db"),
    )
)


# ---------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------


def extract_pairs(
    db_path: Path,
    *,
    min_user_chars: int = 5,
    min_assistant_chars: int = 30,
    max_chars: int = 8000,
) -> list[dict]:
    """Pull (user, assistant) pairs out of the memory.db sessions table.

    Filters:
      - drop trivially short user messages ("hi", "ok", "lol")
      - drop short / template assistant replies
      - drop overflows that the LLM would truncate anyway
      - dedupe on (user, assistant) so repeated test prompts don't bias
        the training data
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """SELECT session_id, role, content, timestamp
           FROM messages
           ORDER BY session_id, timestamp ASC"""
    )

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    pending_user: dict | None = None

    for session_id, role, content, timestamp in cursor.fetchall():
        text = (content or "").strip()
        if not text:
            pending_user = None
            continue

        if role == "user":
            pending_user = {
                "session_id": session_id,
                "content": text,
                "timestamp": timestamp,
            }
            continue

        if role == "assistant" and pending_user is not None:
            user_text = pending_user["content"]
            assistant_text = text

            # Quality filters
            if len(user_text) < min_user_chars:
                pending_user = None
                continue
            if len(assistant_text) < min_assistant_chars:
                pending_user = None
                continue
            if len(user_text) > max_chars or len(assistant_text) > max_chars:
                pending_user = None
                continue

            key = (user_text, assistant_text)
            if key in seen:
                pending_user = None
                continue
            seen.add(key)

            pairs.append({
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "metadata": {
                    "source": "twin-memory",
                    "session_id": session_id,
                    "timestamp": timestamp,
                },
            })
            pending_user = None

    conn.close()
    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> int:
    """Write rows as JSONL; create parents; return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# ---------------------------------------------------------------------
# Optional SIA integration
# ---------------------------------------------------------------------


def _sia_available() -> bool:
    try:
        import self_improving_agent  # noqa: F401
        return True
    except ImportError:
        return False


def run_validate() -> int:
    """Confirm SIA imports cleanly + sanity-check the bridge."""
    from self_improving_agent.observability.cost_tracker import (  # noqa: F401
        begin_run,
        record_usage,
        render_summary,
    )
    from self_improving_agent.reproducibility import set_seed  # noqa: F401
    from self_improving_agent.training.distillation import (  # noqa: F401
        ListQuestionPool,
        run_distillation,
    )
    from self_improving_agent.training.judges import (  # noqa: F401
        EnsembleJudge,
        ProviderJudge,
    )

    print(
        "\nSIA validation: every cross-repo module imported cleanly.\n"
        "  - self_improving_agent.observability.cost_tracker\n"
        "  - self_improving_agent.reproducibility.set_seed\n"
        "  - self_improving_agent.training.distillation (run_distillation)\n"
        "  - self_improving_agent.training.judges (EnsembleJudge / ProviderJudge)\n"
    )
    return 0


def run_distill(pairs: list[dict], out_path: Path, threshold: float) -> int:
    """Run SIA's distillation on the user prompts pulled from twin memory.

    Caps at 30 prompts during proof-of-concept so a single run stays
    well within the Groq free tier.
    """
    import asyncio

    from self_improving_agent.observability.cost_tracker import (
        begin_run,
        render_summary,
    )
    from self_improving_agent.reproducibility import set_seed
    from self_improving_agent.training.distillation import (
        ListQuestionPool,
        run_distillation,
    )

    if not pairs:
        print("No pairs to distill — twin memory is empty or all filtered out.")
        return 2

    set_seed(42)
    begin_run()

    prompts: list[str] = []
    for row in pairs:
        for msg in row["messages"]:
            if msg["role"] == "user":
                prompts.append(msg["content"])
                break

    pool = ListQuestionPool(prompts=prompts[:30])
    print(f"\nRunning SIA distillation on {len(pool.prompts)} prompt(s)...")
    try:
        report = asyncio.run(
            run_distillation(
                pool=pool,
                output_path=out_path.parent / (out_path.stem + ".distilled.jsonl"),
                judge_threshold=threshold,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Distillation failed: {exc}")
        return 1

    print(report.render())
    print()
    print(render_summary())
    return 0 if report.pairs_kept > 0 else 2


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to memory.db (default: data/personal/memory.db).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training_data/twin_pairs.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--run",
        choices=["none", "validate", "distill"],
        default="none",
        help=(
            "After exporting, optionally drive SIA: "
            "'validate' confirms the cross-repo install works, "
            "'distill' invokes SIA's distillation pipeline on the prompts."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Judge threshold (0-10) for kept distilled pairs (default 7.0).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No memory database at {args.db}.")
        print("Run the twin in production for a while to accumulate conversations.")
        return 1

    pairs = extract_pairs(args.db)
    count = write_jsonl(args.out, pairs)
    print(f"Wrote {count} pair(s) to {args.out}")

    if not pairs:
        print(
            "Memory was non-empty but no pairs passed the quality filters.\n"
            "Lower the thresholds or chat with the twin more."
        )

    if args.run == "none":
        print(
            "\nFeed this file to SIA's training pipeline:\n"
            "    python -m self_improving_agent.training.finetune "
            f"--data {args.out}"
        )
        return 0

    if not _sia_available():
        print(
            "\nSIA isn't installed. Install with:\n"
            "    pip install -e '.[sia]'\n"
            "Then re-run with --run distill (or --run validate)."
        )
        return 2

    if args.run == "validate":
        return run_validate()

    if args.run == "distill":
        return run_distill(pairs, args.out, args.threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
