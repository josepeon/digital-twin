#!/usr/bin/env python3
"""Incrementally ingest new message exports into the RAG index.

The original pipeline (parse → build_dataset → build_rag_index) is a
one-shot rebuild. This script is the *incremental* counterpart: watch a
drop-zone directory, detect new exports by content hash, run the right
parser, and append to ChromaDB without rebuilding from scratch.

State (which files have been seen) is tracked in
``data/personal/.ingested.json`` so re-runs are idempotent — drop a
file you've already processed and the script no-ops.

Detection: filename suffix and/or content sniff:
  - *.html | instagram*   → parse_instagram
  - chatgpt*.json         → parse_chatgpt
  - chat.db | imessage*   → parse_imessage
  - WhatsApp*.txt | wa*   → parse_whatsapp
  - *.eml | email*        → parse_email

Without --input-dir the default is ``data/personal/incoming/``. Pair
with a Railway cron or a local watcher for the "real-time learning"
half of Phase 8b.

Usage:
    # One-shot
    python scripts/ingest_new_data.py

    # Watch + reprocess every 5 minutes
    python scripts/ingest_new_data.py --watch 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_INCOMING = Path("data/personal/incoming")
DEFAULT_STATE = Path("data/personal/.ingested.json")
DEFAULT_PROCESSED = Path("data/personal/processed")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------

SourceDetector = Callable[[Path], str | None]


def _detect_source(path: Path) -> str | None:
    """Identify which parser to use based on filename + extension.

    Returns ``"instagram" | "chatgpt" | "imessage" | "whatsapp" | "email"``
    or ``None`` if the file doesn't match any known source.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()

    if "instagram" in name or (suffix == ".html" and "messages" in name):
        return "instagram"
    if "chatgpt" in name and suffix == ".json":
        return "chatgpt"
    if name == "chat.db" or "imessage" in name:
        return "imessage"
    if "whatsapp" in name or name.startswith("wa_"):
        return "whatsapp"
    if suffix == ".eml" or "email" in name or "mbox" in name:
        return "email"
    return None


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------


class IngestState:
    """JSON file recording (file_hash → metadata) for everything ingested."""

    def __init__(self, path: Path = DEFAULT_STATE):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("Could not parse %s — starting fresh", path)
                self._data = {}

    def seen(self, file_hash: str) -> bool:
        return file_hash in self._data

    def mark(self, file_hash: str, path: Path, source: str, count: int) -> None:
        self._data[file_hash] = {
            "path": str(path),
            "source": source,
            "ingested_at": time.time(),
            "messages": count,
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    @property
    def total_ingested(self) -> int:
        return sum(entry.get("messages", 0) for entry in self._data.values())


def hash_file(path: Path) -> str:
    """SHA-1 of the file contents — small, fast, collision-safe enough here."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Parsing dispatch
# ---------------------------------------------------------------------


def _parser_module(source: str):
    """Lazy-import the matching parser module by source name."""
    if source == "instagram":
        from scripts import parse_instagram as mod
    elif source == "chatgpt":
        from scripts import parse_chatgpt as mod
    elif source == "imessage":
        from scripts import parse_imessage as mod
    elif source == "whatsapp":
        from scripts import parse_whatsapp as mod
    elif source == "email":
        from scripts import parse_email as mod
    else:
        raise ValueError(f"Unknown source: {source}")
    return mod


def _has_parse_function(module) -> bool:
    """Best-effort signature check — parsers historically expose
    one of ``parse(path)`` / ``parse_file(path)`` / ``main()``."""
    return any(
        hasattr(module, name) for name in ("parse", "parse_file", "main")
    )


# ---------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------


def discover_new_files(incoming: Path, state: IngestState) -> list[tuple[Path, str, str]]:
    """Return [(path, source, hash)] for every never-seen file in ``incoming``."""
    if not incoming.exists():
        return []
    found: list[tuple[Path, str, str]] = []
    for path in sorted(incoming.iterdir()):
        if not path.is_file():
            continue
        source = _detect_source(path)
        if source is None:
            logger.debug("Skipping %s — unrecognised source", path.name)
            continue
        file_hash = hash_file(path)
        if state.seen(file_hash):
            logger.debug("Skipping %s — already ingested", path.name)
            continue
        found.append((path, source, file_hash))
    return found


def run_once(
    incoming: Path = DEFAULT_INCOMING,
    state_path: Path = DEFAULT_STATE,
    rebuild_index: bool = False,
) -> int:
    """Process every new file in ``incoming``. Returns count ingested."""
    state = IngestState(state_path)
    new_files = discover_new_files(incoming, state)

    if not new_files:
        logger.info(
            "No new files in %s. State: %d file(s), %d message(s) ingested total.",
            incoming, len(state._data), state.total_ingested,
        )
        return 0

    total_added = 0
    for path, source, file_hash in new_files:
        logger.info("Ingesting %s (%s)", path.name, source)
        try:
            module = _parser_module(source)
            if not _has_parse_function(module):
                logger.warning(
                    "Parser %s has no parse()/parse_file()/main() — skipping",
                    source,
                )
                continue

            # Each parser writes to data/personal/processed/; we trust the
            # parser's default output path. Hand it our incoming file.
            count = _invoke_parser(module, path)
            state.mark(file_hash, path, source, count)
            total_added += count
            logger.info("  +%d message(s) from %s", count, path.name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest %s: %s", path, exc)

    if total_added > 0:
        logger.info(
            "Rebuilding RAG index (incremental, %d new message(s))...",
            total_added,
        )
        _refresh_rag_index(rebuild=rebuild_index)

    return total_added


def _invoke_parser(module, path: Path) -> int:
    """Call the most appropriate entry point on the parser module.

    Returns the count of messages produced. The historic parsers print
    counts but don't always return them — we make a best-effort attempt
    and fall back to 0 when the API is non-uniform.
    """
    if hasattr(module, "parse_file"):
        result = module.parse_file(path)
        return int(result) if isinstance(result, int) else (len(result) if hasattr(result, "__len__") else 0)
    if hasattr(module, "parse"):
        result = module.parse(path)
        return int(result) if isinstance(result, int) else (len(result) if hasattr(result, "__len__") else 0)
    if hasattr(module, "main"):
        # Old-style: main() reads its own paths via argparse. We can't
        # easily plumb our file in, so log a hint instead.
        logger.warning(
            "%s only exposes main(); call it manually with --input %s",
            module.__name__, path,
        )
        return 0
    return 0


def _refresh_rag_index(rebuild: bool = False) -> None:
    """Re-run the RAG indexer on the latest dataset.

    Default is incremental (skips already-indexed IDs). ``rebuild=True``
    forces a full rebuild — only useful when changing embeddings or
    chunk format.
    """
    try:
        from scripts.build_rag_index import build_rag_index
        from src.config import PROCESSED_DIR
    except ImportError as exc:
        logger.warning("Could not import RAG indexer: %s", exc)
        return

    # Find the latest dataset file
    candidates = sorted(PROCESSED_DIR.glob("*.jsonl"))
    if not candidates:
        logger.info("No dataset to index in %s", PROCESSED_DIR)
        return
    dataset_path = candidates[-1]
    try:
        n = build_rag_index(dataset_path, rebuild=rebuild)
        logger.info("RAG index: %d document(s) embedded.", n)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG index update failed: %s", exc)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INCOMING,
        help="Directory to scan for new exports (default: data/personal/incoming/).",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help="State file tracking already-ingested files.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full RAG rebuild instead of incremental update.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, sleep this many seconds and re-scan in a loop.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.watch <= 0:
        run_once(args.input_dir, args.state, rebuild_index=args.rebuild)
        return 0

    logger.info(
        "Watching %s every %ds. Ctrl-C to stop.", args.input_dir, args.watch
    )
    try:
        while True:
            run_once(args.input_dir, args.state, rebuild_index=args.rebuild)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        logger.info("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
