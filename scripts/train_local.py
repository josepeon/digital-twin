#!/usr/bin/env python3
"""Train the twin's LoRA fallback locally via SIA's MLX pipeline.

The original Phase 8g plan said "$2-3/hr cloud GPU." This wrapper pivots
to free local training on Apple Silicon — SIA already ships an MLX LoRA
fine-tune pipeline. We just hand it the twin's training data.

Inputs:
  * data/training/jose_train.jsonl — already produced by
    scripts/export_training_data.py (Phase 7g)
  * or training_data/twin_pairs.jsonl from scripts/export_to_sia.py

Output:
  * models/finetuned/<version>/adapters/ — LoRA adapter
  * Registered in SIA's model registry so it can be looked up later

The trained adapter is OPTIONAL. The twin defaults to Groq; the
adapter is wired in src/config.py as a fallback when Groq rate-limits.
Phase 7 build of the offline fallback never finished training; this
closes that loop.

Usage:
    pip install -e '.[sia]'   # installs SIA + mlx

    # Train v1.0 from the Phase 7 export
    python scripts/train_local.py \\
        --data data/training/jose_train.jsonl \\
        --version v1.0-twin

    # Or from the bridge export
    python scripts/train_local.py \\
        --data training_data/twin_pairs.jsonl \\
        --version v1.0-twin

Requirements: macOS + Apple Silicon. mlx-lm is in SIA's [mlx] extra.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DEFAULT_OUT_DIR = Path("models/finetuned")


def _check_sia_available() -> bool:
    try:
        import self_improving_agent  # noqa: F401
        return True
    except ImportError:
        return False


def load_dataset(path: Path) -> list[dict]:
    """Load JSONL examples from ``path``.

    Accepts either of two shapes:
      - SIA / OpenAI chat format: {"messages": [...]}
      - Flat pair format:         {"user": ..., "assistant": ...}
    """
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line: %s", exc)
                continue

            if "messages" in row:
                rows.append(row)
                continue
            # Flat → chat format
            user = row.get("user") or row.get("query")
            assistant = row.get("assistant") or row.get("response")
            if user and assistant:
                rows.append({
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                })
    return rows


async def _run_train(
    dataset: list[dict],
    version: str,
    base_model: str,
    output_dir: Path,
    epochs: int,
    rank: int,
) -> str:
    """Invoke SIA's FineTuner. Returns the adapter path."""
    from self_improving_agent.reproducibility import set_seed
    from self_improving_agent.training.finetune import FineTuner, LoRAConfig
    from self_improving_agent.training.registry import (
        ModelRegistry,
        ModelStatus,
        ModelVersion,
    )

    set_seed(42)  # reproducible runs

    config = LoRAConfig(rank=rank, num_epochs=epochs)
    tuner = FineTuner(
        base_model=base_model,
        output_dir=output_dir,
        lora_config=config,
    )

    # SIA's FineTuner._format_for_mlx expects flat {"input": ..., "output": ...}
    # rows, but our pipeline produces chat-format (optionally with a leading
    # system message). Flatten by finding the first user→assistant pair.
    flat_dataset = []
    for row in dataset:
        if "input" in row and "output" in row:
            flat_dataset.append(row)
            continue
        msgs = row.get("messages") or []
        user_content = None
        for msg in msgs:
            role = msg.get("role")
            if role == "user" and user_content is None:
                user_content = msg.get("content")
            elif role == "assistant" and user_content is not None:
                flat_dataset.append({
                    "input": user_content,
                    "output": msg.get("content", ""),
                })
                user_content = None
                break

    adapter_path = await tuner.train(flat_dataset, version=version, epochs=epochs)

    # Register in SIA's model registry so future runs can look it up
    registry = ModelRegistry(output_dir.parent / "registry.json")
    registry.register(
        ModelVersion(
            version=version,
            base_model=base_model,
            adapter_path=str(adapter_path),
            training_dataset_size=len(flat_dataset),
            status=ModelStatus.CANARY,  # promote to production manually after eval
        )
    )
    return str(adapter_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="JSONL training data (chat or flat format).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-twin",
        help="Adapter version tag (default: v1.0-twin).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"MLX base model (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to write adapters (default: models/finetuned/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without actually training.",
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

    if not _check_sia_available():
        print(
            "self-improving-agent isn't installed. Install with:\n"
            "    pip install -e '.[sia]'\n"
            "Note: SIA's MLX dependency requires macOS + Apple Silicon."
        )
        return 1

    if not args.data.exists():
        print(f"Training data not found: {args.data}")
        return 1

    dataset = load_dataset(args.data)
    if not dataset:
        print(f"No usable examples in {args.data}.")
        return 1

    print(f"Loaded {len(dataset)} training example(s) from {args.data}")
    print(f"Base model:  {args.base_model}")
    print(f"Adapter ver: {args.version}")
    print(f"Epochs:      {args.epochs}")
    print(f"LoRA rank:   {args.rank}")
    print(f"Output:      {args.output_dir}/{args.version}/adapters/")

    if args.dry_run:
        print("\n--dry-run: would train but didn't.")
        return 0

    print("\nTraining... (~20 min on M3 Max for the Phase 7g 19k-pair set)")
    try:
        adapter_path = asyncio.run(
            _run_train(
                dataset=dataset,
                version=args.version,
                base_model=args.base_model,
                output_dir=args.output_dir,
                epochs=args.epochs,
                rank=args.rank,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        return 1

    print(f"\nAdapter saved to: {adapter_path}")
    print(
        "Registered in SIA registry as canary; promote to production "
        "after evaluating with:\n"
        "    python -m tests.eval_identity --quick"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
