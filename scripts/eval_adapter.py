#!/usr/bin/env python3
"""Run the identity eval against a locally trained LoRA adapter.

The default identity eval (tests/eval_identity.py --quick) routes through
PersonalityEngine, which goes to Groq. This script bypasses Groq and runs
the same questions directly against an MLX adapter so we can see what the
canary actually sounds like before promoting it to production.

Usage:
    python scripts/eval_adapter.py
    python scripts/eval_adapter.py --version v1.0-twin --quick
    python scripts/eval_adapter.py --full --output results/v1.0-twin.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.personality_engine import JOSE_SYSTEM_PROMPT  # noqa: E402
from eval_identity import IDENTITY_QUESTIONS, score_response  # noqa: E402


async def _run(version: str, registry_path: Path, quick: bool, output: Path | None) -> int:
    from self_improving_agent.training.registry import ModelRegistry
    from self_improving_agent.providers.mlx_provider import MLXProvider
    from self_improving_agent.providers.base import GenerationConfig

    registry = ModelRegistry(registry_path)
    version_obj = registry.get(version)
    if version_obj is None:
        print(f"Version {version!r} not in registry at {registry_path}")
        return 1

    adapter_abs = (ROOT / version_obj.adapter_path).resolve()
    print(f"Adapter: {adapter_abs}")
    print(f"Base:    {version_obj.base_model}")

    provider = MLXProvider(
        model_path=version_obj.base_model,
        adapter_path=str(adapter_abs),
    )

    questions = IDENTITY_QUESTIONS[:10] if quick else IDENTITY_QUESTIONS
    config = GenerationConfig(temperature=0.7, top_p=0.95, max_tokens=200)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['question']}")
        try:
            resp = await provider.generate(
                messages=[
                    {"role": "system", "content": JOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": q["question"]},
                ],
                config=config,
            )
            text = resp.text.strip()
        except Exception as exc:  # noqa: BLE001
            text = f"[ERROR: {exc}]"

        qr = score_response(q, text)
        results.append(qr)
        icon = "✅" if qr.score >= 0.99 else "⚠️" if qr.score >= 0.5 else "❌"
        print(f"       {icon} score={qr.score:.0%}")
        print(f"       └─ {text[:200]}")
        if qr.notes:
            for note in qr.notes:
                print(f"          {note}")

    overall = sum(r.score for r in results) / len(results) if results else 0.0
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.score)

    print("\n" + "═" * 50)
    print(f"  Adapter:        {version}")
    print(f"  Overall:        {overall:.0%}")
    print(f"  Perfect:        {sum(1 for r in results if r.score >= 0.99)}/{len(results)}")
    print(f"  Partial:        {sum(1 for r in results if 0.01 < r.score < 0.99)}/{len(results)}")
    print(f"  Failed:         {sum(1 for r in results if r.score < 0.01)}/{len(results)}")
    print("\n  Category Breakdown:")
    for cat, scores in sorted(cats.items()):
        s = sum(scores) / len(scores)
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"    {cat:20s} {bar} {s:.0%}")
    print("═" * 50)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "version": version,
            "adapter_path": str(adapter_abs),
            "overall": overall,
            "category_scores": {k: sum(v) / len(v) for k, v in cats.items()},
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "response": r.response,
                    "score": r.score,
                    "notes": r.notes,
                }
                for r in results
            ],
        }, indent=2, ensure_ascii=False))
        print(f"\n  Saved to {output}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0-twin")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "models" / "registry.json",
    )
    parser.add_argument("--quick", action="store_true",
                        help="10 questions only (default: full set).")
    parser.add_argument("--full", action="store_true",
                        help="Run all questions (default).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results.")
    args = parser.parse_args()
    # Default to quick if neither specified explicitly
    quick = args.quick or not args.full

    return asyncio.run(_run(args.version, args.registry, quick, args.output))


if __name__ == "__main__":
    raise SystemExit(main())
