"""
Phase 5 – Turing Test (Interactive Blind Test)

Presents the user with pairs of responses — one from the real Jose (RAG)
and one from the twin — in random order. The user guesses which is real.

Tracks:
  - Accuracy (can the user tell them apart?)
  - Confidence calibration
  - Per-category breakdown

Usage:
    python -m tests.eval_turing                # local engine
    python -m tests.eval_turing --api URL      # deployed API
    python -m tests.eval_turing --rounds 5     # custom round count
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Turing Test Prompts ────────────────────────────────────────────────────────
# Each has a "context" (what triggered the message) and we'll pull a real
# Jose message from RAG for comparison.

TURING_PROMPTS = [
    {
        "context": "Someone asks what Jose does for work",
        "query": "What do you do?",
        "category": "identity",
    },
    {
        "context": "Casual greeting in a group chat",
        "query": "Hey what's up man",
        "category": "casual",
    },
    {
        "context": "Someone asks about a technical topic",
        "query": "Should I learn Python or JavaScript first?",
        "category": "technical",
    },
    {
        "context": "A friend asks about Jose's startup",
        "query": "How's OH going? Any updates?",
        "category": "identity",
    },
    {
        "context": "Someone greets Jose in Spanish",
        "query": "Oye asere qué bolá, cómo tú estás?",
        "category": "spanish",
    },
    {
        "context": "A friend shares something exciting",
        "query": "Bro I just got my first job in tech!",
        "category": "casual",
    },
    {
        "context": "Someone asks Jose's opinion on AI",
        "query": "Do you think AI will replace software engineers?",
        "category": "technical",
    },
    {
        "context": "A family member asks how Jose is doing",
        "query": "Mijo cómo estás? Todo bien?",
        "category": "spanish",
    },
    {
        "context": "Someone asks about Jose's background",
        "query": "Where did you grow up?",
        "category": "identity",
    },
    {
        "context": "A friend asks about what Jose is reading/watching",
        "query": "Seen any good shows lately?",
        "category": "casual",
    },
    {
        "context": "Discussion about building products",
        "query": "What's the hardest part about building a startup?",
        "category": "technical",
    },
    {
        "context": "Someone asks for advice",
        "query": "I wanna switch to software engineering, any tips?",
        "category": "technical",
    },
]


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class TuringRound:
    round_num: int
    query: str
    category: str
    real_text: str
    twin_text: str
    real_was_option: str  # "A" or "B"
    user_guess: str       # "A" or "B"
    correct: bool
    time_secs: float


def _get_real_example(engine: Any, query: str) -> str:
    """Retrieve a real Jose message from RAG for comparison."""
    examples = engine.retrieve(query, n_results=3)
    if examples:
        # Pick the best-matching (first) real example
        return examples[0]["text"]
    return "(No real message found for this context)"


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_turing_test(
    *,
    engine: Any = None,
    api_url: str | None = None,
    api_key: str | None = None,
    rounds: int = 8,
) -> dict[str, Any]:
    """Run the interactive Turing test."""
    import requests

    # Need engine for RAG retrieval even in API mode
    if engine is None:
        from src.personality_engine import PersonalityEngine
        engine = PersonalityEngine()

    prompts = random.sample(TURING_PROMPTS, min(rounds, len(TURING_PROMPTS)))
    results: list[TuringRound] = []

    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  You'll see two responses to each prompt.   │")
    print("  │  One is a REAL Jose message from his data.  │")
    print("  │  The other is the AI twin's response.       │")
    print("  │  Type A or B — which one is the AI twin?    │")
    print("  │  Type Q to quit early.                      │")
    print("  └─────────────────────────────────────────────┘\n")

    for i, p in enumerate(prompts, 1):
        query = p["query"]

        # Get real message
        real_text = _get_real_example(engine, query)

        # Get twin response
        try:
            if api_url:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                resp = requests.post(
                    f"{api_url.rstrip('/')}/ask-jose",
                    json={"query": query},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                twin_text = resp.json()["response"]
            else:
                result = engine.ask(query)
                twin_text = result["response"]
        except Exception as e:
            twin_text = f"[ERROR: {e}]"

        # Randomize position
        real_is_a = random.choice([True, False])
        option_a = real_text if real_is_a else twin_text
        option_b = twin_text if real_is_a else real_text
        real_option = "A" if real_is_a else "B"

        print(f"  ━━ Round {i}/{len(prompts)} ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Context: {p['context']}")
        print(f"  Prompt:  \"{query}\"")
        print()
        print(f"    [A] {option_a}")
        print()
        print(f"    [B] {option_b}")
        print()

        start = time.time()
        while True:
            guess = input("  Which is the AI twin? (A/B/Q): ").strip().upper()
            if guess in ("A", "B", "Q"):
                break
            print("  Please enter A, B, or Q.")
        elapsed = time.time() - start

        if guess == "Q":
            print("\n  Quitting early.\n")
            break

        # The user is guessing which is the TWIN
        # Twin is the opposite of real_option
        twin_option = "B" if real_is_a else "A"
        correct = guess == twin_option

        result = TuringRound(
            round_num=i,
            query=query,
            category=p["category"],
            real_text=real_text,
            twin_text=twin_text,
            real_was_option=real_option,
            user_guess=guess,
            correct=correct,
            time_secs=round(elapsed, 1),
        )
        results.append(result)

        if correct:
            print(f"  ✅ Correct! The twin was option {twin_option}.")
        else:
            print(f"  ❌ Wrong. The twin was option {twin_option}.")
        print()

    # ── Summary ────────────────────────────────────────────────────────────
    if not results:
        return {"total_rounds": 0, "accuracy": 0, "results": []}

    accuracy = sum(1 for r in results if r.correct) / len(results)

    categories: dict[str, list[bool]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.correct)
    category_accuracy = {
        cat: sum(vals) / len(vals)
        for cat, vals in categories.items()
    }

    summary = {
        "total_rounds": len(results),
        "correct_guesses": sum(1 for r in results if r.correct),
        "accuracy": round(accuracy, 4),
        "twin_pass_rate": round(1 - accuracy, 4),  # Lower accuracy = better twin
        "category_accuracy": {k: round(v, 4) for k, v in category_accuracy.items()},
        "avg_decision_time_secs": round(sum(r.time_secs for r in results) / len(results), 1),
        "results": [
            {
                "round": r.round_num,
                "query": r.query,
                "category": r.category,
                "correct": r.correct,
                "time_secs": r.time_secs,
            }
            for r in results
        ],
    }

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Turing Test")
    parser.add_argument("--api", type=str, default=None, help="API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--rounds", type=int, default=8, help="Number of rounds")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Digital Twin — Turing Test                 ║")
    print("║       Can you spot the AI?                   ║")
    print("╚══════════════════════════════════════════════╝")

    if args.api:
        print(f"\n  Mode: Remote API ({args.api})")
    else:
        print("\n  Mode: Local engine")

    from src.personality_engine import PersonalityEngine
    engine = PersonalityEngine()

    summary = run_turing_test(
        engine=engine,
        api_url=args.api,
        api_key=args.api_key,
        rounds=args.rounds,
    )

    # Print report
    print("\n" + "═" * 50)
    print(f"  Rounds Played:    {summary['total_rounds']}")
    print(f"  Your Accuracy:    {summary['accuracy']:.0%}")
    print(f"  Twin Pass Rate:   {summary['twin_pass_rate']:.0%}")
    print(f"  Avg Decision:     {summary['avg_decision_time_secs']}s")
    print()

    if summary.get("category_accuracy"):
        print("  Category Breakdown (your accuracy spotting the twin):")
        for cat, acc in sorted(summary["category_accuracy"].items()):
            indicator = "🤖" if acc > 0.6 else "🎭" if acc < 0.4 else "😐"
            print(f"    {indicator} {cat:15s} {acc:.0%}")
    print()
    if summary["twin_pass_rate"] >= 0.6:
        print("  🎭 The twin fooled you most of the time — it passes!")
    elif summary["twin_pass_rate"] >= 0.4:
        print("  😐 Mixed results — the twin is decent but not perfect.")
    else:
        print("  🤖 You spotted the twin easily — needs improvement.")
    print("═" * 50)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
