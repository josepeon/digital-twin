"""
Phase 5 – Evaluation Runner

Orchestrates all evaluation suites and produces a unified report.

Usage:
    python -m tests.eval_runner                    # full local eval
    python -m tests.eval_runner --api URL --api-key KEY   # remote eval
    python -m tests.eval_runner --quick             # fast subset
    python -m tests.eval_runner --suite identity    # single suite
    python -m tests.eval_runner --suite style       # single suite
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.eval_identity import run_identity_eval
from tests.eval_style import run_style_eval


def _grade(score: float) -> str:
    """Convert 0-1 score to letter grade."""
    if score >= 0.95:
        return "A+"
    if score >= 0.90:
        return "A"
    if score >= 0.85:
        return "A-"
    if score >= 0.80:
        return "B+"
    if score >= 0.75:
        return "B"
    if score >= 0.70:
        return "B-"
    if score >= 0.65:
        return "C+"
    if score >= 0.60:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


def run_full_eval(
    *,
    engine: Any = None,
    api_url: str | None = None,
    api_key: str | None = None,
    quick: bool = False,
    suites: list[str] | None = None,
) -> dict[str, Any]:
    """Run all evaluation suites and return a unified report."""

    target_suites = suites or ["identity", "style"]
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "api" if api_url else "local",
        "quick": quick,
        "suites": {},
    }

    total_start = time.time()

    # ── Identity Evaluation ────────────────────────────────────────────────
    if "identity" in target_suites:
        print("━" * 50)
        print("  SUITE 1: Identity Evaluation")
        print("━" * 50)
        identity_results = run_identity_eval(
            engine=engine,
            api_url=api_url,
            api_key=api_key,
            quick=quick,
        )
        report["suites"]["identity"] = identity_results
        print()

    # ── Style Evaluation ───────────────────────────────────────────────────
    if "style" in target_suites:
        print("━" * 50)
        print("  SUITE 2: Style Evaluation (LLM-as-Judge)")
        print("━" * 50)
        style_results = run_style_eval(
            engine=engine,
            api_url=api_url,
            api_key=api_key,
            quick=quick,
        )
        report["suites"]["style"] = style_results
        print()

    total_time = time.time() - total_start
    report["total_time_secs"] = round(total_time, 1)

    # ── Composite Score ────────────────────────────────────────────────────
    suite_scores = []
    if "identity" in report["suites"]:
        suite_scores.append(("Identity", report["suites"]["identity"]["overall_score"]))
    if "style" in report["suites"]:
        suite_scores.append(("Style", report["suites"]["style"]["overall_score"]))

    if suite_scores:
        composite = sum(s for _, s in suite_scores) / len(suite_scores)
        report["composite_score"] = round(composite, 4)
        report["grade"] = _grade(composite)

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║        DIGITAL TWIN — EVALUATION REPORT             ║")
    print("╠══════════════════════════════════════════════════════╣")

    # Composite
    composite = report.get("composite_score", 0)
    grade = report.get("grade", "?")
    print(f"║                                                      ║")
    print(f"║  Composite Score:  {composite:.0%}   Grade: {grade:3s}                    ║")
    print(f"║  Total Time:       {report.get('total_time_secs', 0):.0f}s                               ║")
    print(f"║                                                      ║")
    print("╠══════════════════════════════════════════════════════╣")

    # Suite breakdown
    for suite_name, suite_data in report.get("suites", {}).items():
        score = suite_data.get("overall_score", 0)
        grade = _grade(score)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"║  {suite_name.upper():12s}  {bar} {score:.0%}  ({grade})     ║")

        # Category details
        cat_scores = suite_data.get("category_scores", {})
        for cat, cat_score in sorted(cat_scores.items()):
            mini_bar = "█" * int(cat_score * 10) + "░" * (10 - int(cat_score * 10))
            print(f"║    {cat:20s} {mini_bar} {cat_score:.0%}           ║")

    print("╠══════════════════════════════════════════════════════╣")

    # Highlights
    all_results = []
    for suite_data in report.get("suites", {}).values():
        all_results.extend(suite_data.get("results", []))

    # Find worst performers
    scored_results = [r for r in all_results if "score" in r or "judge_score" in r]
    if scored_results:
        scored_results.sort(key=lambda r: r.get("score", r.get("judge_score", 0) / 10))
        print("║  AREAS FOR IMPROVEMENT:                              ║")
        for r in scored_results[:3]:
            score = r.get("score", r.get("judge_score", 0) / 10)
            q = r.get("question", r.get("prompt", ""))[:40]
            notes = r.get("notes", [r.get("judge_reason", "")])
            note = notes[0][:40] if notes else ""
            print(f"║    {score:.0%} │ {q:40s} ║")
            if note:
                print(f"║       └─ {note:43s} ║")

    print("╚══════════════════════════════════════════════════════╝")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Evaluation Runner")
    parser.add_argument("--api", type=str, default=None, help="API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--quick", action="store_true", help="Quick mode (subset)")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["identity", "style"],
                        help="Run a single suite")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Digital Twin — Evaluation Runner           ║")
    print("╚══════════════════════════════════════════════╝\n")

    suites = [args.suite] if args.suite else None

    if args.api:
        print(f"  Mode: Remote API ({args.api})")
        report = run_full_eval(
            api_url=args.api,
            api_key=args.api_key or "",
            quick=args.quick,
            suites=suites,
        )
    else:
        print("  Mode: Local engine")
        from src.personality_engine import PersonalityEngine
        engine = PersonalityEngine()
        report = run_full_eval(engine=engine, quick=args.quick, suites=suites)

    print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
