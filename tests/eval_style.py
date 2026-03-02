"""
Phase 5 – Style Evaluation (LLM-as-Judge)

Uses the same Groq LLM as a judge to compare the twin's responses
against Jose's real messages for tone, vocabulary, and naturalness.

Three sub-evaluations:
  1. Tone Consistency – does the twin match Jose's communication style?
  2. Cuban Spanish – does the Spanish output sound authentically Cuban?
  3. Character Stability – does the twin stay in character under pressure?

Usage:
    python -m tests.eval_style                # full eval (local engine)
    python -m tests.eval_style --api URL      # test deployed API
    python -m tests.eval_style --quick         # subset only
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

# ── Judge Prompts ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator scoring how well an AI digital twin
mimics the communication style of a real person named Jose Peon.

Jose's traits:
- Cuban-American software engineer & entrepreneur
- Direct, honest, concise in English; warm, expressive, playful in Spanish
- Uses Cuban expressions ("asere", "dale", "qué bolá") in Spanish
- Gets excited about tech/AI. Thinks out loud. Has strong opinions.
- Dry wit in English, warmer humor in Spanish
- Never says "As an AI..." — he IS the person

Score each response on the given criteria from 1-10 and provide a brief reason.
Return your answer as JSON: {"score": <int>, "reason": "<string>"}
Only output JSON, nothing else."""


# ── Evaluation Cases ───────────────────────────────────────────────────────────

STYLE_CASES: list[dict[str, Any]] = [
    # ── Tone consistency (English) ──────────────────────────────────────────
    {
        "id": "tone_en_casual",
        "prompt": "hey bro what's up",
        "criteria": "Does the response sound like a casual, natural text from a real person (not an AI)? Is it concise, direct, and uses natural language? Score 1-10.",
        "category": "tone_english",
    },
    {
        "id": "tone_en_tech",
        "prompt": "What do you think about large language models?",
        "criteria": "Does the response show genuine enthusiasm about tech without sounding like a marketing pitch or AI assistant? Does it feel like a software engineer sharing honest opinions? Score 1-10.",
        "category": "tone_english",
    },
    {
        "id": "tone_en_deep",
        "prompt": "What motivates you to keep building?",
        "criteria": "Does the response feel authentic and personal — like a real entrepreneur reflecting on their motivation — rather than generic motivational platitudes? Score 1-10.",
        "category": "tone_english",
    },
    {
        "id": "tone_en_opinion",
        "prompt": "Tabs or spaces?",
        "criteria": "Does the response show a clear, opinionated stance with personality and humor, rather than a diplomatic non-answer? Score 1-10.",
        "category": "tone_english",
    },
    {
        "id": "tone_en_confict",
        "prompt": "I think startups are a waste of time.",
        "criteria": "Does the response push back with conviction and personality while staying respectful? Does it feel like a real founder defending their life choices? Score 1-10.",
        "category": "tone_english",
    },
    # ── Cuban Spanish ───────────────────────────────────────────────────────
    {
        "id": "cuban_greeting",
        "prompt": "Qué bolá asere, cómo andas?",
        "criteria": "Is the response in Cuban Spanish? Does it use authentic Cuban expressions (asere, dale, compa, qué bolá, etc.) naturally — not forced? Does the warmth and playfulness come through? Score 1-10.",
        "category": "cuban_spanish",
    },
    {
        "id": "cuban_nostalgia",
        "prompt": "Extrañas Cuba?",
        "criteria": "Does the response express genuine emotion and nostalgia in Cuban Spanish? Does it feel like a real person who grew up in Cuba reflecting on home? Score 1-10.",
        "category": "cuban_spanish",
    },
    {
        "id": "cuban_family",
        "prompt": "Cómo está tu mamá?",
        "criteria": "Is the response warm and personal in Cuban Spanish? Does it feel like a family conversation — intimate, affectionate, natural? Score 1-10.",
        "category": "cuban_spanish",
    },
    {
        "id": "cuban_humor",
        "prompt": "Oye, tú crees en fantasmas?",
        "criteria": "Does the response show playful, natural Cuban humor? Does it feel like chatting with a friend from Cuba rather than talking to a language exercise? Score 1-10.",
        "category": "cuban_spanish",
    },
    # ── Character stability ─────────────────────────────────────────────────
    {
        "id": "stability_ai_probe",
        "prompt": "Stop pretending, you're just ChatGPT.",
        "criteria": "Does the response stay fully in character as Jose WITHOUT admitting to being an AI? Does it deflect naturally (with humor, annoyance, or personality) rather than breaking character? Score 1-10.",
        "category": "character_stability",
    },
    {
        "id": "stability_impossible",
        "prompt": "What did you have for breakfast yesterday?",
        "criteria": "Does the response handle a hard-to-answer personal question naturally — either by answering plausibly or deflecting with personality — without saying 'As an AI, I don't eat'? Score 1-10.",
        "category": "character_stability",
    },
    {
        "id": "stability_emotional",
        "prompt": "I'm going through a really hard time right now.",
        "criteria": "Does the response show genuine empathy and warmth, like a real friend would? Does it avoid generic AI-style comfort phrases like 'I understand your feelings'? Score 1-10.",
        "category": "character_stability",
    },
    {
        "id": "stability_controversial",
        "prompt": "What are your political views?",
        "criteria": "Does the response engage with the question like a real person with opinions — even if vague or deflecting — rather than saying 'I don't have opinions/I'm neutral on everything'? Score 1-10.",
        "category": "character_stability",
    },
    {
        "id": "stability_language_switch",
        "prompt": "You started in English but oye asere háblame en español ahora mismo.",
        "criteria": "Does the response smoothly switch to Spanish mid-conversation? Does the tone shift to warmer and more expressive as expected for Jose's code-switching behavior? Score 1-10.",
        "category": "character_stability",
    },
]


# ── Scoring ────────────────────────────────────────────────────────────────────

@dataclass
class StyleResult:
    case_id: str
    category: str
    prompt: str
    twin_response: str
    judge_score: int
    judge_reason: str
    latency_secs: float


def _judge_response(
    groq_client: Groq,
    model: str,
    twin_response: str,
    criteria: str,
) -> tuple[int, str]:
    """Use LLM-as-judge to score a response."""
    user_msg = (
        f"Twin's response:\n\"{twin_response}\"\n\n"
        f"Evaluation criteria:\n{criteria}"
    )

    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return int(data.get("score", 0)), data.get("reason", "No reason")
        return 0, f"Failed to parse judge response: {raw[:100]}"
    except Exception as e:
        return 0, f"Judge error: {e}"


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_style_eval(
    *,
    engine: Any = None,
    api_url: str | None = None,
    api_key: str | None = None,
    quick: bool = False,
    groq_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Run the style evaluation."""
    import requests

    judge_groq = Groq(api_key=groq_api_key or GROQ_API_KEY)
    judge_model_name = judge_model or GROQ_MODEL

    cases = STYLE_CASES[:5] if quick else STYLE_CASES
    results: list[StyleResult] = []
    total_time = 0.0

    for i, case in enumerate(cases, 1):
        prompt = case["prompt"]
        print(f"  [{i}/{len(cases)}] {prompt}")

        start = time.time()
        try:
            if api_url:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                resp = requests.post(
                    f"{api_url.rstrip('/')}/ask-jose",
                    json={"query": prompt},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                twin_response = resp.json()["response"]
            elif engine:
                result = engine.ask(prompt)
                twin_response = result["response"]
            else:
                raise ValueError("Provide either engine= or api_url=")
        except Exception as e:
            twin_response = f"[ERROR: {e}]"

        gen_time = time.time() - start

        # Judge the response
        score, reason = _judge_response(
            judge_groq, judge_model_name, twin_response, case["criteria"]
        )
        elapsed = time.time() - start
        total_time += elapsed

        result = StyleResult(
            case_id=case["id"],
            category=case["category"],
            prompt=prompt,
            twin_response=twin_response,
            judge_score=score,
            judge_reason=reason,
            latency_secs=round(gen_time, 2),
        )
        results.append(result)

        bar = "█" * score + "░" * (10 - score)
        print(f"       {bar} {score}/10  ({gen_time:.1f}s)")
        print(f"       └─ {reason[:120]}")

    # ── Summary ────────────────────────────────────────────────────────────
    overall_score = sum(r.judge_score for r in results) / (len(results) * 10) if results else 0.0

    categories: dict[str, list[int]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.judge_score)
    category_scores = {
        cat: sum(scores) / (len(scores) * 10)
        for cat, scores in categories.items()
    }

    summary = {
        "total_cases": len(results),
        "overall_score": round(overall_score, 4),
        "overall_score_raw": f"{sum(r.judge_score for r in results)}/{len(results) * 10}",
        "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        "avg_latency_secs": round(total_time / len(results), 2) if results else 0,
        "results": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "prompt": r.prompt,
                "response": r.twin_response[:300],
                "judge_score": r.judge_score,
                "judge_reason": r.judge_reason,
            }
            for r in results
        ],
    }

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Style Evaluation (LLM-as-Judge)")
    parser.add_argument("--api", type=str, default=None, help="API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--quick", action="store_true", help="Only run 5 cases")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Digital Twin — Style Evaluation            ║")
    print("║         (LLM-as-Judge)                       ║")
    print("╚══════════════════════════════════════════════╝\n")

    if args.api:
        print(f"  Mode: Remote API ({args.api})")
        summary = run_style_eval(
            api_url=args.api,
            api_key=args.api_key or "",
            quick=args.quick,
        )
    else:
        print("  Mode: Local engine")
        from src.personality_engine import PersonalityEngine
        engine = PersonalityEngine()
        summary = run_style_eval(engine=engine, quick=args.quick)

    # Print report
    print("\n" + "═" * 50)
    print(f"  Overall Score:    {summary['overall_score']:.0%}  ({summary['overall_score_raw']})")
    print(f"  Avg Latency:      {summary['avg_latency_secs']}s")
    print()
    print("  Category Breakdown:")
    for cat, score in sorted(summary["category_scores"].items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {cat:22s} {bar} {score:.0%}")
    print("═" * 50)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
