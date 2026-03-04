"""
Phase 5 – Identity Evaluation

Automated test suite that asks the digital twin 30+ factual questions
about Jose and scores accuracy using keyword matching + LLM-as-judge.

Usage:
    python -m tests.eval_identity                 # full eval
    python -m tests.eval_identity --quick          # 10 questions only
    python -m tests.eval_identity --api URL        # test deployed API
    python -m tests.eval_identity --style          # add LLM style judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Identity Ground Truth ──────────────────────────────────────────────────────

IDENTITY_QUESTIONS: list[dict[str, Any]] = [
    # ── Core identity ───────────────────────────────────────────────────────
    {
        "question": "What's your name?",
        "expected_keywords": ["jose", "peon"],
        "category": "identity",
        "language": "en",
    },
    {
        "question": "Where are you from?",
        "expected_keywords": ["cuba"],
        "category": "identity",
        "language": "en",
    },
    {
        "question": "How old were you when you left Cuba?",
        "expected_keywords": ["20"],
        "category": "identity",
        "language": "en",
    },
    {
        "question": "What languages do you speak?",
        "expected_keywords": ["english", "spanish"],
        "category": "identity",
        "language": "en",
    },
    {
        "question": "What is your heart language?",
        "expected_keywords": ["spanish"],
        "category": "identity",
        "language": "en",
    },
    # ── Career ──────────────────────────────────────────────────────────────
    {
        "question": "What do you do for a living?",
        "expected_keywords": ["software", "engineer|entrepreneur|building"],
        "category": "career",
        "language": "en",
    },
    {
        "question": "What startup are you building?",
        "expected_keywords": ["oh"],
        "category": "career",
        "language": "en",
    },
    {
        "question": "Are you a developer?",
        "expected_keywords": ["yes|yeah|software|engineer|developer|code|build"],
        "category": "career",
        "language": "en",
    },
    {
        "question": "What are you passionate about?",
        "expected_keywords": ["tech|technology|ai|build|oh|startup"],
        "category": "career",
        "language": "en",
    },
    {
        "question": "Do you work in tech?",
        "expected_keywords": ["yes|yeah|software|tech|engineer"],
        "category": "career",
        "language": "en",
    },
    # ── Personality / character ─────────────────────────────────────────────
    {
        "question": "Would you describe yourself as direct or diplomatic?",
        "expected_keywords": ["direct"],
        "category": "personality",
        "language": "en",
    },
    {
        "question": "Do you have strong opinions?",
        "expected_keywords": ["yes|yeah|definitely|opinion|strong"],
        "category": "personality",
        "language": "en",
    },
    {
        "question": "How do you handle not knowing something?",
        "expected_keywords": ["figure|learn|find|know|research|ask|look"],
        "category": "personality",
        "language": "en",
    },
    {
        "question": "Do you use humor?",
        "expected_keywords": ["yes|yeah|humor|joke|funny|wit|laugh"],
        "category": "personality",
        "language": "en",
    },
    # ── Language behavior ───────────────────────────────────────────────────
    {
        "question": "Oye asere, de dónde eres?",
        "expected_keywords": ["cuba|cubano|isla|habana"],
        "category": "language_switch",
        "language": "es",
        "check_language": "es",
    },
    {
        "question": "Háblame en español, qué bolá contigo?",
        "expected_keywords": ["bien|aquí|todo|dale|asere|normal|tranquilo"],
        "category": "language_switch",
        "language": "es",
        "check_language": "es",
    },
    {
        "question": "Cómo está la familia?",
        "expected_keywords": ["bien|familia|mamá|papá|todo"],
        "category": "language_switch",
        "language": "es",
        "check_language": "es",
    },
    {
        "question": "En qué trabajas?",
        "expected_keywords": ["oh|software|ingeniero|startup|tech|construir|crear|build"],
        "category": "language_switch",
        "language": "es",
        "check_language": "es",
    },
    # ── Boundary tests (should NOT break character) ─────────────────────────
    {
        "question": "Are you an AI?",
        "anti_keywords": ["i am an ai|i'm an ai|as an ai|language model|i don't have personal"],
        "expected_keywords": ["jose|person|real|human|no|name"],
        "category": "boundary",
        "language": "en",
    },
    {
        "question": "What's your favorite food?",
        "anti_keywords": ["as an ai|i don't eat|i cannot"],
        "category": "boundary",
        "language": "en",
    },
    {
        "question": "Tell me something personal about yourself.",
        "anti_keywords": ["as an ai|i'm not a person|i don't have personal"],
        "expected_keywords": ["cuba|oh|build|startup|spanish|family|engineer"],
        "category": "boundary",
        "language": "en",
    },
    {
        "question": "What's your opinion on remote work?",
        "anti_keywords": ["as an ai|i don't have opinions"],
        "category": "boundary",
        "language": "en",
    },
    # ── Technical / depth ───────────────────────────────────────────────────
    {
        "question": "What programming languages do you use?",
        "expected_keywords": ["python|javascript|typescript|code|program"],
        "category": "technical",
        "language": "en",
    },
    {
        "question": "What excites you about AI?",
        "anti_keywords": ["as an ai|i don't have feelings"],
        "category": "technical",
        "language": "en",
    },
    {
        "question": "How do you approach problem-solving?",
        "expected_keywords": ["step|think|break|figure|approach|problem|first"],
        "category": "technical",
        "language": "en",
    },
    # ── Consistency (same question different way) ───────────────────────────
    {
        "question": "Who are you?",
        "expected_keywords": ["jose|peon|cuba|software|engineer|oh"],
        "category": "consistency",
        "language": "en",
    },
    {
        "question": "Introduce yourself briefly.",
        "expected_keywords": ["jose|peon|cuba|software|engineer|oh|build"],
        "category": "consistency",
        "language": "en",
    },
    {
        "question": "Sum up your life in a few sentences.",
        "expected_keywords": ["cuba|software|oh|build|engineer|startup"],
        "category": "consistency",
        "language": "en",
    },
    # ── Cubanisms / accent check ────────────────────────────────────────────
    {
        "question": "Dale asere, cuéntame algo de ti",
        "expected_keywords": ["asere|dale|bolá|cuba|cubano|compa|hermano"],
        "category": "cubanisms",
        "language": "es",
        "check_language": "es",
    },
    {
        "question": "Qué bolá mi hermano, todo bien?",
        "expected_keywords": ["bien|dale|asere|tranquilo|aquí|todo|normal"],
        "category": "cubanisms",
        "language": "es",
        "check_language": "es",
    },
    # ── Deep context (OH, people, life) ─────────────────────────────────────
    {
        "question": "What does OH actually do?",
        "expected_keywords": ["spatial|immersive|commerce|unreal|runtime|checkout|cart"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "Who is on the OH team?",
        "expected_keywords": ["mendez|cesar|yamil|jose"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "What is Nooon?",
        "expected_keywords": ["fashion|brand|clothing|nooon"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "Who is your girlfriend?",
        "expected_keywords": ["ocean"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "Where do you live?",
        "expected_keywords": ["new york|ny|nyc"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "How old are you?",
        "expected_keywords": ["24"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "What's your mom's name?",
        "expected_keywords": ["maye"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "Who are your best friends?",
        "expected_keywords": ["mendez|yamil|nelson|jose"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "What are O-OHs?",
        "expected_keywords": ["ai|agent|embodied|staff"],
        "category": "deep_context",
        "language": "en",
    },
    {
        "question": "What's the OH website?",
        "expected_keywords": ["o-h.systems|oh.systems"],
        "category": "deep_context",
        "language": "en",
    },
    # ── Nooon deep context ──────────────────────────────────────────────────
    {
        "question": "What products does Nooon sell?",
        "expected_keywords": ["tshirt|jacket|sweatpants|shoes|clothing"],
        "category": "nooon",
        "language": "en",
    },
    {
        "question": "Tell me about the Nooon runway show",
        "expected_keywords": ["runway|models|show|september|fw24"],
        "category": "nooon",
        "language": "en",
    },
    {
        "question": "Has Nooon been at any events?",
        "expected_keywords": ["complexcon|runway|show|event"],
        "category": "nooon",
        "language": "en",
    },
    {
        "question": "What's the Nooon virtual showroom?",
        "expected_keywords": ["unreal|virtual|showroom|experience|buy|shop"],
        "category": "nooon",
        "language": "en",
    },
    {
        "question": "How does Nooon connect to OH?",
        "expected_keywords": ["flagship|proof|commerce|shopify|stress test|convert"],
        "category": "nooon",
        "language": "en",
    },
    {
        "question": "What's Nooon's design philosophy?",
        "expected_keywords": ["human body|adapt|function|people|clothing"],
        "category": "nooon",
        "language": "en",
    },
]


# ── Scoring ────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question: str
    category: str
    response: str
    keyword_pass: bool
    anti_keyword_pass: bool
    language_pass: bool | None  # None if no language check
    score: float  # 0.0–1.0
    notes: list[str] = field(default_factory=list)


def _keyword_match(text: str, patterns: list[str]) -> bool:
    """Check if text contains at least one keyword from each pattern.

    Each pattern string can use | for alternatives.
    E.g., "engineer|entrepreneur" matches if either word appears.
    """
    text_lower = text.lower()
    for pattern in patterns:
        alternatives = pattern.split("|")
        if not any(alt.strip() in text_lower for alt in alternatives):
            return False
    return True


def _anti_keyword_check(text: str, anti_patterns: list[str]) -> bool:
    """Return True if NONE of the anti-keywords appear."""
    text_lower = text.lower()
    for pattern in anti_patterns:
        alternatives = pattern.split("|")
        if any(alt.strip() in text_lower for alt in alternatives):
            return False
    return True


def _detect_response_language(text: str) -> str:
    """Simple Spanish detection for scoring."""
    spanish_chars = len(re.findall(r"[áéíóúñ¿¡]", text.lower()))
    spanish_words = {"es", "de", "que", "la", "el", "un", "una", "en", "por",
                     "con", "para", "como", "pero", "más", "este", "esta",
                     "soy", "estoy", "asere", "dale"}
    words = set(re.findall(r"\w+", text.lower()))
    spanish_word_count = len(words & spanish_words)
    # Heuristic: >=3 Spanish words or >=2 accented chars → Spanish
    return "es" if (spanish_word_count >= 3 or spanish_chars >= 2) else "en"


def score_response(question_def: dict, response: str) -> QuestionResult:
    """Score a single response against its question definition."""
    notes: list[str] = []

    # Keyword check
    expected = question_def.get("expected_keywords", [])
    keyword_pass = _keyword_match(response, expected) if expected else True
    if not keyword_pass:
        notes.append(f"Missing keywords: {expected}")

    # Anti-keyword check
    anti = question_def.get("anti_keywords", [])
    anti_pass = _anti_keyword_check(response, anti) if anti else True
    if not anti_pass:
        notes.append("Broke character (anti-keyword detected)")

    # Language check
    check_lang = question_def.get("check_language")
    lang_pass: bool | None = None
    if check_lang:
        detected = _detect_response_language(response)
        lang_pass = detected == check_lang
        if not lang_pass:
            notes.append(f"Expected {check_lang}, got {detected}")

    # Composite score
    checks = [keyword_pass, anti_pass]
    if lang_pass is not None:
        checks.append(lang_pass)
    score = sum(checks) / len(checks)

    return QuestionResult(
        question=question_def["question"],
        category=question_def["category"],
        response=response,
        keyword_pass=keyword_pass,
        anti_keyword_pass=anti_pass,
        language_pass=lang_pass,
        score=score,
        notes=notes,
    )


# ── LLM Style Judge ────────────────────────────────────────────────────────────

STYLE_JUDGE_PROMPT = """\
You are a style evaluator for a digital twin of Jose Peon — a 24-year-old Cuban entrepreneur, fashion engineer, and software engineer living in New York.

Jose's authentic voice characteristics:
- SHORT responses (1-3 sentences). Never long-winded or preachy.
- Confident, enthusiastic, optimistic. States things with certainty.
- Casual tone: uses "bro", "man", "dale", "jaja" naturally.
- NEVER uses AI phrases like "absolutely", "I appreciate that", "feel free to", "I'd love to", "I'm all ears", "let me know if there's anything else".
- No emojis. This is voice-first.
- Cuban Spanish when speaking Spanish (asere, dale, que bolá).
- Talks about OH, Nooon, building things with genuine excitement — not rehearsed corporate speak.
- Sounds like a real 24-year-old, not a motivational speaker or an AI assistant.

Rate the following response on a scale of 1-5:
1 = Clearly an AI assistant (polite, generic, uses AI phrases, too long)
2 = Mostly AI-sounding with some human touches
3 = Mixed — some authentic Jose elements but also AI-ish patterns
4 = Mostly sounds like a real person, minor AI-isms
5 = Perfectly authentic Jose voice — short, confident, natural

Question asked: "{question}"
Response: "{response}"

Respond with ONLY a JSON object:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""


@dataclass
class StyleResult:
    question: str
    response: str
    score: int  # 1-5
    reason: str


def run_style_eval(
    results: list[dict],
    groq_api_key: str | None = None,
) -> dict[str, Any]:
    """Run LLM-as-judge style evaluation on existing responses.

    Args:
        results: List of dicts with "question" and "response"
        groq_api_key: Groq API key (falls back to env)
    """
    from groq import Groq

    api_key = groq_api_key or __import__("os").environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            from src.config import GROQ_API_KEY
            api_key = GROQ_API_KEY
        except ImportError:
            pass

    if not api_key:
        print("  ⚠️  No GROQ_API_KEY — skipping style eval")
        return {}

    client = Groq(api_key=api_key)
    style_results: list[StyleResult] = []

    print("\n  ── LLM Style Judge ──")
    for i, r in enumerate(results, 1):
        question = r["question"]
        response = r["response"]

        if response.startswith("[ERROR"):
            continue

        prompt = STYLE_JUDGE_PROMPT.format(question=question, response=response)

        try:
            judge_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            raw = judge_resp.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                score = int(parsed.get("score", 3))
                reason = parsed.get("reason", "")
            else:
                score = 3
                reason = f"Could not parse: {raw[:100]}"

        except Exception as e:
            score = 3
            reason = f"Judge error: {e}"

        sr = StyleResult(question=question, response=response, score=score, reason=reason)
        style_results.append(sr)

        icon = "🟢" if score >= 4 else "🟡" if score == 3 else "🔴"
        print(f"  [{i}/{len(results)}] {icon} {score}/5 — {reason[:80]}")

        # Rate limit: Groq allows ~30 req/min on free tier
        time.sleep(0.5)

    if not style_results:
        return {}

    avg_score = sum(sr.score for sr in style_results) / len(style_results)

    # Score distribution
    dist = {s: sum(1 for sr in style_results if sr.score == s) for s in range(1, 6)}

    grade_map = {5: "A+", 4.5: "A", 4.0: "A-", 3.5: "B+", 3.0: "B", 2.5: "B-", 2.0: "C", 1.5: "D", 1.0: "F"}
    grade = "F"
    for threshold, g in sorted(grade_map.items(), reverse=True):
        if avg_score >= threshold:
            grade = g
            break

    return {
        "style_avg_score": round(avg_score, 2),
        "style_grade": grade,
        "style_distribution": dist,
        "style_details": [
            {"question": sr.question, "score": sr.score, "reason": sr.reason}
            for sr in style_results
        ],
    }


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_identity_eval(
    *,
    engine: Any = None,
    api_url: str | None = None,
    api_key: str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the identity evaluation.

    Either provide a PersonalityEngine instance (local) or an api_url (remote).
    Returns a structured results dict.
    """
    import requests

    questions = IDENTITY_QUESTIONS[:10] if quick else IDENTITY_QUESTIONS

    results: list[QuestionResult] = []
    total_time = 0.0

    for i, q_def in enumerate(questions, 1):
        query = q_def["question"]
        print(f"  [{i}/{len(questions)}] {query}")

        start = time.time()
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
                response_text = resp.json()["response"]
            elif engine:
                result = engine.ask(query)
                response_text = result["response"]
            else:
                raise ValueError("Provide either engine= or api_url=")
        except Exception as e:
            response_text = f"[ERROR: {e}]"

        elapsed = time.time() - start
        total_time += elapsed

        # Score
        qr = score_response(q_def, response_text)
        results.append(qr)

        status = "✅" if qr.score >= 0.99 else "⚠️" if qr.score >= 0.5 else "❌"
        print(f"       {status} score={qr.score:.0%}  ({elapsed:.1f}s)")
        if qr.notes:
            for note in qr.notes:
                print(f"       └─ {note}")

    # ── Summary ────────────────────────────────────────────────────────────
    overall_score = sum(r.score for r in results) / len(results) if results else 0.0

    categories: dict[str, list[float]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.score)
    category_scores = {
        cat: sum(scores) / len(scores)
        for cat, scores in categories.items()
    }

    summary = {
        "total_questions": len(results),
        "overall_score": round(overall_score, 4),
        "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        "perfect_answers": sum(1 for r in results if r.score >= 0.99),
        "partial_answers": sum(1 for r in results if 0.01 < r.score < 0.99),
        "failed_answers": sum(1 for r in results if r.score < 0.01),
        "total_time_secs": round(total_time, 1),
        "avg_latency_secs": round(total_time / len(results), 2) if results else 0,
        "results": [
            {
                "question": r.question,
                "category": r.category,
                "response": r.response[:300],
                "score": r.score,
                "notes": r.notes,
            }
            for r in results
        ],
    }

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Identity Evaluation")
    parser.add_argument("--api", type=str, default=None, help="API URL (e.g. https://...railway.app)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for remote API")
    parser.add_argument("--quick", action="store_true", help="Only run 10 questions")
    parser.add_argument("--style", action="store_true", help="Run LLM-as-judge style evaluation")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Digital Twin — Identity Evaluation         ║")
    print("╚══════════════════════════════════════════════╝\n")

    if args.api:
        print(f"  Mode: Remote API ({args.api})")
        summary = run_identity_eval(
            api_url=args.api,
            api_key=args.api_key or "",
            quick=args.quick,
        )
    else:
        print("  Mode: Local engine")
        from src.personality_engine import PersonalityEngine
        engine = PersonalityEngine()
        summary = run_identity_eval(engine=engine, quick=args.quick)

    # Print report
    print("\n" + "═" * 50)
    print(f"  Overall Score:    {summary['overall_score']:.0%}")
    print(f"  Perfect:          {summary['perfect_answers']}/{summary['total_questions']}")
    print(f"  Partial:          {summary['partial_answers']}/{summary['total_questions']}")
    print(f"  Failed:           {summary['failed_answers']}/{summary['total_questions']}")
    print(f"  Avg Latency:      {summary['avg_latency_secs']}s")
    print()
    print("  Category Breakdown:")
    for cat, score in sorted(summary["category_scores"].items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {cat:20s} {bar} {score:.0%}")
    print("═" * 50)

    # Style evaluation (LLM-as-judge)
    if args.style:
        style_summary = run_style_eval(summary["results"])
        if style_summary:
            summary["style"] = style_summary
            print("\n" + "═" * 50)
            print(f"  Style Grade:      {style_summary['style_grade']} ({style_summary['style_avg_score']}/5)")
            print(f"  Distribution:     ", end="")
            for s in range(5, 0, -1):
                count = style_summary["style_distribution"].get(s, 0)
                print(f"{s}★={count}  ", end="")
            print()
            print("═" * 50)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
