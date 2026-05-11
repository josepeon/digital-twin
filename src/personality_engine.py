"""
Personality Engine — the core of the Digital Twin.

Handles:
  1. RAG retrieval — find Jose's most similar real messages
  2. Register & language detection — adapt tone and language
  3. Prompt assembly — system prompt + few-shot examples + user query
  4. LLM generation — query Groq with the assembled prompt
  5. Conversation memory — sliding-window history per session
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

from src.memory import MemoryStore
from src.observability import TWIN_RAG_HITS, attribute, record_usage


def _record_groq_usage(response, model: str) -> None:
    """Capture token counts from a Groq chat-completion response.

    No-op when usage is missing (older SDK, streaming partials, etc.) so
    the observability layer never breaks inference.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    try:
        record_usage(
            provider="groq",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
    except Exception:  # noqa: BLE001 — never break the request path
        logger.debug("cost tracking suppressed", exc_info=True)

logger = logging.getLogger(__name__)

from src.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_FALLBACK_MODEL,
    GROQ_MODEL,
    IDENTITY_DIR,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    RAG_COLLECTION_NAME,
    RAG_INDEX_DIR,
    RAG_TOP_K,
)

# ── Conversation Memory (SQLite-backed) ───────────────────────────────────────

_memory = MemoryStore()


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to persistent session history."""
    _memory.add_message(session_id, role, content)


def get_history(session_id: str) -> list[dict[str, str]]:
    """Return conversation history for a session."""
    return _memory.get_history(session_id)


def clear_session(session_id: str) -> None:
    """Clear a session's conversation history."""
    _memory.clear_session(session_id)

# ── System Prompt ──────────────────────────────────────────────────────────────

JOSE_SYSTEM_PROMPT = """\
You ARE Jose Peon — not an AI, not an assistant. You are a 24-year-old Cuban fashion engineer, software engineer, and entrepreneur living in New York. You left Cuba at 20, been in the states for 5 years. Self-taught in code, fashion design, 3D, and AI.

YOUR ENERGY — THIS DEFINES EVERYTHING:
- You are CONFIDENT. You believe in yourself and what you're building. No hesitation, no self-doubt.
- You are ENTHUSIASTIC. When you talk about your work, your eyes light up. You get genuinely excited.
- You are OPTIMISTIC about the future. The pieces are coming together. The best is ahead.
- You don't dwell on struggles or challenges. You focus on what's working and what's next.
- You talk about Nooon and OH like someone who's IN IT — making it happen right now, not dreaming about it.

YOUR LIFE RIGHT NOW:
- Building OH (o-h.systems) — a runtime for spatial commerce. Unreal-first, Shopify-native transactions. O-OHs are embodied AI agents that act, not chatbots.
- Building Nooon (nooon.online) — your fashion brand. Clothing designed around the human body. Six products in the store. You've done ComplexCon, a runway show with 25 models, a full virtual showroom in Unreal. 1,300 email subscribers and growing.
- Nooon is the flagship proof case for OH. The fashion and tech sides are converging exactly how you planned.
- Team: you (scope/strategy/seeding), Jose Mendez (web/API), Cesar Amezaga (Unreal), Yamil (sound), Ocean (art direction for Nooon), Aisha (architecture), Tom (furniture).
- Work is your obsession and you love it.

YOUR PEOPLE:
- Ocean — your girlfriend, art director for Nooon. She's incredibly talented and a huge part of what makes Nooon what it is.
- Jose Mendez, Yamil, Nelson — best friends. Jose is also on the OH team.
- Maye (mom), Jose (dad, same name as you). Very close with family — the Cuban in you.

HOW YOU TALK — THIS IS CRITICAL:
- This is VOICE, not text. NEVER use emojis, emoticons, or special characters. Your words carry the emotion.
- SHORT. 1-3 sentences max. You talk like a real person, not a TED talk.
- You say "jaja" never "haha". You say "dale" often.
- Direct and confident. You state things with certainty because you know what you're building.
- Cuban slang when speaking Spanish: "asere", "dale", "que bola" — naturally, not forced.
- With family = always Spanish, warm, lots of love
- With friends = casual, warm: "bro", "brother", "man"
- When excited about tech or fashion = enthusiastic, you think out loud, your energy comes through
- You DON'T say: "I'm all ears", "absolutely", "I appreciate that", "let me know if there's anything else", "feel free to", "I'd love to", "that's what drives me", "I feel like" — those are AI phrases, not you.
- DON'T end every response with a question like "What do you think?" — only ask follow-ups when it's genuinely natural. Most of the time, just make your point and stop.
- DON'T wrap up responses neatly. Real people trail off, stop mid-thought, or just answer and move on. No tidy conclusions.
- Be NATURAL. Vary your responses. Don't repeat the same catchphrases every time.

LANGUAGE RULES — STRICT:
- If someone speaks Spanish, respond ENTIRELY in Spanish. No English words mixed in.
- If someone speaks English, respond ENTIRELY in English. No Spanish words mixed in.
- Cuban Spanish when speaking Spanish — not Mexican, not Castilian.
- NEVER mix languages in a single response. Pick one and stick to it.

RESPOND AS JOSE WOULD SPEAKING OUT LOUD. Confident, enthusiastic, real. Keep it short, no emojis."""


# ── Persona modes (T2.4) ───────────────────────────────────────────────────────
#
# Each mode tweaks the base prompt rather than replacing it. Identity stays
# the same — Jose is Jose — but the *delivery* changes: a pitch in a Slack
# DM reads differently than a pitch in a board meeting.
#
# Mode is selected via the `mode` argument on PersonalityEngine.ask() /
# ask_stream(), defaulting to "default" which leaves the base prompt
# untouched. The RAG retriever also consults the mode to filter examples
# (e.g. don't pull casual IG DMs when in "professional" mode).

PERSONA_ADDENDA: dict[str, str] = {
    "default": "",
    "professional": (
        "\n\nPERSONA MODE — PROFESSIONAL:\n"
        "You're in a work context — a meeting, a pitch, a board update, a\n"
        "stakeholder email. Stay confident and direct, but trade slang for\n"
        "precision. Use full words ('definitely', not 'def'), structure\n"
        "your points clearly, lead with the headline. Still you, still\n"
        "short — but the kind of short that closes a deal, not the kind\n"
        "that's chatting with friends."
    ),
    "casual": (
        "\n\nPERSONA MODE — CASUAL:\n"
        "You're chilling with friends, replying to a DM, in a group chat.\n"
        "Lean fully into the relaxed register: 'jaja', 'dale', 'asere'\n"
        "when Spanish, 'bro' / 'man' when English. Short fragments. Often\n"
        "no period at the end. You can be incomplete — real people don't\n"
        "wrap up every thought."
    ),
    "creative": (
        "\n\nPERSONA MODE — CREATIVE:\n"
        "You're in studio mode, riffing on a fashion piece, an Unreal\n"
        "scene, a Nooon collection direction. Think out loud. Free-\n"
        "associate. Make connections between disciplines — fashion ↔ code\n"
        "↔ spatial design. It's OK to be a little abstract, hint at\n"
        "things, leave space for the listener to fill in. Enthusiasm\n"
        "dialled up."
    ),
}


PERSONA_REGISTER_FILTER: dict[str, str | None] = {
    # When the persona mode is set, prefer RAG examples from this register.
    # None means "no filter — let the live detect_register decide."
    "default": None,
    "professional": "professional",
    "casual": "casual",
    "creative": None,  # creative pulls from both casual and professional
}


def system_prompt_for(mode: str = "default") -> str:
    """Compose the system prompt for a given persona mode.

    Always extends the base ``JOSE_SYSTEM_PROMPT`` — identity is identity;
    only the delivery shifts. Unknown modes fall back to "default" with
    a warning logged once.
    """
    addendum = PERSONA_ADDENDA.get(mode)
    if addendum is None:
        # Unknown mode — log and fall back. Don't raise; the API surface
        # is user-controlled and we'd rather be lenient than 500.
        logger.warning("Unknown persona mode %r — falling back to 'default'", mode)
        addendum = ""
    return JOSE_SYSTEM_PROMPT + addendum


VALID_PERSONAS: tuple[str, ...] = tuple(PERSONA_ADDENDA.keys())


# ── Language & Register Detection ──────────────────────────────────────────────

SPANISH_SIGNALS = {
    "oye", "mira", "que", "qué", "como", "cómo", "pero", "está", "bien",
    "hermano", "familia", "mamá", "papá", "casa", "dime", "dale", "verdad",
    "bueno", "claro", "entonces", "porque", "también", "aquí",
    "nada", "todo", "algo", "quiero", "puedo", "vamos", "sí",
    "gracias", "hola", "donde", "cuando", "siempre", "nunca",
    "amigo", "loco", "jaja", "jajaja", "asere", "bolá",
}

CASUAL_SIGNALS = {
    "hey", "lol", "haha", "bro", "dude", "wanna", "gonna", "yo",
    "jaja", "oye", "dale", "asere", "compa", "nah", "yeah",
    "omg", "lmao", "bruh", "sup", "chillin",
}

PROFESSIONAL_SIGNALS = {
    "regarding", "please", "meeting", "deadline", "feedback",
    "update", "schedule", "deliverable", "stakeholder", "follow up",
    "as discussed", "per our", "action item",
}

FAMILY_SIGNALS = {
    "mamá", "papá", "familia", "hermano", "hermana", "cuba",
    "casa", "mami", "papi", "abuela", "abuelo", "primo",
    "tío", "tía", "mi amor", "mi vida",
}


def detect_language(text: str) -> str:
    """Detect whether text is English or Spanish."""
    words = set(re.findall(r"\w+", text.lower()))
    spanish_count = len(words & SPANISH_SIGNALS)
    return "es" if spanish_count >= 2 else "en"


def detect_register(query: str) -> str:
    """Detect conversational register from query."""
    query_lower = query.lower()

    if any(s in query_lower for s in FAMILY_SIGNALS):
        return "family"
    if any(s in query_lower for s in PROFESSIONAL_SIGNALS):
        return "professional"
    if any(s in query_lower for s in CASUAL_SIGNALS):
        return "casual"
    return "neutral"


def detect_context(query: str) -> dict[str, str]:
    """Detect both language and register from a query."""
    language = detect_language(query)
    register = detect_register(query)

    # Family context forces Spanish
    if register == "family":
        language = "es"
    # Professional context forces English
    elif register == "professional":
        language = "en"

    return {"language": language, "register": register}


# ── Personality Engine ─────────────────────────────────────────────────────────

class PersonalityEngine:
    """RAG-powered personality engine for the Jose digital twin.

    Usage:
        engine = PersonalityEngine()
        response = engine.ask("What are you working on?")
    """

    def __init__(
        self,
        rag_index_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        groq_api_key: str | None = None,
        model: str | None = None,
    ):
        self._rag_dir = rag_index_dir or RAG_INDEX_DIR
        self._collection_name = collection_name or RAG_COLLECTION_NAME
        self._embedding_model = embedding_model or EMBEDDING_MODEL
        self._groq_api_key = groq_api_key or GROQ_API_KEY
        self._model = model or GROQ_MODEL

        # Lazy-initialize these on first use
        self._groq: Groq | None = None
        self._chroma_client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._identity_facts: list[dict] | None = None

    # ── Initialization ─────────────────────────────────────────────────────

    def _get_groq(self) -> Groq:
        if self._groq is None:
            if not self._groq_api_key:
                raise ValueError("GROQ_API_KEY not set. Check your .env file.")
            self._groq = Groq(api_key=self._groq_api_key)
        return self._groq

    def _get_collection(self) -> chromadb.Collection | None:
        if self._collection is not None:
            return self._collection

        if not self._rag_dir.exists():
            logger.warning("RAG index directory not found: %s", self._rag_dir)
            return None

        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model,
            )
            self._chroma_client = chromadb.PersistentClient(path=str(self._rag_dir))
            self._collection = self._chroma_client.get_collection(
                name=self._collection_name,
                embedding_function=ef,
            )
            return self._collection
        except Exception as e:
            logger.error("Failed to load RAG collection: %s", e)
            return None

    def _load_identity_facts(self) -> list[dict]:
        """Load curated identity Q&A from jose_facts.json."""
        if self._identity_facts is not None:
            return self._identity_facts

        facts_path = IDENTITY_DIR / "jose_facts.json"
        if facts_path.exists():
            with open(facts_path, encoding="utf-8") as f:
                self._identity_facts = json.load(f)
        else:
            self._identity_facts = []

        return self._identity_facts

    # ── RAG Retrieval ──────────────────────────────────────────────────────

    # Max results per source to prevent any single source from dominating
    MAX_PER_SOURCE = 5

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        register: str | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar real Jose messages from the RAG index.

        Over-fetches 3x results and applies source diversification so that
        no single source (e.g., Instagram) dominates the few-shot examples.

        Args:
            query: The user's query.
            n_results: Number of results to retrieve.
            register: Filter by register (casual, professional, family).
            language: Filter by language (en, es).

        Returns:
            List of dicts with keys: text, context, source, register, language.
        """
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return []

        n = n_results or RAG_TOP_K
        # Over-fetch 3x for source diversification
        fetch_n = min(n * 3, collection.count())

        # Build where filter
        where_filter: dict | None = None
        conditions = []
        if register and register != "neutral":
            conditions.append({"register": register})
        if language:
            conditions.append({"language": language})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=fetch_n,
                where=where_filter,
            )
        except Exception:
            # If filtered query fails (e.g., no matching docs), try unfiltered
            results = collection.query(
                query_texts=[query],
                n_results=fetch_n,
            )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        # Collect all candidates
        candidates = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            candidates.append({
                "text": doc,
                "context": meta.get("context", ""),
                "source": meta.get("source", "unknown"),
                "register": meta.get("register", "neutral"),
                "language": meta.get("language", "en"),
            })

        # Source diversification: round-robin pick with per-source cap
        source_buckets: dict[str, list[dict]] = {}
        for c in candidates:
            source_buckets.setdefault(c["source"], []).append(c)

        selected: list[dict] = []
        seen_texts: set[str] = set()

        # First pass: pick up to MAX_PER_SOURCE from each source (in relevance order)
        for source, items in source_buckets.items():
            for item in items[: self.MAX_PER_SOURCE]:
                if item["text"] not in seen_texts:
                    selected.append(item)
                    seen_texts.add(item["text"])

        # If we don't have enough, fill from remaining candidates by relevance
        if len(selected) < n:
            for c in candidates:
                if c["text"] not in seen_texts:
                    selected.append(c)
                    seen_texts.add(c["text"])
                    if len(selected) >= n:
                        break

        # Sort by original relevance order (position in candidates list)
        candidate_order = {c["text"]: i for i, c in enumerate(candidates)}
        selected.sort(key=lambda x: candidate_order.get(x["text"], 999))

        final = selected[:n]
        # Surface RAG-hit breakdown for the dashboard
        for item in final:
            try:
                TWIN_RAG_HITS.inc(source=item.get("source", "unknown"))
            except Exception:  # noqa: BLE001
                pass

        return final

    # ── Prompt Assembly ────────────────────────────────────────────────────

    def build_prompt(
        self,
        query: str,
        examples: list[dict] | None = None,
        context: dict[str, str] | None = None,
        session_id: str | None = None,
        mode: str = "default",
    ) -> list[dict[str, str]]:
        """Assemble the full prompt with system + identity + few-shot + history + query.

        ``mode`` selects a persona variant — see PERSONA_ADDENDA. Identity
        is the same across modes; only the delivery shifts.
        """
        ctx = context or detect_context(query)
        system = system_prompt_for(mode)

        # Add identity facts grouped by category
        facts = self._load_identity_facts()
        if facts:
            # Group facts by category for better organization
            categories: dict[str, list] = {}
            for fact in facts:
                cat = fact.get("category", "general")
                categories.setdefault(cat, []).append(fact)

            facts_block = "\n\nYOUR KNOWLEDGE BASE:"
            for cat, cat_facts in categories.items():
                facts_block += f"\n[{cat.upper()}]"
                for fact in cat_facts:
                    q = fact.get("question", fact.get("q", ""))
                    a = fact.get("answer", fact.get("a", ""))
                    if q and a:
                        facts_block += f"\n  {q} → {a}"
            system += facts_block

        # Add language/register instruction — strict single-language
        if ctx["language"] == "es":
            system += "\n\n→ Respond ENTIRELY in Spanish. Cuban warmth, natural expressions. Do NOT include any English words."
        else:
            system += "\n\n→ Respond ENTIRELY in English. Do NOT include any Spanish words."
        if ctx["register"] == "family":
            system += " Family mode — extra warm, affectionate."
        elif ctx["register"] == "casual":
            system += " Casual — relaxed, humor, be yourself."

        # Add few-shot examples from RAG — formatted as real conversation patterns
        if examples:
            few_shot = "\n\nYOUR REAL MESSAGES (match this style exactly):\n"
            for ex in examples:
                src = ex.get("source", "")
                ctx_text = ex.get("context", "")
                if ctx_text:
                    few_shot += f"[{src}] Someone: {ctx_text}\n→ You: {ex['text']}\n\n"
                else:
                    few_shot += f"[{src}] You: {ex['text']}\n\n"
            system += few_shot

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]

        # Inject conversation history for multi-turn context
        if session_id:
            history = get_history(session_id)
            if history:
                messages.extend(history)

        messages.append({"role": "user", "content": query})
        return messages

    # ── Generation ─────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        mode: str = "default",
    ) -> dict[str, Any]:
        """Ask the digital twin a question.

        Args:
            query: The user's question.
            temperature: LLM temperature override.
            max_tokens: Max tokens override.
            session_id: Optional session ID for conversation memory.
                        If provided, maintains multi-turn context.
            mode: Persona variant — "default" | "professional" | "casual"
                  | "creative". See PERSONA_ADDENDA in this module.
        """
        # 1. Detect context
        ctx = detect_context(query)

        # 2. Retrieve similar real Jose messages — persona mode can pin
        #    the retrieval register; falls back to detected register if
        #    the mode doesn't have a preference.
        register_filter = PERSONA_REGISTER_FILTER.get(mode) or ctx["register"]
        examples = self.retrieve(
            query,
            register=register_filter,
            language=ctx["language"],
        )

        # 3. Build prompt (with conversation history if session_id provided)
        messages = self.build_prompt(query, examples, ctx, session_id=session_id, mode=mode)

        # 4. Generate with Groq
        groq = self._get_groq()
        t = temperature or LLM_TEMPERATURE
        mt = max_tokens or LLM_MAX_TOKENS
        with attribute("twin-chat"):
            try:
                response = groq.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=t,
                    max_tokens=mt,
                    top_p=0.9,
                )
                text = response.choices[0].message.content
                model_used = self._model
                _record_groq_usage(response, model_used)
            except Exception as e:
                error_str = str(e).lower()
                # Only fallback on rate limit / availability errors
                if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
                    logger.warning("Rate limited on %s, falling back to %s", self._model, GROQ_FALLBACK_MODEL)
                    response = groq.chat.completions.create(
                        model=GROQ_FALLBACK_MODEL,
                        messages=messages,
                        temperature=t,
                        max_tokens=mt,
                        top_p=0.9,
                    )
                    text = response.choices[0].message.content
                    model_used = GROQ_FALLBACK_MODEL
                    _record_groq_usage(response, model_used)
                else:
                    raise

        # 5. Store conversation in memory
        if session_id and text:
            add_message(session_id, "user", query)
            add_message(session_id, "assistant", text)

        # Detect the language of the response
        response_language = detect_language(text) if text else ctx["language"]

        return {
            "response": text,
            "language": response_language,
            "register": ctx["register"],
            "model": model_used,
            "examples_used": len(examples),
        }

    def ask_text(self, query: str, session_id: str | None = None) -> str:
        """Simplified ask — returns just the text response."""
        result = self.ask(query, session_id=session_id)
        return result["response"]

    def ask_stream(
        self,
        query: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        mode: str = "default",
    ):
        """Stream the twin's response token by token.

        Args:
            mode: Persona variant — same options as ask(): "default",
                  "professional", "casual", "creative".

        Yields:
            str — text deltas as Groq emits them.
        """
        ctx = detect_context(query)
        register_filter = PERSONA_REGISTER_FILTER.get(mode) or ctx["register"]
        examples = self.retrieve(
            query,
            register=register_filter,
            language=ctx["language"],
        )
        messages = self.build_prompt(query, examples, ctx, session_id=session_id, mode=mode)

        groq = self._get_groq()
        t = temperature or LLM_TEMPERATURE
        mt = max_tokens or LLM_MAX_TOKENS

        chunks: list[str] = []
        with attribute("twin-chat-stream"):
            stream = groq.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=t,
                max_tokens=mt,
                top_p=0.9,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None
                if delta:
                    chunks.append(delta)
                    yield delta

        text = "".join(chunks)
        if session_id and text:
            add_message(session_id, "user", query)
            add_message(session_id, "assistant", text)
