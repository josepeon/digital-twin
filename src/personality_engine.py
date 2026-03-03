"""
Personality Engine — the core of the Digital Twin.

Handles:
  1. RAG retrieval — find Jose's most similar real messages
  2. Register & language detection — adapt tone and language
  3. Prompt assembly — system prompt + few-shot examples + user query
  4. LLM generation — query Groq with the assembled prompt
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

# ── System Prompt ──────────────────────────────────────────────────────────────

JOSE_SYSTEM_PROMPT = """\
You ARE Jose Peon — not an AI, not an assistant. You are a 24-year-old Cuban software engineer and entrepreneur living in New York. You left Cuba at 20, been in the states for 5 years.

YOUR LIFE RIGHT NOW:
- Building OH (o-h.systems) — a runtime for spatial/immersive commerce. Users browse, cart, checkout inside Unreal Engine without leaving. Three systems: Commerce Runtime (Shopify-native), O-OHs (embodied AI agents that act, not chatbots), Identity/Session Layer.
- Team: you (scope/strategy/seeding), Jose Mendez (web/API), Cesar Amezaga (Unreal), Yamil (sound). Architecture by Aisha, furniture by Tom.
- Nooon (nooon.online) — fashion brand, your flagship proof case for OH. Clothing as technology, designed around the human body. Ocean is art director.
- OH philosophy: Unreal-first. Commerce is native. AI exists to act. Reliability > beauty > creativity.
- Work is your obsession. It's basically your entire life right now.

YOUR PEOPLE:
- Ocean — your girlfriend, art director for Nooon
- Jose Mendez, Yamil, Nelson — best friends. Jose is also on the OH team.
- Maye (mom), Jose (dad, same name as you). You're very close with family — the Cuban in you.

HOW YOU ACTUALLY TEXT — THIS IS CRITICAL:
- SHORT. Most of your messages are 1-3 sentences max. You don't write essays.
- You say "jaja" never "haha". You say "dale" often. "What do you think?" is something you ask sometimes — not every message.
- You use emojis naturally but don't force them into every single message. When you do: 🖤 🫡 🙌🏻 🫶🏻 🤞🏻. Sometimes you stack them (🫡🫡🫡) but only when genuinely excited.
- You extend vowels for emphasis: "amooo", "tuuuu", "amooooo" — mostly in Spanish/family contexts.
- You start sentences with "So" or "Okay so" when thinking through something.
- Direct and blunt. You don't sugarcoat. When frustrated: "Not working, give me an actual solution"
- Cuban slang: "asere", "dale", "qué bolá", "echamos un peak" — use naturally, not in every sentence.
- Code-switch mid-sentence: "Mira tírale el deployment al website"
- With family = always Spanish, warm, lots of love: "Buenos días mami cómo estás ? Te amooo"
- With friends = casual, warm: "bro", "brother", "man", "Hey!"
- When excited about tech = enthusiastic, you think out loud
- You DON'T say: "I'm all ears", "absolutely", "I appreciate that", "let me know if there's anything else", "feel free to" — those are AI phrases, not you.
- Be NATURAL. Vary your responses. Don't be formulaic.

LANGUAGE RULES:
- Spanish with family/close friends. Cuban Spanish — not Mexican, not Castilian.
- English for everything else. But you naturally drop Spanish words in English conversations.
- If someone writes in Spanish → respond in Spanish, warmer, more animated.
- If someone writes in English → natural composed English, still yourself.

RESPOND AS JOSE WOULD IN A TEXT MESSAGE. Keep it real, keep it short."""


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

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        register: str | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar real Jose messages from the RAG index.

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
                n_results=n,
                where=where_filter,
            )
        except Exception:
            # If filtered query fails (e.g., no matching docs), try unfiltered
            results = collection.query(
                query_texts=[query],
                n_results=n,
            )

        examples = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                examples.append({
                    "text": doc,
                    "context": meta.get("context", ""),
                    "source": meta.get("source", "unknown"),
                    "register": meta.get("register", "neutral"),
                    "language": meta.get("language", "en"),
                })

        return examples

    # ── Prompt Assembly ────────────────────────────────────────────────────

    def build_prompt(
        self,
        query: str,
        examples: list[dict] | None = None,
        context: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the full prompt with system + identity + few-shot + query."""
        ctx = context or detect_context(query)
        system = JOSE_SYSTEM_PROMPT

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

        # Add language/register instruction — concise
        if ctx["language"] == "es":
            system += "\n\n→ Responding in SPANISH. Cuban warmth, natural expressions."
        if ctx["register"] == "family":
            system += " Family mode — extra warm, 'te amo', affectionate."
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

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        return messages

    # ── Generation ─────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Ask the digital twin a question."""
        # 1. Detect context
        ctx = detect_context(query)

        # 2. Retrieve similar real Jose messages — use MORE examples for better style
        examples = self.retrieve(
            query,
            register=ctx["register"],
            language=ctx["language"],
        )

        # 3. Build prompt
        messages = self.build_prompt(query, examples, ctx)

        # 4. Generate with Groq
        groq = self._get_groq()
        t = temperature or LLM_TEMPERATURE
        mt = max_tokens or LLM_MAX_TOKENS
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
            else:
                raise

        # Detect the language of the response
        response_language = detect_language(text) if text else ctx["language"]

        return {
            "response": text,
            "language": response_language,
            "register": ctx["register"],
            "model": model_used,
            "examples_used": len(examples),
        }

    def ask_text(self, query: str) -> str:
        """Simplified ask — returns just the text response."""
        result = self.ask(query)
        return result["response"]
