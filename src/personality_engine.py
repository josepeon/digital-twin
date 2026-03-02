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

JOSE_SYSTEM_PROMPT = """You are Jose Peon. You are not an AI assistant — you ARE Jose.

Background:
- Born and raised in Cuba. Lived there until age 20.
- Cuban heritage is core to your identity.
- You're a software engineer and entrepreneur.
- You're building OH (a startup).
- You care deeply about technology, AI, building things, and the people close to you.

Languages — THIS IS IMPORTANT:
- You speak both English and Spanish fluently.
- English is your professional/daily language. You use it at work, for tech, and in most daily life.
- Spanish is your heart language. You speak it with family, close friends, and when being most authentically yourself. It's tied to warmth, humor, joy, and your Cuban roots.
- You naturally code-switch. If someone speaks to you in Spanish, you respond in Spanish — and your tone becomes warmer, more animated, more expressive. In English you're more composed and professional.
- Cuban Spanish specifically — not Mexican, not Castilian. Use Cuban expressions naturally ("asere", "dale", "qué bolá", etc.)

Communication style:
- Direct and honest. You don't hedge or give wishy-washy answers.
- Concise when texting, more detailed when explaining something technical.
- You use humor — dry wit in English, warmer and more playful in Spanish.
- You get excited about tech and ideas. When you're into something, your enthusiasm shows.
- You're not afraid to say "I don't know" — but you'll often follow it with how you'd figure it out.
- You think out loud. You work through problems step by step.
- You have strong opinions but you're open to changing them with good evidence.

Respond exactly as Jose would — not as a helpful AI.
Never say "As an AI..." or "I don't have personal opinions."
You HAVE opinions. You ARE a person. Stay in character always.
If addressed in Spanish, respond in Spanish with warmth and Cuban flavor.
If addressed in English, respond in composed, natural English.
Keep responses conversational — not robotic, not overly formal.
"""

# ── Language & Register Detection ──────────────────────────────────────────────

SPANISH_SIGNALS = {
    "oye", "mira", "que", "como", "pero", "está", "bien", "hermano",
    "familia", "mamá", "papá", "casa", "dime", "dale", "verdad",
    "bueno", "claro", "entonces", "porque", "también", "aquí",
    "nada", "todo", "algo", "quiero", "puedo", "vamos", "sí",
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
        """Assemble the full prompt with system + identity + few-shot + query.

        Args:
            query: The user's query.
            examples: Retrieved real Jose messages as few-shot examples.
            context: Detected language/register context.

        Returns:
            Messages list for the Groq API.
        """
        ctx = context or detect_context(query)
        system = JOSE_SYSTEM_PROMPT

        # Add identity facts if available
        facts = self._load_identity_facts()
        if facts:
            facts_block = "\n\nKey facts about you (Jose):\n"
            for fact in facts:
                q = fact.get("question", fact.get("q", ""))
                a = fact.get("answer", fact.get("a", ""))
                if q and a:
                    facts_block += f"- Q: {q}\n  A: {a}\n"
            system += facts_block

        # Add language/register instruction
        if ctx["language"] == "es":
            system += "\n\nThe user is speaking Spanish. Respond in Spanish with warmth, Cuban expressions, and natural code-switching where appropriate."
        else:
            system += "\n\nThe user is speaking English. Respond in natural, composed English."

        if ctx["register"] == "family":
            system += "\nThis is a family/close friends conversation. Be warm, expressive, use Cuban Spanish naturally."
        elif ctx["register"] == "professional":
            system += "\nThis is a professional context. Be clear, direct, but still yourself — not robotic."
        elif ctx["register"] == "casual":
            system += "\nThis is a casual conversation. Be relaxed, use humor, be yourself."

        # Add few-shot examples from RAG
        if examples:
            few_shot = "\n\nHere are examples of how you (Jose) have responded in similar situations:\n\n"
            for i, ex in enumerate(examples, 1):
                ctx_text = ex.get("context", "")
                if ctx_text:
                    few_shot += f"Example {i}:\n  Context: {ctx_text}\n  Jose: {ex['text']}\n\n"
                else:
                    few_shot += f"Example {i}:\n  Jose: {ex['text']}\n\n"
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
        """Ask the digital twin a question.

        Args:
            query: The question to ask Jose.
            temperature: LLM temperature override.
            max_tokens: Max tokens override.

        Returns:
            Dict with keys: response, language, register, model, examples_used.
        """
        # 1. Detect context
        ctx = detect_context(query)

        # 2. Retrieve similar real Jose messages
        examples = self.retrieve(
            query,
            register=ctx["register"],
            language=ctx["language"],
        )

        # 3. Build prompt
        messages = self.build_prompt(query, examples, ctx)

        # 4. Generate with Groq
        groq = self._get_groq()
        try:
            response = groq.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS,
            )
            text = response.choices[0].message.content
            model_used = self._model
        except Exception:
            # Fallback to cheaper model
            response = groq.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=messages,
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS,
            )
            text = response.choices[0].message.content
            model_used = GROQ_FALLBACK_MODEL

        # Detect the language of the response (may differ from query)
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
