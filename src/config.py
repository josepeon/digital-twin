"""
Configuration and settings for the Digital Twin.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "personal"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAG_INDEX_DIR = DATA_DIR / "rag_index"
IDENTITY_DIR = DATA_DIR / "identity"
VOICE_DIR = DATA_DIR / "voice"

# Raw data source directories
RAW_CHATGPT = RAW_DIR / "chatgpt"
RAW_INSTAGRAM = RAW_DIR / "instagram"
RAW_WHATSAPP = RAW_DIR / "whatsapp"
RAW_IMESSAGE = RAW_DIR / "imessage"
RAW_EMAIL = RAW_DIR / "email"

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
TWIN_API_KEY: str = os.getenv("TWIN_API_KEY", "")

# ── Model Config ───────────────────────────────────────────────────────────────
# Groq decommissioned all Llama chat models (and qwen3-32b) on 2026-08-16;
# gpt-oss-120b/20b are the current production replacements.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
# gpt-oss models spend completion tokens on internal reasoning before the
# visible answer, so this must be larger than the old 256.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── RAG Config ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_COLLECTION_NAME: str = "jose_messages"
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "15"))

# ── Voice Config ───────────────────────────────────────────────────────────────
TTS_MODEL: str = os.getenv("TTS_MODEL", "eleven_multilingual_v2")
TTS_MODEL_REALTIME: str = os.getenv("TTS_MODEL_REALTIME", "eleven_flash_v2_5")
TTS_OUTPUT_FORMAT: str = "mp3_44100_128"

# Voice IDs are loaded from files at runtime (created by clone_voice.py)
VOICE_ID_EN_PATH = VOICE_DIR / "voice_id_en.txt"
VOICE_ID_ES_PATH = VOICE_DIR / "voice_id_es.txt"

# ── Data Processing Config ─────────────────────────────────────────────────────
MIN_MESSAGE_LENGTH: int = 5  # Minimum word count for quality filtering
MAX_MESSAGE_LENGTH: int = 2000  # Skip absurdly long pastes
DEDUP_SIMILARITY_THRESHOLD: float = 0.95  # Cosine similarity for near-duplicate detection

# ── Jose's Identity ───────────────────────────────────────────────────────────
JOSE_NAME: str = "Jose Peon"

# Sources ordered by priority
DATA_SOURCES = ["chatgpt", "instagram", "whatsapp", "imessage", "email"]

# Dataset versions
DATASET_VERSIONS = {
    "v0.1": ["chatgpt"],
    "v0.2": ["chatgpt", "instagram"],
    "v0.3": ["chatgpt", "instagram", "whatsapp", "imessage"],
    "v0.4": ["chatgpt", "instagram", "whatsapp", "imessage", "email"],
    "v0.5": ["chatgpt", "instagram", "whatsapp", "imessage", "email"],
    "v0.6": ["chatgpt", "instagram", "whatsapp", "imessage", "email"],
    "v1.0": ["chatgpt", "instagram", "whatsapp", "imessage", "email"],
}
