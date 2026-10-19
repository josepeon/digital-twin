# Digital Twin — Jose Peon

AI digital twin of Jose Peon — RAG personality engine + Groq gpt-oss-120B + ElevenLabs dual-voice clone (EN/ES). WebSocket chat, voice I/O, PWA-installable web UI.

## Architecture

```
User → WebSocket / REST → FastAPI
  → Whisper STT (voice input)
  → RAG retrieval (ChromaDB, source-diversified)
  → System prompt + 32 identity facts + few-shot examples + conversation history
  → Groq Llama 3.3 70B
  → ElevenLabs TTS (cloned voice, EN or ES)
  → Audio + text response
```

## Features

- **Personality engine** — RAG retrieves real Jose messages as few-shot examples, 32 curated identity facts
- **Dual voice clone** — English (composed, confident) + Cuban Spanish (warm, expressive)
- **WebSocket chat** — Real-time bidirectional with REST fallback
- **Voice input** — Groq Whisper STT via `/transcribe` endpoint
- **Persistent memory** — SQLite conversation history with 24h TTL
- **RAG rebalancing** — Source diversification (max 5 per source) prevents Instagram from dominating
- **PWA** — Installable on mobile, service worker for offline shell
- **Style eval** — LLM-as-judge scoring (current: A-, 4.04/5)
- **Identity eval** — 46 automated questions (current: 100%, 46/46)

## Setup

```bash
# Create environment
conda create -n digital-twin python=3.11 -y
conda activate digital-twin

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Groq, ElevenLabs, and API keys

# Run locally
uvicorn src.twin_api:app --host 127.0.0.1 --port 8000
# Open http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/ask-jose` | POST | Text response |
| `/ask-jose/voice` | POST | Audio (MP3) response |
| `/ask-jose/stream` | POST | Streaming audio response |
| `/transcribe` | POST | Speech-to-text (Whisper) |
| `/ws` | WebSocket | Real-time chat (text + audio) |
| `/health` | GET | Health / status check |

All endpoints (except `/health` and `/`) require `Authorization: Bearer <TWIN_API_KEY>`.

## Deploy to Railway

```bash
# Option 1: Git push (auto-deploys from main)
git push origin main

# Option 2: Manual deploy with local data
./deploy.sh              # builds staging dir + railway up
./deploy.sh --dry-run    # builds staging dir only
```

## Evaluation

```bash
# Identity accuracy (keyword matching, 46 questions)
python -m tests.eval_identity

# Identity + style (adds LLM-as-judge scoring)
python -m tests.eval_identity --style

# Quick check (10 questions)
python -m tests.eval_identity --quick

# Test deployed API
python -m tests.eval_identity --api https://digital-twin-production-c355.up.railway.app --api-key YOUR_KEY
```

## Project Structure

```
digital-twin/
├── src/                          # Runtime application code
│   ├── twin_api.py               # FastAPI — REST, WebSocket, STT endpoints
│   ├── personality_engine.py     # RAG retrieval, prompt assembly, LLM generation
│   ├── voice.py                  # ElevenLabs TTS (dual-language)
│   ├── memory.py                 # SQLite conversation memory
│   ├── config.py                 # Environment config and paths
│   ├── models.py                 # Data models (used by scripts/parsers)
│   ├── pii.py                    # PII scrubber (used by scripts/parsers)
│   └── static/                   # Web UI
│       ├── index.html            # Chat interface (WebSocket + voice)
│       ├── manifest.json         # PWA manifest
│       ├── sw.js                 # Service worker
│       └── icon-*.png            # PWA icons
├── scripts/                      # Data processing & utilities (not deployed)
│   ├── parse_*.py                # Source-specific message parsers
│   ├── build_dataset.py          # Merge parsed data into versioned datasets
│   ├── build_rag_index.py        # Build ChromaDB index from dataset
│   ├── clone_voice.py            # ElevenLabs voice cloning setup
│   ├── export_training_data.py   # Export chat pairs for LoRA fine-tuning
│   ├── finetune_lora.py          # LoRA fine-tuning pipeline (unsloth)
│   └── chat_with_jose.py        # CLI chat for local testing
├── tests/                        # Evaluation & test suite
│   ├── eval_identity.py          # 46-question identity eval + LLM style judge
│   ├── eval_style.py             # Standalone style evaluation
│   └── test_*.py                 # Unit tests
├── data/personal/                # Personal data (mostly gitignored)
│   ├── raw/                      # ⛔ Raw message exports
│   ├── processed/                # ⛔ Parsed JSONL files
│   ├── rag_index/                # ✅ ChromaDB embeddings (tracked)
│   ├── identity/                 # ✅ jose_facts.json (tracked)
│   └── voice/                    # ✅ Voice clone IDs (tracked)
├── planning/                     # Design documents
│   └── DIGITAL_TWIN_PLAN.md      # Full project plan (Phases 1-8)
├── Dockerfile                    # Multi-stage Docker build
├── deploy.sh                     # Railway deploy script
├── railway.toml                  # Railway config
└── requirements.txt              # Python dependencies
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq Llama 3.3 70B (fallback: Qwen3 32B) |
| STT | Groq Whisper large-v3 |
| TTS | ElevenLabs Multilingual v2 |
| RAG | ChromaDB + all-MiniLM-L6-v2 |
| API | FastAPI + uvicorn |
| Memory | SQLite (WAL mode) |
| Deploy | Railway (Docker) |
| Frontend | Vanilla HTML/JS, WebSocket, PWA |

See [planning/DIGITAL_TWIN_PLAN.md](planning/DIGITAL_TWIN_PLAN.md) for the full design document and roadmap.
