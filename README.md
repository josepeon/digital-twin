# Digital Twin — Jose Peon

AI digital twin of Jose Peon — RAG personality engine + Groq 70B + ElevenLabs dual-voice clone (EN/ES). Text and voice API that thinks, writes, and speaks as Jose.

## Architecture

```
User query → RAG (ChromaDB) → System prompt + few-shot → Groq Llama 3.3 70B → ElevenLabs TTS → Audio
```

- **Personality**: RAG retrieves real Jose messages as few-shot examples for Groq 70B
- **Voice**: Dual ElevenLabs clones — composed English Jose + warm Cuban Spanish Jose
- **API**: FastAPI service with text and voice endpoints
- **Deployment**: Railway (same infra as OH companion)

## Setup

```bash
# Create conda environment
conda create -n digital-twin python=3.11 -y
conda activate digital-twin

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask-jose` | POST | Text response |
| `/ask-jose/voice` | POST | Audio (MP3) response |
| `/ask-jose/stream` | POST | Streaming audio response |
| `/health` | GET | Health check |

## Project Structure

```
digital-twin/
├── data/personal/         # ⛔ GITIGNORED — personal data, voice samples, RAG index
├── models/                # ⛔ GITIGNORED — local fallback model weights
├── scripts/               # Data parsers, RAG builder, voice cloning
├── src/                   # Core application code
│   ├── personality_engine.py
│   ├── voice.py
│   └── twin_api.py
├── tests/                 # Test suite
└── planning/              # Design documents
```

See [planning/DIGITAL_TWIN_PLAN.md](planning/DIGITAL_TWIN_PLAN.md) for the full design document.
