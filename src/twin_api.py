"""
FastAPI service — the Digital Twin API.

Endpoints:
  POST /ask-jose        → Text response
  POST /ask-jose/voice  → Audio (MP3) response
  POST /ask-jose/stream → Streaming audio response
  GET  /health          → Health check
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.config import TWIN_API_KEY
from src.personality_engine import PersonalityEngine
from src.voice import VoiceEngine

# ── Lifespan ───────────────────────────────────────────────────────────────────

# Shared instances (initialized once at startup)
_engine: PersonalityEngine | None = None
_voice: VoiceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engines on startup, cleanup on shutdown."""
    global _engine, _voice
    _engine = PersonalityEngine()
    _voice = VoiceEngine()
    yield
    _engine = None
    _voice = None


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Digital Twin — Jose Peon",
    description="AI digital twin API. Text and voice responses as Jose.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Narrow this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ───────────────────────────────────────────────────────────────────────

async def verify_api_key(request: Request) -> None:
    """Validate the API key from the Authorization header."""
    if not TWIN_API_KEY:
        return  # No key configured = open access (dev mode)

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.headers.get("X-API-Key", "")

    if token != TWIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Models ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Question or message to Jose")
    language: str | None = Field(None, description="Force language: 'en' or 'es'")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="LLM temperature")


class AskResponse(BaseModel):
    response: str
    language: str
    style_register: str
    model: str
    examples_used: int


class HealthResponse(BaseModel):
    status: str
    rag_index_loaded: bool
    voice_clones_loaded: bool
    groq_model: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — reports engine status."""
    rag_loaded = False
    if _engine:
        collection = _engine._get_collection()
        rag_loaded = collection is not None and collection.count() > 0

    voice_loaded = _voice.has_cloned_voices if _voice else False

    from src.config import GROQ_MODEL
    return HealthResponse(
        status="ok",
        rag_index_loaded=rag_loaded,
        voice_clones_loaded=voice_loaded,
        groq_model=GROQ_MODEL,
    )


@app.post("/ask-jose", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask_jose(req: AskRequest) -> AskResponse:
    """Ask Jose a question — returns text response."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Personality engine not initialized")

    result = _engine.ask(
        query=req.query,
        temperature=req.temperature,
    )

    return AskResponse(
        response=result["response"],
        language=result["language"],
        style_register=result["register"],
        model=result["model"],
        examples_used=result["examples_used"],
    )


@app.post("/ask-jose/voice", dependencies=[Depends(verify_api_key)])
async def ask_jose_voice(req: AskRequest) -> Response:
    """Ask Jose a question — returns MP3 audio response."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Personality engine not initialized")
    if _voice is None:
        raise HTTPException(status_code=503, detail="Voice engine not initialized")

    # Generate text response
    result = _engine.ask(
        query=req.query,
        temperature=req.temperature,
    )

    # Convert to speech
    try:
        audio_bytes = _voice.speak(
            text=result["response"],
            language=result["language"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Response-Text": result["response"][:200],
            "X-Response-Language": result["language"],
            "X-Response-Register": result["register"],
        },
    )


@app.post("/ask-jose/stream", dependencies=[Depends(verify_api_key)])
async def ask_jose_stream(req: AskRequest) -> StreamingResponse:
    """Ask Jose a question — returns streaming MP3 audio."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Personality engine not initialized")
    if _voice is None:
        raise HTTPException(status_code=503, detail="Voice engine not initialized")

    # Generate text response
    result = _engine.ask(
        query=req.query,
        temperature=req.temperature,
    )

    # Stream audio
    def audio_stream():
        yield from _voice.speak_stream(
            text=result["response"],
            language=result["language"],
        )

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "X-Response-Text": result["response"][:200],
            "X-Response-Language": result["language"],
        },
    )


# ── Run ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the API server."""
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.twin_api:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RAILWAY_ENVIRONMENT") is None,  # Only reload in dev
    )


if __name__ == "__main__":
    main()
