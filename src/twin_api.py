"""
FastAPI service — the Digital Twin API.

Endpoints:
  POST  /ask-jose        → Text response
  POST  /ask-jose/voice  → Audio (MP3) response
  POST  /ask-jose/stream → Streaming audio response
  WS    /ws              → WebSocket streaming (text + optional voice)
  GET   /health          → Health check
  GET   /                → Web chat UI
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import TWIN_API_KEY
from src.memory import MemoryStore
from src.observability import (
    attribute as _attribute,
    record_usage as _record_usage,
    registry as _metrics_registry,
)
from src.personality_engine import PersonalityEngine, clear_session, _memory
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

    # Clean up expired sessions on startup
    expired = _memory.cleanup_expired()
    if expired:
        import logging
        logging.getLogger(__name__).info("Cleaned up %d expired sessions", expired)

    yield
    _memory.close()
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

# Static files for web UI
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


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
    session_id: str | None = Field(None, description="Session ID for conversation memory. Omit for stateless.")


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


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    """Prometheus-format metrics. Scrape with any Prometheus server.

    Exposes Groq + ElevenLabs token/character counts, RAG hit breakdown,
    request durations, and active session count — same shape SIA and
    multi_agent_planner emit for unified OH AI observability.
    """
    return Response(
        content=_metrics_registry().render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/ask-jose", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask_jose(req: AskRequest) -> AskResponse:
    """Ask Jose a question — returns text response."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Personality engine not initialized")

    result = _engine.ask(
        query=req.query,
        temperature=req.temperature,
        session_id=req.session_id,
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
        session_id=req.session_id,
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
        session_id=req.session_id,
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


# ── Speech-to-Text ─────────────────────────────────────────────────────────────


@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using Groq Whisper.

    Accepts audio files (webm, mp3, wav, m4a, ogg).
    Returns: {"text": "transcribed text", "language": "en"}
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Read audio bytes
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB limit
        raise HTTPException(status_code=413, detail="Audio file too large (25MB max)")

    # Transcribe with Groq Whisper
    groq = _engine._get_groq()
    loop = asyncio.get_event_loop()

    try:
        filename = audio.filename or "recording.webm"
        with _attribute("twin-stt"):
            result = await loop.run_in_executor(
                None,
                lambda: groq.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model="whisper-large-v3",
                    response_format="json",
                ),
            )
        text = result.text.strip() if result.text else ""
        # Whisper is free on Groq, but track call counts so the dashboard
        # sees STT volume. Token count is approximated by output chars.
        try:
            _record_usage(
                provider="groq-whisper",
                model="whisper-large-v3",
                prompt_tokens=0,
                completion_tokens=len(text),
            )
        except Exception:  # noqa: BLE001
            pass
        return {"text": text, "language": "auto"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")


# ── WebSocket Streaming ────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time streaming.

    Client sends JSON: {"query": "...", "session_id": "...", "voice": false}
    Server streams JSON frames:
      {"type": "text",  "content": "partial text...", "done": false}
      {"type": "text",  "content": "full text",       "done": true, "language": "en"}
      {"type": "audio", "content": "<base64 mp3>"}  (if voice=true)
      {"type": "error", "content": "error message"}
    """
    await ws.accept()

    # Auth check — first message can be {"type": "auth", "key": "..."}
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        first_msg = json.loads(raw)

        if first_msg.get("type") == "auth":
            if TWIN_API_KEY and first_msg.get("key") != TWIN_API_KEY:
                await ws.send_json({"type": "error", "content": "Invalid API key"})
                await ws.close(code=4001)
                return
            # Auth passed, wait for actual query
            raw = await ws.receive_text()
            first_msg = json.loads(raw)
        elif TWIN_API_KEY:
            # No auth message and key is required
            await ws.send_json({"type": "error", "content": "Auth required — send {type: 'auth', key: '...'} first"})
            await ws.close(code=4001)
            return

        # Process first query
        await _handle_ws_query(ws, first_msg)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
        return

    # Keep connection alive for follow-up queries
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_ws_query(ws, msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


async def _handle_ws_query(ws: WebSocket, msg: dict) -> None:
    """Handle a single query over WebSocket."""
    if _engine is None:
        await ws.send_json({"type": "error", "content": "Engine not initialized"})
        return

    query = msg.get("query", "").strip()
    if not query:
        await ws.send_json({"type": "error", "content": "Empty query"})
        return

    session_id = msg.get("session_id", str(uuid.uuid4()))
    want_voice = msg.get("voice", False)

    # Generate text response (runs in thread pool to not block event loop)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _engine.ask(query=query, session_id=session_id),
    )

    text = result["response"]
    lang = result["language"]

    # Send complete text
    await ws.send_json({
        "type": "text",
        "content": text,
        "done": True,
        "language": lang,
        "session_id": session_id,
    })

    # Optionally generate and send voice
    if want_voice and _voice and text:
        try:
            import base64
            audio_bytes = await loop.run_in_executor(
                None,
                lambda: _voice.speak(text=text, language=lang),
            )
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await ws.send_json({
                "type": "audio",
                "content": audio_b64,
            })
        except Exception as e:
            await ws.send_json({"type": "error", "content": f"TTS failed: {e}"})


# ── Web UI ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def web_ui():
    """Serve the web chat interface."""
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    raise HTTPException(status_code=404, detail="Web UI not found")


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
