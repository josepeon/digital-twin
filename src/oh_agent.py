"""
OH platform adapter

Exposes the twin's ``PersonalityEngine`` over OH's spatial-commerce
WebSocket protocol so the twin can appear as Jose's avatar inside the
OH Unreal experience.

Design:

- No new network deps. The adapter rides on the existing FastAPI app —
  one new WebSocket route at /oh/agent.
- The protocol is symmetric: client sends JSON, server replies with
  JSON. Same shape OH uses for its other agents.
- Mode is pinned to "professional" by default (this is a commerce
  context, not a friend DM) but clients can override per-message.
- Streaming uses the existing PersonalityEngine.ask_stream() generator
  so latency is identical to the WS chat path.

Protocol:

    Client → Server:
        {"type": "query",   "content": "Tell me about Nooon", "mode": "creative"?}
        {"type": "presence", "session_id": "...", "visitor_id": "..."}

    Server → Client:
        {"type": "delta",  "content": "Hello, ", "session_id": "..."}
        {"type": "delta",  "content": "I'd love to ", ...}
        {"type": "done",   "session_id": "...", "language": "en"}
        {"type": "error",  "content": "...",     "session_id": "..."}
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = "1.0"

# Default mode for OH commerce context — board-meeting register, not DM
DEFAULT_OH_MODE = "professional"


# ---------------------------------------------------------------------
# Schema validation (lightweight, no Pydantic dep beyond what's already
# in the twin's runtime)
# ---------------------------------------------------------------------


def parse_incoming(raw: str | dict) -> dict[str, Any]:
    """Parse + validate a message from an OH client.

    Returns a dict the adapter can act on. Raises ValueError on
    malformed input — caller is expected to translate that into an
    error frame.
    """
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise ValueError(f"unexpected payload type {type(raw).__name__}")

    msg_type = payload.get("type")
    if msg_type not in ("query", "presence"):
        raise ValueError(f"unknown type {msg_type!r}")

    if msg_type == "query":
        content = payload.get("content", "").strip()
        if not content:
            raise ValueError("query requires non-empty content")
        return {
            "type": "query",
            "content": content,
            "mode": payload.get("mode") or DEFAULT_OH_MODE,
            "session_id": payload.get("session_id"),
            "visitor_id": payload.get("visitor_id"),
        }

    # presence
    return {
        "type": "presence",
        "session_id": payload.get("session_id"),
        "visitor_id": payload.get("visitor_id"),
    }


def encode_delta(content: str, session_id: str | None) -> str:
    return json.dumps({
        "type": "delta",
        "content": content,
        "session_id": session_id,
        "v": PROTOCOL_VERSION,
    })


def encode_done(session_id: str | None, language: str | None = None) -> str:
    return json.dumps({
        "type": "done",
        "session_id": session_id,
        "language": language,
        "v": PROTOCOL_VERSION,
    })


def encode_error(message: str, session_id: str | None = None) -> str:
    return json.dumps({
        "type": "error",
        "content": message,
        "session_id": session_id,
        "v": PROTOCOL_VERSION,
    })


def encode_presence_ack(session_id: str | None) -> str:
    return json.dumps({
        "type": "presence_ack",
        "session_id": session_id,
        "agent": "jose-twin",
        "v": PROTOCOL_VERSION,
    })


# ---------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------


async def stream_response(engine, parsed: dict[str, Any]):
    """Yield WS-framed responses for a parsed query.

    Args:
        engine: PersonalityEngine instance from twin_api lifespan
        parsed: output of ``parse_incoming`` with type == "query"

    Yields:
        str — JSON frames the WS handler ships to the client.

    Catches any engine exception and turns it into an error frame
    rather than dropping the connection. After streaming, always
    emits a "done" frame so the client knows the turn is over.
    """
    session_id = parsed.get("session_id")
    try:
        for delta in engine.ask_stream(
            query=parsed["content"],
            session_id=session_id,
            mode=parsed.get("mode", DEFAULT_OH_MODE),
        ):
            if delta:
                yield encode_delta(delta, session_id)
    except Exception as exc:  # noqa: BLE001 — never close the stream silently
        logger.exception("OH agent stream failed")
        yield encode_error(str(exc), session_id)

    yield encode_done(session_id)
