"""
Observability bootstrap

The twin shares its cost-tracking, metrics, and reproducibility utilities
with ``self_improving_agent`` so all three OH AI repos (digital-twin,
multi_agent_planner, self-improving-agent) report the same way.

On import this module:

  1. Registers the ElevenLabs character-pricing rows that SIA doesn't
     ship — ElevenLabs bills per character, not per token, but SIA's
     existing ``record_usage`` API treats them identically. We document
     this in the metric name (``elevenlabs.<voice>`` provider) so the
     "tokens" label is read as "characters" downstream.
  2. Builds a SIA metric registry for the FastAPI ``/metrics`` endpoint.
  3. Re-exports the SIA helpers callers actually want:
     ``attribute``, ``record_usage``, ``begin_run``, ``render_summary``,
     ``set_seed``.

When SIA isn't installed (e.g. local dev without the [sia] extra),
imports raise ImportError with a clear "pip install -e '.[sia]'" hint.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from self_improving_agent.observability.cost_tracker import (
        ModelPrice,
        attribute,
        begin_run,
        get_tracker,
        record_usage,
        register_price,
        render_summary,
        to_dict,
    )
    from self_improving_agent.observability.metrics import (
        Counter,
        Gauge,
        Histogram,
        MetricRegistry,
        make_sia_registry,
    )
    from self_improving_agent.reproducibility import set_seed
except ImportError as exc:  # pragma: no cover — covered by ImportError test
    raise ImportError(
        "src.observability depends on self-improving-agent. Install with:\n"
        "    pip install -e '.[sia]'\n"
        "or add the package to requirements.txt for production deploy."
    ) from exc


# ---------------------------------------------------------------------
# ElevenLabs pricing — per-1M-characters. SIA doesn't ship this because
# its providers (Groq/OpenAI/MLX) all bill per token.
# ---------------------------------------------------------------------
#
# ElevenLabs current pricing (verified 2026-03-09):
#   Creator tier: $5/mo for 30k chars, then $0.30 per 1k = $300/M chars
#   Pro tier:     $99/mo for 500k chars, then $0.24 per 1k = $240/M chars
#
# We use Creator-tier numbers — over-estimate, never under-report. The
# "model" string is the voice ID we sent; left as the catch-all "*" so
# both EN and ES voices get priced.

_ELEVENLABS_PRICE = ModelPrice(input_per_mtok=0.0, output_per_mtok=300.0)
register_price("elevenlabs", "*", _ELEVENLABS_PRICE)

# Whisper on Groq is currently free tier. Track usage so the dashboard
# counts calls; cost stays at $0.
register_price("groq-whisper", "whisper-large-v3", ModelPrice(0.0, 0.0))


# ---------------------------------------------------------------------
# Metric registry — built once at module import, attached to the app
# at startup via ``observability.registry``.
# ---------------------------------------------------------------------

_registry, _metrics = make_sia_registry()


def registry() -> MetricRegistry:
    """The shared metric registry exposed at /metrics."""
    return _registry


def metric(name: str):
    """Look up a registered metric by SIA's handle name.

    Available: requests_total, request_duration, fallbacks_total,
    tokens_total, version_info.
    """
    return _metrics[name]


# Twin-specific metrics layered on top of SIA's defaults.
TWIN_VOICE_CHARS = Counter(
    "twin_voice_characters_total",
    "Characters sent to ElevenLabs TTS, broken down by voice/language.",
)
_registry.register(TWIN_VOICE_CHARS)

TWIN_RAG_HITS = Counter(
    "twin_rag_hits_total",
    "RAG retrievals served, by source (instagram/chatgpt/imessage/...).",
)
_registry.register(TWIN_RAG_HITS)

TWIN_SESSIONS = Gauge(
    "twin_active_sessions",
    "Active conversation sessions (sessions seen in the last hour).",
)
_registry.register(TWIN_SESSIONS)


# ---------------------------------------------------------------------
# Re-exports for the rest of the package
# ---------------------------------------------------------------------

__all__ = [
    "attribute",
    "begin_run",
    "get_tracker",
    "record_usage",
    "register_price",
    "render_summary",
    "to_dict",
    "set_seed",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricRegistry",
    "registry",
    "metric",
    "TWIN_VOICE_CHARS",
    "TWIN_RAG_HITS",
    "TWIN_SESSIONS",
]
