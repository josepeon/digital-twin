"""T1.1: observability bootstrap — SIA cross-repo wiring."""

from __future__ import annotations

import pytest

from src.observability import (
    TWIN_RAG_HITS,
    TWIN_SESSIONS,
    TWIN_VOICE_CHARS,
    attribute,
    begin_run,
    get_tracker,
    record_usage,
    registry,
    render_summary,
    set_seed,
)


@pytest.fixture(autouse=True)
def _reset():
    begin_run()
    yield
    begin_run()


class TestReExports:
    def test_sia_helpers_callable(self):
        # Should not raise — purely an "import + call" smoke test
        assert callable(record_usage)
        assert callable(attribute)
        assert callable(set_seed)

    def test_set_seed_deterministic(self):
        import random
        set_seed(42)
        first = [random.random() for _ in range(3)]
        set_seed(42)
        second = [random.random() for _ in range(3)]
        assert first == second


class TestPricingRegistration:
    def test_elevenlabs_pricing_registered(self):
        from self_improving_agent.observability.cost_tracker import get_price

        price = get_price("elevenlabs", "any-voice-id")
        assert price is not None
        # Creator-tier rate: $300/M chars on the completion side
        assert price.output_per_mtok == 300.0

    def test_groq_whisper_priced_at_zero(self):
        from self_improving_agent.observability.cost_tracker import get_price

        price = get_price("groq-whisper", "whisper-large-v3")
        assert price is not None
        assert price.output_per_mtok == 0.0


class TestRecording:
    def test_elevenlabs_call_cost_computed(self):
        record_usage(
            provider="elevenlabs",
            model="voice-en",
            prompt_tokens=0,
            completion_tokens=1000,  # 1000 characters
        )
        totals = get_tracker().total()
        assert totals.calls == 1
        # 1000 chars * $300/M = $0.30
        assert totals.cost_usd == pytest.approx(0.30, abs=0.001)

    def test_attribution_picks_up_role(self):
        with attribute("twin-chat"):
            record_usage("groq", "llama-3.3-70b-versatile", 100, 50)
        by_agent = get_tracker().by_agent()
        assert "twin-chat" in by_agent

    def test_summary_renders(self):
        with attribute("twin-voice"):
            record_usage("elevenlabs", "voice-en", 0, 100)
        out = render_summary()
        assert "Cost:" in out
        assert "twin-voice" in out


class TestTwinSpecificMetrics:
    def test_voice_chars_counter_increments(self):
        TWIN_VOICE_CHARS.inc(120, voice="voice-en", language="en")
        rendered = TWIN_VOICE_CHARS.render()
        assert "twin_voice_characters_total" in rendered
        assert "120" in rendered

    def test_rag_hits_counter_per_source(self):
        TWIN_RAG_HITS.inc(source="instagram")
        TWIN_RAG_HITS.inc(source="instagram")
        TWIN_RAG_HITS.inc(source="chatgpt")
        rendered = TWIN_RAG_HITS.render()
        # Two independent series, one per source
        assert rendered.count("twin_rag_hits_total") >= 2

    def test_sessions_gauge_set(self):
        TWIN_SESSIONS.set(7)
        assert "twin_active_sessions 7" in TWIN_SESSIONS.render()


class TestRegistry:
    def test_registry_renders_prometheus_format(self):
        out = registry().render()
        # Each metric appears at least as a # HELP line
        for name in (
            "sia_requests_total",
            "sia_request_duration_seconds",
            "twin_voice_characters_total",
            "twin_rag_hits_total",
            "twin_active_sessions",
        ):
            assert f"# HELP {name}" in out

    def test_registry_handles_empty_state(self):
        # First call before any records — should still produce valid output
        out = registry().render()
        assert "# HELP" in out
        assert out.endswith("\n")
