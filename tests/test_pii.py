"""Tests for the PII scrubber."""

import pytest

from src.pii import scrub_text


class TestScrubEmail:
    def test_scrubs_email_address(self):
        result = scrub_text("Contact me at jose@example.com please")
        assert "[EMAIL]" in result.scrubbed
        assert "jose@example.com" not in result.scrubbed
        assert result.was_modified

    def test_scrubs_multiple_emails(self):
        result = scrub_text("Email jose@example.com or backup@test.org")
        assert result.scrubbed.count("[EMAIL]") == 2


class TestScrubPhone:
    def test_scrubs_us_phone(self):
        result = scrub_text("Call me at 555-123-4567")
        assert "[PHONE]" in result.scrubbed
        assert "555-123-4567" not in result.scrubbed

    def test_scrubs_international_phone(self):
        result = scrub_text("My number is +1 555 123 4567")
        assert "[PHONE]" in result.scrubbed

    def test_scrubs_cuban_phone(self):
        result = scrub_text("Llama al +53 5 1234567")
        assert "[PHONE]" in result.scrubbed


class TestScrubAPIKey:
    def test_scrubs_api_key_pattern(self):
        result = scrub_text("api_key=sk_test_FAKE00KEY00FOR00UNIT00TESTS00ONLY")
        assert "[API_KEY]" in result.scrubbed

    def test_scrubs_groq_key(self):
        result = scrub_text("Use this key: gsk_abcdefghijklmnopqrstuvwx")
        assert "[API_KEY]" in result.scrubbed

    def test_scrubs_github_pat(self):
        result = scrub_text("Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz12")
        assert "[API_KEY]" in result.scrubbed


class TestScrubAddress:
    def test_scrubs_street_address(self):
        result = scrub_text("I live at 123 Main Street")
        assert "[ADDRESS]" in result.scrubbed

    def test_scrubs_avenue(self):
        result = scrub_text("Office is at 456 Park Avenue")
        assert "[ADDRESS]" in result.scrubbed


class TestScrubPassword:
    def test_scrubs_password_mention(self):
        result = scrub_text("password: myS3cretP@ss!")
        assert "[PASSWORD]" in result.scrubbed

    def test_scrubs_spanish_password(self):
        result = scrub_text("la contraseña es abc12345")
        assert "[PASSWORD]" in result.scrubbed


class TestNoFalsePositives:
    def test_normal_text_unchanged(self):
        text = "Hey, I've been working on this AI project and it's going really well"
        result = scrub_text(text)
        assert result.scrubbed == text
        assert not result.was_modified

    def test_spanish_text_unchanged(self):
        text = "Oye asere, qué bolá? Estoy trabajando en algo increíble"
        result = scrub_text(text)
        assert result.scrubbed == text

    def test_empty_text(self):
        result = scrub_text("")
        assert result.scrubbed == ""
        assert not result.was_modified


class TestSelectiveCategories:
    def test_scrub_only_emails(self):
        text = "Email jose@test.com, call 555-1234567"
        result = scrub_text(text, categories=["email"])
        assert "[EMAIL]" in result.scrubbed
        assert "555-1234567" in result.scrubbed  # Phone NOT scrubbed
