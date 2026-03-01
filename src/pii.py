"""
PII (Personally Identifiable Information) scrubber.

Removes or redacts sensitive data from parsed messages before they enter
the RAG index or training dataset:
  - Email addresses
  - Phone numbers (US, international, Cuban formats)
  - Credit card numbers
  - SSN / government IDs
  - API keys and tokens
  - IP addresses
  - Street addresses (heuristic)
  - URLs with auth tokens
  - Passwords mentioned in text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Regex patterns ─────────────────────────────────────────────────────────────

PATTERNS: dict[str, re.Pattern] = {
    # Email addresses
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
    ),
    # Phone numbers — US, international, Cuban (+53), with/without separators
    "phone": re.compile(
        r"(?<!\d)"
        r"(?:\+?\d{1,3}[\s\-.]?)?"       # country code
        r"(?:\(?\d{2,4}\)?[\s\-.]?)?"     # area code
        r"\d{3,4}[\s\-.]?\d{3,4}"         # number
        r"(?!\d)"
    ),
    # Credit card numbers (13-19 digits, possibly separated)
    "credit_card": re.compile(
        r"\b(?:\d[ \-]?){13,19}\b"
    ),
    # US SSN
    "ssn": re.compile(
        r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b"
    ),
    # API keys, tokens, secrets (common patterns)
    "api_key": re.compile(
        r"(?i)"
        r"(?:"
        r"(?:api[_\-]?key|token|secret|password|auth|bearer|sk[_\-]live|sk[_\-]test|pk[_\-]live|pk[_\-]test)"
        r"[\s:=]+['\"]?"
        r"[A-Za-z0-9\-_\.]{16,}"
        r"|"
        r"\b(?:sk|pk|rk|whsec)[_\-][A-Za-z0-9\-_\.]{20,}\b"  # Stripe-style
        r"|"
        r"\bgsk_[A-Za-z0-9\-_]{20,}\b"    # Groq keys
        r"|"
        r"\bghp_[A-Za-z0-9]{36,}\b"       # GitHub PAT
        r"|"
        r"\bxai-[A-Za-z0-9\-_]{20,}\b"    # xAI keys
        r")"
    ),
    # IPv4 addresses
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    # URLs with tokens/auth params
    "auth_url": re.compile(
        r"https?://[^\s]+(?:token|key|auth|secret|password|access)[^\s]*",
        re.IGNORECASE,
    ),
    # Street addresses (heuristic: number + street name + type)
    "street_address": re.compile(
        r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,3}"
        r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Ln|Lane|Rd|Road|Ct|Court|Pl|Place|Way|Cir(?:cle)?)"
        r"\.?\b",
        re.IGNORECASE,
    ),
    # Passwords mentioned in text (handles "password: X", "contraseña es X", "clave es X")
    "password_mention": re.compile(
        r"(?i)(?:password|contrase\w+|clave|passwd)[\s:=]+(?:(?:es|is)\s+)?\S{4,}",
    ),
}

# Redaction placeholder per category
REDACTION_LABELS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "credit_card": "[CREDIT_CARD]",
    "ssn": "[SSN]",
    "api_key": "[API_KEY]",
    "ip_address": "[IP]",
    "auth_url": "[AUTH_URL]",
    "street_address": "[ADDRESS]",
    "password_mention": "[PASSWORD]",
}


@dataclass
class ScrubResult:
    """Result of scrubbing a single text."""

    original: str
    scrubbed: str
    redactions: list[dict[str, str]] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.original != self.scrubbed

    @property
    def redaction_count(self) -> int:
        return len(self.redactions)


def scrub_text(text: str, categories: list[str] | None = None) -> ScrubResult:
    """Scrub PII from a text string.

    Args:
        text: The text to scrub.
        categories: Which PII categories to scrub. If None, scrub all.

    Returns:
        ScrubResult with the scrubbed text and a list of redactions made.
    """
    if not text:
        return ScrubResult(original=text, scrubbed=text)

    cats = categories or list(PATTERNS.keys())
    # Process API keys before phones to prevent phone regex matching digits inside tokens
    priority_order = ["api_key", "credit_card", "ssn", "auth_url", "email", "phone",
                      "ip_address", "street_address", "password_mention"]
    cats = [c for c in priority_order if c in cats] + [c for c in cats if c not in priority_order]
    scrubbed = text
    redactions: list[dict[str, str]] = []

    for cat in cats:
        pattern = PATTERNS.get(cat)
        if pattern is None:
            continue
        label = REDACTION_LABELS[cat]
        for match in pattern.finditer(scrubbed):
            redactions.append({"category": cat, "match": match.group(), "replaced_with": label})
        scrubbed = pattern.sub(label, scrubbed)

    return ScrubResult(original=text, scrubbed=scrubbed, redactions=redactions)


def scrub_messages(messages: list[dict], text_key: str = "text") -> list[dict]:
    """Scrub PII from a list of message dicts in-place.

    Args:
        messages: List of dicts, each containing a text field.
        text_key: The key in each dict that holds the text. Also scrubs "context" if present.

    Returns:
        The same list, with text fields scrubbed.
    """
    for msg in messages:
        if text_key in msg:
            result = scrub_text(msg[text_key])
            msg[text_key] = result.scrubbed
        if "context" in msg:
            result = scrub_text(msg["context"])
            msg["context"] = result.scrubbed
    return messages


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Quick CLI test: pipe text through the scrubber."""
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    result = scrub_text(text)
    print(result.scrubbed)
    if result.redactions:
        print(f"\n--- {result.redaction_count} redaction(s) ---")
        for r in result.redactions:
            print(f"  [{r['category']}] {r['match']!r} → {r['replaced_with']}")


if __name__ == "__main__":
    main()
