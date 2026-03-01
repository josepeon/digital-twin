#!/usr/bin/env python3
"""
Email parser (Gmail / MBOX format).

Parses sent emails from a Gmail export (Google Takeout MBOX) or individual
.eml files into ParsedMessage objects.

Google Takeout exports emails as MBOX files. We only care about SENT mail
(Jose's actual words), not received mail.

Usage:
    python scripts/parse_email.py [--input data/personal/raw/email]
"""

from __future__ import annotations

import argparse
import email
import email.policy
import json
import mailbox
import re
import sys
from datetime import datetime
from email import utils as email_utils
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Language, ParsedMessage, Register, Source
from src.pii import scrub_text


def _detect_language(text: str) -> Language:
    spanish_words = {
        "que", "de", "en", "es", "un", "una", "por", "para", "como",
        "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
        "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso",
    }
    words = set(text.lower().split())
    return Language.ES if len(words & spanish_words) >= 3 else Language.EN


def _detect_register(text: str, subject: str = "") -> Register:
    """Detect email register from content and subject."""
    combined = (text + " " + subject).lower()

    professional_signals = [
        "regarding", "please find", "as discussed", "follow up", "meeting",
        "deadline", "attached", "schedule", "per our", "best regards",
        "sincerely", "dear", "hi team", "update on",
    ]
    if any(s in combined for s in professional_signals):
        return Register.PROFESSIONAL

    family_signals = ["mamá", "papá", "familia", "hermano", "mami", "papi"]
    if any(s in combined for s in family_signals):
        return Register.FAMILY

    return Register.NEUTRAL


def _extract_text_from_email(msg: email.message.Message) -> str:
    """Extract plain text body from an email message."""
    body_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        body_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_parts.append(payload.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                body_parts.append(payload.decode("utf-8", errors="replace"))

    text = "\n".join(body_parts)
    return _clean_email_body(text)


def _clean_email_body(text: str) -> str:
    """Clean up email body text — remove signatures, quoted replies, etc."""
    lines = text.split("\n")
    cleaned: list[str] = []

    for line in lines:
        # Stop at common reply/forward markers
        if re.match(r"^-{3,}\s*(?:Original|Forwarded)\s", line, re.IGNORECASE):
            break
        if re.match(r"^On\s.+wrote:\s*$", line):
            break
        if re.match(r"^El\s.+escribió:\s*$", line):  # Spanish
            break
        if line.startswith(">"):
            continue  # Skip quoted text
        if re.match(r"^Sent from my (iPhone|iPad|Galaxy|Pixel)", line, re.IGNORECASE):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _parse_date(msg: email.message.Message) -> datetime | None:
    """Parse the Date header from an email."""
    date_str = msg.get("Date", "")
    if not date_str:
        return None
    try:
        parsed = email_utils.parsedate_to_datetime(date_str)
        return parsed
    except (ValueError, TypeError):
        return None


def _is_jose_sender(from_header: str, jose_emails: list[str]) -> bool:
    """Check if the From header matches one of Jose's email addresses."""
    from_lower = from_header.lower()
    return any(addr.lower() in from_lower for addr in jose_emails)


# ── MBOX Parser ───────────────────────────────────────────────────────────────

def parse_mbox(
    mbox_path: Path,
    jose_emails: list[str],
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse a Gmail MBOX export for sent emails.

    Args:
        mbox_path: Path to the .mbox file.
        jose_emails: Jose's email addresses (to identify sent mail).
        scrub_pii: Whether to scrub PII.

    Returns:
        List of ParsedMessage objects for Jose's sent emails.
    """
    mbox = mailbox.mbox(str(mbox_path))
    results: list[ParsedMessage] = []

    for msg in mbox:
        from_header = msg.get("From", "")

        # Only Jose's sent mail
        if not _is_jose_sender(from_header, jose_emails):
            continue

        text = _extract_text_from_email(msg)
        if not text.strip():
            continue

        subject = msg.get("Subject", "")
        to_header = msg.get("To", "")

        if scrub_pii:
            text = scrub_text(text).scrubbed
            subject = scrub_text(subject).scrubbed
            to_header = scrub_text(to_header).scrubbed

        pm = ParsedMessage(
            text=text,
            source=Source.EMAIL,
            timestamp=_parse_date(msg),
            context=f"Re: {subject}" if subject else "",
            register=_detect_register(text, subject),
            language=_detect_language(text),
            metadata={
                "subject": subject,
                "to": to_header,
                "type": "sent_email",
            },
        )
        results.append(pm)

    mbox.close()
    return results


# ── EML Parser ────────────────────────────────────────────────────────────────

def parse_eml_files(
    eml_dir: Path,
    jose_emails: list[str],
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse individual .eml files."""
    eml_files = sorted(eml_dir.glob("*.eml"))
    results: list[ParsedMessage] = []

    for eml_file in eml_files:
        with open(eml_file, "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)

        from_header = msg.get("From", "")
        if not _is_jose_sender(from_header, jose_emails):
            continue

        text = _extract_text_from_email(msg)
        if not text.strip():
            continue

        subject = msg.get("Subject", "")

        if scrub_pii:
            text = scrub_text(text).scrubbed
            subject = scrub_text(subject).scrubbed

        pm = ParsedMessage(
            text=text,
            source=Source.EMAIL,
            timestamp=_parse_date(msg),
            context=f"Re: {subject}" if subject else "",
            register=_detect_register(text, subject),
            language=_detect_language(text),
            metadata={
                "subject": subject,
                "type": "sent_email",
                "file": eml_file.name,
            },
        )
        results.append(pm)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_email_export(
    email_dir: Path,
    jose_emails: list[str],
    scrub_pii: bool = True,
) -> list[ParsedMessage]:
    """Parse all email exports (.mbox and .eml files) in a directory."""
    all_messages: list[ParsedMessage] = []

    # MBOX files
    mbox_files = sorted(email_dir.glob("*.mbox"))
    for mbox_file in mbox_files:
        print(f"  Parsing MBOX: {mbox_file.name}...")
        msgs = parse_mbox(mbox_file, jose_emails, scrub_pii)
        print(f"    → {len(msgs)} sent emails")
        all_messages.extend(msgs)

    # EML files
    eml_files = list(email_dir.glob("*.eml"))
    if eml_files:
        print(f"  Parsing {len(eml_files)} .eml files...")
        msgs = parse_eml_files(email_dir, jose_emails, scrub_pii)
        print(f"    → {len(msgs)} sent emails")
        all_messages.extend(msgs)

    return all_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse email exports into training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/personal/raw/email"),
        help="Directory containing .mbox or .eml files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/personal/processed/email_parsed.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--jose-emails",
        type=str,
        nargs="+",
        required=True,
        help="Jose's email address(es)",
    )
    parser.add_argument("--no-scrub", action="store_true", help="Skip PII scrubbing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input directory not found: {args.input}")
        print("   Export: Google Takeout → Gmail → MBOX format")
        sys.exit(1)

    print(f"📂 Parsing email exports: {args.input}")
    messages = parse_email_export(args.input, args.jose_emails, not args.no_scrub)
    print(f"   Total: {len(messages)} sent emails")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(messages)} messages to {args.output}")


if __name__ == "__main__":
    main()
