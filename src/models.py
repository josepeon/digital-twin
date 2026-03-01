"""
Data models for the Digital Twin pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Source(str, Enum):
    """Data source identifiers."""

    CHATGPT = "chatgpt"
    INSTAGRAM = "instagram"
    WHATSAPP = "whatsapp"
    IMESSAGE = "imessage"
    EMAIL = "email"
    IDENTITY = "identity"


class Register(str, Enum):
    """Conversational register / tone."""

    CASUAL = "casual"
    PROFESSIONAL = "professional"
    FAMILY = "family"
    NEUTRAL = "neutral"


class Language(str, Enum):
    """Supported languages."""

    EN = "en"
    ES = "es"


@dataclass
class ParsedMessage:
    """A single parsed message from any data source.

    This is the canonical intermediate format produced by every parser.
    """

    text: str
    source: Source
    timestamp: datetime | None = None
    context: str = ""  # The message/question Jose was responding to
    register: Register = Register.NEUTRAL
    language: Language = Language.EN
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "context": self.context,
            "register": self.register.value,
            "language": self.language.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedMessage:
        return cls(
            text=data["text"],
            source=Source(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            context=data.get("context", ""),
            register=Register(data.get("register", "neutral")),
            language=Language(data.get("language", "en")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationTurn:
    """A single turn in a conversation (for training format)."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class TrainingExample:
    """A formatted training example ready for fine-tuning or RAG indexing."""

    id: str
    messages: list[ConversationTurn]
    source: Source
    register: Register = Register.NEUTRAL
    language: Language = Language.EN
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": [{"role": t.role, "content": t.content} for t in self.messages],
            "source": self.source.value,
            "register": self.register.value,
            "language": self.language.value,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        return cls(
            id=data["id"],
            messages=[ConversationTurn(role=m["role"], content=m["content"]) for m in data["messages"]],
            source=Source(data["source"]),
            register=Register(data.get("register", "neutral")),
            language=Language(data.get("language", "en")),
            quality_score=data.get("quality_score", 0.0),
            metadata=data.get("metadata", {}),
        )
