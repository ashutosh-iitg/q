"""What the store writes, and the errors it raises."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.records import FactKind
from app.triage.policy import Decision


class SessionOwnershipError(LookupError):
    """The session exists but belongs to another household."""


class EpisodeOwnershipError(LookupError):
    """The episode exists but belongs to another member."""


class TurnConflictError(ValueError):
    """An idempotency key was reused with different message text."""


@dataclass(frozen=True)
class EpisodeWrite:
    id: str  # caller generates the ID for a new episode
    member_id: str
    complaint: str
    status: str
    slots: Mapping[str, Any]  # merged into the stored slots
    questions_asked_delta: int
    check_back_after: datetime | None
    related_episode_id: str | None = None
    protocol_id: str | None = None
    findings: Mapping[str, Any] | None = None  # replaces the stored findings when given
    associated: Mapping[str, Any] | None = None  # replaces the stored associated symptoms when given
    urgency: str | None = None  # doctor decisions: kept until a later doctor decision changes it
    handoff_id: str | None = None  # the turn whose intake note handed the episode over; kept when None


@dataclass(frozen=True)
class FactWrite:
    member_id: str
    episode_id: str | None
    kind: FactKind
    subject: str  # normalised on write
    text: str
    effective_at: datetime | None = None


@dataclass(frozen=True)
class TurnWrite:
    household_id: str
    session_id: str
    turn_id: str
    user_text: str
    reply: str
    decision: Decision
    member_id: str | None
    episode: EpisodeWrite | None
    facts: tuple[FactWrite, ...]
    pending_episode_id: str | None


@dataclass(frozen=True)
class MissedFactWrite:
    """A fact the transcript states but the turn missed, found at consolidation."""

    fact: FactWrite
    source_message_id: int


@dataclass(frozen=True)
class ConsolidationWrite:
    """Validated consolidation output. Every ID has been checked against the session's own data."""

    session_id: str
    summary: str | None
    promotions: tuple[tuple[int, str], ...]  # (fact_id, canonical subject)
    duplicates: tuple[tuple[int, int], ...]  # (fact_id, earlier fact it repeats)
    replacements: tuple[tuple[int, int], ...]  # (fact_id, earlier fact it replaces)
    missed_facts: tuple[MissedFactWrite, ...]
