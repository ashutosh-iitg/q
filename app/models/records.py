"""Records loaded from and written to Postgres: members, facts, episodes, messages, sessions."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

FactStatus = Literal["provisional", "consolidated", "verified"]


FactSource = Literal["user_report", "seeded", "clinician"]


FactKind = Literal["condition", "symptom", "observation", "medication", "allergy", "preference", "note"]


@dataclass(frozen=True)
class Member:
    id: str
    name: str
    aliases: tuple[str, ...]
    age_years: float | None
    conditions: tuple[str, ...]  # seeded only


@dataclass(frozen=True)
class MemberFact:
    """Long-term semantic memory. Provisional until consolidation; verified only from a clinician or seed."""

    id: int
    member_id: str
    session_id: str | None
    episode_id: str | None
    kind: FactKind
    subject: str  # normalised label shared by related facts, e.g. "blood_pressure"
    text: str  # the evidence, as reported
    effective_at: datetime | None  # when it was true or measured; created_at is when Q recorded it
    status: FactStatus
    source: FactSource
    source_message_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class Episode:
    """Episodic memory. Durable from the first turn because episodes span sessions."""

    id: str
    member_id: str
    complaint: str
    status: str
    slots: Mapping[str, Any]
    questions_asked: int
    check_back_after: datetime | None
    updated_at: datetime
    protocol_id: str | None = None
    findings: Mapping[str, Any] = MappingProxyType({})  # question ID -> {value, quote, at}
    associated: Mapping[str, Any] = MappingProxyType({})  # symptom -> {status: active|resolved, quote, at}


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    member_id: str | None
    episode_id: str | None
    decision: str | None  # assistant messages only
    created_at: datetime
    session_id: str | None = None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    summary: str
    closed_at: datetime


@dataclass(frozen=True)
class SessionContext:
    """Everything one turn may need, loaded from Postgres in a single round trip."""

    session_id: str
    is_new_session: bool  # first turn: triggers lazy consolidation and any due check-back
    members: tuple[Member, ...]
    facts: tuple[MemberFact, ...]  # not superseded; provisional ones included so lagging consolidation loses nothing
    episodes: tuple[Episode, ...]  # open, plus recently resolved for recurrence checks
    recent_messages: tuple[Message, ...]  # working memory: last N in this session, oldest first
    prior_summaries: tuple[SessionSummary, ...]  # consolidated earlier sessions in this household
    member_hint_id: str | None  # from the last turn, not an override
    pending_episode_id: str | None  # what the pending question was about


@dataclass(frozen=True)
class Clarification:
    """Why the turn can't proceed yet, and what the responder should ask."""

    kind: Literal["member", "episode"]
    needed_info: str
    responder_instructions: str | None


@dataclass(frozen=True)
class SessionRecord:
    id: str
    household_id: str
    status: str
    member_hint_id: str | None
    pending_episode_id: str | None
    started_at: datetime
    last_active_at: datetime
    closed_at: datetime | None


@dataclass(frozen=True)
class ConsolidationInput:
    """Everything consolidation gives the LLM for one closed session."""

    session: SessionRecord
    transcript: tuple[Message, ...]  # every message in the session, oldest first
    members: tuple[Member, ...]
    facts: tuple[MemberFact, ...]  # household long-term memory, not superseded
    episodes: tuple[Episode, ...]  # open, recent, and any this session touched
    prior_summaries: tuple[SessionSummary, ...]

    @property
    def session_facts(self) -> tuple[MemberFact, ...]:
        """This session's facts still awaiting consolidation."""
        return tuple(f for f in self.facts if f.session_id == self.session.id and f.status == "provisional")
