from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.policy import Decision
from app.triage.engine import Triage

FactStatus = Literal["provisional", "consolidated", "verified"]
FactSource = Literal["user_report", "seeded", "clinician"]
FactKind = Literal["condition", "symptom", "observation", "medication", "allergy", "preference", "note"]


class Slots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_hours: float | None
    severity: Literal["mild", "moderate", "severe"] | None
    trend: Literal["better", "same", "worse"] | None
    temperature_c: float | None
    systolic_bp: int | None
    diastolic_bp: int | None


class QuotedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FactKind
    subject: str
    text: str  # exact quote from the current message, for the selected member


class RedFlag(BaseModel):
    """A danger sign that is an emergency whatever the complaint, grouped by body system."""

    model_config = ConfigDict(extra="forbid")

    system: Literal["breathing", "circulation", "neurological", "allergic_reaction", "bleeding"]
    sign: str  # short label, e.g. "struggling to breathe"
    quote: str


class DoctorFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: Literal["severe_pain", "dehydration", "rash_with_fever", "chronic_change"]
    quote: str


class Screen(BaseModel):
    """Safety screen of the current message; needs no member or episode, so it never waits on them."""

    model_config = ConfigDict(extra="forbid")

    red_flags: list[RedFlag]
    doctor_flags: list[DoctorFlag]
    doctor_requested: bool


class MemberResolution(BaseModel):
    """Member agent's final answer, submitted as a tool call."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "none", "unresolved"]  # none: not about a family member's health
    member_id: str | None
    reason: str
    needed_info: str | None  # unresolved only: what would settle it
    responder_instructions: str | None  # unresolved only: how to ask


class EpisodeResolution(BaseModel):
    """Episode agent's final answer, submitted as a tool call."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["existing", "new", "none", "unresolved"]  # none: no personal health episode
    episode_id: str | None  # existing only
    reason: str
    needed_info: str | None
    responder_instructions: str | None


class Answer(BaseModel):
    """The user's answer to one protocol question, with the words that give it."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    value: Literal["present", "absent", "resolved", "unknown"] = Field(description=(
        "Whether the question's finding is there now, whatever wording Q used to ask: present, absent, "
        "resolved when it was there earlier in this illness and has gone, or unknown when the user doesn't know"))
    quote: str  # exact words from the current message


class AssociatedSymptom(BaseModel):
    """Another symptom of the same illness, e.g. the fever that came with a cough, and whether it is still there."""

    model_config = ConfigDict(extra="forbid")

    symptom: str  # short snake_case label, e.g. fever
    status: Literal["active", "resolved"]
    quote: str  # exact words from the current message


class Observation(BaseModel):
    """What the message says about the resolved member and episode."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["symptom_report", "update", "clarification", "general_question", "other"]
    complaint: str | None
    slots: Slots
    answers: list[Answer]
    associated: list[AssociatedSymptom]
    resolved: bool
    facts: list[QuotedFact]


class Extraction(BaseModel):
    """Merged turn findings that policy and persistence read; assembled in code, not by one model call."""

    model_config = ConfigDict(extra="forbid")

    member_id: str | None
    episode_id: str | None  # existing ID; null for a new episode or general question
    intent: Literal["symptom_report", "update", "clarification", "general_question", "doctor_request", "other"]
    complaint: str | None
    slots: Slots
    red_flags: list[RedFlag]
    doctor_flags: list[DoctorFlag]
    resolved: bool
    facts: list[QuotedFact]


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


class TurnState(TypedDict, total=False):
    """Scratch state for one graph run. Durable memory lives in Postgres, not in graph state."""

    household_id: str
    session_id: str
    turn_id: str
    text: str
    profile_id: str | None
    context: SessionContext
    screen: Screen
    member_resolution: MemberResolution
    episode_resolution: EpisodeResolution
    clarification: Clarification | None
    member_id: str | None  # resolved member, before observation
    extraction: Extraction
    selected_episode: Episode | None
    episode_id: str | None
    related_episode_id: str | None
    protocol_id: str | None
    findings: Mapping[str, Any]  # merged; written back whole
    associated: Mapping[str, Any]  # the episode's other symptoms, merged; written back whole
    triage: Triage | None
    replayed: bool
    decision: Decision
    decision_reason: str
    urgency: str | None  # doctor decisions only: doctor_now, doctor_24h or doctor_routine
    handoff_path: str | None
    reply: str
    is_new_session: bool
    unanswered_last_time: bool  # the intake's next question was asked last turn and not answered


class TurnOutput(TypedDict, total=False):
    """What a graph run returns, and so what its trace shows as output: the outcome, not the scratch state."""

    member_id: str | None
    episode_id: str | None
    protocol_id: str | None
    decision: Decision
    decision_reason: str
    urgency: str | None
    handoff_path: str | None
    reply: str
    replayed: bool
    is_new_session: bool
